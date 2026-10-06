// Opt-in, fixed-offset Visium HD candidate preserver for natural BAM records.
//
// This intentionally includes (without modifying) the frozen compatibility
// decoder so the fast, tested E0/E1/E2 half-barcode lookup is shared while the
// production decoder's single-assignment contract remains unchanged.
//
// Build:
//   g++ -std=c++17 -O3 -pthread native/hd_candidate_preserving_reference.cpp \
//       -lhts -lz -o hd_candidate_preserving_reference

#define main hd_r1_anchored_decode_compatibility_main
#include "hd_r1_anchored_decode.cpp"
#undef main

#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <unordered_set>

#include <htslib/sam.h>

namespace {

struct ReferenceArgs {
    std::string bam;
    std::string bc1_oligos;
    std::string bc2_oligos;
    std::string out_dir;
    int shards = 256;
    int bam_threads = 4;
    int compute_threads = std::max(1u, std::thread::hardware_concurrency());
    size_t batch_size = 100000;
    long read_limit = 0;
    int phred_gap_q = 30;
    int phred_missing_q = 30;
    bool omit_candidate_shards = false;
};

struct PreservedCandidate {
    int row2 = -1;
    int col2 = -1;
    int tier = -1;
    int bc1_edit = -1;
    int bc2_edit = -1;
    int bc1_obs_len = -1;
    int bc2_obs_len = -1;
    bool mixed_profile = false;
    double log_sequence_likelihood = -std::numeric_limits<double>::infinity();
};

struct WorkRecord {
    std::string read_id;
    std::string feature;
    std::string raw_umi;
    std::string corrected_umi;
    std::string sr_cb;
    std::string sr_unit_2um;
    std::string observed;
    std::string qualities;
    std::vector<PreservedCandidate> candidates;
    std::string sr_line;
    std::string candidate_lines;
};

struct Counters {
    uint64_t bam_records = 0;
    uint64_t primary_records = 0;
    uint64_t secondary_or_supplementary = 0;
    uint64_t unmapped = 0;
    uint64_t missing_or_zero_xf = 0;
    uint64_t missing_feature = 0;
    uint64_t multiple_feature = 0;
    uint64_t missing_raw_umi = 0;
    uint64_t missing_corrected_umi = 0;
    uint64_t shared_eligible = 0;
    uint64_t missing_cr = 0;
    uint64_t missing_cy = 0;
    uint64_t missing_sr_cb = 0;
    uint64_t missing_sr_unit_2um = 0;
    uint64_t star_candidate_reads = 0;
    uint64_t star_no_candidate_reads = 0;
    uint64_t candidate_rows = 0;
    uint64_t tier_reads[5] = {0, 0, 0, 0, 0};
};

void reference_usage() {
    std::cerr
        << "Usage: hd_candidate_preserving_reference --bam FILE --bc1-oligos FILE "
           "--bc2-oligos FILE --out-dir DIR [options]\n"
        << "  --shards N             Feature-hash shards (default: 256)\n"
        << "  --bam-threads N        BAM decompression threads (default: 4)\n"
        << "  --threads N            Candidate compute threads (default: hardware)\n"
        << "  --batch-size N         Eligible records per compute batch (default: 100000)\n"
        << "  --read-limit N         Stop after N primary BAM records (default: all)\n"
        << "  --phred-gap-q N        Candidate-only gap Q (default: 30)\n"
        << "  --phred-missing-q N    Missing CY Q (default: 30)\n"
        << "  --omit-candidate-shards  Emit only the SR read shards and summary\n";
}

ReferenceArgs parse_reference_args(int argc, char** argv) {
    ReferenceArgs args;
    auto take = [&](int& i, const std::string& option) -> std::string {
        if (++i >= argc) throw std::runtime_error("missing value for " + option);
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--bam") args.bam = take(i, option);
        else if (option == "--bc1-oligos") args.bc1_oligos = take(i, option);
        else if (option == "--bc2-oligos") args.bc2_oligos = take(i, option);
        else if (option == "--out-dir") args.out_dir = take(i, option);
        else if (option == "--shards") args.shards = std::stoi(take(i, option));
        else if (option == "--bam-threads") args.bam_threads = std::stoi(take(i, option));
        else if (option == "--threads") args.compute_threads = std::stoi(take(i, option));
        else if (option == "--batch-size") args.batch_size = std::stoull(take(i, option));
        else if (option == "--read-limit") args.read_limit = std::stol(take(i, option));
        else if (option == "--phred-gap-q") args.phred_gap_q = std::stoi(take(i, option));
        else if (option == "--phred-missing-q") args.phred_missing_q = std::stoi(take(i, option));
        else if (option == "--omit-candidate-shards") args.omit_candidate_shards = true;
        else if (option == "--help" || option == "-h") {
            reference_usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }
    if (args.bam.empty() || args.bc1_oligos.empty() || args.bc2_oligos.empty() ||
        args.out_dir.empty()) {
        throw std::runtime_error("--bam, --bc1-oligos, --bc2-oligos, and --out-dir are required");
    }
    if (args.shards < 1 || args.shards > 4096 || args.bam_threads < 1 ||
        args.compute_threads < 1 || args.batch_size < 1 ||
        args.read_limit < 0 || args.phred_gap_q < 0 || args.phred_missing_q < 0) {
        throw std::runtime_error("invalid numeric option");
    }
    return args;
}

uint64_t fnv1a64(const std::string& value) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char byte : value) {
        hash ^= static_cast<uint64_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

class GzipShards {
  public:
    GzipShards(const std::filesystem::path& dir, int count, const std::string& header)
        : count_(count), paths_(static_cast<size_t>(count)), handles_(static_cast<size_t>(count), nullptr) {
        std::filesystem::create_directories(dir);
        const int width = std::max(3, static_cast<int>(std::to_string(count - 1).size()));
        for (int i = 0; i < count; ++i) {
            std::ostringstream name;
            name << "part-" << std::setw(width) << std::setfill('0') << i << ".tsv.gz";
            paths_[static_cast<size_t>(i)] = dir / name.str();
            gzFile handle = gzopen(paths_[static_cast<size_t>(i)].c_str(), "wb1");
            if (!handle) throw std::runtime_error("cannot open " + paths_[static_cast<size_t>(i)].string());
            gzbuffer(handle, 1 << 20);
            handles_[static_cast<size_t>(i)] = handle;
            write_raw(handle, header);
        }
    }

    ~GzipShards() {
        for (gzFile handle : handles_) {
            if (handle) gzclose(handle);
        }
    }

    void write(const std::string& feature, const std::string& line) {
        const size_t shard = static_cast<size_t>(fnv1a64(feature) % static_cast<uint64_t>(count_));
        write_raw(handles_[shard], line);
    }

  private:
    static void write_raw(gzFile handle, const std::string& value) {
        const int written = gzwrite(handle, value.data(), static_cast<unsigned int>(value.size()));
        if (written != static_cast<int>(value.size())) {
            int error_number = Z_OK;
            const char* message = gzerror(handle, &error_number);
            throw std::runtime_error(std::string("gzip write failed: ") + (message ? message : "unknown"));
        }
    }

    int count_;
    std::vector<std::filesystem::path> paths_;
    std::vector<gzFile> handles_;
};

const char* aux_string(const bam1_t* record, const char tag[2]) {
    uint8_t* value = bam_aux_get(record, tag);
    if (!value) return nullptr;
    const char type = static_cast<char>(*value);
    if (type != 'Z' && type != 'H') return nullptr;
    return bam_aux2Z(value);
}

bool has_multiple_features(const std::string& feature) {
    return feature.find(';') != std::string::npos || feature.find(',') != std::string::npos;
}

int quality_at(const std::string& qualities, size_t index, int missing_q) {
    if (index >= qualities.size()) return missing_q;
    return std::max(0, static_cast<int>(static_cast<unsigned char>(qualities[index])) - 33);
}

double error_probability(int phred) {
    return std::clamp(std::pow(10.0, -static_cast<double>(std::max(0, phred)) / 10.0),
                      1.0e-10, 0.75);
}

double base_log_likelihood(char observed, char candidate, int phred) {
    const char obs = static_cast<char>(std::toupper(static_cast<unsigned char>(observed)));
    const char cand = static_cast<char>(std::toupper(static_cast<unsigned char>(candidate)));
    if ((obs != 'A' && obs != 'C' && obs != 'G' && obs != 'T') ||
        (cand != 'A' && cand != 'C' && cand != 'G' && cand != 'T')) {
        return std::log(0.25);
    }
    const double error = error_probability(phred);
    return obs == cand ? std::log1p(-error) : std::log(error / 3.0);
}

double phred_alignment_log_likelihood(const std::string& observed,
                                      const std::string& qualities,
                                      const std::string& candidate,
                                      int gap_q, int missing_q) {
    std::vector<double> previous(candidate.size() + 1, 0.0);
    std::vector<double> current(candidate.size() + 1, 0.0);
    const double deletion = std::log(error_probability(gap_q));
    for (size_t j = 1; j <= candidate.size(); ++j) previous[j] = previous[j - 1] + deletion;
    for (size_t i = 1; i <= observed.size(); ++i) {
        const int q = quality_at(qualities, i - 1, missing_q);
        const double insertion = std::log(error_probability(q) / 4.0);
        current[0] = previous[0] + insertion;
        for (size_t j = 1; j <= candidate.size(); ++j) {
            current[j] = std::max({
                previous[j - 1] + base_log_likelihood(observed[i - 1], candidate[j - 1], q),
                previous[j] + insertion,
                current[j - 1] + deletion,
            });
        }
        previous.swap(current);
    }
    return previous[candidate.size()];
}

std::vector<PreservedCandidate> preserve_fixed_offset_candidates(
    const std::string& observed, const std::string& qualities, Config& cfg,
    const std::vector<int>& bc1_query_lengths,
    const std::unordered_set<int>& bc2_query_lengths,
    int gap_q, int missing_q) {
    std::unordered_map<uint64_t, PreservedCandidate> best_by_coord;
    int best_tier = std::numeric_limits<int>::max();
    for (int bc1_obs_len : bc1_query_lengths) {
        if (bc1_obs_len <= 0 || bc1_obs_len >= static_cast<int>(observed.size())) continue;
        const int bc2_obs_len = static_cast<int>(observed.size()) - bc1_obs_len;
        if (!bc2_query_lengths.count(bc2_obs_len)) continue;
        HitSpan bc1_hits = cfg.bc1_tiered_h2_lookup.lookup_span(observed.data(), bc1_obs_len);
        HitSpan bc2_hits = cfg.bc2_tiered_h2_lookup.lookup_span(
            observed.data() + bc1_obs_len, bc2_obs_len);
        if (!bc1_hits.found || !bc2_hits.found) continue;
        for (uint16_t i = 0; i < bc1_hits.count; ++i) {
            const PackedHit& left = bc1_hits.data[i];
            if (left.distance > 2) continue;
            for (uint16_t j = 0; j < bc2_hits.count; ++j) {
                const PackedHit& right = bc2_hits.data[j];
                if (right.distance > 2) continue;
                const int tier = static_cast<int>(left.distance) + static_cast<int>(right.distance);
                if (tier > 4 || tier > best_tier) continue;
                if (tier < best_tier) {
                    best_tier = tier;
                    best_by_coord.clear();
                }
                const int row2 = static_cast<int>(right.idx);
                const int col2 = static_cast<int>(left.idx);
                const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(row2)) << 32) |
                                     static_cast<uint32_t>(col2);
                PreservedCandidate candidate;
                candidate.row2 = row2;
                candidate.col2 = col2;
                candidate.tier = tier;
                candidate.bc1_edit = static_cast<int>(left.distance);
                candidate.bc2_edit = static_cast<int>(right.distance);
                candidate.bc1_obs_len = bc1_obs_len;
                candidate.bc2_obs_len = bc2_obs_len;
                auto found = best_by_coord.find(key);
                if (found == best_by_coord.end()) {
                    best_by_coord.emplace(key, candidate);
                } else {
                    PreservedCandidate& old = found->second;
                    if (old.bc1_edit != candidate.bc1_edit || old.bc2_edit != candidate.bc2_edit ||
                        old.bc1_obs_len != candidate.bc1_obs_len) {
                        old.mixed_profile = true;
                    }
                    const auto new_key = std::make_tuple(candidate.bc1_edit, candidate.bc2_edit,
                                                         candidate.bc1_obs_len);
                    const auto old_key = std::make_tuple(old.bc1_edit, old.bc2_edit,
                                                         old.bc1_obs_len);
                    if (new_key < old_key) {
                        const bool mixed = old.mixed_profile;
                        old = candidate;
                        old.mixed_profile = mixed;
                    }
                }
            }
        }
    }

    std::vector<PreservedCandidate> output;
    output.reserve(best_by_coord.size());
    for (auto& item : best_by_coord) {
        PreservedCandidate candidate = item.second;
        const std::string full = cfg.bc1_oligos[static_cast<size_t>(candidate.col2)] +
                                 cfg.bc2_oligos[static_cast<size_t>(candidate.row2)];
        candidate.log_sequence_likelihood = phred_alignment_log_likelihood(
            observed, qualities, full, gap_q, missing_q);
        output.push_back(candidate);
    }
    std::sort(output.begin(), output.end(), [](const PreservedCandidate& left,
                                               const PreservedCandidate& right) {
        return std::tie(left.row2, left.col2) < std::tie(right.row2, right.col2);
    });
    return output;
}

std::string tier_profile(const PreservedCandidate& candidate) {
    if (candidate.mixed_profile) return "mixed";
    if (candidate.tier == 2 && candidate.bc1_edit == 1 && candidate.bc2_edit == 1) {
        return "balanced_1_1";
    }
    if (candidate.tier == 2) return "single_half_2_0_or_0_2";
    return "direct";
}

void process_and_emit_batch(
    std::vector<WorkRecord>& batch,
    Config& cfg,
    const std::vector<int>& bc1_lengths,
    const std::unordered_set<int>& bc2_length_set,
    const ReferenceArgs& args,
    GzipShards& candidate_shards,
    GzipShards& sr_shards,
    Counters& counts) {
    if (batch.empty()) return;
    std::atomic<size_t> next{0};
    const int worker_count = std::min<int>(
        args.compute_threads, static_cast<int>(batch.size()));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));
    for (int thread_index = 0; thread_index < worker_count; ++thread_index) {
        workers.emplace_back([&]() {
            while (true) {
                const size_t index = next.fetch_add(1, std::memory_order_relaxed);
                if (index >= batch.size()) break;
                WorkRecord& record = batch[index];
                if (!record.observed.empty()) {
                    record.candidates = preserve_fixed_offset_candidates(
                        record.observed, record.qualities, cfg, bc1_lengths,
                        bc2_length_set, args.phred_gap_q, args.phred_missing_q);
                }
                const int min_tier =
                    record.candidates.empty() ? -1 : record.candidates.front().tier;
                std::ostringstream sr_line;
                sr_line << record.read_id << '\t' << record.feature << '\t'
                        << record.raw_umi << '\t' << record.corrected_umi << '\t'
                        << record.sr_cb << '\t' << record.sr_unit_2um << '\t'
                        << record.candidates.size() << '\t' << min_tier << '\n';
                record.sr_line = sr_line.str();
                if (!args.omit_candidate_shards) {
                    std::ostringstream candidate_lines;
                    for (const PreservedCandidate& candidate : record.candidates) {
                        candidate_lines << record.read_id << '\t' << record.feature << '\t'
                                        << record.raw_umi << '\t' << record.corrected_umi << '\t'
                                        << record.sr_cb << '\t' << candidate.tier << '\t'
                                        << record.candidates.size() << '\t' << candidate.row2 << '\t'
                                        << candidate.col2 << '\t' << candidate.bc1_edit << '\t'
                                        << candidate.bc2_edit << '\t' << candidate.bc1_obs_len << '\t'
                                        << candidate.bc2_obs_len << '\t' << tier_profile(candidate) << '\t'
                                        << std::setprecision(17)
                                        << candidate.log_sequence_likelihood << '\n';
                    }
                    record.candidate_lines = candidate_lines.str();
                }
            }
        });
    }
    for (std::thread& worker : workers) worker.join();

    for (const WorkRecord& record : batch) {
        const int min_tier = record.candidates.empty() ? -1 : record.candidates.front().tier;
        if (record.candidates.empty()) {
            ++counts.star_no_candidate_reads;
        } else {
            ++counts.star_candidate_reads;
            ++counts.tier_reads[min_tier];
            counts.candidate_rows += record.candidates.size();
        }

        sr_shards.write(record.feature, record.sr_line);
        if (!record.candidate_lines.empty()) {
            candidate_shards.write(record.feature, record.candidate_lines);
        }
    }
    batch.clear();
}

std::string reference_json_escape(const std::string& value) {
    std::string output;
    output.reserve(value.size());
    for (char byte : value) {
        switch (byte) {
            case '\\': output += "\\\\"; break;
            case '"': output += "\\\""; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default: output += byte; break;
        }
    }
    return output;
}

void write_summary(const ReferenceArgs& args, const Counters& counts) {
    const std::filesystem::path path = std::filesystem::path(args.out_dir) / "summary.json";
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out << "{\n"
        << "  \"schema\": \"star_spatial.hd.candidate_preserving_reference.v1\",\n"
        << "  \"bam\": \"" << reference_json_escape(args.bam) << "\",\n"
        << "  \"bc1_oligos\": \"" << reference_json_escape(args.bc1_oligos) << "\",\n"
        << "  \"bc2_oligos\": \"" << reference_json_escape(args.bc2_oligos) << "\",\n"
        << "  \"fixed_full_barcode_offset\": 0,\n"
        << "  \"candidate_shards_materialized\": "
        << (args.omit_candidate_shards ? "false" : "true") << ",\n"
        << "  \"max_half_edit\": 2,\n"
        << "  \"max_paired_edit\": 4,\n"
        << "  \"shards\": " << args.shards << ",\n"
        << "  \"compute_threads\": " << args.compute_threads << ",\n"
        << "  \"batch_size\": " << args.batch_size << ",\n"
        << "  \"phred_gap_q\": " << args.phred_gap_q << ",\n"
        << "  \"phred_missing_q\": " << args.phred_missing_q << ",\n"
        << "  \"prohibited_prior_flags\": {\"spatial\": false, \"image\": false, "
           "\"expression\": false, \"cell_type\": false, \"neighborhood\": false, "
           "\"graph\": false, \"space_ranger_cb\": false},\n"
        << "  \"counts\": {\n"
        << "    \"bam_records\": " << counts.bam_records << ",\n"
        << "    \"primary_records\": " << counts.primary_records << ",\n"
        << "    \"secondary_or_supplementary\": " << counts.secondary_or_supplementary << ",\n"
        << "    \"unmapped\": " << counts.unmapped << ",\n"
        << "    \"missing_or_zero_xf\": " << counts.missing_or_zero_xf << ",\n"
        << "    \"missing_feature\": " << counts.missing_feature << ",\n"
        << "    \"multiple_feature\": " << counts.multiple_feature << ",\n"
        << "    \"missing_raw_umi\": " << counts.missing_raw_umi << ",\n"
        << "    \"missing_corrected_umi\": " << counts.missing_corrected_umi << ",\n"
        << "    \"shared_eligible\": " << counts.shared_eligible << ",\n"
        << "    \"missing_cr\": " << counts.missing_cr << ",\n"
        << "    \"missing_cy\": " << counts.missing_cy << ",\n"
        << "    \"missing_sr_cb\": " << counts.missing_sr_cb << ",\n"
        << "    \"missing_sr_unit_2um\": " << counts.missing_sr_unit_2um << ",\n"
        << "    \"star_candidate_reads\": " << counts.star_candidate_reads << ",\n"
        << "    \"star_no_candidate_reads\": " << counts.star_no_candidate_reads << ",\n"
        << "    \"candidate_rows\": " << counts.candidate_rows << ",\n"
        << "    \"tier_reads\": {\"H0\": " << counts.tier_reads[0]
        << ", \"H1\": " << counts.tier_reads[1]
        << ", \"H2\": " << counts.tier_reads[2]
        << ", \"H3\": " << counts.tier_reads[3]
        << ", \"H4\": " << counts.tier_reads[4] << "}\n"
        << "  }\n"
        << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const ReferenceArgs args = parse_reference_args(argc, argv);
        std::filesystem::create_directories(args.out_dir);

        Config cfg;
        load_lines({args.bc1_oligos}, cfg.bc1_oligos);
        load_lines({args.bc2_oligos}, cfg.bc2_oligos);
        if (cfg.bc1_oligos.empty() || cfg.bc2_oligos.empty()) {
            throw std::runtime_error("empty oligo table");
        }
        cfg.grid_cols = static_cast<int>(cfg.bc1_oligos.size());
        cfg.grid_rows = static_cast<int>(cfg.bc2_oligos.size());
        cfg.bc1_len.reserve(cfg.bc1_oligos.size());
        cfg.bc2_len.reserve(cfg.bc2_oligos.size());
        for (const std::string& value : cfg.bc1_oligos) cfg.bc1_len.push_back(value.size());
        for (const std::string& value : cfg.bc2_oligos) cfg.bc2_len.push_back(value.size());
        const std::vector<int> bc1_lengths = edit_query_lengths(unique_sorted_lengths(cfg.bc1_len), 2);
        const std::vector<int> bc2_lengths = edit_query_lengths(unique_sorted_lengths(cfg.bc2_len), 2);
        const std::unordered_set<int> bc2_length_set(bc2_lengths.begin(), bc2_lengths.end());
        cfg.bc1_tiered_h2_lookup.build_best_edit_universe_for_query_lengths(
            cfg.bc1_oligos, bc1_lengths, 2, "reference_bc1_h0_h2",
            TargetSliceMode::Full, false, false);
        cfg.bc2_tiered_h2_lookup.build_best_edit_universe_for_query_lengths(
            cfg.bc2_oligos, bc2_lengths, 2, "reference_bc2_h0_h2",
            TargetSliceMode::Full, false, false);

        const std::string candidate_header =
            "read_id\tfeature_id\traw_umi\tsr_corrected_umi\tsr_cb\tmin_tier\t"
            "candidate_count\trow2\tcol2\tbc1_edit\tbc2_edit\tbc1_obs_len\t"
            "bc2_obs_len\ttier_profile\tlog_sequence_likelihood\n";
        const std::string sr_header =
            "read_id\tfeature_id\traw_umi\tsr_corrected_umi\tsr_cb\tsr_unit_2um\t"
            "star_candidate_count\tstar_min_tier\n";
        GzipShards candidate_shards(std::filesystem::path(args.out_dir) / "candidate_shards",
                                    args.shards, candidate_header);
        GzipShards sr_shards(std::filesystem::path(args.out_dir) / "sr_read_shards",
                             args.shards, sr_header);

        samFile* input = sam_open(args.bam.c_str(), "rb");
        if (!input) throw std::runtime_error("cannot open BAM: " + args.bam);
        hts_set_threads(input, args.bam_threads);
        bam_hdr_t* header = sam_hdr_read(input);
        bam1_t* record = bam_init1();
        if (!header || !record) throw std::runtime_error("cannot initialize BAM reader");

        Counters counts;
        std::vector<WorkRecord> batch;
        batch.reserve(args.batch_size);
        while (sam_read1(input, header, record) >= 0) {
            ++counts.bam_records;
            if (record->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY)) {
                ++counts.secondary_or_supplementary;
                continue;
            }
            if (args.read_limit > 0 &&
                counts.primary_records >= static_cast<uint64_t>(args.read_limit)) break;
            ++counts.primary_records;
            if (record->core.flag & BAM_FUNMAP) {
                ++counts.unmapped;
                continue;
            }
            uint8_t* xf_aux = bam_aux_get(record, "xf");
            if (!xf_aux || bam_aux2i(xf_aux) == 0) {
                ++counts.missing_or_zero_xf;
                continue;
            }
            const char* gx_value = aux_string(record, "GX");
            if (!gx_value || !*gx_value) {
                ++counts.missing_feature;
                continue;
            }
            const std::string feature(gx_value);
            if (has_multiple_features(feature)) {
                ++counts.multiple_feature;
                continue;
            }
            const char* ur_value = aux_string(record, "UR");
            if (!ur_value || !*ur_value) {
                ++counts.missing_raw_umi;
                continue;
            }
            const char* ub_value = aux_string(record, "UB");
            if (!ub_value || !*ub_value) {
                ++counts.missing_corrected_umi;
                continue;
            }
            ++counts.shared_eligible;
            const char* cb_value = aux_string(record, "CB");
            const char* sb_value = aux_string(record, "sb");
            if (!cb_value || !*cb_value) ++counts.missing_sr_cb;
            if (!sb_value || !*sb_value) ++counts.missing_sr_unit_2um;
            const char* cr_value = aux_string(record, "CR");
            const char* cy_value = aux_string(record, "CY");
            if (!cr_value || !*cr_value) ++counts.missing_cr;
            if (!cy_value || !*cy_value) ++counts.missing_cy;
            WorkRecord work;
            work.read_id = bam_get_qname(record);
            work.feature = feature;
            work.raw_umi = ur_value;
            work.corrected_umi = ub_value;
            work.sr_cb = cb_value ? cb_value : "";
            work.sr_unit_2um = sb_value ? sb_value : "";
            work.observed = cr_value ? cr_value : "";
            work.qualities = cy_value ? cy_value : "";
            batch.push_back(std::move(work));
            if (batch.size() >= args.batch_size) {
                process_and_emit_batch(batch, cfg, bc1_lengths, bc2_length_set,
                                       args, candidate_shards, sr_shards, counts);
            }
            if (counts.primary_records % 10000000ULL == 0) {
                std::cerr << "processed primary records=" << counts.primary_records
                          << " shared_eligible=" << counts.shared_eligible
                          << " candidate_reads=" << counts.star_candidate_reads << "\n";
            }
        }
        process_and_emit_batch(batch, cfg, bc1_lengths, bc2_length_set,
                               args, candidate_shards, sr_shards, counts);
        bam_destroy1(record);
        bam_hdr_destroy(header);
        sam_close(input);
        write_summary(args, counts);
        std::cerr << "complete primary_records=" << counts.primary_records
                  << " shared_eligible=" << counts.shared_eligible
                  << " candidate_reads=" << counts.star_candidate_reads
                  << " candidate_rows=" << counts.candidate_rows << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        reference_usage();
        return 2;
    }
}
