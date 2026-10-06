#include <htslib/sam.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Coordinate {
    int row = -1;
    int col = -1;
    bool valid() const { return row >= 0 && col >= 0; }
};

struct Candidate {
    Coordinate coordinate;
    int e1 = -1;
    int e2 = -1;
    int obs1 = -1;
    int obs2 = -1;
};

struct Disagreement {
    std::string read_id;
    Coordinate sr;
    int tier = -1;
    int candidate_count = 0;
    std::string shape;
    bool short_overlap = false;
    std::vector<Candidate> candidates;
    bool found = false;
};

struct Split {
    int e1 = 999;
    int e2 = 999;
    int sum = 999;
    int position = -1;
};

struct Metrics {
    uint64_t bam_records = 0;
    uint64_t primary_records = 0;
    uint64_t disagreements_loaded = 0;
    uint64_t disagreements_found = 0;
    uint64_t duplicate_bam_records = 0;
    uint64_t missing_cr = 0;
    uint64_t candidate_count_mismatch = 0;
    uint64_t star_split_validation_failures = 0;
    std::map<std::string, uint64_t> tier_relation;
    std::map<std::string, uint64_t> whole_relation;
    std::map<std::string, uint64_t> sr_edit_class;
    std::map<std::string, uint64_t> parent_relation;
    std::map<int, uint64_t> tier;
    std::map<int, uint64_t> candidate_count;
    std::map<int, uint64_t> sr_split_sum;
    std::map<int, uint64_t> sr_split_max_half;
    std::map<int, uint64_t> sr_split_offset;
    std::map<std::pair<int, int>, uint64_t> sr_split_pair;
};

uint64_t hash_read(std::string_view value) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char ch : value) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool gz_line(gzFile handle, std::string& line) {
    line.clear();
    char buffer[16384];
    while (true) {
        char* got = gzgets(handle, buffer, sizeof(buffer));
        if (!got) return !line.empty();
        line += got;
        if (!line.empty() && line.back() == '\n') {
            line.pop_back();
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return true;
        }
        if (gzeof(handle)) return true;
    }
}

std::vector<std::string_view> split_view(std::string_view value, char delimiter) {
    std::vector<std::string_view> output;
    size_t begin = 0;
    while (true) {
        const size_t end = value.find(delimiter, begin);
        output.emplace_back(value.substr(begin, end == std::string_view::npos ? end : end - begin));
        if (end == std::string_view::npos) return output;
        begin = end + 1;
    }
}

Coordinate unit(std::string_view value) {
    constexpr std::string_view prefix = "s_002um_";
    if (value.substr(0, prefix.size()) != prefix) return {};
    size_t position = prefix.size();
    auto number = [&](int& out) {
        if (position >= value.size() || value[position] < '0' || value[position] > '9') return false;
        out = 0;
        while (position < value.size() && value[position] >= '0' && value[position] <= '9') {
            out = out * 10 + (value[position++] - '0');
        }
        return true;
    };
    Coordinate coordinate;
    if (!number(coordinate.row) || position >= value.size() || value[position++] != '_' ||
        !number(coordinate.col)) return {};
    if (position < value.size() && value.substr(position) != "-1") return {};
    return coordinate;
}

std::vector<std::string> lines(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open oligos: " + path);
    std::vector<std::string> output;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) output.push_back(line);
    }
    return output;
}

int edit(std::string_view observed, std::string_view target) {
    std::array<int, 64> previous{}, current{};
    if (target.size() >= previous.size()) throw std::runtime_error("sequence too long");
    for (size_t j = 0; j <= target.size(); ++j) previous[j] = static_cast<int>(j);
    for (size_t i = 1; i <= observed.size(); ++i) {
        current[0] = static_cast<int>(i);
        for (size_t j = 1; j <= target.size(); ++j) {
            current[j] = std::min({
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (observed[i - 1] != target[j - 1]),
            });
        }
        std::swap(previous, current);
    }
    return previous[target.size()];
}

Split best_split(std::string_view raw, std::string_view bc1, std::string_view bc2) {
    Split best;
    for (size_t position = 1; position < raw.size(); ++position) {
        const int e1 = edit(raw.substr(0, position), bc1);
        const int e2 = edit(raw.substr(position), bc2);
        const int sum = e1 + e2;
        const auto key = std::make_tuple(sum, std::max(e1, e2), e1, static_cast<int>(position));
        const auto old = std::make_tuple(best.sum, std::max(best.e1, best.e2), best.e1, best.position);
        if (key < old) best = {e1, e2, sum, static_cast<int>(position)};
    }
    return best;
}

std::string compare(int left, int right) {
    if (left < right) return "lower";
    if (left > right) return "greater";
    return "equal";
}

std::string half_class(int observed_length, int target_length, int distance) {
    if (distance == 0) return "exact";
    const int delta = observed_length - target_length;
    if (delta < 0) return distance == -delta ? "deletion_frame" : "deletion_mixed";
    if (delta > 0) return distance == delta ? "insertion_frame" : "insertion_mixed";
    return "same_length_edit";
}

Candidate parse_candidate(std::string_view value) {
    const auto fields = split_view(value, ',');
    if (fields.size() != 7) throw std::runtime_error("candidate record does not have seven fields");
    return {
        {std::stoi(std::string(fields[0])), std::stoi(std::string(fields[1]))},
        std::stoi(std::string(fields[2])), std::stoi(std::string(fields[3])),
        std::stoi(std::string(fields[4])), std::stoi(std::string(fields[5])),
    };
}

using Index = std::unordered_map<uint64_t, std::vector<Disagreement>>;

Index load_disagreements(const std::string& directory, std::vector<uint64_t>& bloom, Metrics& metrics) {
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gz") paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    Index index;
    index.reserve(4000000);
    for (const auto& path : paths) {
        gzFile input = gzopen(path.c_str(), "rb");
        if (!input) throw std::runtime_error("cannot open disagreement shard");
        std::string line;
        if (!gz_line(input, line)) throw std::runtime_error("empty disagreement shard");
        while (gz_line(input, line)) {
            if (line.empty()) continue;
            const auto fields = split_view(line, '\t');
            if (fields.size() != 11) throw std::runtime_error("disagreement row does not have eleven fields");
            if (fields[5] != "sr_selected_nonminimum") continue;
            Disagreement item;
            item.read_id = std::string(fields[0]);
            item.sr = unit(fields[4]);
            item.tier = std::stoi(std::string(fields[6]));
            item.candidate_count = std::stoi(std::string(fields[7]));
            item.shape = std::string(fields[8]);
            item.short_overlap = fields[9] == "1";
            if (!item.sr.valid()) throw std::runtime_error("invalid SR coordinate");
            for (std::string_view candidate : split_view(fields[10], ';')) {
                item.candidates.push_back(parse_candidate(candidate));
            }
            if (static_cast<int>(item.candidates.size()) != item.candidate_count) {
                ++metrics.candidate_count_mismatch;
            }
            const uint64_t hash = hash_read(item.read_id);
            auto& bucket = index[hash];
            for (const auto& old : bucket) {
                if (old.read_id == item.read_id) throw std::runtime_error("duplicate disagreement read id");
            }
            bucket.push_back(std::move(item));
            const uint64_t bit = hash & ((1ULL << 28) - 1);
            bloom[bit >> 6] |= 1ULL << (bit & 63);
            ++metrics.disagreements_loaded;
        }
        gzclose(input);
    }
    return index;
}

Disagreement* find(Index& index, const std::vector<uint64_t>& bloom, std::string_view id) {
    const uint64_t hash = hash_read(id);
    const uint64_t bit = hash & ((1ULL << 28) - 1);
    if (!(bloom[bit >> 6] & (1ULL << (bit & 63)))) return nullptr;
    auto found = index.find(hash);
    if (found == index.end()) return nullptr;
    for (auto& item : found->second) if (std::string_view(item.read_id) == id) return &item;
    return nullptr;
}

const char* aux_string(bam1_t* record, const char tag[2]) {
    uint8_t* value = bam_aux_get(record, tag);
    if (!value || (*value != 'Z' && *value != 'H')) return nullptr;
    return bam_aux2Z(value);
}

template <typename Key>
void json_hist(std::ostream& output, const std::map<Key, uint64_t>& values) {
    output << '{';
    bool first = true;
    for (const auto& [key, value] : values) {
        if (!first) output << ',';
        output << "\n    \"" << key << "\": " << value;
        first = false;
    }
    if (!values.empty()) output << '\n';
    output << "  }";
}

void json_pairs(std::ostream& output, const std::map<std::pair<int, int>, uint64_t>& values) {
    output << '{';
    bool first = true;
    for (const auto& [key, value] : values) {
        if (!first) output << ',';
        output << "\n    \"" << key.first << '+' << key.second << "\": " << value;
        first = false;
    }
    if (!values.empty()) output << '\n';
    output << "  }";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string bam_path, disagreement_dir, bc1_path, bc2_path, output_path, metrics_path;
        int threads = 8;
        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];
            if (index + 1 >= argc) throw std::runtime_error("missing argument value");
            const std::string value = argv[++index];
            if (option == "--bam") bam_path = value;
            else if (option == "--disagreements-dir") disagreement_dir = value;
            else if (option == "--bc1") bc1_path = value;
            else if (option == "--bc2") bc2_path = value;
            else if (option == "--output") output_path = value;
            else if (option == "--metrics") metrics_path = value;
            else if (option == "--threads") threads = std::stoi(value);
            else throw std::runtime_error("unknown argument: " + option);
        }
        if (bam_path.empty() || disagreement_dir.empty() || bc1_path.empty() || bc2_path.empty() ||
            output_path.empty() || metrics_path.empty()) throw std::runtime_error("missing required argument");

        const auto bc1 = lines(bc1_path);
        const auto bc2 = lines(bc2_path);
        Metrics metrics;
        std::vector<uint64_t> bloom(1ULL << 22, 0);
        Index disagreements = load_disagreements(disagreement_dir, bloom, metrics);

        gzFile output = gzopen(output_path.c_str(), "wb6");
        if (!output) throw std::runtime_error("cannot open output ledger");
        const std::string header =
            "read_id\tmin_tier\tcandidate_count\tshape_signature\tshort_bc2_overlap\traw_cr\t"
            "sr_row\tsr_col\tcandidate_min_whole_edit\tsr_whole_edit\tsr_best_bc1_edit\t"
            "sr_best_bc2_edit\tsr_best_sum\tsr_best_split\tsr_split_offset\tsr_edit_class\t"
            "sr_tier_relation\tsr_whole_relation\tsame_8um_parent\tsame_16um_parent\n";
        gzwrite(output, header.data(), header.size());

        samFile* input = sam_open(bam_path.c_str(), "r");
        if (!input) throw std::runtime_error("cannot open BAM");
        hts_set_threads(input, threads);
        bam_hdr_t* header_record = sam_hdr_read(input);
        bam1_t* record = bam_init1();
        if (!header_record || !record) throw std::runtime_error("BAM initialization failed");
        while (sam_read1(input, header_record, record) >= 0) {
            ++metrics.bam_records;
            if (record->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY)) continue;
            ++metrics.primary_records;
            Disagreement* item = find(disagreements, bloom, bam_get_qname(record));
            if (!item) continue;
            if (item->found) {
                ++metrics.duplicate_bam_records;
                continue;
            }
            item->found = true;
            ++metrics.disagreements_found;
            const char* raw_value = aux_string(record, "CR");
            if (!raw_value || !*raw_value) {
                ++metrics.missing_cr;
                continue;
            }
            const std::string_view raw(raw_value);
            if (static_cast<size_t>(item->sr.col) >= bc1.size() ||
                static_cast<size_t>(item->sr.row) >= bc2.size()) throw std::runtime_error("SR coordinate outside oligos");

            int minimum_whole = 999;
            bool same8 = false;
            bool same16 = false;
            for (const Candidate& candidate : item->candidates) {
                if (candidate.coordinate.col < 0 || candidate.coordinate.row < 0 ||
                    static_cast<size_t>(candidate.coordinate.col) >= bc1.size() ||
                    static_cast<size_t>(candidate.coordinate.row) >= bc2.size()) {
                    throw std::runtime_error("candidate coordinate outside oligos");
                }
                if (candidate.obs1 <= 0 || candidate.obs2 <= 0 ||
                    static_cast<size_t>(candidate.obs1 + candidate.obs2) != raw.size()) {
                    ++metrics.star_split_validation_failures;
                } else {
                    const int e1 = edit(raw.substr(0, candidate.obs1), bc1[candidate.coordinate.col]);
                    const int e2 = edit(raw.substr(candidate.obs1, candidate.obs2), bc2[candidate.coordinate.row]);
                    if (e1 != candidate.e1 || e2 != candidate.e2 || e1 + e2 != item->tier) {
                        ++metrics.star_split_validation_failures;
                    }
                }
                minimum_whole = std::min(
                    minimum_whole,
                    edit(raw, bc1[candidate.coordinate.col] + bc2[candidate.coordinate.row])
                );
                same8 = same8 || (
                    candidate.coordinate.row / 4 == item->sr.row / 4 &&
                    candidate.coordinate.col / 4 == item->sr.col / 4
                );
                same16 = same16 || (
                    candidate.coordinate.row / 8 == item->sr.row / 8 &&
                    candidate.coordinate.col / 8 == item->sr.col / 8
                );
            }

            const std::string& sr_bc1 = bc1[item->sr.col];
            const std::string& sr_bc2 = bc2[item->sr.row];
            const int sr_whole = edit(raw, sr_bc1 + sr_bc2);
            const Split sr = best_split(raw, sr_bc1, sr_bc2);
            const int offset = sr.position - static_cast<int>(sr_bc1.size());
            const std::string edit_class =
                half_class(sr.position, sr_bc1.size(), sr.e1) + "+" +
                half_class(static_cast<int>(raw.size()) - sr.position, sr_bc2.size(), sr.e2);
            const std::string tier_relation = compare(sr.sum, item->tier);
            const std::string whole_relation = compare(sr_whole, minimum_whole);
            const std::string parent_relation = same8 ? "same_8um" : (same16 ? "same_16um_only" : "cross_16um");

            ++metrics.tier[item->tier];
            ++metrics.candidate_count[item->candidate_count];
            ++metrics.sr_split_sum[sr.sum];
            ++metrics.sr_split_max_half[std::max(sr.e1, sr.e2)];
            ++metrics.sr_split_offset[offset];
            ++metrics.sr_split_pair[{sr.e1, sr.e2}];
            ++metrics.tier_relation[tier_relation];
            ++metrics.whole_relation[whole_relation];
            ++metrics.sr_edit_class[edit_class];
            ++metrics.parent_relation[parent_relation];

            std::ostringstream row;
            row << item->read_id << '\t' << item->tier << '\t' << item->candidate_count << '\t'
                << item->shape << '\t' << (item->short_overlap ? 1 : 0) << '\t' << raw << '\t'
                << item->sr.row << '\t' << item->sr.col << '\t' << minimum_whole << '\t'
                << sr_whole << '\t' << sr.e1 << '\t' << sr.e2 << '\t' << sr.sum << '\t'
                << sr.position << '\t' << offset << '\t' << edit_class << '\t' << tier_relation
                << '\t' << whole_relation << '\t' << (same8 ? 1 : 0) << '\t' << (same16 ? 1 : 0) << '\n';
            const std::string text = row.str();
            gzwrite(output, text.data(), text.size());
        }
        bam_destroy1(record);
        bam_hdr_destroy(header_record);
        sam_close(input);
        gzclose(output);

        std::ofstream meta(metrics_path);
        meta << "{\n"
             << "  \"bam_records\": " << metrics.bam_records << ",\n"
             << "  \"primary_records\": " << metrics.primary_records << ",\n"
             << "  \"disagreements_loaded\": " << metrics.disagreements_loaded << ",\n"
             << "  \"disagreements_found\": " << metrics.disagreements_found << ",\n"
             << "  \"disagreements_not_found\": " << metrics.disagreements_loaded - metrics.disagreements_found << ",\n"
             << "  \"duplicate_bam_records\": " << metrics.duplicate_bam_records << ",\n"
             << "  \"missing_cr\": " << metrics.missing_cr << ",\n"
             << "  \"candidate_count_mismatch\": " << metrics.candidate_count_mismatch << ",\n"
             << "  \"star_split_validation_failures\": " << metrics.star_split_validation_failures << ",\n"
             << "  \"tier_histogram\": "; json_hist(meta, metrics.tier); meta << ",\n"
             << "  \"candidate_count_histogram\": "; json_hist(meta, metrics.candidate_count); meta << ",\n"
             << "  \"sr_split_sum_histogram\": "; json_hist(meta, metrics.sr_split_sum); meta << ",\n"
             << "  \"sr_split_max_half_histogram\": "; json_hist(meta, metrics.sr_split_max_half); meta << ",\n"
             << "  \"sr_split_offset_histogram\": "; json_hist(meta, metrics.sr_split_offset); meta << ",\n"
             << "  \"sr_split_pair_histogram\": "; json_pairs(meta, metrics.sr_split_pair); meta << ",\n"
             << "  \"sr_tier_relation\": "; json_hist(meta, metrics.tier_relation); meta << ",\n"
             << "  \"sr_whole_relation\": "; json_hist(meta, metrics.whole_relation); meta << ",\n"
             << "  \"sr_edit_class\": "; json_hist(meta, metrics.sr_edit_class); meta << ",\n"
             << "  \"parent_relation\": "; json_hist(meta, metrics.parent_relation); meta << "\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 2;
    }
}
