#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct Candidate {
    int row = -1;
    int col = -1;
    int e1 = -1;
    int e2 = -1;
    int obs1 = -1;
    int obs2 = -1;
    std::string profile;
};

struct Group {
    std::string read_id;
    std::string feature_id;
    std::string raw_umi;
    std::string corrected_umi;
    std::string sr_cb;
    int tier = -1;
    std::vector<Candidate> candidates;
};

struct Metrics {
    uint64_t candidate_rows = 0;
    uint64_t read_groups = 0;
    uint64_t sr_selected_minimum = 0;
    uint64_t sr_selected_nonminimum = 0;
    uint64_t sr_unassigned = 0;
    uint64_t sr_invalid_coordinate = 0;
    uint64_t disagreement_groups = 0;
    uint64_t overlap_groups = 0;
};

std::vector<std::string> split(const std::string& line, char delimiter = '\t') {
    std::vector<std::string> output;
    size_t begin = 0;
    while (true) {
        const size_t end = line.find(delimiter, begin);
        output.emplace_back(line.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) return output;
        begin = end + 1;
    }
}

bool getline_gz(gzFile handle, std::string& line) {
    line.clear();
    char buffer[8192];
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

void gz_write(gzFile handle, const std::string& text) {
    if (gzwrite(handle, text.data(), static_cast<unsigned int>(text.size())) !=
        static_cast<int>(text.size())) {
        throw std::runtime_error("gzip write failed");
    }
}

std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open " + path);
    std::vector<std::string> output;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) output.push_back(line);
    }
    return output;
}

std::pair<int, int> parse_unit(const std::string& value) {
    const std::string prefix = "s_002um_";
    if (value.compare(0, prefix.size(), prefix) != 0) return {-1, -1};
    const size_t separator = value.find('_', prefix.size());
    if (separator == std::string::npos) return {-1, -1};
    const size_t dash = value.find('-', separator + 1);
    try {
        return {
            std::stoi(value.substr(prefix.size(), separator - prefix.size())),
            std::stoi(value.substr(separator + 1, dash - separator - 1)),
        };
    } catch (...) {
        return {-1, -1};
    }
}

uint64_t coordinate_key(int row, int col) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) |
           static_cast<uint32_t>(col);
}

std::string shape_signature(const Group& group) {
    std::set<std::pair<int, int>> shapes;
    for (const Candidate& candidate : group.candidates) {
        shapes.emplace(candidate.e1, candidate.e2);
    }
    std::ostringstream output;
    bool first = true;
    for (const auto& shape : shapes) {
        if (!first) output << ';';
        output << shape.first << '+' << shape.second;
        first = false;
    }
    return output.str();
}

std::string candidate_ledger(const Group& group) {
    std::ostringstream output;
    for (size_t i = 0; i < group.candidates.size(); ++i) {
        if (i) output << ';';
        const Candidate& candidate = group.candidates[i];
        output << candidate.row << ',' << candidate.col << ',' << candidate.e1 << ','
               << candidate.e2 << ',' << candidate.obs1 << ',' << candidate.obs2 << ','
               << candidate.profile;
    }
    return output.str();
}

bool short_bc2_overlap(
    const Group& group,
    const std::pair<int, int>& sr,
    const std::vector<std::string>& bc2,
    const std::vector<std::vector<int>>& suffix_rows) {
    if (sr.first < 0) return false;
    for (const Candidate& candidate : group.candidates) {
        if (candidate.e1 != 1 || candidate.e2 != 0 || candidate.obs2 != 15 ||
            candidate.row < 0 || static_cast<size_t>(candidate.row) >= bc2.size() ||
            bc2[candidate.row].size() != 15) {
            continue;
        }
        for (int suffix_row : suffix_rows[candidate.row]) {
            if (suffix_row == sr.first) return true;
        }
    }
    return false;
}

void process_group(const Group& group,
                   const std::vector<std::string>& bc2,
                   const std::vector<std::vector<int>>& suffix_rows,
                   gzFile output,
                   Metrics& metrics) {
    if (group.read_id.empty()) return;
    ++metrics.read_groups;
    std::unordered_set<uint64_t> minimum;
    for (const Candidate& candidate : group.candidates) {
        minimum.insert(coordinate_key(candidate.row, candidate.col));
    }
    const std::pair<int, int> sr = parse_unit(group.sr_cb);
    std::string status;
    if (group.sr_cb.empty()) {
        status = "sr_unassigned";
        ++metrics.sr_unassigned;
    } else if (sr.first < 0 || sr.second < 0) {
        status = "sr_invalid_coordinate";
        ++metrics.sr_invalid_coordinate;
    } else if (minimum.count(coordinate_key(sr.first, sr.second))) {
        status = "sr_selected_minimum";
        ++metrics.sr_selected_minimum;
    } else {
        status = "sr_selected_nonminimum";
        ++metrics.sr_selected_nonminimum;
    }
    if (status == "sr_selected_minimum") return;

    ++metrics.disagreement_groups;
    const bool overlap = status == "sr_selected_nonminimum" &&
        short_bc2_overlap(group, sr, bc2, suffix_rows);
    if (overlap) ++metrics.overlap_groups;
    std::ostringstream row;
    row << group.read_id << '\t' << group.feature_id << '\t' << group.raw_umi << '\t'
        << group.corrected_umi << '\t' << group.sr_cb << '\t' << status << '\t'
        << group.tier << '\t' << group.candidates.size() << '\t'
        << shape_signature(group) << '\t' << (overlap ? 1 : 0) << '\t'
        << candidate_ledger(group) << '\n';
    gz_write(output, row.str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string input_path;
        std::string bc2_path;
        std::string output_path;
        std::string metrics_path;
        std::string shard;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + option);
            const std::string value = argv[++i];
            if (option == "--input") input_path = value;
            else if (option == "--bc2") bc2_path = value;
            else if (option == "--output") output_path = value;
            else if (option == "--metrics") metrics_path = value;
            else if (option == "--shard") shard = value;
            else throw std::runtime_error("unknown option: " + option);
        }
        if (input_path.empty() || bc2_path.empty() || output_path.empty() ||
            metrics_path.empty() || shard.empty()) {
            throw std::runtime_error("missing required argument");
        }

        const std::vector<std::string> bc2 = read_lines(bc2_path);
        std::unordered_map<std::string, std::vector<int>> exact;
        for (size_t i = 0; i < bc2.size(); ++i) exact[bc2[i]].push_back(static_cast<int>(i));
        std::vector<std::vector<int>> suffix_rows(bc2.size());
        for (size_t i = 0; i < bc2.size(); ++i) {
            if (bc2[i].size() != 15) continue;
            auto found = exact.find(bc2[i].substr(1));
            if (found != exact.end()) suffix_rows[i] = found->second;
        }

        gzFile input = gzopen(input_path.c_str(), "rb");
        if (!input) throw std::runtime_error("cannot open input");
        gzFile output = gzopen(output_path.c_str(), "wb6");
        if (!output) throw std::runtime_error("cannot open output");
        gz_write(output,
            "read_id\tfeature_id\traw_umi\tsr_corrected_umi\tsr_cb\tstatus\t"
            "min_tier\tcandidate_count\tshape_signature\tshort_bc2_overlap\t"
            "candidates\n");

        std::string line;
        if (!getline_gz(input, line)) throw std::runtime_error("empty input");
        const std::vector<std::string> expected = {
            "read_id", "feature_id", "raw_umi", "sr_corrected_umi", "sr_cb",
            "min_tier", "candidate_count", "row2", "col2", "bc1_edit",
            "bc2_edit", "bc1_obs_len", "bc2_obs_len", "tier_profile",
            "log_sequence_likelihood",
        };
        if (split(line) != expected) throw std::runtime_error("unexpected input schema");

        Metrics metrics;
        Group group;
        while (getline_gz(input, line)) {
            if (line.empty()) continue;
            const std::vector<std::string> fields = split(line);
            if (fields.size() != expected.size()) throw std::runtime_error("malformed row");
            if (!group.read_id.empty() && fields[0] != group.read_id) {
                process_group(group, bc2, suffix_rows, output, metrics);
                group = Group{};
            }
            if (group.read_id.empty()) {
                group.read_id = fields[0];
                group.feature_id = fields[1];
                group.raw_umi = fields[2];
                group.corrected_umi = fields[3];
                group.sr_cb = fields[4];
                group.tier = std::stoi(fields[5]);
            }
            group.candidates.push_back({
                std::stoi(fields[7]), std::stoi(fields[8]),
                std::stoi(fields[9]), std::stoi(fields[10]),
                std::stoi(fields[11]), std::stoi(fields[12]), fields[13],
            });
            ++metrics.candidate_rows;
        }
        process_group(group, bc2, suffix_rows, output, metrics);
        gzclose(input);
        gzclose(output);

        std::ofstream meta(metrics_path);
        meta << "{\n"
             << "  \"shard\": \"" << shard << "\",\n"
             << "  \"candidate_rows\": " << metrics.candidate_rows << ",\n"
             << "  \"read_groups\": " << metrics.read_groups << ",\n"
             << "  \"sr_selected_minimum\": " << metrics.sr_selected_minimum << ",\n"
             << "  \"sr_selected_nonminimum\": " << metrics.sr_selected_nonminimum << ",\n"
             << "  \"sr_unassigned\": " << metrics.sr_unassigned << ",\n"
             << "  \"sr_invalid_coordinate\": " << metrics.sr_invalid_coordinate << ",\n"
             << "  \"disagreement_groups\": " << metrics.disagreement_groups << ",\n"
             << "  \"short_bc2_overlap_groups\": " << metrics.overlap_groups << "\n"
             << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 2;
    }
}
