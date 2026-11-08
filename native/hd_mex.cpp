#include "hd_mex.hpp"

#include "hd_parallel.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <charconv>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace visium_hd::mex {

MappedFile::MappedFile(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat info;
    if (::fstat(fd, &info) != 0) {
        ::close(fd);
        throw std::runtime_error("cannot stat " + path);
    }
    size_ = static_cast<std::size_t>(info.st_size);
    if (size_ > 0) {
        void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("cannot map " + path);
        }
        ::madvise(mapped, size_, MADV_SEQUENTIAL);
        data_ = static_cast<const char*>(mapped);
    }
    ::close(fd);
}

MappedFile::~MappedFile() {
    if (data_) ::munmap(const_cast<char*>(data_), size_);
}

namespace {

const char* line_end(const char* at, const char* end) {
    const void* found = std::memchr(at, '\n', static_cast<std::size_t>(end - at));
    return found ? static_cast<const char*>(found) : end;
}

// Chunk boundaries at line starts, so that each chunk holds whole lines.
std::vector<const char*> line_chunks(const char* begin, const char* end, std::size_t chunks) {
    std::vector<const char*> out{begin};
    const std::size_t size = static_cast<std::size_t>(end - begin);
    for (std::size_t k = 1; k < chunks; ++k) {
        const char* at = begin + size * k / chunks;
        if (at <= out.back()) continue;
        at = line_end(at - 1, end);
        if (at < end) ++at;
        if (at > out.back() && at < end) out.push_back(at);
    }
    out.push_back(end);
    return out;
}

bool unsigned_field(const char*& at, const char* end, std::uint64_t& value) {
    while (at < end && (*at == ' ' || *at == '\t')) ++at;
    const std::from_chars_result r = std::from_chars(at, end, value);
    if (r.ec != std::errc() || r.ptr == at) return false;
    at = r.ptr;
    return true;
}

}  // namespace

Dimensions dimensions(const char* data, std::size_t size, const std::string& path) {
    const char* at = data;
    const char* end = data + size;
    const char* first = line_end(at, end);
    const std::string banner(at, first);
    if (banner.rfind("%%MatrixMarket matrix coordinate", 0) != 0) throw std::runtime_error("not a Matrix Market coordinate file: " + path);
    Dimensions d;
    d.real = banner.find(" real ") != std::string::npos;
    if (!d.real && banner.find(" integer ") == std::string::npos) throw std::runtime_error("unsupported Matrix Market field: " + path);
    at = first < end ? first + 1 : end;
    while (at < end) {
        const char* next = line_end(at, end);
        if (at < next && *at != '%') {
            const char* field = at;
            if (!unsigned_field(field, next, d.rows) || !unsigned_field(field, next, d.columns)
                || !unsigned_field(field, next, d.entries)) {
                throw std::runtime_error("bad Matrix Market dimensions: " + path);
            }
            d.data_offset = static_cast<std::size_t>((next < end ? next + 1 : end) - data);
            return d;
        }
        at = next < end ? next + 1 : end;
    }
    throw std::runtime_error("missing Matrix Market dimensions: " + path);
}

std::uint64_t matrix_columns(const std::string& path) {
    const MappedFile file(path);
    return dimensions(file.data(), file.size(), path).columns;
}

std::vector<double> column_totals(const std::string& path, int threads) {
    const MappedFile file(path);
    const Dimensions d = dimensions(file.data(), file.size(), path);
    const char* begin = file.data() + d.data_offset;
    const char* end = file.data() + file.size();
    const std::vector<const char*> bounds = line_chunks(begin, end, static_cast<std::size_t>(std::max(threads, 1)) * 16);
    struct Run {
        std::uint64_t column;
        double sum;
    };
    std::vector<std::vector<Run>> runs(bounds.size() - 1);
    std::vector<std::uint64_t> entries(bounds.size() - 1, 0);
    parallel_for(bounds.size() - 1, threads, [&](std::size_t k) {
        std::vector<Run>& out = runs[k];
        std::uint64_t count = 0;
        const char* at = bounds[k];
        const char* stop = bounds[k + 1];
        while (at < stop) {
            const char* next = line_end(at, stop);
            const char* field = at;
            while (field < next && (*field == ' ' || *field == '\t' || *field == '\r')) ++field;
            if (field < next && *field != '%') {
                std::uint64_t row = 0, column = 0;
                if (!unsigned_field(field, next, row) || !unsigned_field(field, next, column)) {
                    throw std::runtime_error("bad Matrix Market entry in " + path);
                }
                while (field < next && (*field == ' ' || *field == '\t')) ++field;
                double value = 0.0;
                const char* value_end = next;
                while (value_end > field && (value_end[-1] == '\r' || value_end[-1] == ' ')) --value_end;
                const std::from_chars_result r = std::from_chars(field, value_end, value);
                if (r.ec != std::errc() || r.ptr != value_end) throw std::runtime_error("bad Matrix Market value in " + path);
                if (row < 1 || row > d.rows || column < 1 || column > d.columns) {
                    throw std::runtime_error("Matrix Market entry outside the matrix in " + path);
                }
                if (!out.empty() && out.back().column == column) {
                    out.back().sum += value;
                } else {
                    out.push_back(Run{column, value});
                }
                ++count;
            }
            at = next < stop ? next + 1 : stop;
        }
        entries[k] = count;
    });
    std::uint64_t total_entries = 0;
    std::vector<double> totals(static_cast<std::size_t>(d.columns), 0.0);
    for (std::size_t k = 0; k < runs.size(); ++k) {
        total_entries += entries[k];
        for (const Run& run : runs[k]) totals[run.column - 1] += run.sum;
    }
    if (total_entries != d.entries) throw std::runtime_error("Matrix Market entry count differs from its header: " + path);
    return totals;
}

namespace {

bool parse_int(const char* at, const char* end, std::int64_t& value) {
    if (at == end) return false;
    const std::from_chars_result r = std::from_chars(at, end, value);
    return r.ec == std::errc() && r.ptr == end;
}

}  // namespace

Barcodes read_barcodes(const std::string& path, int bin_um, int side, bool skip_blank, int threads) {
    const MappedFile file(path);
    const char* begin = file.data();
    const char* end = begin + file.size();
    char prefix[16];
    std::snprintf(prefix, sizeof(prefix), "s_%03dum_", bin_um);
    const std::size_t prefix_length = std::strlen(prefix);
    const std::vector<const char*> bounds = line_chunks(begin, end, static_cast<std::size_t>(std::max(threads, 1)) * 16);
    const std::size_t chunks = bounds.size() - 1;
    struct Part {
        std::string text;
        std::vector<std::uint64_t> length;
        std::vector<std::int32_t> row, col;
    };
    std::vector<Part> parts(chunks);
    parallel_for(chunks, threads, [&](std::size_t k) {
        Part& part = parts[k];
        const char* at = bounds[k];
        const char* stop = bounds[k + 1];
        while (at < stop) {
            const char* next = line_end(at, stop);
            // line.strip().split('\t')[0]
            const char* a = at;
            const char* b = next;
            while (a < b && std::strchr(" \t\r\f\v", *a)) ++a;
            while (b > a && std::strchr(" \t\r\f\v", b[-1])) --b;
            const void* tab = std::memchr(a, '\t', static_cast<std::size_t>(b - a));
            const char* code_end = tab ? static_cast<const char*>(tab) : b;
            if (a == b && skip_blank) {
                at = next < stop ? next + 1 : stop;
                continue;
            }
            const std::string_view code(a, static_cast<std::size_t>(code_end - a));
            if (code.size() <= prefix_length || code.compare(0, prefix_length, prefix) != 0) {
                throw std::runtime_error("barcode is not of " + std::to_string(bin_um) + " um: " + std::string(code));
            }
            const char* row_begin = code.data() + prefix_length;
            const char* code_stop = code.data() + code.size();
            const char* underscore = static_cast<const char*>(std::memchr(row_begin, '_', static_cast<std::size_t>(code_stop - row_begin)));
            const char* dash = underscore ? static_cast<const char*>(std::memchr(underscore + 1, '-', static_cast<std::size_t>(code_stop - underscore - 1))) : nullptr;
            std::int64_t row = -1, col = -1;
            if (!underscore || !dash || !parse_int(row_begin, underscore, row) || !parse_int(underscore + 1, dash, col)
                || std::memchr(dash + 1, '_', static_cast<std::size_t>(code_stop - dash - 1))) {
                throw std::runtime_error("malformed barcode: " + std::string(code));
            }
            if (row < 0 || row >= side || col < 0 || col >= side) throw std::runtime_error("barcode outside the grid: " + std::string(code));
            part.text.append(code.data(), code.size());
            part.length.push_back(code.size());
            part.row.push_back(static_cast<std::int32_t>(row));
            part.col.push_back(static_cast<std::int32_t>(col));
            at = next < stop ? next + 1 : stop;
        }
    });
    Barcodes out;
    std::size_t rows = 0, bytes = 0;
    for (const Part& part : parts) {
        rows += part.row.size();
        bytes += part.text.size();
    }
    out.text.reserve(bytes);
    out.begin.reserve(rows + 1);
    out.row.reserve(rows);
    out.col.reserve(rows);
    out.begin.push_back(0);
    for (const Part& part : parts) {
        out.text += part.text;
        for (std::size_t i = 0; i < part.row.size(); ++i) {
            out.begin.push_back(out.begin.back() + part.length[i]);
            out.row.push_back(part.row[i]);
            out.col.push_back(part.col[i]);
        }
    }
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(side) * static_cast<std::size_t>(side), 0);
    for (std::size_t i = 0; i < rows; ++i) {
        std::uint8_t& flag = seen[static_cast<std::size_t>(out.row[i]) * side + out.col[i]];
        if (flag) throw std::runtime_error("duplicate barcode in " + path);
        flag = 1;
    }
    return out;
}

}  // namespace visium_hd::mex
