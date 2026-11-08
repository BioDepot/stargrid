#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace visium_hd::mex {

// A read-only view of a whole file, memory-mapped.
class MappedFile {
public:
    explicit MappedFile(const std::string& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    const char* data() const { return data_; }
    std::size_t size() const { return size_; }

private:
    const char* data_ = nullptr;
    std::size_t size_ = 0;
};

struct Dimensions {
    std::uint64_t rows = 0, columns = 0, entries = 0;
    bool real = false;
    std::size_t data_offset = 0;   // first entry line
};

// The header of a Matrix Market coordinate file.
Dimensions dimensions(const char* data, std::size_t size, const std::string& path);
std::uint64_t matrix_columns(const std::string& path);

// The sum of each column of a Matrix Market coordinate matrix (features x barcodes), parsed in
// parallel.
std::vector<double> column_totals(const std::string& path, int threads);

// The barcodes of one bin size, s_<bin>um_<row>_<column>-1, in file order: the barcode text without
// line ends (row i is text[begin[i], begin[i + 1])), and the bin row and column of each.
struct Barcodes {
    std::string text;
    std::vector<std::uint64_t> begin;
    std::vector<std::int32_t> row, col;
    std::size_t size() const { return row.size(); }
};

// Reads barcodes.tsv (the first tab-separated field of each line), requiring every barcode to be of
// `bin_um` um, inside a grid of `side` bins, and unique. `skip_blank` drops blank lines (as the
// 2 um count loader does); otherwise a blank line is an error (as the positions writer requires).
Barcodes read_barcodes(const std::string& path, int bin_um, int side, bool skip_blank, int threads);

}  // namespace visium_hd::mex
