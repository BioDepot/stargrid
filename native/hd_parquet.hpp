#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace visium_hd::parquet {

// A minimal Apache Parquet writer for flat tables of nullable string, int32, double and boolean
// columns without nulls: PLAIN encoding, one data page per column chunk, pages compressed with
// GZIP (zlib) in parallel. Readers see the same Arrow types that pyarrow would write.
enum class Type { String, Int32, Double, Boolean };

struct Column {
    std::string name;
    Type type = Type::Double;
    // String: row i is text + begin[i], begin[i + 1] - begin[i] bytes long (begin has rows + 1 items).
    const char* text = nullptr;
    const std::uint64_t* begin = nullptr;
    const std::int32_t* int32s = nullptr;
    const double* doubles = nullptr;
    const std::uint8_t* booleans = nullptr;
};

struct Options {
    std::size_t row_group_rows = 1u << 18;
    int compression_level = 1;
    int threads = 1;
    std::string created_by = "visium_hd_count_register";
};

void write(const std::string& path, const std::vector<Column>& columns, std::size_t rows, const Options& options);

}  // namespace visium_hd::parquet
