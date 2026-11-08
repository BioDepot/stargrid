#include "hd_parquet.hpp"

#include "hd_parallel.hpp"

#include <zlib.h>

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace visium_hd::parquet {

namespace {

// Physical types, encodings and codecs of the Parquet format (parquet.thrift).
constexpr int kBoolean = 0, kInt32 = 1, kDouble = 5, kByteArray = 6;
constexpr int kPlain = 0, kRle = 3;
constexpr int kOptional = 1;
constexpr int kUtf8 = 0;
constexpr int kGzip = 2;
constexpr int kDataPage = 0;

// The Thrift compact protocol, as much of it as the file metadata needs.
class Compact {
public:
    std::string out;

    void i32(int id, std::int64_t value) {
        field(id, 5);
        varint(zigzag(value));
    }
    void i64(int id, std::int64_t value) {
        field(id, 6);
        varint(zigzag(value));
    }
    void binary(int id, const std::string& value) {
        field(id, 8);
        raw_binary(value);
    }
    void begin_struct(int id) {
        field(id, 12);
        stack_.push_back(last_);
        last_ = 0;
    }
    void end_struct() {
        out += '\0';
        last_ = stack_.back();
        stack_.pop_back();
    }
    void begin_list(int id, int element_type, std::size_t size) {
        field(id, 9);
        if (size < 15) {
            out += static_cast<char>((size << 4) | element_type);
        } else {
            out += static_cast<char>(0xF0 | element_type);
            varint(size);
        }
    }
    void begin_element() {
        stack_.push_back(last_);
        last_ = 0;
    }
    void end_element() { end_struct(); }
    void raw_i32(std::int64_t value) { varint(zigzag(value)); }
    void raw_binary(const std::string& value) {
        varint(value.size());
        out += value;
    }
    void stop() { out += '\0'; }

private:
    void field(int id, int type) {
        const int delta = id - last_;
        if (delta > 0 && delta <= 15) {
            out += static_cast<char>((delta << 4) | type);
        } else {
            out += static_cast<char>(type);
            varint(zigzag(id));
        }
        last_ = id;
    }
    void varint(std::uint64_t value) {
        while (value >= 0x80) {
            out += static_cast<char>((value & 0x7F) | 0x80);
            value >>= 7;
        }
        out += static_cast<char>(value);
    }
    static std::uint64_t zigzag(std::int64_t value) {
        return (static_cast<std::uint64_t>(value) << 1) ^ static_cast<std::uint64_t>(value >> 63);
    }

    int last_ = 0;
    std::vector<int> stack_;
};

void put4(std::string& out, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) out += static_cast<char>((value >> (8 * i)) & 0xFF);
}

int physical(Type type) {
    switch (type) {
        case Type::String: return kByteArray;
        case Type::Int32: return kInt32;
        case Type::Double: return kDouble;
        case Type::Boolean: return kBoolean;
    }
    return kDouble;
}

std::string gzip(const std::string& data, int level) {
    z_stream stream{};
    if (deflateInit2(&stream, level, Z_DEFLATED, MAX_WBITS + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
        throw std::runtime_error("deflateInit2 failed");
    }
    std::string out(deflateBound(&stream, static_cast<uLong>(data.size())), '\0');
    stream.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(data.data()));
    stream.avail_in = static_cast<uInt>(data.size());
    stream.next_out = reinterpret_cast<Bytef*>(out.data());
    stream.avail_out = static_cast<uInt>(out.size());
    const int status = deflate(&stream, Z_FINISH);
    out.resize(stream.total_out);
    deflateEnd(&stream);
    if (status != Z_STREAM_END) throw std::runtime_error("gzip compression failed");
    return out;
}

struct Chunk {
    std::string bytes;                 // page header and compressed page
    std::uint64_t uncompressed = 0;    // page header and uncompressed page
};

Chunk encode(const Column& column, std::size_t first, std::size_t count, int level) {
    std::string body;
    {
        // Definition levels: every value present, as one RLE run of 1s, prefixed by its length.
        std::string levels;
        std::uint64_t header = static_cast<std::uint64_t>(count) << 1;
        while (header >= 0x80) {
            levels += static_cast<char>((header & 0x7F) | 0x80);
            header >>= 7;
        }
        levels += static_cast<char>(header);
        levels += '\1';
        put4(body, static_cast<std::uint32_t>(levels.size()));
        body += levels;
    }
    switch (column.type) {
        case Type::String: {
            const std::uint64_t bytes = column.begin[first + count] - column.begin[first];
            body.reserve(body.size() + 4 * count + bytes);
            for (std::size_t i = first; i < first + count; ++i) {
                const std::uint64_t length = column.begin[i + 1] - column.begin[i];
                put4(body, static_cast<std::uint32_t>(length));
                body.append(column.text + column.begin[i], length);
            }
            break;
        }
        case Type::Int32:
            body.append(reinterpret_cast<const char*>(column.int32s + first), 4 * count);
            break;
        case Type::Double:
            body.append(reinterpret_cast<const char*>(column.doubles + first), 8 * count);
            break;
        case Type::Boolean: {
            std::string bits((count + 7) / 8, '\0');
            for (std::size_t i = 0; i < count; ++i) {
                if (column.booleans[first + i]) bits[i / 8] = static_cast<char>(bits[i / 8] | (1 << (i % 8)));
            }
            body += bits;
            break;
        }
    }
    if (body.size() > 0x7FFFFFFF) throw std::runtime_error("Parquet page too large");
    const std::string compressed = gzip(body, level);
    Compact header;
    header.i32(1, kDataPage);
    header.i32(2, static_cast<std::int64_t>(body.size()));
    header.i32(3, static_cast<std::int64_t>(compressed.size()));
    header.begin_struct(5);
    header.i32(1, static_cast<std::int64_t>(count));
    header.i32(2, kPlain);
    header.i32(3, kRle);
    header.i32(4, kRle);
    header.end_struct();
    header.stop();
    Chunk chunk;
    chunk.uncompressed = header.out.size() + body.size();
    chunk.bytes = header.out + compressed;
    return chunk;
}

}  // namespace

void write(const std::string& path, const std::vector<Column>& columns, std::size_t rows, const Options& options) {
    if (columns.empty()) throw std::runtime_error("a Parquet table needs at least one column");
    for (const Column& column : columns) {
        const bool ready = (column.type == Type::String && column.text && column.begin)
            || (column.type == Type::Int32 && column.int32s) || (column.type == Type::Double && column.doubles)
            || (column.type == Type::Boolean && column.booleans);
        if (!ready && rows > 0) throw std::runtime_error("Parquet column without data: " + column.name);
    }
    const std::size_t per_group = std::max<std::size_t>(options.row_group_rows, 1);
    const std::size_t groups = rows == 0 ? 0 : (rows + per_group - 1) / per_group;
    std::vector<Chunk> chunks(groups * columns.size());
    parallel_for(chunks.size(), options.threads, [&](std::size_t task) {
        const std::size_t group = task / columns.size();
        const std::size_t first = group * per_group;
        chunks[task] = encode(columns[task % columns.size()], first, std::min(per_group, rows - first),
                              options.compression_level);
    });

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write " + path);
    output.write("PAR1", 4);
    std::uint64_t offset = 4;
    std::vector<std::uint64_t> offsets(chunks.size());
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        offsets[i] = offset;
        output.write(chunks[i].bytes.data(), static_cast<std::streamsize>(chunks[i].bytes.size()));
        offset += chunks[i].bytes.size();
    }

    Compact meta;
    meta.i32(1, 1);
    meta.begin_list(2, 12, columns.size() + 1);
    meta.begin_element();
    meta.binary(4, "schema");
    meta.i32(5, static_cast<std::int64_t>(columns.size()));
    meta.end_element();
    for (const Column& column : columns) {
        meta.begin_element();
        meta.i32(1, physical(column.type));
        meta.i32(3, kOptional);
        meta.binary(4, column.name);
        if (column.type == Type::String) {
            meta.i32(6, kUtf8);
            meta.begin_struct(10);   // LogicalType
            meta.begin_struct(1);    // STRING
            meta.end_struct();
            meta.end_struct();
        }
        meta.end_element();
    }
    meta.i64(3, static_cast<std::int64_t>(rows));
    meta.begin_list(4, 12, groups);
    for (std::size_t group = 0; group < groups; ++group) {
        const std::size_t count = std::min(per_group, rows - group * per_group);
        std::uint64_t group_bytes = 0;
        meta.begin_element();
        meta.begin_list(1, 12, columns.size());
        for (std::size_t c = 0; c < columns.size(); ++c) {
            const std::size_t task = group * columns.size() + c;
            group_bytes += chunks[task].uncompressed;
            meta.begin_element();
            meta.i64(2, static_cast<std::int64_t>(offsets[task]));
            meta.begin_struct(3);
            meta.i32(1, physical(columns[c].type));
            meta.begin_list(2, 5, 2);
            meta.raw_i32(kPlain);
            meta.raw_i32(kRle);
            meta.begin_list(3, 8, 1);
            meta.raw_binary(columns[c].name);
            meta.i32(4, kGzip);
            meta.i64(5, static_cast<std::int64_t>(count));
            meta.i64(6, static_cast<std::int64_t>(chunks[task].uncompressed));
            meta.i64(7, static_cast<std::int64_t>(chunks[task].bytes.size()));
            meta.i64(9, static_cast<std::int64_t>(offsets[task]));
            meta.end_struct();
            meta.end_element();
        }
        meta.i64(2, static_cast<std::int64_t>(group_bytes));
        meta.i64(3, static_cast<std::int64_t>(count));
        meta.end_element();
    }
    meta.binary(6, options.created_by);
    meta.stop();
    output.write(meta.out.data(), static_cast<std::streamsize>(meta.out.size()));
    std::string tail;
    put4(tail, static_cast<std::uint32_t>(meta.out.size()));
    tail += "PAR1";
    output.write(tail.data(), static_cast<std::streamsize>(tail.size()));
    if (!output) throw std::runtime_error("cannot write " + path);
}

}  // namespace visium_hd::parquet
