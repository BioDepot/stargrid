#include "hd_npy.hpp"

#include <zlib.h>

#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace visium_hd::npy {

namespace {

std::vector<char> read_bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open array file: " + path);
    const std::streamsize size = input.tellg();
    input.seekg(0);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    if (size > 0 && !input.read(bytes.data(), size)) throw std::runtime_error("cannot read array file: " + path);
    return bytes;
}

std::uint64_t little(const char* at, int bytes) {
    std::uint64_t value = 0;
    for (int i = bytes - 1; i >= 0; --i) value = (value << 8) | static_cast<unsigned char>(at[i]);
    return value;
}

void put(std::string& out, std::uint64_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) out += static_cast<char>((value >> (8 * i)) & 0xFF);
}

std::string between(const std::string& header, const std::string& key) {
    const std::size_t k = header.find("'" + key + "'");
    if (k == std::string::npos) throw std::runtime_error("npy header lacks " + key);
    const std::size_t colon = header.find(':', k);
    if (colon == std::string::npos) throw std::runtime_error("malformed npy header");
    return header.substr(colon + 1);
}

Array parse_npy(const char* data, std::size_t size, const std::string& where) {
    if (size < 10 || std::memcmp(data, "\x93NUMPY", 6) != 0) throw std::runtime_error("not a .npy array: " + where);
    const int major = static_cast<unsigned char>(data[6]);
    std::size_t header_length = 0;
    std::size_t offset = 0;
    if (major == 1) {
        header_length = static_cast<std::size_t>(little(data + 8, 2));
        offset = 10;
    } else if (major == 2 || major == 3) {
        if (size < 12) throw std::runtime_error("truncated .npy header: " + where);
        header_length = static_cast<std::size_t>(little(data + 8, 4));
        offset = 12;
    } else {
        throw std::runtime_error("unsupported .npy version: " + where);
    }
    if (offset + header_length > size) throw std::runtime_error("truncated .npy header: " + where);
    const std::string header(data + offset, header_length);
    Array array;
    {
        const std::string rest = between(header, "descr");
        const std::size_t q0 = rest.find('\'');
        const std::size_t q1 = rest.find('\'', q0 + 1);
        if (q0 == std::string::npos || q1 == std::string::npos) throw std::runtime_error("malformed npy descr: " + where);
        array.descr = rest.substr(q0 + 1, q1 - q0 - 1);
    }
    {
        const std::string rest = between(header, "fortran_order");
        const std::size_t t = rest.find_first_not_of(' ');
        if (rest.compare(t, 4, "True") == 0) throw std::runtime_error("Fortran-order arrays are not supported: " + where);
    }
    {
        const std::string rest = between(header, "shape");
        const std::size_t open = rest.find('(');
        const std::size_t close = rest.find(')', open);
        if (open == std::string::npos || close == std::string::npos) throw std::runtime_error("malformed npy shape: " + where);
        std::stringstream dims(rest.substr(open + 1, close - open - 1));
        std::string token;
        while (std::getline(dims, token, ',')) {
            const std::size_t a = token.find_first_not_of(' ');
            if (a == std::string::npos) continue;
            array.shape.push_back(static_cast<std::size_t>(std::stoull(token.substr(a))));
        }
    }
    const std::size_t bytes = array.count() * array.item_size();
    if (offset + header_length + bytes > size) throw std::runtime_error("truncated .npy data: " + where);
    array.data.assign(data + offset + header_length, data + offset + header_length + bytes);
    return array;
}

struct Entry {
    std::string name;
    int method = 0;
    std::uint64_t compressed = 0;
    std::uint64_t uncompressed = 0;
    std::uint64_t local_offset = 0;
};

std::vector<Entry> central_directory(const std::vector<char>& zip, const std::string& path) {
    const std::size_t size = zip.size();
    if (size < 22) throw std::runtime_error("not a zip archive: " + path);
    std::size_t eocd = std::string::npos;
    for (std::size_t at = size - 22 + 1; at-- > (size > 22 + 65535 ? size - 22 - 65535 : 0);) {
        if (little(zip.data() + at, 4) == 0x06054b50) {
            eocd = at;
            break;
        }
    }
    if (eocd == std::string::npos) throw std::runtime_error("zip end record not found: " + path);
    std::uint64_t entries = little(zip.data() + eocd + 10, 2);
    std::uint64_t cd_offset = little(zip.data() + eocd + 16, 4);
    if ((entries == 0xFFFF || cd_offset == 0xFFFFFFFF) && eocd >= 20
        && little(zip.data() + eocd - 20, 4) == 0x07064b50) {
        const std::uint64_t record = little(zip.data() + eocd - 20 + 8, 8);
        if (record + 56 > size || little(zip.data() + record, 4) != 0x06064b50) {
            throw std::runtime_error("bad zip64 end record: " + path);
        }
        entries = little(zip.data() + record + 32, 8);
        cd_offset = little(zip.data() + record + 48, 8);
    }
    std::vector<Entry> out;
    std::size_t at = static_cast<std::size_t>(cd_offset);
    for (std::uint64_t i = 0; i < entries; ++i) {
        if (at + 46 > size || little(zip.data() + at, 4) != 0x02014b50) throw std::runtime_error("bad zip directory: " + path);
        Entry entry;
        entry.method = static_cast<int>(little(zip.data() + at + 10, 2));
        entry.compressed = little(zip.data() + at + 20, 4);
        entry.uncompressed = little(zip.data() + at + 24, 4);
        const std::size_t name_length = static_cast<std::size_t>(little(zip.data() + at + 28, 2));
        const std::size_t extra_length = static_cast<std::size_t>(little(zip.data() + at + 30, 2));
        const std::size_t comment_length = static_cast<std::size_t>(little(zip.data() + at + 32, 2));
        entry.local_offset = little(zip.data() + at + 42, 4);
        entry.name.assign(zip.data() + at + 46, name_length);
        std::size_t extra = at + 46 + name_length;
        const std::size_t extra_end = extra + extra_length;
        while (extra + 4 <= extra_end) {
            const std::uint64_t id = little(zip.data() + extra, 2);
            const std::size_t length = static_cast<std::size_t>(little(zip.data() + extra + 2, 2));
            if (id == 0x0001) {
                std::size_t field = extra + 4;
                if (entry.uncompressed == 0xFFFFFFFF) { entry.uncompressed = little(zip.data() + field, 8); field += 8; }
                if (entry.compressed == 0xFFFFFFFF) { entry.compressed = little(zip.data() + field, 8); field += 8; }
                if (entry.local_offset == 0xFFFFFFFF) { entry.local_offset = little(zip.data() + field, 8); }
            }
            extra += 4 + length;
        }
        out.push_back(entry);
        at += 46 + name_length + extra_length + comment_length;
    }
    return out;
}

std::vector<char> member_bytes(const std::vector<char>& zip, const Entry& entry, const std::string& path) {
    const std::size_t at = static_cast<std::size_t>(entry.local_offset);
    if (at + 30 > zip.size() || little(zip.data() + at, 4) != 0x04034b50) throw std::runtime_error("bad zip member header: " + path);
    const std::size_t begin = at + 30 + static_cast<std::size_t>(little(zip.data() + at + 26, 2))
        + static_cast<std::size_t>(little(zip.data() + at + 28, 2));
    if (begin + entry.compressed > zip.size()) throw std::runtime_error("truncated zip member: " + path);
    std::vector<char> out(static_cast<std::size_t>(entry.uncompressed));
    if (entry.method == 0) {
        if (entry.compressed != entry.uncompressed) throw std::runtime_error("bad stored zip member: " + path);
        std::memcpy(out.data(), zip.data() + begin, out.size());
        return out;
    }
    if (entry.method != 8) throw std::runtime_error("unsupported zip compression in " + path);
    if (entry.compressed >= 0xFFFFFFFFull || entry.uncompressed >= 0xFFFFFFFFull) {
        throw std::runtime_error("deflated zip member too large: " + path);
    }
    z_stream stream{};
    if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) throw std::runtime_error("inflateInit2 failed");
    stream.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(zip.data() + begin));
    stream.avail_in = static_cast<uInt>(entry.compressed);
    stream.next_out = reinterpret_cast<Bytef*>(out.data());
    stream.avail_out = static_cast<uInt>(entry.uncompressed);
    const int status = inflate(&stream, Z_FINISH);
    const bool complete = status == Z_STREAM_END && stream.avail_out == 0;
    inflateEnd(&stream);
    if (!complete) throw std::runtime_error("cannot inflate zip member of " + path);
    return out;
}

bool ends_with(const std::string& text, const std::string& tail) {
    return text.size() >= tail.size() && text.compare(text.size() - tail.size(), tail.size(), tail) == 0;
}

std::string npy_header(const std::string& descr, const std::vector<std::size_t>& shape) {
    std::string dims = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        dims += std::to_string(shape[i]);
        if (shape.size() == 1 || i + 1 < shape.size()) dims += ",";
        if (i + 1 < shape.size()) dims += " ";
    }
    dims += ")";
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + dims + ", }";
    const std::size_t total = 10 + header.size() + 1;
    header += std::string((64 - total % 64) % 64, ' ') + "\n";
    std::string out("\x93NUMPY\x01\x00", 8);
    put(out, header.size(), 2);
    return out + header;
}

}  // namespace

std::size_t Array::count() const {
    std::size_t n = 1;
    for (std::size_t d : shape) n *= d;
    return n;
}

std::size_t Array::item_size() const {
    if (descr.size() < 3) throw std::runtime_error("unsupported npy dtype: " + descr);
    return static_cast<std::size_t>(std::stoul(descr.substr(2)));
}

Array read(const std::string& path, const std::string& key) {
    const std::vector<char> bytes = read_bytes(path);
    if (!ends_with(path, ".npz")) return parse_npy(bytes.data(), bytes.size(), path);
    const std::vector<Entry> entries = central_directory(bytes, path);
    for (const Entry& entry : entries) {
        if (key.empty() || entry.name == key + ".npy") {
            const std::vector<char> member = member_bytes(bytes, entry, path);
            return parse_npy(member.data(), member.size(), path + ":" + entry.name);
        }
    }
    throw std::runtime_error("array " + key + " not found in " + path);
}

namespace {

template <class T>
void convert(const Array& array, std::vector<double>& out) {
    const T* values = reinterpret_cast<const T*>(array.data.data());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<double>(values[i]);
}

}  // namespace

std::vector<double> as_doubles(const Array& array) {
    const char order = array.descr[0];
    if (order == '>') throw std::runtime_error("big-endian arrays are not supported");
    const std::string type = array.descr.substr(1);
    std::vector<double> out(array.count());
    if (type == "b1" || type == "u1") convert<std::uint8_t>(array, out);
    else if (type == "i1") convert<std::int8_t>(array, out);
    else if (type == "u2") convert<std::uint16_t>(array, out);
    else if (type == "i2") convert<std::int16_t>(array, out);
    else if (type == "u4") convert<std::uint32_t>(array, out);
    else if (type == "i4") convert<std::int32_t>(array, out);
    else if (type == "u8") convert<std::uint64_t>(array, out);
    else if (type == "i8") convert<std::int64_t>(array, out);
    else if (type == "f4") convert<float>(array, out);
    else if (type == "f8") convert<double>(array, out);
    else throw std::runtime_error("unsupported npy dtype: " + array.descr);
    return out;
}

std::vector<std::uint8_t> as_flags(const Array& array) {
    if (array.descr == "|b1" || array.descr == "|u1") {
        std::vector<std::uint8_t> out(array.count());
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = array.data[i] != 0;
        return out;
    }
    const std::vector<double> values = as_doubles(array);
    std::vector<std::uint8_t> out(values.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = values[i] != 0.0;   // NaN counts as set, as in numpy
    return out;
}

void write_npy(const std::string& path, const Member& member) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    const std::string header = npy_header(member.descr, member.shape);
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(static_cast<const char*>(member.data), static_cast<std::streamsize>(member.bytes));
    if (!output) throw std::runtime_error("cannot write " + path);
}

void write_npz(const std::string& path, const std::vector<Member>& members) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write " + path);
    std::string directory;
    std::uint64_t offset = 0;
    for (const Member& member : members) {
        const std::string header = npy_header(member.descr, member.shape);
        const std::uint64_t size = header.size() + member.bytes;
        if (size >= 0xFFFFFFFFull || offset >= 0xFFFFFFFFull) throw std::runtime_error("npz member too large: " + path);
        uLong crc = crc32_z(0L, reinterpret_cast<const Bytef*>(header.data()), header.size());
        crc = crc32_z(crc, static_cast<const Bytef*>(member.data), member.bytes);
        const std::string name = member.name + ".npy";
        std::string local;
        put(local, 0x04034b50, 4);
        put(local, 20, 2);
        put(local, 0, 2);
        put(local, 0, 2);
        put(local, 0, 2);
        put(local, 0x21, 2);
        put(local, crc, 4);
        put(local, size, 4);
        put(local, size, 4);
        put(local, name.size(), 2);
        put(local, 0, 2);
        local += name;
        output.write(local.data(), static_cast<std::streamsize>(local.size()));
        output.write(header.data(), static_cast<std::streamsize>(header.size()));
        output.write(static_cast<const char*>(member.data), static_cast<std::streamsize>(member.bytes));
        put(directory, 0x02014b50, 4);
        put(directory, 20, 2);
        put(directory, 20, 2);
        put(directory, 0, 2);
        put(directory, 0, 2);
        put(directory, 0, 2);
        put(directory, 0x21, 2);
        put(directory, crc, 4);
        put(directory, size, 4);
        put(directory, size, 4);
        put(directory, name.size(), 2);
        put(directory, 0, 2);
        put(directory, 0, 2);
        put(directory, 0, 2);
        put(directory, 0, 2);
        put(directory, 0, 4);
        put(directory, offset, 4);
        directory += name;
        offset += local.size() + size;
    }
    std::string end;
    put(end, 0x06054b50, 4);
    put(end, 0, 2);
    put(end, 0, 2);
    put(end, members.size(), 2);
    put(end, members.size(), 2);
    put(end, directory.size(), 4);
    put(end, offset, 4);
    put(end, 0, 2);
    output.write(directory.data(), static_cast<std::streamsize>(directory.size()));
    output.write(end.data(), static_cast<std::streamsize>(end.size()));
    if (!output) throw std::runtime_error("cannot write " + path);
}

}  // namespace visium_hd::npy
