#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace visium_hd::npy {

// An array read from a NumPy .npy file or a member of an .npz archive (stored or deflated,
// including the zip64 members that numpy.savez writes).
struct Array {
    std::string descr;                 // e.g. "|b1", "<f8", "<u4"
    std::vector<std::size_t> shape;
    std::vector<char> data;            // C order, little-endian

    std::size_t count() const;
    std::size_t item_size() const;
};

Array read(const std::string& path, const std::string& key = "");   // key: .npz member, "" for the first

// The array as flags (nonzero) or as doubles, whatever its numeric type.
std::vector<std::uint8_t> as_flags(const Array& array);
std::vector<double> as_doubles(const Array& array);

struct Member {
    std::string name;                  // without ".npy"
    std::string descr;
    std::vector<std::size_t> shape;
    const void* data;
    std::size_t bytes;
};

// An uncompressed .npz archive that numpy.load reads; members are not compressed so that writing
// costs no more than copying the bytes.
void write_npz(const std::string& path, const std::vector<Member>& members);
void write_npy(const std::string& path, const Member& member);

}  // namespace visium_hd::npy
