#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace visium_hd {

// SHA-256 (FIPS 180-4).
class Sha256 {
public:
    Sha256();
    void update(const void* data, std::size_t size);
    void update(const std::string& text) { update(text.data(), text.size()); }
    std::string hex();   // finishes the digest

private:
    void block(const unsigned char* data);
    std::array<std::uint32_t, 8> state_;
    unsigned char buffer_[64];
    std::size_t buffered_ = 0;
    std::uint64_t length_ = 0;
};

std::string sha256_file(const std::string& path);

// The digest of a directory tree as score_hd_flex_registration_roi.tree_sha256 computes it: for every
// file, in the order of its path components, the 8-byte little-endian length of its relative path,
// that path, and its contents.
std::string tree_sha256(const std::string& root);

}  // namespace visium_hd
