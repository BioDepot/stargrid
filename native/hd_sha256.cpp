#include "hd_sha256.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace visium_hd {

namespace {

constexpr std::uint32_t kRound[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

}  // namespace

Sha256::Sha256()
    : state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {}

void Sha256::block(const unsigned char* p) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16) | (std::uint32_t(p[4 * i + 2]) << 8)
            | std::uint32_t(p[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
        const std::uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + kRound[i] + w[i];
        const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256::update(const void* data, std::size_t size) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    length_ += size;
    if (buffered_) {
        const std::size_t take = std::min(size, 64 - buffered_);
        std::memcpy(buffer_ + buffered_, p, take);
        buffered_ += take;
        p += take;
        size -= take;
        if (buffered_ < 64) return;
        block(buffer_);
        buffered_ = 0;
    }
    for (; size >= 64; p += 64, size -= 64) block(p);
    std::memcpy(buffer_, p, size);
    buffered_ = size;
}

std::string Sha256::hex() {
    const std::uint64_t bits = length_ * 8;
    const unsigned char one = 0x80;
    update(&one, 1);
    const unsigned char zero = 0;
    while (buffered_ != 56) update(&zero, 1);
    unsigned char tail[8];
    for (int i = 0; i < 8; ++i) tail[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
    update(tail, 8);
    std::string out;
    char digits[9];
    for (std::uint32_t word : state_) {
        std::snprintf(digits, sizeof(digits), "%08x", word);
        out += digits;
    }
    return out;
}

namespace {

void add_file(Sha256& digest, const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path);
    std::vector<char> buffer(1 << 20);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        digest.update(buffer.data(), static_cast<std::size_t>(input.gcount()));
    }
}

}  // namespace

std::string sha256_file(const std::string& path) {
    Sha256 digest;
    add_file(digest, path);
    return digest.hex();
}

std::string tree_sha256(const std::string& root) {
    namespace fs = std::filesystem;
    std::vector<std::vector<std::string>> files;
    for (const fs::directory_entry& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        std::vector<std::string> parts;
        for (const fs::path& part : fs::relative(entry.path(), root)) parts.push_back(part.string());
        files.push_back(parts);
    }
    std::sort(files.begin(), files.end());   // pathlib orders paths by their components
    Sha256 digest;
    for (const std::vector<std::string>& parts : files) {
        std::string relative;
        for (std::size_t i = 0; i < parts.size(); ++i) relative += (i ? "/" : "") + parts[i];
        unsigned char length[8];
        for (int i = 0; i < 8; ++i) length[i] = static_cast<unsigned char>(static_cast<std::uint64_t>(relative.size()) >> (8 * i));
        digest.update(length, 8);
        digest.update(relative);
        add_file(digest, (fs::path(root) / relative).string());
    }
    return digest.hex();
}

}  // namespace visium_hd
