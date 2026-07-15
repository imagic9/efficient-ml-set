// SHA-256 per FIPS 180-4. See hashing.hpp for why this is here rather than linked.

#include "wildlife_trigger/hashing.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace wildlife_trigger {
namespace {

// First 32 bits of the fractional parts of the cube roots of the first 64 primes.
constexpr std::array<uint32_t, 64> kK = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t rotr(uint32_t value, int bits) {
    return (value >> bits) | (value << (32 - bits));
}

class Sha256 {
  public:
    void update(const uint8_t *data, size_t length) {
        total_ += length;
        while (length > 0) {
            const size_t take = std::min(length, sizeof(buffer_) - buffered_);
            std::memcpy(buffer_ + buffered_, data, take);
            buffered_ += take;
            data += take;
            length -= take;
            if (buffered_ == sizeof(buffer_)) {
                compress(buffer_);
                buffered_ = 0;
            }
        }
    }

    std::string hex_digest() {
        // Padding: 0x80, then zeros, then the message length in bits as big-endian
        // 64-bit. If fewer than 8 bytes remain, the length spills into another block.
        const uint64_t bit_length = total_ * 8;
        const uint8_t one = 0x80;
        update(&one, 1);
        const uint8_t zero = 0x00;
        while (buffered_ != 56) {
            update(&zero, 1);
        }
        // total_ is now wrong (padding counted), which is fine: bit_length was saved.
        uint8_t length_bytes[8];
        for (int i = 0; i < 8; ++i) {
            length_bytes[i] = static_cast<uint8_t>(bit_length >> (56 - 8 * i));
        }
        update(length_bytes, 8);

        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const uint32_t word : state_) {
            out << std::setw(8) << word;
        }
        return out.str();
    }

  private:
    void compress(const uint8_t *block) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
                   (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
                   static_cast<uint32_t>(block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 =
                rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 =
                rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];

        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t temp1 = h + S1 + ch + kK[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = S0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    // First 32 bits of the fractional parts of the square roots of the first 8 primes.
    uint32_t state_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t buffer_[64] = {};
    size_t buffered_ = 0;
    uint64_t total_ = 0;
};

}  // namespace

std::string sha256_bytes(const uint8_t *data, size_t length) {
    Sha256 hasher;
    hasher.update(data, length);
    return hasher.hex_digest();
}

std::string sha256_string(const std::string &text) {
    return sha256_bytes(reinterpret_cast<const uint8_t *>(text.data()), text.size());
}

std::string sha256_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open file for hashing: " + path);
    }
    Sha256 hasher;
    std::vector<char> chunk(1 << 20);
    while (file.read(chunk.data(), static_cast<std::streamsize>(chunk.size())) ||
           file.gcount() > 0) {
        hasher.update(reinterpret_cast<const uint8_t *>(chunk.data()),
                      static_cast<size_t>(file.gcount()));
        if (!file) {
            break;
        }
    }
    return hasher.hex_digest();
}

}  // namespace wildlife_trigger
