// SHA-256, implemented here rather than pulled in.
//
// The policy binds thresholds to a specific model and class map by hash (DESIGN §4),
// so the C++ side must compute the same digest Python records. That is the only
// cryptographic need in the whole application: no signatures, no MAC, no secrets, and
// nothing adversarial -- the threat is a stale file, not an attacker.
//
// The alternatives were OpenSSL (a dependency the Pi bundle would then need, against
// DESIGN §11's no-extra-package rule) or vendoring a second third-party source for
// ~60 lines of well-specified arithmetic. FIPS 180-4 is short, fixed forever, and
// tested here against the standard vectors.
//
// Do not reuse this for anything security-bearing without revisiting that reasoning:
// it is a plain implementation with no side-channel hardening, because file
// fingerprinting needs none.

#ifndef WILDLIFE_TRIGGER_HASHING_HPP
#define WILDLIFE_TRIGGER_HASHING_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace wildlife_trigger {

// Lowercase hex, 64 characters -- the format `sha256sum` and Python's hexdigest
// produce, so the strings can be compared directly.
std::string sha256_bytes(const uint8_t *data, size_t length);
std::string sha256_string(const std::string &text);

// Streams the file; a model is ~14 MB and need not be resident to be hashed.
// Throws std::runtime_error if the file cannot be read.
std::string sha256_file(const std::string &path);

}  // namespace wildlife_trigger

#endif  // WILDLIFE_TRIGGER_HASHING_HPP
