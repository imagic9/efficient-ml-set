// Logging convention -- DESIGN §11 component 7 ("CLI and logging").
//
// The application's output discipline is simple and load-bearing: stdout carries
// ONLY complete machine-readable JSON/JSONL evidence, and stderr carries the human
// and diagnostic chatter. A latency table is piped from stdout; a stray log line
// there would corrupt it. This header is the single place that convention lives,
// so every subcommand emits diagnostics the same way instead of each reinventing a
// `std::cerr <<`.
//
// Levels and their on-the-wire prefixes are chosen to preserve the contract the
// rest of the system already depends on:
//
//   error -> "error: <msg>"      (validate.policy_rejections greps stderr for
//                                  "error:" to confirm a bad policy was refused;
//                                  the level MUST keep that substring)
//   warn  -> "warning: <msg>"
//   info  -> "<msg>"             (the concise human output DESIGN §11 asks for --
//                                  no tag, so a result summary reads cleanly)
//   debug -> "debug: <msg>"      (suppressed unless WILDLIFE_LOG_LEVEL=debug)
//
// The threshold is read once from WILDLIFE_LOG_LEVEL (debug|info|warn|error),
// defaulting to info. A suppressed line builds no string and touches no stream, so
// leaving debug() calls in the hot path costs a single integer comparison.
//
// Header-only on purpose: it depends on nothing but <iostream>/<sstream>, so the
// always-built core library and the ORT-linked app library share one copy without a
// new translation unit, and the unit test links it with no ORT present.

#ifndef WILDLIFE_TRIGGER_LOGGING_HPP
#define WILDLIFE_TRIGGER_LOGGING_HPP

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace wildlife_trigger {
namespace log {

enum class Level { debug = 0, info = 1, warn = 2, error = 3 };

// Format a message exactly as it appears on stderr, without writing it. Exposed so
// the convention can be asserted in a unit test rather than by eyeballing output.
inline std::string format(Level level, const std::string &message) {
    switch (level) {
        case Level::error:
            return "error: " + message;
        case Level::warn:
            return "warning: " + message;
        case Level::debug:
            return "debug: " + message;
        case Level::info:
            break;
    }
    return message;
}

// The active threshold, parsed once from WILDLIFE_LOG_LEVEL. A function-local static
// gives thread-safe one-time init (C++11) and keeps the environment read out of the
// per-line path.
inline Level threshold() {
    static const Level cached = [] {
        const char *raw = std::getenv("WILDLIFE_LOG_LEVEL");
        if (raw == nullptr) return Level::info;
        const std::string value(raw);
        if (value == "debug") return Level::debug;
        if (value == "info") return Level::info;
        if (value == "warn" || value == "warning") return Level::warn;
        if (value == "error") return Level::error;
        return Level::info;  // an unrecognised value is not a reason to go silent
    }();
    return cached;
}

inline bool enabled(Level level) {
    return static_cast<int>(level) >= static_cast<int>(threshold());
}

// A one-line stream. Built by info()/warn()/error()/debug(), streamed into with <<,
// and flushed to stderr by its destructor at the end of the full expression:
//
//     log::info() << "wrote " << path;
//
// When the level is below the threshold the line is inert -- operator<< does
// nothing and no string is built. C++17 guaranteed copy elision means the factory
// returns it without a move, so the ostringstream member need not be movable.
class Line {
  public:
    Line(Level level, bool active) : level_(level), active_(active) {}
    Line(const Line &) = delete;
    Line &operator=(const Line &) = delete;

    ~Line() {
        if (active_) {
            std::cerr << format(level_, buffer_.str()) << "\n";
        }
    }

    template <typename T>
    Line &operator<<(const T &value) {
        if (active_) buffer_ << value;
        return *this;
    }

  private:
    Level level_;
    bool active_;
    std::ostringstream buffer_;
};

inline Line at(Level level) { return Line(level, enabled(level)); }
inline Line debug() { return at(Level::debug); }
inline Line info() { return at(Level::info); }
inline Line warn() { return at(Level::warn); }
inline Line error() { return at(Level::error); }

}  // namespace log
}  // namespace wildlife_trigger

#endif  // WILDLIFE_TRIGGER_LOGGING_HPP
