// Unit test for the logging convention (DESIGN §11 component 7).
//
// Asserts the on-the-wire format of each level -- because the exact prefixes are a
// contract, not cosmetics: validate.policy_rejections greps stderr for "error:" to
// confirm a bad policy was refused, and a result summary must read cleanly with no
// tag. It also pins the level ordering the threshold relies on.
//
// The environment threshold is read once per process, so this test asserts the
// default (info) rather than trying to mutate a cached static; the parsing itself
// is exercised by run_e1_foundation.sh, which drives the CLI under
// WILDLIFE_LOG_LEVEL=debug and =error.

#include "wildlife_trigger/logging.hpp"

#include <cassert>
#include <cstdio>
#include <string>

namespace log = wildlife_trigger::log;
using wildlife_trigger::log::Level;

int main() {
    // The prefixes are the contract. error keeps the "error:" substring that
    // policy_rejections depends on; info is untagged so the human summary is clean.
    assert(log::format(Level::error, "boom") == "error: boom");
    assert(log::format(Level::warn, "careful") == "warning: careful");
    assert(log::format(Level::info, "SHUTTER_TRIGGER=1") == "SHUTTER_TRIGGER=1");
    assert(log::format(Level::debug, "trace") == "debug: trace");

    // The "error:" grep contract, stated as the test that would fail if the prefix
    // ever drifts to e.g. "[error]".
    assert(log::format(Level::error, "x").find("error:") != std::string::npos);

    // Level ordering underpins the threshold comparison.
    assert(static_cast<int>(Level::debug) < static_cast<int>(Level::info));
    assert(static_cast<int>(Level::info) < static_cast<int>(Level::warn));
    assert(static_cast<int>(Level::warn) < static_cast<int>(Level::error));

    // Default threshold is info: debug is suppressed, everything else passes.
    assert(!log::enabled(Level::debug));
    assert(log::enabled(Level::info));
    assert(log::enabled(Level::warn));
    assert(log::enabled(Level::error));

    // A suppressed line must be inert and safe to stream into; an active one must
    // not throw. Neither writes anything this test inspects -- the point is that
    // building and destroying a Line is well-behaved on both branches.
    log::debug() << "suppressed " << 42 << " " << 3.14;
    log::info() << "active " << 7;

    std::printf("PASS\n");
    return 0;
}
