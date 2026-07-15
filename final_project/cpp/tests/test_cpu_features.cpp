#include "wildlife_trigger/cpu_features.hpp"

#include <cassert>
#include <cstdio>

using wildlife_trigger::CpuFeatures;

int main() {
    const CpuFeatures f = wildlife_trigger::detect_cpu_features();
    std::printf("detected: %s\n", wildlife_trigger::describe(f).c_str());

#if defined(__aarch64__)
    // Every ARMv8-A CPU has NEON; its absence means the probe is broken rather
    // than that the hardware is exotic.
    assert(f.asimd && "aarch64 without asimd means HWCAP reading is broken");

    // Feature implications, not hardware trivia: these catch a probe that mixes
    // up HWCAP and HWCAP2 bit positions, which is the realistic failure here.
    assert((!f.sve2 || f.sve) && "sve2 without sve is impossible");
    assert((!f.i8mm || f.asimd) && "i8mm without asimd is impossible");
#endif

    // The Pi 5 predicate must be self-consistent: it cannot hold while any
    // gx10-only feature is present. This is the assertion the qemu rehearsal
    // depends on, so it is worth pinning.
    if (f.looks_like_pi5()) {
        assert(!f.i8mm && !f.sve && !f.sve2 && !f.bf16);
        assert(f.asimd && f.asimddp);
        std::printf("cpu profile: matches Cortex-A76 / Pi 5\n");
    } else {
        std::printf("cpu profile: does NOT match Pi 5 (expected on gx10 native)\n");
    }
    std::printf("PASS\n");
    return 0;
}
