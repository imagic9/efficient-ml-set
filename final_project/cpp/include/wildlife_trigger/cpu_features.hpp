// Report the ARM64 CPU features the kernel advertises to userspace.
//
// This is not diagnostics for its own sake. ONNX Runtime / MLAS selects its
// kernels from these HWCAP bits at runtime, so they decide whether an INT8
// convolution takes an i8mm path, a dotprod path, or plain NEON -- and therefore
// the accumulation order, and therefore the numbers. DESIGN §4 requires every
// run to record them, because gx10 (Cortex-X925: i8mm + SVE2) and the Pi 5
// (Cortex-A76: dotprod only) differ here, and `qemu-aarch64 -cpu cortex-a76`
// exists precisely to make gx10 answer the way the Pi would.

#ifndef WILDLIFE_TRIGGER_CPU_FEATURES_HPP
#define WILDLIFE_TRIGGER_CPU_FEATURES_HPP

#include <string>

namespace wildlife_trigger {

struct CpuFeatures {
    bool asimd = false;    // NEON
    bool asimddp = false;  // dot product: Pi 5 yes, Pi 4 no
    bool sve = false;      // gx10 only
    bool sve2 = false;     // gx10 only
    bool i8mm = false;     // gx10 only
    bool bf16 = false;     // gx10 only

    // True when the feature set matches what a Cortex-A76 advertises, i.e. what
    // ORT would see on a Pi 5. Under `qemu-aarch64 -cpu cortex-a76` this holds on
    // gx10, which is how the parity rehearsal is validated as actually in effect.
    bool looks_like_pi5() const {
        return asimd && asimddp && !sve && !sve2 && !i8mm && !bf16;
    }
};

// Reads AT_HWCAP / AT_HWCAP2. On a non-ARM64 build every field stays false.
CpuFeatures detect_cpu_features();

// One-line summary for logs and provenance, e.g. "asimd,asimddp".
std::string describe(const CpuFeatures &features);

}  // namespace wildlife_trigger

#endif  // WILDLIFE_TRIGGER_CPU_FEATURES_HPP
