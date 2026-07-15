#include "wildlife_trigger/cpu_features.hpp"

#include <string>

#if defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace wildlife_trigger {

CpuFeatures detect_cpu_features() {
    CpuFeatures f;
#if defined(__aarch64__)
    const unsigned long hwcap = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
    f.asimd = (hwcap & HWCAP_ASIMD) != 0;
    f.asimddp = (hwcap & HWCAP_ASIMDDP) != 0;
    f.sve = (hwcap & HWCAP_SVE) != 0;
    f.sve2 = (hwcap2 & HWCAP2_SVE2) != 0;
    f.i8mm = (hwcap2 & HWCAP2_I8MM) != 0;
    f.bf16 = (hwcap2 & HWCAP2_BF16) != 0;
#endif
    return f;
}

std::string describe(const CpuFeatures &f) {
    std::string out;
    const auto add = [&out](bool present, const char *name) {
        if (!present) return;
        if (!out.empty()) out += ",";
        out += name;
    };
    add(f.asimd, "asimd");
    add(f.asimddp, "asimddp");
    add(f.sve, "sve");
    add(f.sve2, "sve2");
    add(f.i8mm, "i8mm");
    add(f.bf16, "bf16");
    return out.empty() ? "none" : out;
}

}  // namespace wildlife_trigger
