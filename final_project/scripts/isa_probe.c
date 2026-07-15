/* Report the ARM64 CPU features the kernel advertises to userspace.
 *
 * ORT/MLAS picks its kernels from these HWCAP bits at runtime, so this is what
 * decides whether an INT8 convolution runs an i8mm path, a dotprod path, or
 * plain NEON -- and therefore which accumulation order, and therefore which
 * numbers. DESIGN §4 uses it to establish that gx10 and the Pi differ, and that
 * `qemu-aarch64 -cpu cortex-a76` reproduces the Pi's answer.
 *
 * Build and run:
 *     gcc -O2 -o isa_probe scripts/isa_probe.c
 *     ./isa_probe                                 # gx10 native
 *     qemu-aarch64 -cpu cortex-a76 ./isa_probe    # Pi 5 model
 *     qemu-aarch64 -cpu cortex-a72 ./isa_probe    # Pi 4 model
 *
 * Expected (measured 2026-07-15):
 *     native gx10   asimd Y  asimddp Y  sve Y  sve2 Y  i8mm Y  bf16 Y
 *     cortex-a76    asimd Y  asimddp Y  sve n  sve2 n  i8mm n  bf16 n
 *     cortex-a72    asimd Y  asimddp n  sve n  sve2 n  i8mm n  bf16 n
 */

#include <stdio.h>
#include <sys/auxv.h>
#include <asm/hwcap.h>

int main(void) {
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);

    struct feature {
        const char *name;
        int present;
    } features[] = {
        {"asimd (NEON)",    !!(hwcap  & HWCAP_ASIMD)},
        {"asimddp (dotprod)", !!(hwcap  & HWCAP_ASIMDDP)},
        {"sve",             !!(hwcap  & HWCAP_SVE)},
        {"sve2",            !!(hwcap2 & HWCAP2_SVE2)},
        {"i8mm",            !!(hwcap2 & HWCAP2_I8MM)},
        {"bf16",            !!(hwcap2 & HWCAP2_BF16)},
    };

    printf("AT_HWCAP=0x%lx AT_HWCAP2=0x%lx\n", hwcap, hwcap2);
    for (unsigned i = 0; i < sizeof features / sizeof *features; i++)
        printf("  %-20s %s\n", features[i].name,
               features[i].present ? "YES" : "no");
    return 0;
}
