#!/bin/sh
# Wildlife Trigger — F1 host preflight (issue #77). FAIL CLOSED.
#
# Gate E proved the bundle on clean Ubuntu 24.04 ARM64. It did NOT prove that an
# arbitrary rented physical host matches the Pi 5 / Ubuntu 24.04 contract the prebuilt
# binary was built for (`-mcpu=cortex-a76`, glibc 2.39, OpenCV `.406`). This runs BEFORE
# install.sh mutates anything and refuses, loudly and non-zero, on a host outside that
# contract — so a mismatch costs a message here, not a day of rental time after `apt`
# has already changed the target and `ldd` fails late.
#
# Three gates:
#   1. architecture is aarch64;
#   2. OS is **Ubuntu 24.04** (the build target: glibc 2.39 + apt OpenCV 4.6.0 as
#      libopencv-*406t64). Another distro/version ships a different glibc and OpenCV
#      soname, so the prebuilt binary would fault or fail to load;
#   3. the CPU provides **asimddp** — the ARMv8.2 dot-product feature Cortex-A76 (Pi 5)
#      has and Cortex-A72 (Pi 4) lacks, and which the INT8 kernels dispatch on. Gating on
#      the ISA feature (not the literal CPU part) is deliberate: it accepts a Pi 5 AND a
#      dev host like gx10 (so the E8 dry run still runs), while refusing a Pi 4. Whether
#      the host is *literally* a Pi 5 (`is_pi5_a76`) is RECORDED for the Phase F evidence,
#      never confused with a Pi result (DESIGN §12.4).
#
# Machine-readable facts go to stdout as KEY=VALUE (install.sh captures them into
# environment.json); the human summary and any refusal reason go to stderr. Exit 0 iff
# all gates pass.
#
# Inputs are overridable for testing (the refusal paths must be provable without a real
# Pi 4 / wrong-OS host):
#   WT_UNAME_M    (default `uname -m`)
#   WT_OS_RELEASE (default /etc/os-release)
#   WT_CPUINFO    (default /proc/cpuinfo)
#
# POSIX sh. Usage:  ./preflight.sh

set -eu

OS_RELEASE="${WT_OS_RELEASE:-/etc/os-release}"
CPUINFO="${WT_CPUINFO:-/proc/cpuinfo}"
ARCH="${WT_UNAME_M:-$(uname -m)}"

field() {  # $1=key $2=file — first "key : value" or "KEY=value", trimmed
    grep -m1 "$1" "$2" 2>/dev/null | cut -d: -f2- 2>/dev/null | sed 's/^[ =]*//; s/"//g; s/[[:space:]]*$//'
}

OS_CODENAME="$(grep -m1 '^VERSION_CODENAME=' "${OS_RELEASE}" 2>/dev/null | cut -d= -f2 | tr -d '"' || true)"
OS_ID="$(grep -m1 '^ID=' "${OS_RELEASE}" 2>/dev/null | cut -d= -f2 | tr -d '"' || true)"
OS_VERSION="$(grep -m1 '^VERSION_ID=' "${OS_RELEASE}" 2>/dev/null | cut -d= -f2 | tr -d '"' || true)"
KERNEL="$(uname -r 2>/dev/null || echo unknown)"

CPU_PART="$(field 'CPU part' "${CPUINFO}")"
CPU_IMPL="$(field 'CPU implementer' "${CPUINFO}")"
CPU_MODEL="$(field '^Model' "${CPUINFO}")"
[ -n "${CPU_MODEL}" ] || CPU_MODEL="$(field 'model name' "${CPUINFO}")"
if grep -m1 'Features' "${CPUINFO}" 2>/dev/null | grep -qw asimddp; then
    HAS_ASIMDDP=1
else
    HAS_ASIMDDP=0
fi
# Cortex-A76 (Raspberry Pi 5) is implementer 0x41 (ARM), part 0xd0b.
if [ "${CPU_IMPL}" = "0x41" ] && [ "${CPU_PART}" = "0xd0b" ]; then
    IS_PI5_A76=1
else
    IS_PI5_A76=0
fi
GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo unknown)"

# --- the gates -------------------------------------------------------------
reasons=""
add() { reasons="${reasons}${reasons:+; }$1"; }

[ "${ARCH}" = "aarch64" ] || add "architecture is '${ARCH}', not aarch64"

{ [ "${OS_ID}" = "ubuntu" ] && [ "${OS_VERSION}" = "24.04" ]; } || add \
"OS is '${OS_ID:-?} ${OS_VERSION:-?} (${OS_CODENAME:-?})', not Ubuntu 24.04. The prebuilt bundle is built on Ubuntu 24.04 (glibc 2.39) and apt-installs OpenCV 4.6.0 from Ubuntu's libopencv-*406t64; another distro/version ships a different glibc and/or OpenCV soname. Rebuild the app on the Pi against its apt OpenCV (see README) — do not run this bundle here"

[ "${HAS_ASIMDDP}" = "1" ] || add \
"CPU lacks the 'asimddp' feature (ARMv8.2 dot-product). This bundle targets Cortex-A76 (Pi 5); Cortex-A72 (Pi 4) lacks it and the binary may fault or mis-execute. Use the documented Pi 4 contingency (see README) — do not run this bundle here"

if [ -z "${reasons}" ]; then PASSED=1; else PASSED=0; fi

# --- machine-readable facts to stdout (install.sh evals these) -------------
cat <<EOF
WT_ARCH='${ARCH}'
WT_OS_ID='${OS_ID}'
WT_OS_CODENAME='${OS_CODENAME}'
WT_OS_VERSION='${OS_VERSION}'
WT_KERNEL='${KERNEL}'
WT_CPU_IMPL='${CPU_IMPL}'
WT_CPU_PART='${CPU_PART}'
WT_CPU_MODEL='${CPU_MODEL}'
WT_HAS_ASIMDDP='${HAS_ASIMDDP}'
WT_IS_PI5_A76='${IS_PI5_A76}'
WT_GLIBC='${GLIBC}'
WT_PREFLIGHT_PASSED='${PASSED}'
WT_PREFLIGHT_REASONS='${reasons}'
EOF

# --- human summary to stderr -----------------------------------------------
{
    echo "  preflight: arch=${ARCH} os=${OS_ID:-?}/${OS_CODENAME:-?} cpu=${CPU_MODEL:-part ${CPU_PART:-?}}"
    echo "             asimddp=${HAS_ASIMDDP} is_pi5_a76=${IS_PI5_A76} glibc=${GLIBC}"
    if [ "${PASSED}" = "1" ]; then
        echo "  preflight PASSED (host satisfies the Pi 5 / Ubuntu 24.04 ISA contract)"
        [ "${IS_PI5_A76}" = "1" ] || echo "  note: this is NOT a literal Cortex-A76 Pi 5 — a run here is diagnostic, never a Pi result (DESIGN 12.4)"
    else
        echo "  preflight FAILED — refusing before any change to this host:"
        printf '    - %s\n' "${reasons}"
    fi
} >&2

[ "${PASSED}" = "1" ]
