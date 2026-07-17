#!/bin/sh
# Wildlife Trigger — Raspberry Pi installer (E7).
#
# Runs ON the Pi (or any clean glibc-2.36+ ARM64 host), with no access to the
# training machine. It proves the bundle arrived intact, proves the binary can load
# on this glibc, resolves OpenCV, and leaves a ready-to-run install.
#
# What it installs and why:
#   - libonnxruntime.so travels IN the bundle (no apt package exists for the pinned
#     build; P0 proved it needs only GLIBC_2.27, so the same file that ran in the
#     build container runs here). run.sh points the loader at it.
#   - OpenCV does NOT travel in the bundle: Debian's libopencv_imgcodecs drags a
#     ~50-library GDAL/poppler/database closure that is impractical to carry. Instead
#     this installs the OpenCV 4.6.0 runtime from apt. Raspberry Pi OS Bookworm is
#     Debian bookworm, so its libopencv-*406 packages are the SAME version the binary
#     was linked against — a byte-compatible soname (.406). (A Trixie Pi ships a
#     different soname; see README for that contingency.)
#
# POSIX sh, no bashisms: a fresh Pi OS /bin/sh is dash.
#
# Usage:  ./install.sh [--no-apt]      (--no-apt skips the OpenCV apt step)

set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
NO_APT=0
[ "${1:-}" = "--no-apt" ] && NO_APT=1

echo "=== Wildlife Trigger install (bundle at ${HERE}) ==="

# 1. Integrity: what arrived must be what left.
echo "--- verifying MANIFEST.sha256"
( cd "${HERE}" && sha256sum -c MANIFEST.sha256 >/dev/null ) \
    && echo "    all files verified" \
    || { echo "    FAIL: checksum mismatch — the bundle is corrupt or altered" >&2; exit 1; }

# 2. Loadability: prove the binary's required GLIBC symbols exist on THIS host before
#    the first run, not during a field deployment.
echo "--- checking the binary loads on this glibc"
HOST_GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo unknown)"
echo "    host glibc: ${HOST_GLIBC}"
if command -v objdump >/dev/null 2>&1; then
    NEED="$(objdump -T "${HERE}/bin/wildlife_trigger" "${HERE}/lib/"libonnxruntime.so.* 2>/dev/null \
            | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sed 's/GLIBC_//' | sort -V | tail -1)"
    echo "    highest GLIBC symbol required: ${NEED:-none}"
else
    echo "    objdump absent; skipping symbol scan (ldd below still gates)"
fi

# 3. OpenCV runtime from apt (unless suppressed). The exact 4.6.0 .406 soname.
if [ "${NO_APT}" -eq 0 ]; then
    echo "--- installing the OpenCV 4.6.0 runtime (apt)"
    if command -v apt-get >/dev/null 2>&1; then
        SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
        ${SUDO} apt-get update -qq
        ${SUDO} apt-get install -y --no-install-recommends \
            libopencv-core406 libopencv-imgproc406 libopencv-imgcodecs406
        echo "    OpenCV runtime installed"
    else
        echo "    apt-get absent; install libopencv-{core,imgproc,imgcodecs}406 by hand" >&2
    fi
else
    echo "--- skipping apt (per --no-apt); OpenCV must already be present"
fi

# 4. Resolve every NEEDED library now, so a missing dep is a message here, not a
#    cryptic loader failure later. run.sh supplies the bundle's lib/ for ORT.
echo "--- resolving shared libraries"
if LD_LIBRARY_PATH="${HERE}/lib" ldd "${HERE}/bin/wildlife_trigger" | grep -q 'not found'; then
    echo "    FAIL: unresolved libraries:" >&2
    LD_LIBRARY_PATH="${HERE}/lib" ldd "${HERE}/bin/wildlife_trigger" | grep 'not found' >&2
    echo "    (on Bookworm run without --no-apt; on Trixie see README)" >&2
    exit 1
fi
echo "    all libraries resolve"

echo
echo "=== installed. Try the demo:"
echo "      ${HERE}/run_demo.sh"
