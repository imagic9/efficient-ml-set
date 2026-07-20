#!/bin/sh
# Wildlife Trigger — Raspberry Pi installer (E7, hardened for F1 by issue #77).
#
# Runs ON the Pi (or any clean Ubuntu 24.04 ARM64 host), with no access to the
# training machine. It fails closed on a host outside the Pi 5 / Ubuntu 24.04 contract
# BEFORE touching the system, then proves the bundle arrived intact, installs OpenCV,
# resolves libraries, and records a machine-readable environment.json.
#
# What it installs and why:
#   - libonnxruntime.so travels IN the bundle (no apt package exists for the pinned
#     build; P0 proved it needs only GLIBC_2.27, so the same file that ran in the
#     build container runs here). run.sh points the loader at it.
#   - OpenCV does NOT travel in the bundle: its libopencv_imgcodecs drags a
#     ~50-library GDAL/poppler/database closure that is impractical to carry. Instead
#     this installs the OpenCV 4.6.0 runtime from apt. Ubuntu 24.04 ships that 4.6.0 as
#     the libopencv-*406t64 packages (the t64 = 64-bit time_t rename), which carry the
#     SAME .406 soname the binary was linked against. (Another distro/version ships a
#     different soname; preflight refuses it. See README for the rebuild contingency.)
#
# POSIX sh, no bashisms: a fresh Ubuntu /bin/sh is dash.
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

# 2. FAIL-CLOSED preflight (issue #77): refuse a host outside the Pi 5 / Ubuntu 24.04
#    contract BEFORE apt mutates anything. preflight.sh prints KEY=VALUE facts on
#    stdout and the human summary / refusal reason on stderr.
echo "--- host preflight (fail closed)"
if ! facts="$("${HERE}/preflight.sh")"; then
    echo "    install refused: this host is not the validated Pi 5 / Ubuntu 24.04 target." >&2
    echo "    Nothing was changed. See the reasons above and deploy/pi/README.md." >&2
    exit 1
fi
eval "${facts}"

# 3. OpenCV runtime from apt (unless suppressed). The exact 4.6.0 .406 soname; the
#    preflight already proved this host is Ubuntu 24.04, where it is available as the
#    t64-renamed packages. Names must match configs/env/pins.env OPENCV_RUNTIME_PKGS.
DID_APT=false
if [ "${NO_APT}" -eq 0 ]; then
    echo "--- installing the OpenCV 4.6.0 runtime (apt)"
    if command -v apt-get >/dev/null 2>&1; then
        SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
        ${SUDO} apt-get update -qq
        # Confirm the .406 candidate exists before mutating (belt-and-suspenders on
        # top of the Ubuntu 24.04 gate).
        if ! apt-cache policy libopencv-core406t64 2>/dev/null | grep -q 'Candidate:'; then
            echo "    FAIL: libopencv-core406t64 has no apt candidate on this host" >&2
            echo "    (expected on Ubuntu 24.04; if you see this, the apt sources are wrong)" >&2
            exit 1
        fi
        ${SUDO} apt-get install -y --no-install-recommends \
            libopencv-core406t64 libopencv-imgproc406t64 libopencv-imgcodecs406t64
        DID_APT=true
        echo "    OpenCV runtime installed"
    else
        echo "    apt-get absent; install libopencv-{core,imgproc,imgcodecs}406t64 by hand" >&2
    fi
else
    echo "--- skipping apt (per --no-apt); OpenCV must already be present"
fi

# 4. Resolve every NEEDED library now, so a missing dep is a message here, not a
#    cryptic loader failure later. run.sh supplies the bundle's lib/ for ORT.
echo "--- resolving shared libraries"
LIBS_OK=true
if LD_LIBRARY_PATH="${HERE}/lib" ldd "${HERE}/bin/wildlife_trigger" | grep -q 'not found'; then
    LIBS_OK=false
    echo "    FAIL: unresolved libraries:" >&2
    LD_LIBRARY_PATH="${HERE}/lib" ldd "${HERE}/bin/wildlife_trigger" | grep 'not found' >&2
    echo "    (on Ubuntu 24.04 run without --no-apt; on another OS see README)" >&2
fi
${LIBS_OK} && echo "    all libraries resolve"

# 5. Machine-readable host-environment record (issue #77): OS, kernel, arch, CPU
#    identity/features, glibc, OpenCV/ORT versions, and the install outcome. This is
#    the artifact a real F1 install leaves for the Phase F evidence.
ORT_VER="$(grep -m1 '"onnxruntime_version"' "${HERE}/BUNDLE.json" 2>/dev/null | sed 's/.*: *"//; s/".*//')"
OPENCV_VER="$(dpkg-query -W -f='${Version}' libopencv-core406t64 2>/dev/null || echo 'not-installed')"
bool() { if [ "$1" = "1" ] || [ "$1" = "true" ]; then echo true; else echo false; fi; }
OUTCOME=installed; ${LIBS_OK} || OUTCOME=libraries_unresolved
cat > "${HERE}/environment.json" <<EOF
{
  "kind": "f1_host_environment",
  "os": {"id": "${WT_OS_ID}", "version_id": "${WT_OS_VERSION}", "codename": "${WT_OS_CODENAME}"},
  "kernel": "${WT_KERNEL}",
  "arch": "${WT_ARCH}",
  "cpu": {"implementer": "${WT_CPU_IMPL}", "part": "${WT_CPU_PART}", "model": "${WT_CPU_MODEL}",
          "asimddp": $(bool "${WT_HAS_ASIMDDP}"), "is_pi5_a76": $(bool "${WT_IS_PI5_A76}")},
  "glibc": "${WT_GLIBC}",
  "onnxruntime_version": "${ORT_VER}",
  "opencv_runtime_installed": "${OPENCV_VER}",
  "preflight": {"passed": $(bool "${WT_PREFLIGHT_PASSED}"), "reasons": "${WT_PREFLIGHT_REASONS}"},
  "install": {"outcome": "${OUTCOME}", "opencv_apt": ${DID_APT}, "libraries_resolved": ${LIBS_OK}},
  "note": "A latency is a Pi result only when measured on a Pi (DESIGN 12.4). is_pi5_a76=false is a diagnostic host, never a Pi verdict."
}
EOF
echo "    wrote environment.json (is_pi5_a76=$(bool "${WT_IS_PI5_A76}"), opencv=${OPENCV_VER})"

${LIBS_OK} || { echo "install FAILED: unresolved libraries" >&2; exit 1; }
echo
echo "=== installed. Try the demo:"
echo "      ${HERE}/run_demo.sh"
