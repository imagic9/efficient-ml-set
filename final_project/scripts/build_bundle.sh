#!/usr/bin/env bash
# Build and stage the ARM64 deployment bundle (PLAN A4, provisional; E7 hardens it).
#
# A bundle is what actually reaches the Pi. Everything about this script exists to
# make that transfer boring:
#
#   - it is built inside wildlife-trigger-target:bookworm, so every object links
#     against glibc 2.36 -- gx10's native 2.39 would produce a binary the Pi's loader
#     refuses outright;
#   - libonnxruntime.so travels WITH the bundle. P0 measured it needing only
#     GLIBC_2.27, so the same file serves the container and the Pi, and shipping it
#     removes any question of what the Pi's apt would have installed;
#   - every file is checksummed into MANIFEST.sha256, so what arrives can be proved
#     to be what left;
#   - the GLIBC audit runs against the staged binary, not the build tree, because the
#     staged binary is the artifact.
#
# Deliberately NOT bundled: the session-optimized ONNX graph. ORT warns that a graph
# serialized above ORT_ENABLE_EXTENDED "should only be used in the same environment
# the model was optimized in" (measured in P0). The Pi gets the ordinary model and
# optimizes it itself.
#
# OpenCV is NOT bundled either, and that is a known gap E7 must close: bookworm apt
# gives 4.6.0 and a Trixie Pi would give another soname. See pins.env.
#
# Usage:  scripts/build_bundle.sh <staging_dir> [artifacts_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"
STAGING="${1:?usage: build_bundle.sh <staging_dir> [artifacts_dir]}"
ARTIFACTS="${2:-${PROJECT_ROOT}/results/a4/artifacts}"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

BUILD_DIR="/work/build/bundle"
REL_STAGING="${STAGING#"${PROJECT_ROOT}"/}"
REL_ARTIFACTS="${ARTIFACTS#"${PROJECT_ROOT}"/}"

echo "Building the deployment bundle in ${TARGET_IMAGE_TAG}"
echo "  staging  : ${STAGING}"
echo "  artifacts: ${ARTIFACTS}"

rm -rf "${STAGING}"
mkdir -p "${STAGING}"

docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${TARGET_IMAGE_TAG}" bash -lc "
set -euo pipefail

# -mcpu=cortex-a76 targets the Pi 5. Never 'native': on gx10 that is a Cortex-X925
# with i8mm and SVE2 the Pi does not have, and the CMake guard rejects it anyway.
cmake -S /work/cpp -B ${BUILD_DIR} \
    -DCMAKE_BUILD_TYPE=Release \
    -DWILDLIFE_CPU_TARGET=${QEMU_PI5_CPU} >/dev/null
cmake --build ${BUILD_DIR} --target wildlife_trigger -j\"\$(nproc)\" >/dev/null

STAGE=/work/${REL_STAGING}
mkdir -p \"\$STAGE/bin\" \"\$STAGE/lib\" \"\$STAGE/models\" \"\$STAGE/policies\"

install -m 0755 ${BUILD_DIR}/wildlife_trigger \"\$STAGE/bin/\"

# Copy the real file behind the soname chain, then recreate the links the loader
# looks for. cp -P alone would stage a dangling symlink.
ORT_REAL=\"\$(readlink -f ${ORT_PREFIX}/lib/libonnxruntime.so)\"
install -m 0644 \"\$ORT_REAL\" \"\$STAGE/lib/\$(basename \"\$ORT_REAL\")\"
ln -sf \"\$(basename \"\$ORT_REAL\")\" \"\$STAGE/lib/libonnxruntime.so.1\"
ln -sf \"\$(basename \"\$ORT_REAL\")\" \"\$STAGE/lib/libonnxruntime.so\"

cp -f /work/${REL_ARTIFACTS}/*.onnx \"\$STAGE/models/\" 2>/dev/null || true
cp -f /work/${REL_ARTIFACTS}/class_map.json \"\$STAGE/policies/\" 2>/dev/null || true
cp -f /work/${REL_ARTIFACTS}/*_v0.json \"\$STAGE/policies/\" 2>/dev/null || true

# The launcher. The Pi has no reason to know where the libraries went, and setting
# LD_LIBRARY_PATH by hand at 3am in a field is how a deployment fails.
cat > \"\$STAGE/bin/run.sh\" <<'LAUNCHER'
#!/bin/sh
# Bundle launcher: resolve libonnxruntime.so from the bundle, not from the system.
here=\"\$(cd \"\$(dirname \"\$0\")/..\" && pwd)\"
LD_LIBRARY_PATH=\"\$here/lib:\${LD_LIBRARY_PATH:-}\" exec \"\$here/bin/wildlife_trigger\" \"\$@\"
LAUNCHER
chmod 0755 \"\$STAGE/bin/run.sh\"

echo
echo 'GLIBC audit against the STAGED binary:'
/work/scripts/audit_target_compat.sh \"\$STAGE/bin/wildlife_trigger\" | sed 's/^/  /'

# Checksums last, over everything staged. Sorted so the manifest is reproducible.
cd \"\$STAGE\"
find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
echo
echo \"staged \$(wc -l < MANIFEST.sha256) files\"
"

echo
echo "Bundle contents:"
find "${STAGING}" -type f -o -type l | sort | sed "s|${STAGING}|  .|"
echo
echo "Bundle size: $(du -sh "${STAGING}" | cut -f1)"
echo
echo "Verify on the target with:"
echo "  cd <bundle> && sha256sum -c MANIFEST.sha256"
echo "  ./bin/run.sh self-test --model models/<m>.onnx --class-map policies/class_map.json \\"
echo "      --policy policies/bobcat_v0.json --image <frame>.jpg"
