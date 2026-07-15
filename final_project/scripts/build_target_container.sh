#!/usr/bin/env bash
# Build the target-compatible ARM64 container (PLAN A2).
#
# All pins come from configs/env/pins.env so this script and the Dockerfile
# cannot disagree about what is being built.
#
# Usage:  scripts/build_target_container.sh [--no-cache]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

echo "Building ${TARGET_IMAGE_TAG}"
echo "  base: ${TARGET_BASE_IMAGE}@${TARGET_BASE_DIGEST}"
echo "  ort : ${ORT_VERSION}"

docker build \
    -f "${PROJECT_ROOT}/docker/Dockerfile.target" \
    -t "${TARGET_IMAGE_TAG}" \
    --build-arg "TARGET_BASE_IMAGE=${TARGET_BASE_IMAGE}" \
    --build-arg "TARGET_BASE_DIGEST=${TARGET_BASE_DIGEST}" \
    --build-arg "ORT_VERSION=${ORT_VERSION}" \
    --build-arg "ORT_SHA256=${ORT_SHA256}" \
    --build-arg "ORT_URL=${ORT_URL}" \
    --build-arg "ORT_PREFIX=${ORT_PREFIX}" \
    --build-arg "UID=$(id -u)" \
    --build-arg "GID=$(id -g)" \
    "$@" \
    "${PROJECT_ROOT}/docker"

echo
echo "Verifying the target contract inside the image:"
docker run --rm "${TARGET_IMAGE_TAG}" sh -c '
    echo "  distro : $(. /etc/os-release && echo "$PRETTY_NAME")"
    echo "  glibc  : $(ldd --version | head -1 | grep -oE "[0-9]+\.[0-9]+$")"
    echo "  gcc    : $(gcc -dumpversion)"
    echo "  cmake  : $(cmake --version | head -1 | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")"
    echo "  opencv : $(pkg-config --modversion opencv4 2>/dev/null || echo unknown)"
    echo "  qemu   : $(qemu-aarch64 --version | head -1 | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")"
    # Resolve the symlink: libonnxruntime.so.1 is a link and would report "1".
    echo "  ort    : $(basename "$(readlink -f /opt/onnxruntime/lib/libonnxruntime.so)" | sed "s/^libonnxruntime\.so\.//")"
'

echo
echo "Run it with:"
echo "  docker run --rm -it -v \"${PROJECT_ROOT}:/work\" ${TARGET_IMAGE_TAG}"
