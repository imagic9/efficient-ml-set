#!/usr/bin/env bash
# End-to-end check of the target-compatible environment (PLAN A2 gate).
#
# Runs entirely inside the bookworm container: build the C++ with the Pi's
# compiler and glibc, prove the result can load on the Pi, and prove it behaves
# as the Pi would under HWCAP emulation.
#
# This is the A2 evidence that Gate A depends on, so it must be runnable
# unattended and must fail loudly.
#
# Usage:  scripts/verify_target_env.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${TARGET_IMAGE_TAG}" bash -euo pipefail -c '
BUILD=/tmp/target_build

echo "### 1. Build with the target toolchain"
echo "  gcc $(gcc -dumpfullversion)  glibc $(ldd --version | head -1 | grep -oE "[0-9]+\.[0-9]+$")"
cmake -S cpp -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DWILDLIFE_CPU_TARGET=cortex-a76 >/dev/null
cmake --build "$BUILD" -j"$(nproc)" >/dev/null
echo "  built with -mcpu=cortex-a76"

echo
echo "### 2. Can it load on the Pi?"
scripts/audit_target_compat.sh "$BUILD/tests/test_cpu_features" "$ORT_PREFIX/lib/libonnxruntime.so"

echo
echo "### 3. Native run in the container (gx10 CPU, sees i8mm/SVE2)"
"$BUILD/tests/test_cpu_features"

echo
echo "### 4. Under qemu -cpu cortex-a76 (the Pi 5 feature set)"
qemu-aarch64 -cpu cortex-a76 "$BUILD/tests/test_cpu_features"

echo
echo "### 5. ctest, natively and through emulation"
ctest --test-dir "$BUILD" --output-on-failure 2>&1 | tail -3

echo
echo "### 6. ORT links and reports its version"
cat > /tmp/ortver.cpp <<CPP
#include <onnxruntime_cxx_api.h>
#include <cstdio>
int main() {
  std::printf("  ORT %s\n", Ort::GetVersionString().c_str());
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "probe");
  std::printf("  session env constructed OK\n");
  return 0;
}
CPP
g++ -O2 -std=c++17 -I"$ORT_PREFIX/include" /tmp/ortver.cpp \
    -L"$ORT_PREFIX/lib" -lonnxruntime -Wl,-rpath,"$ORT_PREFIX/lib" -o /tmp/ortver
/tmp/ortver
scripts/audit_target_compat.sh /tmp/ortver | tail -1

echo
echo "### 7. ORT under the Pi feature set"
qemu-aarch64 -cpu cortex-a76 /tmp/ortver
'

echo
echo "TARGET ENVIRONMENT VERIFIED"
