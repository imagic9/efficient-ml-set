#!/usr/bin/env bash
# E6 — native-vs-target build-and-test (PLAN E6, DESIGN §11).
#
# Build and run the whole suite under TWO toolchains on gx10: the native one
# (Ubuntu 24.04, gcc 13, glibc 2.39 — reproduced by wildlife-trigger-native so the
# host stays untouched) and the target one (Debian bookworm, gcc 12, glibc 2.36).
# Both carry the same pinned ORT and OpenCV 4.6.0, so a decision difference would be
# a compiler/glibc effect, not a library gap. Only the target build ships; the native
# build is a portability witness.
#
# For each toolchain: ctest, the on-device self-test on M0, and run-dataset over the
# full benchmark_val_1000. The gate asserts both green and byte-for-byte the same
# decisions across the two builds. Correctness only (DESIGN §12.4).
#
# Usage:  scripts/run_e6_native_vs_target.sh [output_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
OUT="${1:-${PROJECT_ROOT}/results/e6/native_vs_target}"

FREEZE="results/model_selection/pre_pi_freeze.json"
CLASS_MAP="artifacts/class_map.json"
POLICY="artifacts/policies/bobcat_v1.json"
FIXTURE="tests/fixtures/frame_1024x747.jpg"
MANIFEST="data/manifests/benchmark_val_1000.jsonl"
IMAGES_ROOT="data/raw/extracted/eccv_18_all_images_sm"

cd "${PROJECT_ROOT}"
[[ -d "${IMAGES_ROOT}" ]] || { echo "no images root at ${IMAGES_ROOT}" >&2; exit 2; }

read -r M0_ONNX THRESH < <("${PYTHON}" -c "
import json
m0 = next(m for m in json.load(open('${FREEZE}'))['models'] if m['model_id'] == 'M0')
print(m0['onnx']['artifact'], m0['policy']['threshold'])")
[[ -f "${M0_ONNX}" ]] || { echo "M0 ONNX not found at ${M0_ONNX}" >&2; exit 2; }

REL_OUT="${OUT#"${PROJECT_ROOT}"/}"
rm -rf "${OUT}"
mkdir -p "${OUT}"

echo "=============================================================="
echo "E6 native-vs-target build-and-test on M0 -> ${OUT}"
echo "=============================================================="

# Ensure the native image exists (the target image is already built).
if ! docker image inspect "${NATIVE_IMAGE_TAG}" >/dev/null 2>&1; then
    echo
    echo "--- native image ${NATIVE_IMAGE_TAG} not found; building it"
    bash "${PROJECT_ROOT}/scripts/build_native_container.sh"
fi

# $1=image  $2=cpu_target ("" for generic)  $3=build_subdir  $4=label
build_and_test() {
    local image="$1" cpu_target="$2" build_dir="$3" label="$4"
    local cpu_flag=""
    [[ -n "${cpu_target}" ]] && cpu_flag="-DWILDLIFE_CPU_TARGET=${cpu_target}"
    echo
    echo "--- [${label}] build + ctest in ${image}"
    docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${image}" bash -lc "
        set -euo pipefail
        cmake -S /work/cpp -B /work/${build_dir} -DCMAKE_BUILD_TYPE=Release ${cpu_flag} >/dev/null
        cmake --build /work/${build_dir} -j\"\$(nproc)\" >/dev/null
        cd /work/${build_dir} && ctest --output-on-failure 2>&1 | tail -3 | sed 's/^/    /'
        cd /work
        echo '    --- self-test on M0'
        /work/${build_dir}/wildlife_trigger self-test \
            --model /work/${M0_ONNX} --class-map /work/${CLASS_MAP} \
            --policy /work/${POLICY} --image /work/${FIXTURE} \
            --output /work/${REL_OUT}/selftest_${label}.json >/dev/null 2>&1 \
            && echo '    self-test PASSED' || echo '    self-test FAILED'
        echo '    --- run-dataset over benchmark_val_1000'
        /work/${build_dir}/wildlife_trigger run-dataset \
            --model /work/${M0_ONNX} --class-map /work/${CLASS_MAP} --policy /work/${POLICY} \
            --manifest /work/${MANIFEST} --images-root /work/${IMAGES_ROOT} \
            --threads 1 --on-corrupt fail --output /work/${REL_OUT}/rundataset_${label}.jsonl \
            2>&1 | tail -1 | sed 's/^/    /'
    "
}

# Native build: generic ARMv8 (runs on gx10, never shipped). Target build: cortex-a76.
build_and_test "${NATIVE_IMAGE_TAG}" "" "build/e6native" "native"
build_and_test "${TARGET_IMAGE_TAG}" "${QEMU_PI5_CPU}" "build/e6target" "target"

# ctest counts, re-run cheaply just to read the pass/total for the record.
ctest_count() {  # $1=image $2=build_dir
    docker run --rm -v "${PROJECT_ROOT}:/work" -w "/work/$2" "$1" bash -lc \
        "ctest 2>&1 | grep -oE '[0-9]+% tests passed, [0-9]+ tests failed out of [0-9]+'" \
        | "${PYTHON}" -c "import sys,re; m=re.search(r'(\d+) tests failed out of (\d+)', sys.stdin.read()); t=int(m.group(2)); f=int(m.group(1)); print(f'{t-f}/{t}')"
}
NATIVE_CTEST="$(ctest_count "${NATIVE_IMAGE_TAG}" build/e6native)"
TARGET_CTEST="$(ctest_count "${TARGET_IMAGE_TAG}" build/e6target)"

echo
echo "--- E6 native-vs-target gate (both green + cross-toolchain decision parity)"
"${PYTHON}" -m wildlife_trigger.validate.native_vs_target \
    --native-jsonl "${OUT}/rundataset_native.jsonl" \
    --target-jsonl "${OUT}/rundataset_target.jsonl" \
    --native-selftest "${OUT}/selftest_native.json" \
    --target-selftest "${OUT}/selftest_target.json" \
    --native-ctest "${NATIVE_CTEST}" \
    --target-ctest "${TARGET_CTEST}" \
    --output "${OUT}/../native_vs_target.json"
