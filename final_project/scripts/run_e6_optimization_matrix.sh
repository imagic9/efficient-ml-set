#!/usr/bin/env bash
# E6 — the inference-pipeline optimization matrix, on M0 (PLAN E6, DESIGN §11/§12).
#
# One factor at a time, off a fixed baseline (fused preprocessing, full decode,
# ORT_ENABLE_ALL, CPU arena on, one thread). Each cell is a `benchmark` run whose
# self-describing pipeline_config lets the collator attribute the latency delta to the
# single knob that moved. The baseline model is M0 — Phase E8 requires the baseline in
# the measurement matrix, and reduced-decode/preprocess are model-independent anyway.
#
# DIAGNOSTIC ONLY. These are gx10 latencies; DESIGN §12.4 makes a latency a Pi result
# only when measured ON a Pi, and every cell's measured_on_pi is false. The matrix
# ranks which knobs are worth carrying to the Pi, where the measurement that counts is
# taken. INT8 thread scaling in particular is a Phase-F measurement, not this table.
#
# Usage:  scripts/run_e6_optimization_matrix.sh [output_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
OUT="${1:-${PROJECT_ROOT}/results/e6/optimization_matrix}"
BUILD_DIR="/work/build/e6opt"
IN_CONTAINER=(docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${TARGET_IMAGE_TAG}")

FREEZE="results/model_selection/pre_pi_freeze.json"
CLASS_MAP="artifacts/class_map.json"
POLICY="artifacts/policies/bobcat_v1.json"
FIXTURE="tests/fixtures/frame_1024x747.jpg"
WARMUP=10
ITERS=200

cd "${PROJECT_ROOT}"

M0_ONNX="$("${PYTHON}" -c "
import json
m0 = next(m for m in json.load(open('${FREEZE}'))['models'] if m['model_id'] == 'M0')
print(m0['onnx']['artifact'])")"
[[ -f "${M0_ONNX}" ]] || { echo "M0 ONNX not found at ${M0_ONNX}" >&2; exit 2; }

REL_OUT="${OUT#"${PROJECT_ROOT}"/}"
rm -rf "${OUT}"
mkdir -p "${OUT}"

echo "=============================================================="
echo "E6 optimization matrix on M0 (DIAGNOSTIC — not a Pi result) -> ${OUT}"
echo "=============================================================="

echo
echo "--- build inside ${TARGET_IMAGE_TAG} (target: ${QEMU_PI5_CPU})"
"${IN_CONTAINER[@]}" bash -lc "
    set -euo pipefail
    cmake -S /work/cpp -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release \
        -DWILDLIFE_CPU_TARGET=${QEMU_PI5_CPU} >/dev/null
    cmake --build ${BUILD_DIR} -j\"\$(nproc)\" >/dev/null
    echo '    built'
"

bench() {  # $1=cell_label  $2..=extra flags
    local label="$1"; shift
    echo "    ${label}: $*"
    "${IN_CONTAINER[@]}" bash -lc "
        ${BUILD_DIR}/wildlife_trigger benchmark \
            --model /work/${M0_ONNX} --class-map /work/${CLASS_MAP} \
            --policy /work/${POLICY} --image /work/${FIXTURE} \
            --warmup ${WARMUP} --iterations ${ITERS} $* \
            --output /work/${REL_OUT}/bench_${label}.json
    " >/dev/null 2> >(sed 's/^/        /' >&2)
}

echo
echo "--- ${ITERS} iterations per cell, one factor at a time off the baseline"
bench baseline
bench preprocess_reference --preprocess reference
bench decode_half          --decode half
bench decode_quarter       --decode quarter
bench graph_extended       --graph-opt extended
bench arena_off            --arena off
bench threads_2            --threads 2
bench threads_4            --threads 4

echo
echo "--- collate the matrix (diagnostic; asserts one-factor-at-a-time and off-Pi)"
"${PYTHON}" -m wildlife_trigger.validate.optimization_matrix \
    --dir "${OUT}" \
    --output "${OUT}/../optimization_matrix.json"
