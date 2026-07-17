#!/usr/bin/env bash
# E6 — the reduced-decode accuracy / decision-drift check (PLAN E6, DESIGN §11).
#
# Reduced JPEG decode (IMREAD_REDUCED_COLOR_2/4) is the one optimization-matrix knob
# that changes the tensor, so it is NOT preprocessing parity: DESIGN §11 keeps it only
# if the validation bobcat metrics hold within a predeclared tolerance. This runs the
# C++ dataset runner over the whole benchmark manifest three times per model — full,
# half, quarter decode — and the decode_drift gate compares each reduced variant to the
# model's own full-decode run and to the ground-truth labels.
#
# The safety-critical tolerance is ZERO lost true detections (a missed bobcat is the
# product's core failure); a small fraction of new false fires is tolerated. Threads
# are held at 1: this is about decisions, not latency, and the matrix owns latency.
#
# Correctness only. No latency here is a result (DESIGN §12.4).
#
# Usage:  scripts/run_e6_decode_drift.sh [output_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
OUT="${1:-${PROJECT_ROOT}/results/e6/decode_drift}"
BUILD_DIR="/work/build/e6opt"
IN_CONTAINER=(docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${TARGET_IMAGE_TAG}")

FREEZE="results/model_selection/pre_pi_freeze.json"
CLASS_MAP="artifacts/class_map.json"
MANIFEST="data/manifests/benchmark_val_1000.jsonl"
IMAGES_ROOT="data/raw/extracted/eccv_18_all_images_sm"

cd "${PROJECT_ROOT}"
[[ -d "${IMAGES_ROOT}" ]] || { echo "no images root at ${IMAGES_ROOT}" >&2; exit 2; }

REL_OUT="${OUT#"${PROJECT_ROOT}"/}"
rm -rf "${OUT}"
mkdir -p "${OUT}"

echo "=============================================================="
echo "E6 reduced-decode drift on M0/M2/M4 over benchmark_val_1000 -> ${OUT}"
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

# Model table from the freeze: id -> onnx, policy, threshold.
mapfile -t MODELS < <("${PYTHON}" -c "
import json
d = json.load(open('${FREEZE}'))
for m in d['models']:
    if m['model_id'] in ('M0', 'M2', 'M4'):
        print(m['model_id'], m['onnx']['artifact'], m['policy']['path'], m['policy']['threshold'])
")

run_dataset() {  # $1=onnx $2=policy $3=decode $4=out
    "${IN_CONTAINER[@]}" bash -lc "
        ${BUILD_DIR}/wildlife_trigger run-dataset \
            --model /work/$1 --class-map /work/${CLASS_MAP} --policy /work/$2 \
            --manifest /work/${MANIFEST} --images-root /work/${IMAGES_ROOT} \
            --decode $3 --threads 1 --on-corrupt fail --output /work/$4
    " 2>&1 | tail -1 | sed 's/^/        /'
}

SPECS=()
for row in "${MODELS[@]}"; do
    read -r MID ONNX POLICY THRESH <<< "${row}"
    echo
    echo "--- ${MID}: run-dataset full / half / quarter"
    for DECODE in full half quarter; do
        echo "    ${DECODE}..."
        run_dataset "${ONNX}" "${POLICY}" "${DECODE}" "${REL_OUT}/${MID}_${DECODE}.jsonl"
    done
    SPECS+=("${MID}:${THRESH}")
done

echo
echo "--- E6 decode-drift gate (bobcat metrics must hold; latency not judged)"
"${PYTHON}" -m wildlife_trigger.validate.decode_drift \
    --dir "${OUT}" \
    --models "${SPECS[@]}" \
    --variants half quarter \
    --manifest "${MANIFEST}" \
    --output "${OUT}/../decode_drift.json"
