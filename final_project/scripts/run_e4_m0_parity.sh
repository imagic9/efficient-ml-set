#!/usr/bin/env bash
# E4 — the dataset runner, proven on M0 at corpus scale (PLAN E4, DESIGN §10/§11).
#
# The optimized ladder (M1-M4) already passed P4 via run_d1_p3p4.sh. M0 — the FP32
# baseline every other model is measured against — had only fixture-level py<->cpp
# parity (p_ort_cpp) and P2, not the full run-dataset confusion-matrix parity that
# E4's "match Python validation outputs and confusion matrix" box demands. This runs
# exactly that comparison for M0, reusing validate.p4_dataset_parity (generalized to a
# candidate without evaluation.json).
#
# Same idiom as run_d1_p3p4.sh p4: the C++ run-dataset runs inside the target
# container, the comparator runs in the pinned venv, and its exit code IS the verdict.
#
# Usage:  scripts/run_e4_m0_parity.sh [output_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
OUT="${1:-${PROJECT_ROOT}/results/e4}"
BUILD_DIR="/work/build/e4"
IN_CONTAINER=(docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${TARGET_IMAGE_TAG}")

FREEZE="results/model_selection/pre_pi_freeze.json"
CLASS_MAP="artifacts/class_map.json"
POLICY="artifacts/policies/bobcat_v1.json"
IMAGES_ROOT="data/raw/extracted/eccv_18_all_images_sm"
CANDIDATE="results/training/c2/c2_m0_fp32_seed42_20260716T061203Z"

cd "${PROJECT_ROOT}"

# Model identity from the freeze — the single source of truth for what M0 is.
read -r M0_ONNX M0_SHA < <("${PYTHON}" -c "
import json
m0 = next(m for m in json.load(open('${FREEZE}'))['models'] if m['model_id'] == 'M0')
print(m0['onnx']['artifact'], m0['onnx']['sha256'])")

[[ -f "${M0_ONNX}" ]] || { echo "M0 ONNX not found at ${M0_ONNX}" >&2; exit 2; }
[[ -f "${CANDIDATE}/predictions.npz" ]] || { echo "no M0 predictions.npz at ${CANDIDATE}" >&2; exit 2; }
[[ -d "${IMAGES_ROOT}" ]] || { echo "no images root at ${IMAGES_ROOT}" >&2; exit 2; }

REL_OUT="${OUT#"${PROJECT_ROOT}"/}"
rm -rf "${OUT}"
mkdir -p "${OUT}"

echo "=============================================================="
echo "E4 dataset-runner parity on M0 -> ${OUT}"
echo "  M0: ${M0_ONNX}  (${M0_SHA:0:16}...)"
echo "=============================================================="

echo
echo "--- build + ctest inside ${TARGET_IMAGE_TAG} (target: ${QEMU_PI5_CPU})"
"${IN_CONTAINER[@]}" bash -lc "
    set -euo pipefail
    cmake -S /work/cpp -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release \
        -DWILDLIFE_CPU_TARGET=${QEMU_PI5_CPU} >/dev/null
    cmake --build ${BUILD_DIR} -j\"\$(nproc)\" >/dev/null
    cd ${BUILD_DIR} && ctest --output-on-failure 2>&1 | tail -3 | sed 's/^/    /'
"

echo
echo "--- run-dataset: both validation manifests on M0, in manifest order"
for split in cis_val_clean trans_val; do
    echo "    ${split}..."
    "${IN_CONTAINER[@]}" bash -lc "
        ${BUILD_DIR}/wildlife_trigger run-dataset \
            --model /work/${M0_ONNX} \
            --class-map /work/${CLASS_MAP} \
            --policy /work/${POLICY} \
            --manifest /work/data/manifests/${split}.jsonl \
            --images-root /work/${IMAGES_ROOT} \
            --output /work/${REL_OUT}/cpp_${split}.jsonl \
            --on-corrupt fail --threads 1
    " 2>&1 | tail -1 | sed 's/^/    /'
done

echo
echo "--- P4: corpus-scale Python-vs-C++ comparison (ordered ids, labels, scores,"
echo "        decisions, confusion matrix) for M0"
"${PYTHON}" -m wildlife_trigger.validate.p4_dataset_parity \
    --candidate "${CANDIDATE}" \
    --policy "${POLICY}" \
    --cpp-dir "${OUT}" \
    --model-sha256 "${M0_SHA}" \
    --model-path "${M0_ONNX}" \
    --label "M0" \
    --output "${OUT}/p4_dataset_parity_m0.json"

# The per-frame JSONL stays local (large); the committed evidence is the report.
