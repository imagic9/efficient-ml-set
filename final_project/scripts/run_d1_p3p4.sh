#!/usr/bin/env bash
# D1 — P3/P4 evidence for the selected M1 candidate (PLAN D1, DESIGN §10).
#
# Same idiom as run_c4_parity.sh: the C++ halves run inside the target
# container (built for the Pi's CPU), the Python halves run in the pinned venv,
# and the comparators' exit codes ARE the gates' verdicts.
#
# Phase p3: ort_probe on P1's canonical tensors + the real infer CLI with the
#           candidate policy, then validate.p3_quantized (which re-runs the
#           full validation evaluation and demands exact equality, and attaches
#           the passing report to the policy).
# Phase p4: run-dataset over both validation manifests, then
#           validate.p4_dataset_parity (ordered ids, labels, scores, decisions,
#           confusion matrices).
#
# Usage:  scripts/run_d1_p3p4.sh <p3|p4> <method> [candidate-root]
#   e.g.  scripts/run_d1_p3p4.sh p3 percentile            # D1 (m1_ptq default)
#         scripts/run_d1_p3p4.sh p4 lr3e-5 m2_qat         # D2 arms
#
# The candidate-root names both the directory under results/optimize/ and the
# policy id (bobcat_<root>_<method>_v1) — the naming convention every D-phase
# calibration uses, so one driver serves the whole ladder.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
BUILD_DIR="/work/build/d1"
IN_CONTAINER=(docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${TARGET_IMAGE_TAG}")

PHASE="${1:?usage: run_d1_p3p4.sh <p3|p4> <method> [candidate-root]}"
METHOD="${2:?usage: run_d1_p3p4.sh <p3|p4> <method> [candidate-root]}"
ROOT_NAME="${3:-m1_ptq}"

# P1's canonical tensors came from the M0 run's parity evidence; they are
# preprocessing artifacts (model-independent), which is why P3 may reuse them.
P1_RUN_ID="c2_m0_fp32_seed42_20260716T061203Z"

CANDIDATE="results/optimize/${ROOT_NAME}/${METHOD}"
POLICY="artifacts/policies/bobcat_${ROOT_NAME}_${METHOD}_v1.json"

cd "${PROJECT_ROOT}"
[[ -d "${CANDIDATE}" ]] || { echo "no candidate at ${CANDIDATE}" >&2; exit 2; }
[[ -f "${POLICY}" ]] || { echo "no policy at ${POLICY}" >&2; exit 2; }

build_cpp() {
    echo "--- build + ctest inside ${TARGET_IMAGE_TAG}"
    "${IN_CONTAINER[@]}" bash -lc "
        set -euo pipefail
        cmake -S /work/cpp -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release \
            -DWILDLIFE_CPU_TARGET=${QEMU_PI5_CPU} >/dev/null
        cmake --build ${BUILD_DIR} -j\"\$(nproc)\" >/dev/null
        cd ${BUILD_DIR} && ctest --output-on-failure 2>&1 | tail -3 | sed 's/^/    /'
    "
}

phase_p3() {
    local out="${CANDIDATE}/p3"
    rm -rf "${out}" "${CANDIDATE}/p3_coverage" "${CANDIDATE}/p3_reevaluation"
    mkdir -p "${out}/logits" "${out}/scratch"

    build_cpp

    echo
    echo "--- logits layer: ort_probe (C++) on P1's canonical tensors"
    local count=0
    for tensor in $(ls "results/parity/${P1_RUN_ID}/p1/"*.fused.bin | head -10); do
        local name
        name="$(basename "${tensor}" .fused.bin)"
        "${IN_CONTAINER[@]}" bash -lc "
            ${BUILD_DIR}/ort_probe \
                --model /work/${CANDIDATE}/model.onnx \
                --input-bin /work/results/parity/${P1_RUN_ID}/p1/${name}.fused.bin \
                --output-bin /work/${out}/logits/${name}.bin \
                --optimized-out /work/${out}/scratch/${name}.opt.onnx \
                --profile-prefix /work/${out}/scratch/${name}
        " >/dev/null 2>&1
        count=$((count + 1))
    done
    echo "    ${count} tensors probed"

    echo
    echo "--- decision layer: the real infer CLI with the candidate policy"
    local images=("tests/fixtures/frame_1024x747.jpg")
    if [[ -d "data/raw/extracted/eccv_18_all_images_sm" ]]; then
        while IFS= read -r file; do
            images+=("data/raw/extracted/eccv_18_all_images_sm/${file}")
        done < <("${PYTHON}" -c "
import json
golden = json.load(open('tests/fixtures/golden_raw.json'))
for e in golden['fixtures'][:3]:
    print(e['file_name'])")
    fi
    for image in "${images[@]}"; do
        local name
        name="$(basename "${image%.*}")"
        "${IN_CONTAINER[@]}" bash -lc "
            ${BUILD_DIR}/wildlife_trigger infer \
                --model /work/${CANDIDATE}/model.onnx \
                --class-map /work/artifacts/class_map.json \
                --policy /work/${POLICY} \
                --image /work/${image} \
                --output /work/${out}/infer_${name}.json
        " >/dev/null 2>&1
    done
    echo "    ${#images[@]} images inferred"

    echo
    echo "--- P3: the four registered checks"
    "${PYTHON}" -m wildlife_trigger.validate.p3_quantized \
        --candidate "${CANDIDATE}" \
        --policy "${POLICY}" \
        --p1-dir "results/parity/${P1_RUN_ID}/p1" \
        --cpp-logits-dir "${out}/logits" \
        --infer-dir "${out}" \
        --output "${CANDIDATE}/p3_quantized.json"
}

phase_p4() {
    local out="${CANDIDATE}/p4"
    rm -rf "${out}"
    mkdir -p "${out}"

    build_cpp

    echo
    echo "--- run-dataset: both validation manifests, in manifest order"
    for split in cis_val_clean trans_val; do
        echo "    ${split}..."
        "${IN_CONTAINER[@]}" bash -lc "
            ${BUILD_DIR}/wildlife_trigger run-dataset \
                --model /work/${CANDIDATE}/model.onnx \
                --class-map /work/artifacts/class_map.json \
                --policy /work/${POLICY} \
                --manifest /work/data/manifests/${split}.jsonl \
                --images-root /work/data/raw/extracted/eccv_18_all_images_sm \
                --output /work/${out}/cpp_${split}.jsonl \
                --threads 1
        " 2>&1 | tail -1 | sed 's/^/    /'
    done

    echo
    echo "--- P4: corpus-scale comparison"
    "${PYTHON}" -m wildlife_trigger.validate.p4_dataset_parity \
        --candidate "${CANDIDATE}" \
        --policy "${POLICY}" \
        --cpp-dir "${out}" \
        --output "${CANDIDATE}/p4_dataset_parity.json"

    # The per-frame JSONL stays on this machine (results/**/*.jsonl under p4/ is
    # large); the committed evidence is p4_dataset_parity.json.
}

case "${PHASE}" in
    p3) phase_p3 ;;
    p4) phase_p4 ;;
    *) echo "unknown phase: ${PHASE} (expected: p3 | p4)" >&2; exit 2 ;;
esac
