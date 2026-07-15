#!/usr/bin/env bash
# PLAN A4 — the mandatory early C++ vertical slice, end to end and unattended.
#
# Proves the thin path works before any data exists and before any training:
#   saved JPEG -> C++ decode/preprocess -> ORT -> policy -> SHUTTER_TRIGGER JSON,
# plus a schema-valid benchmark, a system-monitor snapshot, and an installable ARM64
# bundle -- natively and under `qemu-aarch64 -cpu cortex-a76`.
#
# The point of building this now is that a vertical slice discovers integration
# problems while they are still cheap. Everything it runs on is synthetic and says so.
# No number here is a result: DESIGN §12.4 makes a latency a Pi number only when it is
# measured on a Pi.
#
# Usage:  scripts/run_a4_slice.sh [output_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"
OUT="${1:-${PROJECT_ROOT}/results/a4}"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
ARTIFACTS="${OUT}/artifacts"
EVIDENCE="${OUT}/evidence"
BUNDLE="${OUT}/bundle"
FIXTURE="${PROJECT_ROOT}/tests/fixtures/frame_1024x747.jpg"
BUILD_DIR="/work/build/a4"
IN_CONTAINER=(docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${TARGET_IMAGE_TAG}")

rm -rf "${EVIDENCE}"
mkdir -p "${ARTIFACTS}" "${EVIDENCE}"
cd "${PROJECT_ROOT}"

REL_OUT="${OUT#"${PROJECT_ROOT}"/}"
REL_FIXTURE="${FIXTURE#"${PROJECT_ROOT}"/}"

echo "=============================================================="
echo "A4 vertical slice -> ${OUT}"
echo "=============================================================="

# --- 1. vendored sources ------------------------------------------------------
echo
echo "--- vendored third-party integrity"
"${HERE}/verify_vendored.sh" | sed 's/^/    /'

# --- 2. the smoke artifacts ---------------------------------------------------
echo
echo "--- smoke model, class map and policies (deterministic, NOT M0)"
"${PYTHON}" -m wildlife_trigger.models.smoke --output-dir "${ARTIFACTS}" >/dev/null
cp "${ARTIFACTS}/class_map.json" "${EVIDENCE}/class_map.json"
MODEL="${ARTIFACTS}/smoke_mobilenetv2_16.onnx"

echo "--- deterministic JPEG fixture at the dominant CCT geometry"
"${PYTHON}" -m wildlife_trigger.validate.image_fixture \
    --output "${FIXTURE}" --report "${EVIDENCE}/image_fixture.json" >/dev/null
"${PYTHON}" -c "
import json; d = json.load(open('${EVIDENCE}/image_fixture.json'))
print('    %s  %dx%d  %d bytes  sha256 %s...' % (
    d['path'].split('/')[-1], d['width'], d['height'], d['bytes'], d['sha256'][:12]))
"

# --- 3. build and unit-test in the target container ---------------------------
echo
echo "--- build + ctest inside ${TARGET_IMAGE_TAG}"
"${IN_CONTAINER[@]}" bash -lc "
    set -euo pipefail
    cmake -S /work/cpp -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release \
        -DWILDLIFE_CPU_TARGET=${QEMU_PI5_CPU} >/dev/null
    cmake --build ${BUILD_DIR} -j\"\$(nproc)\" >/dev/null
    cd ${BUILD_DIR} && ctest --output-on-failure 2>&1 | tail -4 | sed 's/^/    /'
"

# --- 4. the slice, natively and under the Pi's ISA ----------------------------
run_slice() {
    local mode="$1" runner=""
    if [[ "${mode}" == "qemu" ]]; then
        runner="qemu-aarch64 -L / -cpu ${QEMU_PI5_CPU}"
    fi

    "${IN_CONTAINER[@]}" bash -lc "
        set -euo pipefail
        ${runner} ${BUILD_DIR}/wildlife_trigger infer \
            --model /work/${REL_OUT}/artifacts/smoke_mobilenetv2_16.onnx \
            --class-map /work/${REL_OUT}/artifacts/class_map.json \
            --policy /work/${REL_OUT}/artifacts/bobcat_v0.json \
            --image /work/${REL_FIXTURE} \
            --profile-prefix /work/${REL_OUT}/evidence/ort_profile_${mode} \
            --output /work/${REL_OUT}/evidence/infer.${mode}.json \
    " > "${EVIDENCE}/infer.${mode}.stdout" 2> "${EVIDENCE}/infer.${mode}.stderr"

    "${PYTHON}" -c "
import json
d = json.load(open('${EVIDENCE}/infer.${mode}.json'))
print('    %-6s SHUTTER_TRIGGER=%d  top1=%-8s %.4f  cpu=%-28s pi5=%s' % (
    '${mode}', d['decision']['SHUTTER_TRIGGER'], d['decision']['top1']['class'],
    d['decision']['top1']['score'], d['environment']['cpu_features'],
    d['environment']['looks_like_pi5']))
"
}

echo
echo "--- infer: JPEG -> decode -> preprocess -> ORT -> policy -> SHUTTER_TRIGGER"
run_slice native
run_slice qemu

echo
echo "--- multi-target policy on the same model, no reload (DESIGN §4)"
"${IN_CONTAINER[@]}" bash -lc "
    ${BUILD_DIR}/wildlife_trigger infer \
        --model /work/${REL_OUT}/artifacts/smoke_mobilenetv2_16.onnx \
        --class-map /work/${REL_OUT}/artifacts/class_map.json \
        --policy /work/${REL_OUT}/artifacts/bobcat_coyote_v0.json \
        --image /work/${REL_FIXTURE} \
        --output /work/${REL_OUT}/evidence/infer.multi_target.json
" >/dev/null 2>&1
"${PYTHON}" -c "
import json; d = json.load(open('${EVIDENCE}/infer.multi_target.json'))
print('    targets:', [t['class'] for t in d['decision']['targets']],
      '-> SHUTTER_TRIGGER=%d' % d['decision']['SHUTTER_TRIGGER'])
"

# --- 5. the policy loader must refuse bad policies through the real CLI --------
echo
echo "--- policy loader rejections (through the CLI, not just ctest)"
"${PYTHON}" -m wildlife_trigger.validate.policy_rejections \
    --artifacts "${ARTIFACTS}" \
    --image "${FIXTURE}" \
    --binary "${BUILD_DIR}/wildlife_trigger" \
    --image-rel "/work/${REL_FIXTURE}" \
    --artifacts-rel "/work/${REL_OUT}/artifacts" \
    --project-root "${PROJECT_ROOT}" \
    --image-tag "${TARGET_IMAGE_TAG}" \
    --report "${EVIDENCE}/policy_rejections.json"

# --- 6. benchmark and self-test -----------------------------------------------
echo
echo "--- benchmark (smoke check of the timing path; NOT a Pi result)"
"${IN_CONTAINER[@]}" bash -lc "
    ${BUILD_DIR}/wildlife_trigger benchmark \
        --model /work/${REL_OUT}/artifacts/smoke_mobilenetv2_16.onnx \
        --class-map /work/${REL_OUT}/artifacts/class_map.json \
        --policy /work/${REL_OUT}/artifacts/bobcat_v0.json \
        --image /work/${REL_FIXTURE} \
        --warmup 10 --iterations 100 --threads 1 \
        --output /work/${REL_OUT}/evidence/benchmark.native.json
" >/dev/null 2> >(sed 's/^/    /' >&2)

echo
echo "--- self-test"
"${IN_CONTAINER[@]}" bash -lc "
    ${BUILD_DIR}/wildlife_trigger self-test \
        --model /work/${REL_OUT}/artifacts/smoke_mobilenetv2_16.onnx \
        --class-map /work/${REL_OUT}/artifacts/class_map.json \
        --policy /work/${REL_OUT}/artifacts/bobcat_v0.json \
        --image /work/${REL_FIXTURE} \
        --output /work/${REL_OUT}/evidence/self_test.native.json
" >/dev/null 2> >(sed 's/^/    /' >&2)

# --- 7. the deployment bundle -------------------------------------------------
echo
echo "--- ARM64 deployment bundle"
"${HERE}/build_bundle.sh" "${BUNDLE}" "${ARTIFACTS}" | sed 's/^/    /'

echo
echo "--- verifying the staged bundle runs from its own launcher"
REL_BUNDLE="${BUNDLE#"${PROJECT_ROOT}"/}"
"${IN_CONTAINER[@]}" bash -lc "
    set -euo pipefail
    cd /work/${REL_BUNDLE} && sha256sum -c MANIFEST.sha256 >/dev/null && echo '    checksums OK'
    ./bin/run.sh self-test \
        --model models/smoke_mobilenetv2_16.onnx \
        --class-map policies/class_map.json \
        --policy policies/bobcat_v0.json \
        --image /work/${REL_FIXTURE} \
        --output /work/${REL_OUT}/evidence/bundle_self_test.json
" >/dev/null 2> >(sed 's/^/    /' >&2)

"${PYTHON}" -m wildlife_trigger.validate.bundle_audit \
    --bundle "${BUNDLE}" \
    --project-root "${PROJECT_ROOT}" \
    --image-tag "${TARGET_IMAGE_TAG}" \
    --report "${EVIDENCE}/bundle_audit.json"

# --- 8. the gate --------------------------------------------------------------
echo
echo "--- A4 gate"
"${PYTHON}" -m wildlife_trigger.validate.a4_gate \
    --evidence "${EVIDENCE}" \
    --bundle "${BUNDLE}" \
    --report "${OUT}/a4_gate.json"
