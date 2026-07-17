#!/usr/bin/env bash
# Gate E6 — the C++ application is correct before performance claims (PLAN E6).
#
# Consolidation, not a re-run: retain the ALL and EXTENDED optimized graphs + profiles
# for M0 (so the graph-level comparison rests on artifacts, not on inferring execution
# from node names), then cite every shortlisted model's parity chain (P1-P4) and the E6
# experiments (QEMU parity, native-vs-target, optimization matrix, decode-drift) into
# one gate verdict.
#
# Usage:  scripts/run_e6_gate.sh [output_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
OUT="${1:-${PROJECT_ROOT}/results/e6}"
BUILD_DIR="/work/build/e6opt"
IN_CONTAINER=(docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${TARGET_IMAGE_TAG}")

FREEZE="results/model_selection/pre_pi_freeze.json"
CLASS_MAP="artifacts/class_map.json"
POLICY="artifacts/policies/bobcat_v1.json"
FIXTURE="tests/fixtures/frame_1024x747.jpg"
GRAPHS="${OUT}/graphs"
REL_GRAPHS="${GRAPHS#"${PROJECT_ROOT}"/}"

cd "${PROJECT_ROOT}"

M0_ONNX="$("${PYTHON}" -c "
import json
m0 = next(m for m in json.load(open('${FREEZE}'))['models'] if m['model_id'] == 'M0')
print(m0['onnx']['artifact'])")"
[[ -f "${M0_ONNX}" ]] || { echo "M0 ONNX not found at ${M0_ONNX}" >&2; exit 2; }

rm -rf "${GRAPHS}"
mkdir -p "${GRAPHS}"

echo "=============================================================="
echo "Gate E6 — correctness before performance -> ${OUT}"
echo "=============================================================="

echo
echo "--- build inside ${TARGET_IMAGE_TAG}"
"${IN_CONTAINER[@]}" bash -lc "
    set -euo pipefail
    cmake -S /work/cpp -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release \
        -DWILDLIFE_CPU_TARGET=${QEMU_PI5_CPU} >/dev/null
    cmake --build ${BUILD_DIR} -j\"\$(nproc)\" >/dev/null
    echo '    built'
"

echo
echo "--- retain the ALL and EXTENDED optimized graphs + profiles for M0 (inspection)"
for LEVEL in all extended; do
    "${IN_CONTAINER[@]}" bash -lc "
        ${BUILD_DIR}/wildlife_trigger infer \
            --model /work/${M0_ONNX} --class-map /work/${CLASS_MAP} \
            --policy /work/${POLICY} --image /work/${FIXTURE} \
            --graph-opt ${LEVEL} \
            --optimized-model /work/${REL_GRAPHS}/opt_${LEVEL}.onnx \
            --profile-prefix /work/${REL_GRAPHS}/prof_${LEVEL}
    " >/dev/null 2>&1
    echo "    ORT_ENABLE_${LEVEL}: $(ls -1 "${GRAPHS}/opt_${LEVEL}.onnx" 2>/dev/null && du -h "${GRAPHS}/opt_${LEVEL}.onnx" | cut -f1)"
done

echo
echo "--- consolidate Gate E6"
"${PYTHON}" -m wildlife_trigger.validate.e6_gate \
    --freeze "${FREEZE}" \
    --results-root results \
    --e6-dir "${OUT}" \
    --output "${OUT}/e6_gate.json"
