#!/usr/bin/env bash
# PLAN A3 — the P0 toolchain spike, end to end and unattended.
#
# Produces the three model forms, then proves each one executes in the C++ ONNX
# Runtime inside the target-compatible bookworm container, both natively and under
# `qemu-aarch64 -cpu cortex-a76`.
#
# The emulated pass is not a formality. gx10 is a Cortex-X925 with i8mm and SVE2;
# the Pi 5's Cortex-A76 has neither. A quantized path that only works because gx10
# has i8mm is a P0 *failure*, and this is the cheapest place on earth to discover
# it -- the alternative is discovering it during a one-shot Pi rental.
#
# No timing is produced anywhere here. QEMU models no caches and no memory
# bandwidth (DESIGN §12.4), and the native run is on the wrong microarchitecture.
# This script answers "does it execute, and as what dtype" and nothing else.
#
# Usage:  scripts/run_p0_spike.sh [output_dir]
#         Default output: results/p0/

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"
OUT="${1:-${PROJECT_ROOT}/results/p0}"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
MODELS="${OUT}/models"
EVIDENCE="${OUT}/evidence"
BUILD_DIR="/work/build/p0"

if [[ ! -x "${PYTHON}" ]]; then
    echo "No training venv at ${PYTHON}. Run scripts/setup_gx10.sh first." >&2
    exit 1
fi

mkdir -p "${MODELS}" "${EVIDENCE}"
cd "${PROJECT_ROOT}"

echo "=============================================================="
echo "P0 spike -> ${OUT}"
echo "=============================================================="

# --- 1. the three model forms ------------------------------------------------
# Every export is opset-guarded by wildlife_trigger.models.export: opset 9 and any
# non-contract opset are rejected there, not here.

echo
echo "--- M0: FP32 ImageNet MobileNetV2, opset ${P0_OPSET_EXPECTED:-17}"
"${PYTHON}" -m wildlife_trigger.models.export \
    --output "${MODELS}/m0_fp32.onnx" \
    --describe-json "${EVIDENCE}/m0_fp32.export.json" >/dev/null

echo "--- M1: static S8S8 QDQ PTQ (synthetic calibration -- toolchain proof)"
"${PYTHON}" -m wildlife_trigger.optimize.ptq \
    --input "${MODELS}/m0_fp32.onnx" \
    --output "${MODELS}/m1_ptq.onnx" \
    --synthetic-calibration 32 \
    --describe-json "${EVIDENCE}/m1_ptq.export.json" >/dev/null

echo "--- M2: QAT candidate 1 (synthetic training -- toolchain proof)"
"${PYTHON}" -m wildlife_trigger.optimize.qat \
    --output "${MODELS}/m2_qat.onnx" \
    --steps 20 \
    --describe-json "${EVIDENCE}/m2_qat.export.json" >/dev/null

echo "--- opset contract across all three"
"${PYTHON}" -m wildlife_trigger.validate.opset_parity \
    --models "${MODELS}/m0_fp32.onnx" "${MODELS}/m1_ptq.onnx" "${MODELS}/m2_qat.onnx" \
    --report "${EVIDENCE}/opset_parity.json"

# --- 2. a shared input fixture ----------------------------------------------
# One blob both call sites read, so a Python-vs-C++ output difference is a real
# difference and not two different inputs.

echo
echo "--- input fixture"
"${PYTHON}" -m wildlife_trigger.validate.fixture \
    --output "${EVIDENCE}/input_1x3x224x224.bin" \
    --shape 1 3 224 224

# --- 3. Python ORT reference -------------------------------------------------

echo
for model in m0_fp32 m1_ptq m2_qat; do
    echo "--- python ORT coverage: ${model}"
    set +e
    "${PYTHON}" -m wildlife_trigger.validate.ort_coverage \
        --model "${MODELS}/${model}.onnx" \
        --label "${model}-python" \
        --workdir "${EVIDENCE}/python/${model}" \
        --input-bin "${EVIDENCE}/input_1x3x224x224.bin" \
        --report "${EVIDENCE}/${model}.python.coverage.json" >/dev/null
    echo "    verdict exit=$?  (0=integer 2=not-integer 1=could-not-run)"
    set -e
done

# --- 4. C++ ORT in the target container, native and emulated -----------------

echo
echo "--- building ort_probe inside ${TARGET_IMAGE_TAG}"
docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${TARGET_IMAGE_TAG}" bash -lc "
    set -euo pipefail
    cmake -S /work/cpp -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release \
        -DWILDLIFE_CPU_TARGET=${QEMU_PI5_CPU} >/dev/null
    cmake --build ${BUILD_DIR} --target ort_probe -j\"\$(nproc)\" >/dev/null
    echo '    built; GLIBC audit:'
    /work/scripts/audit_target_compat.sh ${BUILD_DIR}/ort_probe | sed 's/^/      /'
"

# Relative paths inside the container's /work mount.
REL_OUT="${OUT#"${PROJECT_ROOT}"/}"

for mode in native qemu; do
    echo
    echo "--- C++ ORT probe: ${mode}"
    for model in m0_fp32 m1_ptq m2_qat; do
        runner=""
        if [[ "${mode}" == "qemu" ]]; then
            # -L / so the emulated binary resolves the container's own loader and
            # libraries rather than the host's.
            runner="qemu-aarch64 -L / -cpu ${QEMU_PI5_CPU}"
        fi

        set +e
        docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${TARGET_IMAGE_TAG}" bash -lc "
            set -euo pipefail
            mkdir -p /work/${REL_OUT}/evidence/cpp-${mode}/${model}
            ${runner} ${BUILD_DIR}/ort_probe \
                --model /work/${REL_OUT}/models/${model}.onnx \
                --optimized-out /work/${REL_OUT}/evidence/cpp-${mode}/${model}/optimized.onnx \
                --profile-prefix /work/${REL_OUT}/evidence/cpp-${mode}/${model}/profile \
                --input-bin /work/${REL_OUT}/evidence/input_1x3x224x224.bin \
                --output-bin /work/${REL_OUT}/evidence/cpp-${mode}/${model}/output.bin
        " > "${EVIDENCE}/${model}.cpp-${mode}.probe.json"
        probe_status=$?
        set -e

        if [[ ${probe_status} -ne 0 ]]; then
            echo "    ${model}: PROBE FAILED (exit ${probe_status})"
            cat "${EVIDENCE}/${model}.cpp-${mode}.probe.json"
            continue
        fi

        "${PYTHON}" -c "
import json, sys
d = json.load(open('${EVIDENCE}/${model}.cpp-${mode}.probe.json'))
print('    %-8s cpu=%-28s looks_like_pi5=%s argmax=%s' % (
    '${model}', d['cpu_features'], d['looks_like_pi5'], d['output_argmax']))
"
        profile_json="$(ls -t "${EVIDENCE}/cpp-${mode}/${model}"/profile*.json | head -1)"
        set +e
        "${PYTHON}" -m wildlife_trigger.validate.ort_coverage \
            --model "${MODELS}/${model}.onnx" \
            --label "${model}-cpp-${mode}" \
            --from-artifacts \
            --optimized "${EVIDENCE}/cpp-${mode}/${model}/optimized.onnx" \
            --profile "${profile_json}" \
            --report "${EVIDENCE}/${model}.cpp-${mode}.coverage.json" >/dev/null
        echo "             verdict exit=$?  (0=integer 2=not-integer 1=could-not-run)"
        set -e
    done
done

# --- 5. the gate -------------------------------------------------------------

echo
echo "--- P0 gate"
"${PYTHON}" -m wildlife_trigger.validate.p0_gate \
    --evidence "${EVIDENCE}" \
    --report "${OUT}/p0_gate.json"
