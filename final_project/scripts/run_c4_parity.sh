#!/usr/bin/env bash
# C4 — the parity evidence runs (PLAN C4, DESIGN §10).
#
# Phase p1: preprocessing parity. Builds the C++ application in the target
# container, dumps the preprocessed tensor for every fixture through BOTH C++
# paths (fused = the shipping hot path, reference = DESIGN §11's unfused
# primitives), then compares them against the canonical Python tensors under the
# tolerances registered in DESIGN §10. The 20 golden CCT fixtures need the raw
# image archive (gx10); the committed synthetic supplement runs anywhere.
#
# The comparator's exit code is the gate's verdict: this script failing IS P1
# failing, and nothing downstream may proceed from it.
#
# Usage:  scripts/run_c4_parity.sh p1 <run_dir>
#   e.g.  scripts/run_c4_parity.sh p1 results/training/c2/c2_m0_fp32_seed42_20260716T061203Z

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
BUILD_DIR="/work/build/c4"
IN_CONTAINER=(docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${TARGET_IMAGE_TAG}")

PHASE="${1:?usage: run_c4_parity.sh <p1> <run_dir>}"
RUN_DIR="${2:?usage: run_c4_parity.sh <p1> <run_dir>}"

cd "${PROJECT_ROOT}"

RUN_ID="$("${PYTHON}" - "${RUN_DIR}" <<'EOF'
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
history = json.loads((run_dir / "history.json").read_text())
summary = run_dir / "run_summary.json"
run_id = history["run_name"]
if summary.exists():
    run_id = json.loads(summary.read_text()).get("run_id", run_id)
print(run_id)
EOF
)"
PARITY="${PROJECT_ROOT}/results/parity/${RUN_ID}"

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

phase_p1() {
    local out="${PARITY}/p1"
    rm -rf "${out}"
    mkdir -p "${out}"
    local rel_out="results/parity/${RUN_ID}/p1"

    build_cpp

    echo
    echo "--- dump-tensor: every fixture through both C++ paths"
    # (name, image path) pairs: the committed supplement plus, when the raw
    # archive is present, the 20 frozen goldens.
    local list
    list="$("${PYTHON}" - <<'EOF'
import json
from pathlib import Path

pairs = []
manifest = json.loads(Path("tests/fixtures/p1_supplement/manifest.json").read_text())
for name, entry in sorted(manifest["fixtures"].items()):
    pairs.append((name, entry["path"]))

images_dir = Path("data/raw/extracted/eccv_18_all_images_sm")
if images_dir.is_dir():
    golden = json.loads(Path("tests/fixtures/golden_raw.json").read_text())
    for entry in golden["fixtures"]:
        pairs.append((entry["image_id"], str(images_dir / entry["file_name"])))
else:
    print("WARNING: raw archive missing; goldens skipped", flush=True)

for name, path in pairs:
    print(f"{name}\t{path}")
EOF
)"
    local count=0
    while IFS=$'\t' read -r name image; do
        [[ "${name}" == WARNING:* ]] && { echo "    ${name} ${image}"; continue; }
        for mode in fused reference; do
            "${IN_CONTAINER[@]}" bash -lc "
                ${BUILD_DIR}/wildlife_trigger dump-tensor \
                    --image /work/${image} \
                    --preprocess ${mode} \
                    --output-bin /work/${rel_out}/${name}.${mode}.bin \
                    --output /work/${rel_out}/${name}.${mode}.json
            " >/dev/null 2>&1
        done
        count=$((count + 1))
    done <<< "${list}"
    echo "    ${count} fixtures dumped (x2 modes)"

    echo
    echo "--- compare: Python canonical vs C++ reference vs C++ fused"
    local golden_args=()
    if [[ -d "data/raw/extracted/eccv_18_all_images_sm" ]]; then
        golden_args=(--golden tests/fixtures/golden_raw.json
                     --images-dir data/raw/extracted/eccv_18_all_images_sm)
    fi
    "${PYTHON}" -m wildlife_trigger.validate.p1_preprocess \
        --cpp-dir "${out}" \
        --supplement tests/fixtures/p1_supplement/manifest.json \
        "${golden_args[@]}" \
        --output "${PARITY}/p1_preprocess.json"

    # The .bin/.json intermediates stay on this machine (results/**/*.bin is
    # gitignored); the committed evidence is p1_preprocess.json.
}

phase_ort() {
    # Initial ORT python-vs-C++ parity (DESIGN §10). Requires: the exported ONNX
    # in the run dir, P1's dumped tensors (run `p1` first), and the policy
    # ALREADY re-bound by wildlife_trigger.rebind_policy — the C++ loader
    # refuses an unbound policy, which is the loud failure working as designed.
    local out="${PARITY}/ort"
    rm -rf "${out}"
    mkdir -p "${out}/logits" "${out}/scratch"
    local rel_out="results/parity/${RUN_ID}/ort"
    local rel_run="${RUN_DIR#"${PROJECT_ROOT}"/}"
    rel_run="${rel_run#./}"

    local run_name
    run_name="$("${PYTHON}" -c "
import json; print(json.load(open('${RUN_DIR}/history.json'))['run_name'])")"
    local onnx="${rel_run}/${run_name}.onnx"

    build_cpp

    echo
    echo "--- logits layer: ort_probe (C++) on P1's canonical tensors"
    local count=0
    for tensor in $(ls "${PARITY}/p1/"*.fused.bin | head -10); do
        local name
        name="$(basename "${tensor}" .fused.bin)"
        "${IN_CONTAINER[@]}" bash -lc "
            ${BUILD_DIR}/ort_probe \
                --model /work/${onnx} \
                --input-bin /work/results/parity/${RUN_ID}/p1/${name}.fused.bin \
                --output-bin /work/${rel_out}/logits/${name}.bin \
                --optimized-out /work/${rel_out}/scratch/${name}.opt.onnx \
                --profile-prefix /work/${rel_out}/scratch/${name}
        " >/dev/null 2>&1
        count=$((count + 1))
    done
    echo "    ${count} tensors probed"

    echo
    echo "--- decision layer: the real infer CLI with the re-bound policy"
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
                --model /work/${onnx} \
                --class-map /work/artifacts/class_map.json \
                --policy /work/artifacts/policies/bobcat_v1.json \
                --image /work/${image} \
                --output /work/${rel_out}/infer_${name}.json
        " >/dev/null 2>&1
    done
    echo "    ${#images[@]} images inferred"

    echo
    echo "--- compare both layers"
    "${PYTHON}" -m wildlife_trigger.validate.ort_cpp_parity \
        --run "${RUN_DIR}" \
        --p1-dir "${PARITY}/p1" \
        --cpp-logits-dir "${out}/logits" \
        --infer-dir "${out}" \
        --policy artifacts/policies/bobcat_v1.json \
        --output "${PARITY}/p_ort_cpp.json"
}

case "${PHASE}" in
    p1) phase_p1 ;;
    ort) phase_ort ;;
    *) echo "unknown phase: ${PHASE} (expected: p1 | ort)" >&2; exit 2 ;;
esac
