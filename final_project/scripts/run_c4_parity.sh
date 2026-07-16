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

case "${PHASE}" in
    p1) phase_p1 ;;
    *) echo "unknown phase: ${PHASE} (expected: p1)" >&2; exit 2 ;;
esac
