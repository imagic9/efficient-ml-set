#!/usr/bin/env bash
# E6 — the pre-rental ISA rehearsal: native gx10 vs `qemu-aarch64 -cpu cortex-a76`
# for the whole shortlist (M0, M2, M4), on the C++ dataset runner (PLAN E6, §12.2).
#
# QEMU withholds the build host's i8mm/sve2, so ORT dispatches the Pi's kernels. Running
# the SAME binary on the SAME frames native and emulated, then diffing the scores, is the
# rehearsal of the §12.2 parity claim — in minutes, before the rental clock starts. The
# registered expectation: the FP32 arm (M0) shifts, the INT8 arms (M2/M4) stay near-bitwise.
#
# CORRECTNESS ONLY. Emulated latency models no caches or memory bandwidth and is never a
# result (§12.4); the runner's timings under QEMU are ignored.
#
# Usage:  scripts/run_e6_qemu_parity.sh [output_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
OUT="${1:-${PROJECT_ROOT}/results/e6}"
BUILD_DIR="/work/build/e6"
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
echo "E6 pre-rental ISA parity (native vs qemu ${QEMU_PI5_CPU}) -> ${OUT}"
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

# A small deterministic stratified slice — QEMU is ~10-50x slower than native, so this
# is sized for the rehearsal, not the corpus. Seeded by a sort on image_id.
echo
echo "--- stratified slice of benchmark_val_1000 (deterministic, QEMU-sized)"
"${PYTHON}" - "${MANIFEST}" "${IMAGES_ROOT}" "${OUT}/e6_slice.jsonl" <<'PY'
import json, os, sys
from collections import defaultdict
manifest, images_root, out = sys.argv[1:4]
rows = [json.loads(l) for l in open(manifest) if l.strip()]
per = {"empty": 25, "other": 25, "rare": 25, "bobcat": 30, "threshold_adjacent": 20, "multi_label": 2}
by = defaultdict(list)
for r in rows:
    if os.path.exists(os.path.join(images_root, r["file_name"])):
        by[r["benchmark_stratum"]].append(r)
chosen = []
for s in ("empty", "other", "rare", "bobcat", "threshold_adjacent", "multi_label"):
    chosen.extend(sorted(by.get(s, []), key=lambda r: r["image_id"])[: per[s]])
chosen.sort(key=lambda r: r["image_id"])
with open(out, "w") as f:
    for r in chosen:
        f.write(json.dumps({"file_name": r["file_name"], "image_id": r["image_id"],
                            "seq_id": r.get("seq_id", ""), "labels": r["labels"]}) + "\n")
print(f"    {len(chosen)} frames across {len({r['benchmark_stratum'] for r in chosen})} strata")
PY

# Model table from the freeze: id -> onnx, policy, threshold, int8?
mapfile -t MODELS < <("${PYTHON}" -c "
import json
d = json.load(open('${FREEZE}'))
for m in d['models']:
    if m['model_id'] in ('M0', 'M2', 'M4'):
        int8 = '1' if m['kind'] != 'fp32_baseline' else '0'
        print(m['model_id'], m['onnx']['artifact'], m['policy']['path'], m['policy']['threshold'], int8)
")

run_dataset() {  # $1=runner_prefix $2=onnx $3=policy $4=out
    "${IN_CONTAINER[@]}" bash -lc "
        $1 ${BUILD_DIR}/wildlife_trigger run-dataset \
            --model /work/$2 --class-map /work/${CLASS_MAP} --policy /work/$3 \
            --manifest /work/${REL_OUT}/e6_slice.jsonl --images-root /work/${IMAGES_ROOT} \
            --output /work/$4 --on-corrupt fail --threads 1
    " 2>&1 | tail -1 | sed 's/^/        /'
}

PAIRS=()
INT8=()
for row in "${MODELS[@]}"; do
    read -r MID ONNX POLICY THRESH IS_INT8 <<< "${row}"
    echo
    echo "--- ${MID}: run-dataset native + qemu ${QEMU_PI5_CPU}"
    echo "    native..."
    run_dataset "" "${ONNX}" "${POLICY}" "${REL_OUT}/native_${MID}.jsonl"
    echo "    qemu..."
    run_dataset "qemu-aarch64 -L / -cpu ${QEMU_PI5_CPU}" "${ONNX}" "${POLICY}" "${REL_OUT}/qemu_${MID}.jsonl"
    PAIRS+=("${MID}:${REL_OUT}/native_${MID}.jsonl:${REL_OUT}/qemu_${MID}.jsonl:${THRESH}")
    [[ "${IS_INT8}" == "1" ]] && INT8+=("${MID}")
done

echo
echo "--- E6 ISA parity comparison (correctness only; emulated latency ignored)"
"${PYTHON}" -m wildlife_trigger.validate.qemu_parity \
    --pairs "${PAIRS[@]}" \
    --int8 "${INT8[@]}" \
    --output "${OUT}/qemu_parity.json"
