#!/usr/bin/env bash
# E1 — the C++ foundation, hardened and exercised against the REAL M0 baseline.
#
# A4 proved the vertical slice runs at all, on a synthetic smoke network whose every
# number it declared fake. E1 hardens that foundation and proves it against the actual
# frozen FP32 baseline (M0), the way DESIGN §11 / PLAN E1 require ("harden the smoke
# implementation using M0"). Nothing here is a Pi latency result — DESIGN §12.4 makes a
# latency a Pi number only when measured on a Pi; the benchmark below is a timing-path
# smoke check and says so in its own provenance field.
#
# Same idiom as run_d1_p3p4.sh: the C++ runs inside the target container (built for the
# Pi's CPU, run natively and under `-cpu cortex-a76`), the gate runs in the pinned venv,
# and validate.e1_gate's exit code IS the verdict.
#
# What it establishes, all against M0:
#   - the app builds under the pinned target toolchain and ctest is green
#     (including the new logging unit test);
#   - self-test, infer (native + QEMU Pi-ISA), and benchmark all run on M0;
#   - the dataset runner reproduces M0's own precomputed operating point over a
#     stratified slice of benchmark_val_1000;
#   - the leveled logging convention behaves under WILDLIFE_LOG_LEVEL.
#
# Usage:  scripts/run_e1_foundation.sh [output_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
OUT="${1:-${PROJECT_ROOT}/results/e1}"
EVIDENCE="${OUT}/evidence"
BUILD_DIR="/work/build/e1"
IN_CONTAINER=(docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${TARGET_IMAGE_TAG}")

FREEZE="results/model_selection/pre_pi_freeze.json"
MANIFEST="data/manifests/benchmark_val_1000.jsonl"
CLASS_MAP="artifacts/class_map.json"
POLICY="artifacts/policies/bobcat_v1.json"
IMAGES_ROOT="data/raw/extracted/eccv_18_all_images_sm"

cd "${PROJECT_ROOT}"

# The M0 ONNX path and hash are read from the freeze, never hard-coded: the freeze is
# the single source of truth for what M0 is, and a stale timestamp here would silently
# test a different file.
M0_ONNX="$("${PYTHON}" -c "
import json
m0 = next(m for m in json.load(open('${FREEZE}'))['models'] if m['model_id'] == 'M0')
print(m0['onnx']['artifact'])")"

[[ -f "${M0_ONNX}" ]] || { echo "M0 ONNX not found at ${M0_ONNX} (is it on this host?)" >&2; exit 2; }
[[ -f "${POLICY}" ]] || { echo "no policy at ${POLICY}" >&2; exit 2; }
[[ -d "${IMAGES_ROOT}" ]] || { echo "no images root at ${IMAGES_ROOT}" >&2; exit 2; }

REL_EVIDENCE="${EVIDENCE#"${PROJECT_ROOT}"/}"
rm -rf "${EVIDENCE}"
mkdir -p "${EVIDENCE}"

echo "=============================================================="
echo "E1 C++ foundation on real M0 -> ${OUT}"
echo "  M0: ${M0_ONNX}"
echo "=============================================================="

# --- 1. vendored integrity + build + ctest in the target container ------------
echo
echo "--- vendored third-party integrity"
"${HERE}/verify_vendored.sh" | sed 's/^/    /'

echo
echo "--- build + ctest inside ${TARGET_IMAGE_TAG} (target: ${QEMU_PI5_CPU})"
"${IN_CONTAINER[@]}" bash -lc "
    set -euo pipefail
    cmake -S /work/cpp -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release \
        -DWILDLIFE_CPU_TARGET=${QEMU_PI5_CPU} >/dev/null
    cmake --build ${BUILD_DIR} -j\"\$(nproc)\" >/dev/null
    cd ${BUILD_DIR} && ctest --output-on-failure
" 2>&1 | tee "${EVIDENCE}/ctest.txt" | tail -6 | sed 's/^/    /'

# --- 2. a deterministic, stratified slice of benchmark_val_1000 ---------------
# Stratum coverage guarantees the dataset check sees empty, rare, other, bobcat and
# threshold-adjacent frames rather than whatever the first N happened to be. Seeded by
# a sort on image_id, so it is byte-identical on every run and every host.
echo
echo "--- stratified slice of benchmark_val_1000 (deterministic)"
"${PYTHON}" - "${MANIFEST}" "${IMAGES_ROOT}" "${EVIDENCE}/e1_slice.jsonl" "${EVIDENCE}/bobcat_frame.json" <<'PY'
import json, os, sys
from collections import defaultdict

manifest, images_root, slice_out, bobcat_out = sys.argv[1:5]
rows = [json.loads(l) for l in open(manifest) if l.strip()]

per_stratum = {"empty": 50, "other": 50, "rare": 50, "bobcat": 60,
               "threshold_adjacent": 30, "multi_label": 2}
by_stratum = defaultdict(list)
for r in rows:
    if os.path.exists(os.path.join(images_root, r["file_name"])):
        by_stratum[r["benchmark_stratum"]].append(r)

chosen = []
for stratum in ("empty", "other", "rare", "bobcat", "threshold_adjacent", "multi_label"):
    picks = sorted(by_stratum.get(stratum, []), key=lambda r: r["image_id"])
    chosen.extend(picks[: per_stratum[stratum]])
chosen.sort(key=lambda r: r["image_id"])

with open(slice_out, "w") as f:
    for r in chosen:
        f.write(json.dumps({
            "file_name": r["file_name"],
            "image_id": r["image_id"],
            "seq_id": r.get("seq_id", ""),
            "labels": r["labels"],
        }) + "\n")

bobcat = next(r for r in chosen if "bobcat" in r["labels"])
json.dump({"file_name": bobcat["file_name"], "image_id": bobcat["image_id"]},
          open(bobcat_out, "w"))
print(f"    {len(chosen)} frames across "
      f"{len({r['benchmark_stratum'] for r in chosen})} strata; "
      f"bobcat frame {bobcat['image_id'][:12]}...")
PY

BOBCAT_FILE="$("${PYTHON}" -c "import json;print(json.load(open('${EVIDENCE}/bobcat_frame.json'))['file_name'])")"
BOBCAT_IMAGE="/work/${IMAGES_ROOT}/${BOBCAT_FILE}"

# --- 3. self-test on M0 (native) ----------------------------------------------
echo
echo "--- self-test on M0"
"${IN_CONTAINER[@]}" bash -lc "
    ${BUILD_DIR}/wildlife_trigger self-test \
        --model /work/${M0_ONNX} \
        --class-map /work/${CLASS_MAP} \
        --policy /work/${POLICY} \
        --image ${BOBCAT_IMAGE} \
        --output /work/${REL_EVIDENCE}/self_test.m0.json
" >/dev/null 2> >(sed 's/^/    /' >&2)

# --- 4. infer on a real bobcat frame: native + Pi ISA -------------------------
echo
echo "--- infer on the bobcat frame (native + qemu ${QEMU_PI5_CPU})"
"${IN_CONTAINER[@]}" bash -lc "
    ${BUILD_DIR}/wildlife_trigger infer \
        --model /work/${M0_ONNX} --class-map /work/${CLASS_MAP} \
        --policy /work/${POLICY} --image ${BOBCAT_IMAGE} \
        --output /work/${REL_EVIDENCE}/infer.m0.native.json
" >/dev/null 2> "${EVIDENCE}/log.default.stderr"
sed 's/^/    native  /' "${EVIDENCE}/log.default.stderr"

"${IN_CONTAINER[@]}" bash -lc "
    qemu-aarch64 -L / -cpu ${QEMU_PI5_CPU} ${BUILD_DIR}/wildlife_trigger infer \
        --model /work/${M0_ONNX} --class-map /work/${CLASS_MAP} \
        --policy /work/${POLICY} --image ${BOBCAT_IMAGE} \
        --output /work/${REL_EVIDENCE}/infer.m0.qemu.json
" >/dev/null 2> >(sed 's/^/    qemu    /' >&2)

# --- 5. the logging convention under its threshold ----------------------------
# Four probes captured to stderr files the gate reads. Default: debug is suppressed and
# the info summary shows. debug: the model-contract debug lines appear. error: the info
# summary is silenced. bad policy: the error keeps its "error:" prefix and a non-zero exit.
echo
echo "--- logging convention (WILDLIFE_LOG_LEVEL)"
"${IN_CONTAINER[@]}" bash -lc "
    WILDLIFE_LOG_LEVEL=debug ${BUILD_DIR}/wildlife_trigger infer \
        --model /work/${M0_ONNX} --class-map /work/${CLASS_MAP} \
        --policy /work/${POLICY} --image ${BOBCAT_IMAGE}
" >/dev/null 2> "${EVIDENCE}/log.debug.stderr"

"${IN_CONTAINER[@]}" bash -lc "
    WILDLIFE_LOG_LEVEL=error ${BUILD_DIR}/wildlife_trigger infer \
        --model /work/${M0_ONNX} --class-map /work/${CLASS_MAP} \
        --policy /work/${POLICY} --image ${BOBCAT_IMAGE}
" >/dev/null 2> "${EVIDENCE}/log.error.stderr"

set +e
"${IN_CONTAINER[@]}" bash -lc "
    ${BUILD_DIR}/wildlife_trigger infer \
        --model /work/${M0_ONNX} --class-map /work/${CLASS_MAP} \
        --policy /work/${REL_EVIDENCE}/does_not_exist.json --image ${BOBCAT_IMAGE}
" >/dev/null 2> "${EVIDENCE}/log.badpolicy.stderr"
echo "$?" > "${EVIDENCE}/log.badpolicy.exit"
set -e
echo "    default/debug/error/bad-policy captured"

# --- 6. benchmark on M0 (native; a timing-path smoke, NOT a Pi result) --------
echo
echo "--- benchmark on M0 (NOT a Pi result — DESIGN §12.4)"
"${IN_CONTAINER[@]}" bash -lc "
    ${BUILD_DIR}/wildlife_trigger benchmark \
        --model /work/${M0_ONNX} --class-map /work/${CLASS_MAP} \
        --policy /work/${POLICY} --image ${BOBCAT_IMAGE} \
        --warmup 10 --iterations 100 --threads 1 \
        --output /work/${REL_EVIDENCE}/benchmark.m0.json
" >/dev/null 2> >(sed 's/^/    /' >&2)

# --- 7. dataset runner over the slice, on M0 (native) -------------------------
echo
echo "--- run-dataset over the stratified slice, on M0"
"${IN_CONTAINER[@]}" bash -lc "
    ${BUILD_DIR}/wildlife_trigger run-dataset \
        --model /work/${M0_ONNX} --class-map /work/${CLASS_MAP} \
        --policy /work/${POLICY} \
        --manifest /work/${REL_EVIDENCE}/e1_slice.jsonl \
        --images-root /work/${IMAGES_ROOT} \
        --output /work/${REL_EVIDENCE}/run_dataset.m0.jsonl \
        --on-corrupt fail --threads 1
" 2>&1 | tail -1 | sed 's/^/    /'

# --- 8. the gate --------------------------------------------------------------
echo
echo "--- E1 gate"
"${PYTHON}" -m wildlife_trigger.validate.e1_gate \
    --evidence "${EVIDENCE}" \
    --freeze "${FREEZE}" \
    --manifest "${MANIFEST}" \
    --report "${OUT}/e1_gate.json"
