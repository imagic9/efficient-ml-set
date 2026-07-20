#!/bin/sh
# F4 frozen full-test evaluation on gx10 (PLAN F4). The one-time test-set opening.
#
# Runs the FROZEN deployment artifacts — the E7 bundle's exact C++ binary + bundled ORT +
# M0/M2 ONNX + policies — through run-dataset on the full sealed cis_test and trans_test
# splits, then computes operating-point metrics at the frozen threshold (eval_frozen_test.py).
#
# The C++ run happens INSIDE the target container (wildlife-trigger-target:ubuntu2404), which
# carries the exact OpenCV 4.6.0 .406 the binary links — the gx10 host has no system OpenCV
# (E-phase idiom: run_e4_m0_parity.sh, run_e8_dry_run.sh). Metrics run on the host venv.
# gx10 evaluation of ACCURACY only (never latency; §12.4). Test labels are scored at the frozen
# validation threshold, never used to select (F3 already froze it).
#
# POSIX sh. Usage (gx10):  scripts/run_f4_frozen_test.sh
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
IMG=wildlife-trigger-target:ubuntu2404
OUT="${ROOT}/results/f4"; mkdir -p "${OUT}"

echo "=== F4 frozen full-test — C++ run-dataset inside ${IMG} (test set opened once) ==="
docker run --rm -v "${ROOT}:/work" -w /work "${IMG}" sh -c '
    set -eu
    B=/work/results/e7/bundle
    IMAGES=/work/data/raw/extracted/eccv_18_all_images_sm
    for M in M0 M2; do
        for SPLIT in cis_test trans_test; do
            echo "--- ${M} ${SPLIT}"
            "$B/bin/run.sh" run-dataset --model "$B/models/${M}.onnx" \
                --class-map "$B/policies/class_map.json" --policy "$B/policies/${M}.json" \
                --manifest "/work/data/manifests/${SPLIT}.jsonl" --images-root "$IMAGES" \
                --on-corrupt fail --output "/work/results/f4/gx10_test_${M}_${SPLIT}.jsonl" 2>&1 \
                | grep -iE "run-dataset" || true
        done
    done
'

echo "=== metrics at the frozen threshold (host venv) ==="
. "${HOME}/venvs/wildlife_trigger/bin/activate"
for M in M0 M2; do
    python3 scripts/eval_frozen_test.py --model-id "${M}" \
        --policy "results/e7/bundle/policies/${M}.json" --manifests-dir data/manifests \
        --pred-cis "results/f4/gx10_test_${M}_cis_test.jsonl" \
        --pred-trans "results/f4/gx10_test_${M}_trans_test.jsonl" \
        --output "results/f4/frozen_test_${M}.json"
done
echo "=== F4 frozen full-test done -> ${OUT} ==="
