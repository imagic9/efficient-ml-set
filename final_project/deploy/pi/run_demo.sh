#!/bin/sh
# Wildlife Trigger — Raspberry Pi demo (E7).
#
# Exercises the deployed application end to end with the bundled artifacts and the
# small sample manifest, so a fresh install can be seen working in one command before
# any real benchmark. Runs the on-device self-test (no fixtures needed), a single
# infer on a sample frame, a short benchmark, and run-dataset over the sample slice.
#
# Latency printed here is a SMOKE CHECK of the timing path on whatever host runs it.
# It is a Pi result only when this runs ON a Pi (DESIGN §12.4); Phase F takes the
# measurement that counts.
#
# POSIX sh. Usage:  ./run_demo.sh [MODEL]      MODEL in {M0,M2,M4}, default M0

set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-M0}"
RUN="${HERE}/bin/run.sh"
ONNX="${HERE}/models/${MODEL}.onnx"
POLICY="${HERE}/policies/${MODEL}.json"
CLASS_MAP="${HERE}/policies/class_map.json"
MANIFEST="${HERE}/data/manifest.jsonl"
IMAGES="${HERE}/data/images"

[ -f "${ONNX}" ] || { echo "no model ${MODEL} in bundle (have: $(ls "${HERE}/models"))" >&2; exit 2; }

# First image in the sample set, for the single-frame infer/benchmark.
SAMPLE="$(ls "${IMAGES}"/*.jpg 2>/dev/null | head -1)"
[ -n "${SAMPLE}" ] || { echo "no sample images in ${IMAGES}" >&2; exit 2; }

echo "=== Wildlife Trigger demo — model ${MODEL} ==="
echo "    binary : ${RUN}"
echo "    sample : ${SAMPLE}"
echo

echo "--- self-test (asserts contract on this host; no fixtures)"
"${RUN}" self-test --model "${ONNX}" --class-map "${CLASS_MAP}" \
    --policy "${POLICY}" --image "${SAMPLE}" >/dev/null && echo "    self-test PASSED"

echo
echo "--- infer on one frame"
"${RUN}" infer --model "${ONNX}" --class-map "${CLASS_MAP}" \
    --policy "${POLICY}" --image "${SAMPLE}" 2>/dev/null \
    | grep -E '"SHUTTER_TRIGGER"|"top1"|"class"|"score"' | head -6 | sed 's/^/    /' || true

echo
echo "--- benchmark (SMOKE CHECK — a Pi result only if this is a Pi; DESIGN §12.4)"
"${RUN}" benchmark --model "${ONNX}" --class-map "${CLASS_MAP}" \
    --policy "${POLICY}" --image "${SAMPLE}" --warmup 5 --iterations 50 2>&1 >/dev/null \
    | grep -iE 'end-to-end' | sed 's/^/    /' || true

echo
echo "--- run-dataset over the sample slice ($(wc -l < "${MANIFEST}") frames)"
"${RUN}" run-dataset --model "${ONNX}" --class-map "${CLASS_MAP}" --policy "${POLICY}" \
    --manifest "${MANIFEST}" --images-root "${IMAGES}" \
    --on-corrupt fail --output "${HERE}/demo_predictions.jsonl" 2>&1 \
    | grep -iE 'run-dataset' | sed 's/^/    /' || true

echo
echo "=== demo complete. Predictions -> ${HERE}/demo_predictions.jsonl"
echo "    For a real benchmark on a Pi, see the F-phase profiling in PLAN §8."
