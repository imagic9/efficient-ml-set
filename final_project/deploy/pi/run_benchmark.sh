#!/bin/sh
# Wildlife Trigger — one-command benchmark (E7/E8, Gate E).
#
# Benchmarks every bundled model — the FP32 baseline M0 and the INT8 candidates
# M2/M4 — on one sample frame, writing a machine-readable benchmark_<MODEL>.json per
# model plus a benchmark_matrix.json index. This is the one command a Pi operator runs
# to produce the measurement matrix Gate E and Phase F consume; the baseline is always
# in it, so an optimized model is never reported without the thing it is measured
# against.
#
# The numbers are a Pi result ONLY when this runs on a Pi (DESIGN §12.4). On any other
# host — gx10, a container — they are a smoke check of the timing path.
#
# POSIX sh. Usage:  ./run_benchmark.sh [warmup] [iterations]

set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${HERE}/bin/run.sh"
CLASS_MAP="${HERE}/policies/class_map.json"
IMAGES="${HERE}/data/images"
WARMUP="${1:-10}"
ITERS="${2:-100}"

SAMPLE="$(ls "${IMAGES}"/*.jpg 2>/dev/null | head -1)"
[ -n "${SAMPLE}" ] || { echo "no sample images in ${IMAGES}" >&2; exit 2; }

echo "=== Wildlife Trigger benchmark matrix (warmup ${WARMUP}, iterations ${ITERS}) ==="
echo "    sample: ${SAMPLE}"
echo "    NOTE: a Pi result only if this host is a Pi (DESIGN §12.4)."

MODELS=""
for MODEL in M0 M2 M4; do
    ONNX="${HERE}/models/${MODEL}.onnx"
    POLICY="${HERE}/policies/${MODEL}.json"
    [ -f "${ONNX}" ] || continue
    echo "--- ${MODEL}"
    "${RUN}" benchmark --model "${ONNX}" --class-map "${CLASS_MAP}" \
        --policy "${POLICY}" --image "${SAMPLE}" \
        --warmup "${WARMUP}" --iterations "${ITERS}" \
        --output "${HERE}/benchmark_${MODEL}.json" 2>&1 >/dev/null \
        | grep -iE 'end-to-end' | sed 's/^/    /' || true
    MODELS="${MODELS} ${MODEL}"
done

# A tiny index so the reporting code has one file to open. The baseline is named
# explicitly so a consumer can assert it is present.
{
    echo '{'
    echo '  "kind": "benchmark_matrix",'
    echo '  "baseline": "M0",'
    printf '  "models": ['
    first=1
    for MODEL in ${MODELS}; do
        [ "${first}" -eq 1 ] || printf ', '
        printf '"%s"' "${MODEL}"
        first=0
    done
    echo '],'
    echo '  "note": "per-model benchmark_<MODEL>.json; a Pi result only on a Pi (DESIGN 12.4)"'
    echo '}'
} > "${HERE}/benchmark_matrix.json"

echo
echo "=== wrote benchmark_matrix.json +$(echo "${MODELS}" | wc -w) per-model files"
