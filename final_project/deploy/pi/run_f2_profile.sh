#!/bin/sh
# Wildlife Trigger — F2 Pi validation performance profiling (PLAN §8 F2).
#
# Runs ON the Pi. Pins the CPU governor to `performance` (a documented, reversible
# DVFS control so the M0-vs-M2/M4 comparison is not confounded by frequency scaling —
# NOT model tuning), benchmarks the frozen shortlist under the shipping config, then
# sweeps the E6 knob matrix one factor at a time for the FP32 baseline M0, and captures
# an ORT profile to identify the bottleneck.
#
# The benchmark's human summary goes to stderr; the machine-readable percentiles go to
# the --output JSON. This script keeps both: f2/bench_<tag>.json + f2/bench_<tag>.log.
#
# A latency here is a real Pi result (this is a Pi; DESIGN §12.4). Reduced decode
# (half/quarter) is measured for the report but stays REJECTED on accuracy grounds
# (E6 decode-drift gate); the thread count is only a candidate until F3 re-checks parity.
#
# POSIX sh. Usage:  ./run_f2_profile.sh [warmup] [iterations]
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${HERE}/bin/run.sh"
CM="${HERE}/policies/class_map.json"
OUT="${HERE}/f2"; mkdir -p "${OUT}"
WARMUP="${1:-20}"
ITERS="${2:-300}"
SAMPLE="$(ls "${HERE}/data/images"/*.jpg 2>/dev/null | head -1)"
[ -n "${SAMPLE}" ] || { echo "no sample images" >&2; exit 2; }

echo "=== F2 profiling (warmup ${WARMUP}, iters ${ITERS}) sample=$(basename "${SAMPLE}") ==="

echo "--- pin performance governor (documented DVFS control; reversible)"
for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
    echo performance > "${c}" 2>/dev/null || true
done
echo "    governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" \
     "cur_freq_kHz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)"

# $1=model $2=extra-args $3=tag
bench() {
    _m="$1"; _extra="$2"; _tag="$3"
    _json="${OUT}/bench_${_tag}.json"
    _log="${OUT}/bench_${_tag}.log"
    # shellcheck disable=SC2086
    "${RUN}" benchmark --model "${HERE}/models/${_m}.onnx" --class-map "${CM}" \
        --policy "${HERE}/policies/${_m}.json" --image "${SAMPLE}" \
        --warmup "${WARMUP}" --iterations "${ITERS}" ${_extra} \
        --output "${_json}" >/dev/null 2> "${_log}" || true
    printf '    %-22s ' "${_tag}:"
    grep -iE 'end-to-end' "${_log}" || echo "(no summary — see ${_log})"
}

echo "--- baseline shortlist (shipping: threads 1, decode full, graph all, arena on)"
bench M0 "" "M0_base"
bench M2 "" "M2_base"
bench M4 "" "M4_base"

echo "--- M0 knob matrix (one factor at a time, off the shipping baseline)"
bench M0 "--threads 2"            "M0_threads2"
bench M0 "--threads 3"            "M0_threads3"
bench M0 "--threads 4"            "M0_threads4"
bench M0 "--graph-opt extended"   "M0_graph_extended"
bench M0 "--arena off"            "M0_arena_off"
bench M0 "--preprocess reference" "M0_preprocess_reference"
bench M0 "--decode half"          "M0_decode_half"
bench M0 "--decode quarter"       "M0_decode_quarter"

echo "--- ORT profile for M0 (bottleneck identification)"
"${RUN}" benchmark --model "${HERE}/models/M0.onnx" --class-map "${CM}" \
    --policy "${HERE}/policies/M0.json" --image "${SAMPLE}" \
    --warmup 5 --iterations 30 --profile-prefix "${OUT}/ort_profile_M0" \
    >/dev/null 2>&1 || true
ls "${OUT}"/ort_profile_M0* 2>/dev/null | sed 's/^/    profile: /' || true

echo "--- thermal / throttle after the sweep"
{ vcgencmd measure_temp; vcgencmd get_throttled; } 2>/dev/null | sed 's/^/    /' || true
echo "    cur_freq_kHz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)"

echo "=== F2 done -> ${OUT} ==="
