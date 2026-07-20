#!/bin/sh
# Wildlife Trigger — F4 Pi benchmark + parity capture (PLAN §8 F4). Runs ON the Pi.
#
# The measurement that counts (DESIGN §12.4): the frozen deployment config (threads=1, full
# decode, ORT_ENABLE_ALL, arena on, fused, 256x192) benchmarked on the real Raspberry Pi CM5.
#   1. M0 baseline vs the frozen winner M2 (and M4 for the optimization-ladder table),
#      >=1000 iterations x 3 SEPARATE process repetitions each (per-run JSON).
#   2. run-dataset over the bundled parity slice for M0 and M2 -> per-frame predictions the
#      gx10 side diffs against the frozen C++/ORT reference (score deltas, decisions).
# Pins the performance governor; records thermal/throttle around every run. No tuning, no
# config change (post-freeze). Outputs to ./f4/.
#
# POSIX sh. Usage:  ./run_f4_pi_benchmark.sh [warmup] [iterations] [reps]
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${HERE}/bin/run.sh"
CM="${HERE}/policies/class_map.json"
WARMUP="${1:-50}"
ITERS="${2:-1000}"
REPS="${3:-3}"
OUT="${HERE}/${4:-f4}"; mkdir -p "${OUT}"   # $4 lets F5 write a separate repeat dir
MANIFEST="${HERE}/data/manifest.jsonl"
IMAGES="${HERE}/data/images"
SAMPLE="$(ls "${IMAGES}"/*.jpg 2>/dev/null | head -1)"
[ -n "${SAMPLE}" ] || { echo "no sample images" >&2; exit 2; }

echo "=== F4 Pi benchmark (frozen config: threads=1, full decode, ALL, arena on, fused) ==="
for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
    echo performance > "${c}" 2>/dev/null || true
done
echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" \
     "freq_kHz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)"
thermal() { printf '    thermal: '; vcgencmd measure_temp 2>/dev/null; vcgencmd get_throttled 2>/dev/null | sed 's/^/    /'; }

echo "--- latency: >=${ITERS} iters x ${REPS} reps, threads=1 (frozen)"
for M in M0 M2 M4; do
    [ -f "${HERE}/models/${M}.onnx" ] || continue
    rep=1
    while [ "${rep}" -le "${REPS}" ]; do
        json="${OUT}/bench_${M}_rep${rep}.json"
        log="${OUT}/bench_${M}_rep${rep}.log"
        "${RUN}" benchmark --model "${HERE}/models/${M}.onnx" --class-map "${CM}" \
            --policy "${HERE}/policies/${M}.json" --image "${SAMPLE}" \
            --warmup "${WARMUP}" --iterations "${ITERS}" --threads 1 \
            --output "${json}" >/dev/null 2> "${log}" || true
        printf '    %-3s rep%s: ' "${M}" "${rep}"
        grep -iE 'end-to-end' "${log}" || echo "(no summary)"
        rep=$((rep + 1))
    done
    thermal
done

echo "--- parity: run-dataset over the bundled slice ($(wc -l < "${MANIFEST}") frames), threads=1"
for M in M0 M2; do
    "${RUN}" run-dataset --model "${HERE}/models/${M}.onnx" --class-map "${CM}" \
        --policy "${HERE}/policies/${M}.json" --manifest "${MANIFEST}" \
        --images-root "${IMAGES}" --on-corrupt fail \
        --output "${OUT}/pi_parity_${M}.jsonl" 2>&1 | grep -iE 'run-dataset' | sed "s/^/    ${M}: /" || true
done

echo "=== F4 Pi benchmark done -> ${OUT} ==="
