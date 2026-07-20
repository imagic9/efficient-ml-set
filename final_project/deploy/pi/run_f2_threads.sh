#!/bin/sh
# Wildlife Trigger — F2 thread-scaling matrix (PLAN §8 F2). Runs ON the Pi.
#
# The E6 knob matrix was M0-only and found threads>1 regressing on gx10 (20 cores,
# intra-op overhead > gain on this MobileNetV2). On the 4-core Cortex-A76 Pi the
# opposite holds, so this sweeps the FULL shortlist × threads {1,2,3,4} under an
# otherwise-shipping config to decide a single uniform thread count and to DECOMPOSE
# the speedup into a model part (quantization/pruning) and an inference part (threads).
#
# Threads change FP32 reduction order, so any thread count other than the frozen
# shipping default (1) is a CANDIDATE only until F3 re-checks parity; INT8 QDQ accumulates
# exactly in int32 and is thread-invariant. Real Pi results (DESIGN §12.4).
#
# POSIX sh. Usage:  ./run_f2_threads.sh [warmup] [iterations]
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${HERE}/bin/run.sh"
CM="${HERE}/policies/class_map.json"
OUT="${HERE}/f2"; mkdir -p "${OUT}"
WARMUP="${1:-20}"
ITERS="${2:-300}"
SAMPLE="$(ls "${HERE}/data/images"/*.jpg 2>/dev/null | head -1)"
[ -n "${SAMPLE}" ] || { echo "no sample images" >&2; exit 2; }

echo "=== F2 thread-scaling matrix (warmup ${WARMUP}, iters ${ITERS}) ==="
for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
    echo performance > "${c}" 2>/dev/null || true
done
echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" \
     "freq_kHz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)"

for M in M0 M2 M4; do
    for T in 1 2 3 4; do
        tag="${M}_t${T}"
        json="${OUT}/thr_${tag}.json"
        log="${OUT}/thr_${tag}.log"
        "${RUN}" benchmark --model "${HERE}/models/${M}.onnx" --class-map "${CM}" \
            --policy "${HERE}/policies/${M}.json" --image "${SAMPLE}" \
            --warmup "${WARMUP}" --iterations "${ITERS}" --threads "${T}" \
            --output "${json}" >/dev/null 2> "${log}" || true
        printf '    %-8s ' "${tag}:"
        grep -iE 'end-to-end' "${log}" || echo "(no summary)"
    done
done

echo "--- thermal / throttle after"
{ vcgencmd measure_temp; vcgencmd get_throttled; } 2>/dev/null | sed 's/^/    /' || true
echo "=== thread matrix done -> ${OUT} ==="
