#!/usr/bin/env bash
# E5 — the benchmark + system monitor, on M0 (PLAN E5, DESIGN §11/§12).
#
# The BenchmarkRunner and SystemMonitor (benchmark.cpp) were built in A4 and their
# schema was gated by a4_gate. E5 closes the two boxes A4 left open: the percentile
# CALCULATION (now unit-tested in test_benchmark, run below via ctest) and the
# performance-target report (200 ms / 5 FPS primary, 100 ms / 10 FPS aspirational),
# which the benchmark now emits with the target values and _on_this_host flags.
#
# Nothing here is a Pi result: DESIGN §12.4 makes a latency a Pi number only when
# measured ON a Pi, and the output's measured_on_pi flag is false.
#
# Usage:  scripts/run_e5_benchmark.sh [output_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
OUT="${1:-${PROJECT_ROOT}/results/e5}"
BUILD_DIR="/work/build/e5"
IN_CONTAINER=(docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${TARGET_IMAGE_TAG}")

FREEZE="results/model_selection/pre_pi_freeze.json"
CLASS_MAP="artifacts/class_map.json"
POLICY="artifacts/policies/bobcat_v1.json"
FIXTURE="tests/fixtures/frame_1024x747.jpg"

cd "${PROJECT_ROOT}"

M0_ONNX="$("${PYTHON}" -c "
import json
m0 = next(m for m in json.load(open('${FREEZE}'))['models'] if m['model_id'] == 'M0')
print(m0['onnx']['artifact'])")"
[[ -f "${M0_ONNX}" ]] || { echo "M0 ONNX not found at ${M0_ONNX}" >&2; exit 2; }

REL_OUT="${OUT#"${PROJECT_ROOT}"/}"
rm -rf "${OUT}"
mkdir -p "${OUT}"

echo "=============================================================="
echo "E5 benchmark + system monitor on M0 -> ${OUT}"
echo "=============================================================="

echo
echo "--- build + ctest inside ${TARGET_IMAGE_TAG} (incl. the new benchmark test)"
"${IN_CONTAINER[@]}" bash -lc "
    set -euo pipefail
    cmake -S /work/cpp -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release \
        -DWILDLIFE_CPU_TARGET=${QEMU_PI5_CPU} >/dev/null
    cmake --build ${BUILD_DIR} -j\"\$(nproc)\" >/dev/null
    cd ${BUILD_DIR} && ctest --output-on-failure 2>&1 | tail -8 | sed 's/^/    /'
"

echo
echo "--- benchmark on M0 (NOT a Pi result — DESIGN §12.4)"
"${IN_CONTAINER[@]}" bash -lc "
    ${BUILD_DIR}/wildlife_trigger benchmark \
        --model /work/${M0_ONNX} --class-map /work/${CLASS_MAP} \
        --policy /work/${POLICY} --image /work/${FIXTURE} \
        --warmup 10 --iterations 200 --threads 1 \
        --output /work/${REL_OUT}/benchmark_m0.json
" >/dev/null 2> >(sed 's/^/    /' >&2)

echo
echo "--- E5 checks: schema, percentile ordering, targets report"
"${PYTHON}" - "${OUT}/benchmark_m0.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
fail = []

if d.get("schema_version") != 1:
    fail.append("schema_version != 1")
if d.get("measured_iterations", 0) <= 0:
    fail.append("no measured iterations")

# Percentile ordering per stage (the calculation itself is unit-tested in test_benchmark).
for stage, v in d["stages_ms"].items():
    if not (v["min"] <= v["p50"] <= v["p95"] <= v["p99"] <= v["max"]):
        fail.append(f"{stage}: percentiles not ordered")

# The performance-target report (E5), honest about not being a Pi verdict.
pt = d.get("performance_targets", {})
if pt.get("measured_on_pi") is not False:
    fail.append("performance_targets.measured_on_pi must be present and false off-Pi")
for tier in ("primary", "aspirational"):
    t = pt.get(tier, {})
    if "met_on_this_host" not in t or "p95_end_to_end_ms" not in t or "min_fps" not in t:
        fail.append(f"performance_targets.{tier} incomplete")
if pt.get("primary", {}).get("p95_end_to_end_ms") != 200.0:
    fail.append("primary target must be 200 ms")
if pt.get("aspirational", {}).get("p95_end_to_end_ms") != 100.0:
    fail.append("aspirational target must be 100 ms")

# The system monitor must report RSS and be honest about absent sensors.
sysd = d["system"]
if sysd.get("peak_rss_kib", 0) <= 0:
    fail.append("peak_rss_kib not captured")
for sensor in ("cpu_temperature_c", "cpu_frequency_khz", "throttling"):
    val = sysd.get(sensor)
    if not (isinstance(val, str) or isinstance(val, (int, float))):
        fail.append(f"{sensor} is neither a value nor 'unavailable'")

e2e = d["stages_ms"]["end_to_end"]
print(f"    end-to-end p50={e2e['p50']:.2f}ms p95={e2e['p95']:.2f}ms  "
      f"({d['fps']['end_to_end_from_p50']:.1f} FPS)")
print(f"    targets (Pi, NOT measured here): primary 200ms/5FPS "
      f"met_on_this_host={pt['primary']['met_on_this_host']}; "
      f"aspirational 100ms/10FPS met_on_this_host={pt['aspirational']['met_on_this_host']}")
print(f"    system: peak_rss={sysd['peak_rss_kib']}KiB  "
      f"temp={sysd['cpu_temperature_c']}  throttling={sysd['throttling']}")

if fail:
    print("E5 FAILED:")
    for f in fail:
        print(f"    FAIL: {f}")
    sys.exit(1)
print("E5 checks PASSED")
PY
