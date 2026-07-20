#!/usr/bin/env bash
# E7/F1 — prove the fail-closed host preflight (issue #77).
#
# The preflight refuses a host outside the Pi 5 / Ubuntu 24.04 contract BEFORE
# install.sh mutates anything. This exercises the success path and every refusal path —
# without a physical Pi 4 or a wrong-OS host — by feeding preflight.sh synthetic
# CPU/OS/arch inputs (its testing overrides) and a real non-Ubuntu container. Each
# refusal must exit non-zero with an actionable reason and change nothing.
#
# Scenarios:
#   R1 success  — real gx10 host in an Ubuntu 24.04 container: PASS, is_pi5_a76=0 (dev host)
#   R2 pi5-sim  — A76 cpuinfo fixture:                    PASS, is_pi5_a76=1
#   R3 pi4      — A72 cpuinfo fixture (no asimddp):        REFUSE
#   R4 wrong-os — real debian:bookworm-slim /etc/os-release: REFUSE (not Ubuntu 24.04)
#   R5 arch     — WT_UNAME_M=x86_64:                       REFUSE
#
# Usage:  scripts/run_e7_preflight.sh [output_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
OUT="${1:-${PROJECT_ROOT}/results/e7}"
RAW="${OUT}/preflight_raw"
FX=/work/tests/fixtures/preflight
PF=/work/deploy/pi/preflight.sh
TARGET="${TARGET_IMAGE_TAG}"

cd "${PROJECT_ROOT}"
rm -rf "${RAW}"; mkdir -p "${RAW}"

echo "=============================================================="
echo "E7/F1 host preflight — success + refusal paths (issue #77)"
echo "=============================================================="

# Run preflight.sh in a container with the repo mounted (it refuses before any
# mutation, so a plain mount is fine). Writes <name>.rc / .out / .err into RAW.
scenario() {  # $1=name $2=image  $3..=env assignments
    local name="$1" image="$2"; shift 2
    set +e
    docker run --rm -v "${PROJECT_ROOT}:/work" -w /work "${image}" \
        env "$@" sh "${PF}" > "${RAW}/${name}.out" 2> "${RAW}/${name}.err"
    echo $? > "${RAW}/${name}.rc"
    set -e
    echo "  ${name}: exit $(cat "${RAW}/${name}.rc")"
}

echo
echo "--- running scenarios"
scenario R1 "${TARGET}"
scenario R2 "${TARGET}" "WT_CPUINFO=${FX}/cpuinfo_pi5_a76"
scenario R3 "${TARGET}" "WT_CPUINFO=${FX}/cpuinfo_pi4_a72"
scenario R4 "debian:bookworm-slim"
scenario R5 "${TARGET}" "WT_UNAME_M=x86_64"

echo
echo "--- E7/F1 preflight verdict"
"${PYTHON}" -m wildlife_trigger.validate.preflight_check \
    --raw-dir "${RAW}" --output "${OUT}/preflight.json"
