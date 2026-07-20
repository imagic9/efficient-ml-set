#!/usr/bin/env bash
# Gate E — full ARM64 dry run of the deployment bundle (PLAN E8).
#
# Runs the EXACT future Pi commands — install.sh, run_benchmark.sh (the one-command
# benchmark matrix), run_demo.sh — UNATTENDED, in a clean ubuntu:24.04 container
# (the rented Pi's OS) with only the bundle mounted (no repo, no training env). Then parses the
# machine-readable outputs with the reporting code (e8_dry_run.py) and records a
# known-good dry-run log for later diffing against the real Pi (Phase F).
#
# gx10 dry-run latency is diagnostic only (DESIGN §12.4). Gate E is the last gate
# before the rental: do not rent the Pi before it passes.
#
# Usage:  scripts/run_e8_dry_run.sh [output_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
OUT="${1:-${PROJECT_ROOT}/results/e8}"
STAGING="${PROJECT_ROOT}/results/e7/bundle"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT}"

echo "=============================================================="
echo "Gate E — full ARM64 dry run (exact Pi commands, unattended)"
echo "=============================================================="

# Build a fresh bundle so the dry run reflects the current tree, not a stale stage.
echo
echo "--- build a fresh bundle"
bash "${PROJECT_ROOT}/scripts/build_bundle.sh" "${STAGING}" >/dev/null 2>&1 \
    && echo "    staged $(wc -l < "${STAGING}/MANIFEST.sha256") files"

echo
echo "--- unattended dry run in a clean ${TARGET_BASE_IMAGE} (no repo, no training env)"
DRY_LOG="${OUT}/dry_run.log"
set +e
docker run --rm -v "${STAGING}:/opt/bundle" \
    "${TARGET_BASE_IMAGE}@${TARGET_BASE_DIGEST}" \
    sh -c 'cd /opt/bundle && ./install.sh && ./run_benchmark.sh && ./run_demo.sh' \
    > "${DRY_LOG}" 2>&1
DRY_RC=$?
set -e
tail -20 "${DRY_LOG}" | sed 's/^/    /'
echo "    (full log: ${DRY_LOG}, exit ${DRY_RC})"

echo
echo "--- Gate E: parse the machine-readable outputs (baseline included, schema, off-Pi)"
"${PYTHON}" -m wildlife_trigger.validate.e8_dry_run \
    --bundle "${STAGING}" \
    --dry-run-rc "${DRY_RC}" \
    --output "${OUT}/dry_run.json"
