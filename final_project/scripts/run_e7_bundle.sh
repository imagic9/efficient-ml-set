#!/usr/bin/env bash
# E7 — build, audit, and clean-install-test the deployment bundle (PLAN E7).
#
# Three steps, each answering a question the Pi rental cannot afford to answer late:
#   1. build_bundle.sh stages the bundle in the target container;
#   2. bundle_audit.py proves completeness, checksums, and target glibc;
#   3. the clean-install test copies the bundle into a fresh debian:bookworm-slim
#      container with NO OpenCV, NO ORT, and NO access to the repo, then runs
#      install.sh + run_demo.sh — proving the bundle installs and runs on a clean
#      target-compatible host without the training environment or unbundled artifacts.
#
# The demo latency is a smoke check on gx10, never a Pi result (DESIGN §12.4).
#
# Usage:  scripts/run_e7_bundle.sh [output_dir]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

PYTHON="${VENV_PATH}/bin/python"
OUT="${1:-${PROJECT_ROOT}/results/e7}"
STAGING="${PROJECT_ROOT}/results/e7/bundle"

cd "${PROJECT_ROOT}"
mkdir -p "${OUT}"

echo "=============================================================="
echo "E7 deployment bundle: build -> audit -> clean install test"
echo "=============================================================="

echo
echo "--- 1/3 build_bundle.sh"
bash "${PROJECT_ROOT}/scripts/build_bundle.sh" "${STAGING}" 2>&1 | sed 's/^/    /'

echo
echo "--- 2/3 bundle_audit.py (completeness, checksums, glibc)"
"${PYTHON}" -m wildlife_trigger.validate.bundle_audit \
    --bundle "${STAGING}" --project-root "${PROJECT_ROOT}" \
    --image-tag "${TARGET_IMAGE_TAG}" --report "${OUT}/bundle_audit.json" | sed 's/^/    /'

echo
echo "--- 3/3 clean-install test in a fresh ${TARGET_BASE_IMAGE} (no OpenCV/ORT, no repo)"
CLEAN_LOG="${OUT}/clean_install.log"
set +e
docker run --rm -v "${STAGING}:/opt/bundle" \
    "${TARGET_BASE_IMAGE}@${TARGET_BASE_DIGEST}" \
    sh -c 'cd /opt/bundle && ./install.sh && ./run_demo.sh' \
    > "${CLEAN_LOG}" 2>&1
CLEAN_RC=$?
set -e
sed 's/^/    /' "${CLEAN_LOG}"

echo
echo "--- E7 verdict"
"${PYTHON}" - "${OUT}/bundle_audit.json" "${CLEAN_LOG}" "${CLEAN_RC}" "${OUT}/e7_bundle.json" <<'PY'
import json, sys
audit_path, log_path, rc, out = sys.argv[1:5]
audit = json.load(open(audit_path))
log = open(log_path).read()
clean = {
    "exit_code": int(rc),
    "opencv_installed": "OpenCV runtime installed" in log,
    "self_test_passed": "self-test PASSED" in log,
    "demo_complete": "demo complete" in log,
    "libs_resolved": "all libraries resolve" in log,
}
clean["passed"] = int(rc) == 0 and clean["self_test_passed"] and clean["demo_complete"]
report = {
    "gate": "E7 deployment bundle (PLAN E7)",
    "audit_passed": audit["passed"],
    "audit": {k: audit[k] for k in ("file_count", "checksums_verified", "max_glibc",
                                    "missing_paths")},
    "clean_install": clean,
    "verdict": {"passed": bool(audit["passed"] and clean["passed"])},
}
json.dump(report, open(out, "w"), indent=2)
open(out, "a").write("\n")
print(f"    audit: {'PASS' if audit['passed'] else 'FAIL'} "
      f"({audit['file_count']} files, glibc {audit['max_glibc']})")
print(f"    clean install: {'PASS' if clean['passed'] else 'FAIL'} "
      f"(opencv={clean['opencv_installed']}, self-test={clean['self_test_passed']}, "
      f"demo={clean['demo_complete']})")
print(f"E7 bundle {'PASSED' if report['verdict']['passed'] else 'FAILED'}; wrote {out}")
sys.exit(0 if report["verdict"]["passed"] else 1)
PY
