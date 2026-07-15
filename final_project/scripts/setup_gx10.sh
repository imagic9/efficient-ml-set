#!/usr/bin/env bash
# Create the isolated gx10 Python/GPU training environment (PLAN A2).
#
# Deliberately NOT ~/efficientml/venv: that venv has
# include-system-site-packages=true and inherits ~5 GB of unrelated packages from
# ~/.local. A lockfile taken from it would describe an environment we do not
# control and cannot reproduce.
#
# Usage:  scripts/setup_gx10.sh [--recreate]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${HERE}")"

# shellcheck source=../configs/env/pins.env
source "${PROJECT_ROOT}/configs/env/pins.env"

if [[ "${1:-}" == "--recreate" ]]; then
    echo "Removing ${VENV_PATH}"
    rm -rf "${VENV_PATH}"
fi

if [[ ! -d "${VENV_PATH}" ]]; then
    echo "Creating isolated venv at ${VENV_PATH}"
    # No --system-site-packages: that isolation is the whole point.
    "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

PIP="${VENV_PATH}/bin/pip"
"${PIP}" install -q --upgrade pip setuptools wheel

echo "Verifying isolation before installing anything"
if "${VENV_PATH}/bin/python" -c "import sys; sys.exit(0 if len(sys.path) and not any('/.local/' in p for p in sys.path) else 1)"; then
    echo "  no ~/.local leakage"
else
    echo "  ERROR: ~/.local is on sys.path; the venv is not isolated" >&2
    exit 1
fi

echo "Installing torch ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
# Pinned because A0 verified this build on GB10 (compute capability 12.1, CUDA
# 13.0, driver 580.142). Known-working on Blackwell beats newest.
"${PIP}" install -q \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

echo "Installing the project and its dev extras"
"${PIP}" install -q -e "${PROJECT_ROOT}[dev]"

echo
echo "Verifying the environment"
"${VENV_PATH}/bin/python" - <<'PY'
import torch
print(f"  torch          {torch.__version__}")
print(f"  cuda available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device         {torch.cuda.get_device_name(0)} "
          f"cap={torch.cuda.get_device_capability(0)}")
else:
    raise SystemExit("  ERROR: CUDA is unavailable; training would silently fall to CPU")

import onnxruntime as ort
print(f"  onnxruntime    {ort.__version__} providers={ort.get_available_providers()}")
import cv2
print(f"  opencv         {cv2.__version__}")
import torch_pruning
print(f"  torch-pruning  {torch_pruning.__version__}")
PY

echo
echo "Writing the lockfile"
# Exclude the editable install of the project itself: a lockfile should pin
# dependencies, not restate the thing being built.
"${PIP}" freeze --exclude-editable > "${PROJECT_ROOT}/requirements.lock"
echo "  $(wc -l < "${PROJECT_ROOT}/requirements.lock") pinned packages -> requirements.lock"

echo
echo "Activate with:  source ${VENV_PATH}/bin/activate"
