#!/bin/sh
# F3 confirmation seeds — non-gating M2 QAT variability runs (PLAN F3, DESIGN §8.5).
#
# Retrains the SELECTED transformation (M2 QAT, lr 5e-5, 6 epochs) from the frozen M0
# checkpoint with seeds 17 and 73, into ISOLATED output roots so the seed-42 deployment
# artifact is never touched. Measures seed variability only; must NOT gate the freeze, any
# trial day, or Gate F, but must finish before Gate G. Runs on gx10 (GPU). Sequential —
# one GPU, one run at a time.
#
# Launch detached:  nohup scripts/run_confirmation_seeds.sh > results/f3/confirmation_seeds.log 2>&1 < /dev/null &
set -eu

cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
. "${HOME}/venvs/wildlife_trigger/bin/activate"

echo "=== confirmation seeds start $(date -u +%FT%TZ) — python $(command -v python) ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "device", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
PY

for S in 17 73; do
    echo "=== $(date -u +%FT%TZ) confirmation seed ${S} START ==="
    python -m wildlife_trigger.optimize.qat_train \
        --config "configs/optimize/m2_qat_seed${S}.yaml" --lr 5e-5
    echo "=== $(date -u +%FT%TZ) confirmation seed ${S} DONE (exit $?) ==="
done
echo "=== ALL CONFIRMATION SEEDS DONE $(date -u +%FT%TZ) ==="
