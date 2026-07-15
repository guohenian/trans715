#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
test -f experiments/phase2/runs/scale5000/diagnostic/checkpoint_epoch_131.pt || { echo "checkpoint_epoch_131.pt from the first diagnostic is required" >&2; exit 2; }
python -m building_simplify.pipeline train-diagnostic --config experiments/phase2/configs/scale5000_diagnostic_eos4.json --precision bf16 --output-dir experiments/phase2/runs/scale5000/diagnostic_eos4
