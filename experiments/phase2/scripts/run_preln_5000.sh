#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
test -f experiments/phase2/runs/scale5000/dropout0/dropout0-reviewed || { echo "dropout0-reviewed marker is required" >&2; exit 2; }
python -m building_simplify.pipeline train-baseline --config experiments/phase2/configs/scale5000_preln.json --precision bf16 --output-dir experiments/phase2/runs/scale5000/preln
