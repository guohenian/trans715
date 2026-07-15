#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m building_simplify.pipeline train-diagnostic --config experiments/phase2/configs/scale5000_diagnostic.json --precision bf16 --output-dir experiments/phase2/runs/scale5000/diagnostic
