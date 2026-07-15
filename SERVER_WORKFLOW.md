# Server workflow

The public entry point is `python -m building_simplify.pipeline`. Los Angeles is used for training and validation. New York is exported as the frozen final test set.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-server.txt
python -m pytest -q
```

## 1. Export all purpose datasets

Place the four source directories at the workspace root using their existing names, then run:

```bash
python -m building_simplify.pipeline prepare-all \
  --workspace . --output-root datasets --seed 20260713
```

This command projects Los Angeles to EPSG:32611 and New York to EPSG:32618, creates the 80/20 Los Angeles split, retains 10% of simple training rectangles, trains one Los Angeles BPE vocabulary per scale, and exports:

```text
datasets/scale{5000,10000}/
  train/{atomic.jsonl,bpe.jsonl}
  validation/{atomic.jsonl,bpe.jsonl}
  test_new_york/{atomic.jsonl,bpe.jsonl}
  unmatched/{ids.jsonl,source_atomic.jsonl}
  diagnostic_512/{atomic.jsonl,bpe.jsonl}
  audits/*.json
  vocab/bpe_vocab.txt
```

Completed projection and scale manifests are hash-checked and reused. Pass `--force` only when intentionally rebuilding. Review both `audits/manifest.json` files before training.

## 2. Train baseline and diagnostic models

### Precision policy

Training accepts `--precision auto|bf16|fp32`. `auto` selects BF16 only on a CUDA device that reports BF16 support and otherwise uses FP32. FP16 is intentionally not used. On the RTX 5090 server, use BF16 explicitly:

```bash
--device cuda --precision bf16
```

The trainer checks every loss and gradient norm for NaN/Inf. A failure writes `failure.json`, saves `checkpoint_last_finite.pt` when model parameters are still finite, and stops immediately. Epoch checkpoints contain optimizer, both learning-rate schedulers, random states, resolved precision and the full training configuration.

Unchanged Post-LN baseline:

```bash
python -m building_simplify.pipeline train-baseline \
  --datasets-root datasets --scale 5000 \
  --output-dir runs/scale5000/baseline --device cuda --precision bf16
```

Fixed 512-complex-sample overfit diagnostic (`dropout=0`, weight decay and scheduled sampling disabled, maximum 3000 steps):

```bash
python -m building_simplify.pipeline train-diagnostic \
  --datasets-root datasets --scale 5000 \
  --output-dir runs/scale5000/diagnostic --device cuda --precision bf16
```

Repeat for scale 10000. The diagnostic gate is teacher-forced accuracy >= 0.995 and greedy exact >= 0.98. After it passes, use `train-baseline` with one explicit change at a time, such as `--dropout 0`, `--pre-ln`, or the larger capacity arguments.

For the isolated 1:5000 phase-two workflow, run only the first script initially:

```bash
bash experiments/phase2/scripts/run_diagnostic_5000.sh
```

After reviewing the diagnostic metrics, create the documented approval marker and run `run_dropout0_5000.sh`. Pre-LN and larger-model scripts also require their previous experiment's review marker. Configurations, commands and acceptance gates are documented in `experiments/phase2/README.md`; outputs remain under ignored `experiments/phase2/runs/`.

## 3. Evaluate frozen predictions

Save greedy predictions with `building_simplify.train --eval-only-checkpoint --prediction-output`, then aggregate them:

```bash
python -m building_simplify.pipeline evaluate \
  --predictions runs/scale5000/new_york_predictions.jsonl \
  --vocab datasets/scale5000/vocab/bpe_vocab.txt \
  --output runs/scale5000/new_york_metrics.json
```

New York is evaluated once after model selection. The formal target is complex greedy exact >= 0.60 for each scale separately.

Infer only ArcGIS-unmatched Los Angeles FIDs:

```bash
python -m building_simplify.pipeline infer-unmatched \
  --datasets-root datasets --scale 5000 \
  --checkpoint runs/scale5000/chosen/checkpoint_epoch_30.pt \
  --output runs/scale5000/unmatched_predictions.shp --device cuda
```

`unmatched/ids.jsonl` means ArcGIS supplied no target record. Pairing, geometry, or tokenization failures are separate in `audits/preparation_failures.jsonl`.
