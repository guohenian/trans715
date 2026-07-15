import json
from pathlib import Path

import pytest
import torch

from building_simplify.pipeline import parse_args
from building_simplify.train import (
    NonFiniteTrainingError,
    ensure_finite_tensor,
    resolve_precision_mode,
)


def test_auto_precision_uses_fp32_on_cpu():
    policy = resolve_precision_mode("auto", "cpu")

    assert policy.mode == "fp32"
    assert not policy.autocast_enabled
    assert policy.autocast_dtype is None


def test_auto_precision_uses_bf16_on_supported_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    policy = resolve_precision_mode("auto", "cuda")

    assert policy.mode == "bf16"
    assert policy.autocast_enabled
    assert policy.autocast_dtype is torch.bfloat16


def test_explicit_bf16_rejects_unsupported_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(RuntimeError, match="BF16"):
        resolve_precision_mode("bf16", "cuda")


def test_non_finite_tensor_writes_failure_report(tmp_path: Path):
    context = {
        "epoch": 8,
        "global_step": 218000,
        "source_max_len": 1500,
        "target_max_len": 1450,
        "precision": "bf16",
        "learning_rate": 1e-4,
    }

    with pytest.raises(NonFiniteTrainingError, match="loss"):
        ensure_finite_tensor(torch.tensor(float("nan")), "loss", tmp_path, context)

    report = json.loads((tmp_path / "failure.json").read_text(encoding="utf-8"))
    assert report["failure_stage"] == "loss"
    assert report["epoch"] == 8
    assert report["global_step"] == 218000
    assert report["source_max_len"] == 1500
    assert report["target_max_len"] == 1450
    assert report["precision"] == "bf16"


def test_pipeline_config_loads_defaults_and_cli_overrides(tmp_path: Path):
    config = tmp_path / "experiment.json"
    config.write_text(
        json.dumps({
            "datasets_root": "datasets",
            "scale": 5000,
            "output_dir": "runs/from-config",
            "batch_size": 8,
            "precision": "bf16",
        }),
        encoding="utf-8",
    )

    args = parse_args([
        "train-baseline", "--config", str(config), "--batch-size", "4"
    ])

    assert args.scale == 5000
    assert args.output_dir == Path("runs/from-config")
    assert args.batch_size == 4
    assert args.precision == "bf16"
