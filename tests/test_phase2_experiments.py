import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "experiments" / "phase2" / "configs"
SCRIPTS = ROOT / "experiments" / "phase2" / "scripts"


def _load(name: str) -> dict:
    return json.loads((CONFIGS / f"scale5000_{name}.json").read_text(encoding="utf-8"))


def test_phase2_configs_follow_one_variable_experiment_order():
    diagnostic = _load("diagnostic")
    dropout0 = _load("dropout0")
    preln = _load("preln")
    larger = _load("larger")

    assert diagnostic["command"] == "train-diagnostic"
    assert diagnostic["max_steps"] == 3000
    assert diagnostic["epochs"] * ((512 + diagnostic["batch_size"] - 1) // diagnostic["batch_size"]) >= diagnostic["max_steps"]
    assert diagnostic["dropout"] == 0.0
    assert diagnostic["weight_decay"] == 0.0
    assert diagnostic["scheduled_sampling_max"] == 0.0

    assert dropout0["command"] == "train-baseline"
    assert dropout0["dropout"] == 0.0
    assert not dropout0["pre_ln"]
    assert dropout0["d_model"] == 256
    assert preln == {**dropout0, "name": "preln", "output_dir": "experiments/phase2/runs/scale5000/preln", "pre_ln": True, "requires": "dropout0-reviewed"}
    assert larger == {**preln, "name": "larger", "output_dir": "experiments/phase2/runs/scale5000/larger", "d_model": 384, "num_layers": 6, "dim_feedforward": 1536, "requires": "preln-reviewed"}


def test_phase2_scripts_are_isolated_and_use_bf16():
    names = ["diagnostic", "dropout0", "preln", "larger"]
    contents = [(SCRIPTS / f"run_{name}_5000.sh").read_text(encoding="utf-8") for name in names]

    assert all("--precision bf16" in content for content in contents)
    assert len({line for content in contents for line in content.splitlines() if "--output-dir" in line}) == 4
    assert "train-diagnostic" in contents[0]
    assert all("train-baseline" in content for content in contents[1:])
