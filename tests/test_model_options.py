import torch
import inspect

from building_simplify.model import BuildingTransformer
from building_simplify.train import _average_completed_loss, _seed_everything, _training_limit_reached, evaluate_full_greedy_from_config, strip_pad_tokens


def test_pre_ln_is_optional_and_defaults_to_existing_post_ln():
    post_ln = BuildingTransformer(1000, d_model=32, nhead=4, num_layers=1, dim_feedforward=64)
    pre_ln = BuildingTransformer(1000, d_model=32, nhead=4, num_layers=1, dim_feedforward=64, pre_ln=True)

    assert not post_ln.transformer.encoder.layers[0].norm_first
    assert pre_ln.transformer.encoder.layers[0].norm_first
    source = torch.tensor([[725, 723, 10, 722]])
    target = torch.tensor([[723, 10, 722]])
    assert pre_ln(source, target).shape == (1, 3, 1000)


def test_full_greedy_evaluation_accepts_persistent_prediction_output():
    assert "prediction_output" in inspect.signature(evaluate_full_greedy_from_config).parameters


def test_training_seed_repeats_torch_random_sequence():
    _seed_everything(20260713)
    first = torch.rand(4)
    _seed_everything(20260713)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_strip_pad_only_removes_trailing_batch_padding():
    assert strip_pad_tokens([1, 720, 2, 722, 720, 720]) == [1, 720, 2, 722]


def test_training_step_limit_and_partial_epoch_average():
    assert _training_limit_reached(3000, 3000)
    assert not _training_limit_reached(2999, 3000)
    assert _average_completed_loss(12.0, 3) == 4.0
