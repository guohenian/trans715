import json
from dataclasses import asdict
from pathlib import Path

import pytest
from shapely.geometry import Polygon

from building_simplify.bpe import BPEVocabulary
from building_simplify.evaluation import evaluate_prediction_jsonl, geometry_metrics, token_metrics, tokenization_geometry_metrics, vertex_bucket
from building_simplify.geometry import make_source_tokens, make_target_tokens


def test_token_metrics_distinguish_exact_and_edit_similarity():
    exact = token_metrics([1, 2, 722], [1, 2, 722])
    changed = token_metrics([1, 3, 722], [1, 2, 722])

    assert exact["greedy_exact"] == 1.0
    assert exact["token_edit_similarity"] == 1.0
    assert changed["greedy_exact"] == 0.0
    assert 0.0 < changed["token_edit_similarity"] < 1.0
    assert changed["eos_position_error"] == 0.0


def test_identical_geometry_has_ideal_detailed_metrics():
    polygon = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])
    metrics = geometry_metrics(polygon, polygon)

    assert metrics["outline_iou"] == pytest.approx(1.0)
    assert metrics["mean_boundary_distance"] == pytest.approx(0.0)
    assert metrics["perimeter_relative_error"] == pytest.approx(0.0)
    assert metrics["minimum_rectangle_iou"] == pytest.approx(1.0)
    assert metrics["minimum_rectangle_angle_error_deg"] == pytest.approx(0.0)


def test_tokenization_metrics_only_compute_required_reconstruction_errors():
    polygon = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])
    metrics = tokenization_geometry_metrics(polygon, polygon)
    assert set(metrics) == {"outline_iou", "mean_boundary_distance", "perimeter_relative_error", "area_relative_error"}
    assert metrics["outline_iou"] == pytest.approx(1.0)


def test_vertex_bucket_uses_declared_ranges():
    assert vertex_bucket(4) == "4"
    assert vertex_bucket(8) == "5-8"
    assert vertex_bucket(16) == "9-16"
    assert vertex_bucket(17) == "17+"


def test_prediction_jsonl_is_aggregated_by_difficulty(tmp_path: Path):
    polygon = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])
    _, frame = make_source_tokens(polygon, 5000)
    target = make_target_tokens(polygon, frame)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({
            "raw_fid": 1,
            "difficulty": "simple_rectangle",
            "source_vertex_count": 4,
            "source_frame": asdict(frame),
            "pred_tokens": target,
            "target_tokens": target,
        }) + "\n",
        encoding="utf-8",
    )
    vocab_path = tmp_path / "vocab.txt"
    BPEVocabulary(token_atoms={}).save(vocab_path)

    report = evaluate_prediction_jsonl(predictions, vocab_path, tmp_path / "report.json")

    assert report["groups"]["overall"]["sample_count"] == 1
    assert report["groups"]["simple_rectangle"]["greedy_exact"] == pytest.approx(1.0)
    assert report["groups"]["vertices:4"]["outline_iou"] == pytest.approx(1.0)


def test_prediction_report_tracks_invalid_reference_separately(tmp_path: Path):
    polygon = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])
    _, frame = make_source_tokens(polygon, 5000)
    target = make_target_tokens(polygon, frame)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({
        "raw_fid": 1,
        "difficulty": "complex",
        "source_vertex_count": 5,
        "source_frame": asdict(frame),
        "pred_tokens": target,
        "target_tokens": [723, 722],
    }) + "\n", encoding="utf-8")
    vocab_path = tmp_path / "vocab.txt"
    BPEVocabulary(token_atoms={}).save(vocab_path)

    report = evaluate_prediction_jsonl(predictions, vocab_path, tmp_path / "report.json")

    assert report["groups"]["overall"]["invalid_reference_count"] == 1
    assert "outline_iou" not in report["groups"]["overall"]


def test_invalid_prediction_counts_as_zero_in_all_sample_iou(tmp_path: Path):
    polygon = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])
    _, frame = make_source_tokens(polygon, 5000)
    target = make_target_tokens(polygon, frame)
    rows = [
        {"raw_fid": 1, "difficulty": "complex", "source_vertex_count": 5, "source_frame": asdict(frame), "pred_tokens": target, "target_tokens": target},
        {"raw_fid": 2, "difficulty": "complex", "source_vertex_count": 5, "source_frame": asdict(frame), "pred_tokens": [723, 722], "target_tokens": target},
    ]
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    vocab_path = tmp_path / "vocab.txt"
    BPEVocabulary(token_atoms={}).save(vocab_path)

    report = evaluate_prediction_jsonl(predictions, vocab_path, tmp_path / "report.json")

    assert report["groups"]["overall"]["outline_iou"] == pytest.approx(1.0)
    assert report["groups"]["overall"]["all_sample_outline_iou"] == pytest.approx(0.5)
