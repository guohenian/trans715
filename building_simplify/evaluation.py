from __future__ import annotations

import math
import json
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Sequence

import shapefile
from shapely.geometry import Polygon, shape

from .bpe import BPEVocabulary
from .config import SPECIAL_TOKENS
from .geometry import TokenFrame, clean_polygon, decode_polygon


def _frame_from_dict(data: dict) -> TokenFrame:
    payload = dict(data)
    payload["inner_starts"] = tuple(tuple(pair) for pair in payload.get("inner_starts", ()))
    return TokenFrame(**payload)


def _edit_distance(first: Sequence[int], second: Sequence[int]) -> int:
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, start=1):
        current = [i]
        for j, right in enumerate(second, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def _eos_position(tokens: Sequence[int], eos_token: int) -> int:
    try:
        return list(tokens).index(eos_token)
    except ValueError:
        return len(tokens)


def token_metrics(prediction: Sequence[int], target: Sequence[int], eos_token: int = 722) -> dict[str, float]:
    prediction = [int(token) for token in prediction]
    target = [int(token) for token in target]
    maximum_length = max(len(prediction), len(target), 1)
    edit_distance = _edit_distance(prediction, target)
    return {
        "greedy_exact": float(prediction == target),
        "token_edit_similarity": 1.0 - (edit_distance / maximum_length),
        "token_length_error": float(abs(len(prediction) - len(target))),
        "eos_position_error": float(abs(_eos_position(prediction, eos_token) - _eos_position(target, eos_token))),
    }


def _mean_boundary_distance(first: Polygon, second: Polygon, samples: int = 128) -> float:
    distances: list[float] = []
    for source, target in ((first.boundary, second.boundary), (second.boundary, first.boundary)):
        if source.length <= 1e-12:
            continue
        for index in range(samples):
            point = source.interpolate(index / samples, normalized=True)
            distances.append(point.distance(target))
    return sum(distances) / len(distances) if distances else math.inf


def _minimum_rectangle_properties(polygon: Polygon) -> tuple[Polygon, float, float, float]:
    rectangle = polygon.minimum_rotated_rectangle
    coords = list(rectangle.exterior.coords)[:-1]
    edges = []
    for index in range(4):
        dx = coords[(index + 1) % 4][0] - coords[index][0]
        dy = coords[(index + 1) % 4][1] - coords[index][1]
        edges.append((math.hypot(dx, dy), math.degrees(math.atan2(dy, dx)) % 180.0))
    lengths = sorted((length for length, _ in edges), reverse=True)
    angle = max(edges, key=lambda item: item[0])[1]
    return rectangle, angle, lengths[0], lengths[-1]


def _angle_error(first: float, second: float) -> float:
    difference = abs(first - second) % 180.0
    return min(difference, 180.0 - difference)


def geometry_metrics(prediction: Polygon, target: Polygon) -> dict[str, float]:
    if prediction.is_empty or target.is_empty:
        return {key: math.inf for key in (
            "outline_iou", "mean_boundary_distance", "hausdorff_distance", "perimeter_relative_error",
            "area_relative_error", "minimum_rectangle_iou", "minimum_rectangle_angle_error_deg",
            "minimum_rectangle_long_error", "minimum_rectangle_short_error",
        )}
    union = prediction.union(target).area
    outline_iou = prediction.intersection(target).area / union if union > 1e-12 else 0.0
    pred_rectangle, pred_angle, pred_long, pred_short = _minimum_rectangle_properties(prediction)
    target_rectangle, target_angle, target_long, target_short = _minimum_rectangle_properties(target)
    rectangle_union = pred_rectangle.union(target_rectangle).area
    return {
        "outline_iou": outline_iou,
        "mean_boundary_distance": _mean_boundary_distance(prediction, target),
        "hausdorff_distance": prediction.boundary.hausdorff_distance(target.boundary),
        "perimeter_relative_error": abs(prediction.length - target.length) / max(target.length, 1e-12),
        "area_relative_error": abs(prediction.area - target.area) / max(target.area, 1e-12),
        "minimum_rectangle_iou": pred_rectangle.intersection(target_rectangle).area / rectangle_union if rectangle_union > 1e-12 else 0.0,
        "minimum_rectangle_angle_error_deg": _angle_error(pred_angle, target_angle),
        "minimum_rectangle_long_error": abs(pred_long - target_long) / max(target_long, 1e-12),
        "minimum_rectangle_short_error": abs(pred_short - target_short) / max(target_short, 1e-12),
    }


def tokenization_geometry_metrics(prediction: Polygon, target: Polygon) -> dict[str, float]:
    if prediction.is_empty or target.is_empty:
        return {
            "outline_iou": 0.0,
            "mean_boundary_distance": math.inf,
            "perimeter_relative_error": math.inf,
            "area_relative_error": math.inf,
        }
    union = prediction.union(target).area
    return {
        "outline_iou": prediction.intersection(target).area / union if union > 1e-12 else 0.0,
        "mean_boundary_distance": _mean_boundary_distance(prediction, target, samples=16),
        "perimeter_relative_error": abs(prediction.length - target.length) / max(target.length, 1e-12),
        "area_relative_error": abs(prediction.area - target.area) / max(target.area, 1e-12),
    }


def vertex_bucket(vertex_count: int) -> str:
    if vertex_count <= 4:
        return "4"
    if vertex_count <= 8:
        return "5-8"
    if vertex_count <= 16:
        return "9-16"
    return "17+"


def _finite_mean(total: float, count: int) -> float:
    return total / count if count else math.nan


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def evaluate_prediction_jsonl(
    prediction_path: Path | str,
    vocab_path: Path | str,
    output_path: Path | str,
) -> dict:
    vocab = BPEVocabulary.load(vocab_path)
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    metric_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    counts: dict[str, int] = defaultdict(int)
    invalid: dict[str, int] = defaultdict(int)
    invalid_reference: dict[str, int] = defaultdict(int)
    all_sample_iou_sum: dict[str, float] = defaultdict(float)
    valid_reference_count: dict[str, int] = defaultdict(int)

    with Path(prediction_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            groups = [
                "overall",
                str(row.get("difficulty", "unknown")),
                f"vertices:{vertex_bucket(int(row.get('source_vertex_count', 0)))}",
            ]
            prediction = [int(token) for token in row["pred_tokens"]]
            target = [int(token) for token in row["target_tokens"]]
            metrics = token_metrics(prediction, target)
            frame = _frame_from_dict(row["source_frame"])
            try:
                pred_polygon = decode_polygon(vocab.decode(prediction), frame)
                if pred_polygon.is_empty or not pred_polygon.is_valid:
                    raise ValueError("invalid predicted polygon")
            except Exception:
                pred_polygon = None
            try:
                target_polygon = decode_polygon(vocab.decode(target), frame)
                if target_polygon.is_empty or not target_polygon.is_valid:
                    raise ValueError("invalid reference polygon")
            except Exception:
                target_polygon = None
            if pred_polygon is not None and target_polygon is not None:
                metrics.update(geometry_metrics(pred_polygon, target_polygon))

            for group in groups:
                counts[group] += 1
                if pred_polygon is None:
                    invalid[group] += 1
                if target_polygon is None:
                    invalid_reference[group] += 1
                else:
                    valid_reference_count[group] += 1
                    if pred_polygon is not None:
                        all_sample_iou_sum[group] += metrics.get("outline_iou", 0.0)
                for key, value in metrics.items():
                    if math.isfinite(float(value)):
                        sums[group][key] += float(value)
                        metric_counts[group][key] += 1

    report_groups: dict[str, dict[str, float | int]] = {}
    for group in sorted(counts):
        sample_count = counts[group]
        values: dict[str, float | int] = {
            "sample_count": sample_count,
            "invalid_geometry_count": invalid[group],
            "invalid_geometry_rate": invalid[group] / sample_count if sample_count else 0.0,
            "invalid_reference_count": invalid_reference[group],
            "invalid_reference_rate": invalid_reference[group] / sample_count if sample_count else 0.0,
        }
        for key, total in sums[group].items():
            values[key] = _finite_mean(total, metric_counts[group][key])
        if valid_reference_count[group]:
            values["all_sample_outline_iou"] = all_sample_iou_sum[group] / valid_reference_count[group]
        report_groups[group] = values

    report = {"prediction_path": str(prediction_path), "vocab_path": str(vocab_path), "groups": report_groups}
    output_path = Path(output_path)
    _write_json_atomic(output_path, report)
    return report


def audit_bpe_jsonl(raw_path: Path | str, bpe_path: Path | str, vocab_path: Path | str) -> dict[str, int | str]:
    vocab = BPEVocabulary.load(vocab_path)
    rows = roundtrip_failures = invalid_token_count = row_alignment_failures = 0
    with Path(raw_path).open("r", encoding="utf-8") as raw_handle, Path(bpe_path).open("r", encoding="utf-8") as bpe_handle:
        for raw_line, bpe_line in zip_longest(raw_handle, bpe_handle):
            if raw_line is None or bpe_line is None:
                row_alignment_failures += 1
                continue
            if not raw_line.strip() and not bpe_line.strip():
                continue
            if not raw_line.strip() or not bpe_line.strip():
                row_alignment_failures += 1
                continue
            raw = json.loads(raw_line)
            bpe = json.loads(bpe_line)
            rows += 1
            if raw.get("sample_id") != bpe.get("sample_id"):
                row_alignment_failures += 1
            for field in ("source_tokens", "target_tokens"):
                encoded = [int(token) for token in bpe[field]]
                invalid_token_count += sum(
                    not (0 <= token < 720 or token in SPECIAL_TOKENS or token in vocab.token_atoms)
                    for token in encoded
                )
                if vocab.decode(encoded) != [int(token) for token in raw[field]]:
                    roundtrip_failures += 1
    return {
        "raw_path": str(raw_path),
        "bpe_path": str(bpe_path),
        "vocab_path": str(vocab_path),
        "rows": rows,
        "roundtrip_failures": roundtrip_failures,
        "invalid_token_count": invalid_token_count,
        "row_alignment_failures": row_alignment_failures,
    }


def categorize_failure(message: str) -> str:
    text = message.lower()
    if "empty" in text or "zero area" in text:
        return "empty_geometry"
    if "not polygon" in text or "no polygon" in text:
        return "non_polygon"
    if "invalid" in text or "self-intersection" in text:
        return "invalid_geometry"
    if "ring" in text or "token" in text or "candidate" in text:
        return "decode_error"
    return "other_error"


def vertex_count(polygon: Polygon) -> int:
    return max(0, len(polygon.exterior.coords) - 1) + sum(max(0, len(ring.coords) - 1) for ring in polygon.interiors)


def unmatched_geometry_metrics(raw: Polygon, prediction: Polygon) -> dict[str, float]:
    valid = not prediction.is_empty and prediction.is_valid and prediction.area > 1e-12
    if not valid:
        return {"valid_geometry": 0.0}
    detailed = geometry_metrics(prediction, raw)
    raw_vertices = vertex_count(raw)
    prediction_vertices = vertex_count(prediction)
    return {
        "valid_geometry": 1.0,
        "vertex_compression_ratio": (raw_vertices - prediction_vertices) / raw_vertices if raw_vertices else 0.0,
        "area_ratio": prediction.area / max(raw.area, 1e-12),
        "perimeter_ratio": prediction.length / max(raw.length, 1e-12),
        "minimum_rectangle_iou": detailed["minimum_rectangle_iou"],
        "minimum_rectangle_angle_error_deg": detailed["minimum_rectangle_angle_error_deg"],
    }


def evaluate_unmatched_shapefile(
    raw_path: Path | str,
    prediction_path: Path | str,
    unmatched_jsonl: Path | str,
    output_path: Path | str,
    failures_path: Path | str | None = None,
) -> dict:
    requested: set[int] = set()
    with Path(unmatched_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                requested.add(int(json.loads(line)["raw_fid"]))
    sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    matched = missing_prediction = 0
    failure_categories: dict[str, int] = {}
    resolved_failures = Path(failures_path) if failures_path is not None else Path(prediction_path).with_suffix(".failures.jsonl")
    if resolved_failures.exists():
        with resolved_failures.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    category = categorize_failure(str(json.loads(line).get("error", "")))
                    failure_categories[category] = failure_categories.get(category, 0) + 1
    with shapefile.Reader(str(raw_path)) as raw_reader, shapefile.Reader(str(prediction_path)) as prediction_reader:
        prediction_names = [field[0] for field in prediction_reader.fields[1:]]
        if "raw_fid" not in prediction_names:
            raise ValueError("Prediction shapefile is missing raw_fid")
        raw_fid_index = prediction_names.index("raw_fid")
        prediction_by_fid = {int(record[raw_fid_index]): index for index, record in enumerate(prediction_reader.iterRecords())}
        for raw_fid in sorted(requested):
            prediction_index = prediction_by_fid.get(raw_fid)
            if prediction_index is None:
                missing_prediction += 1
                continue
            raw_polygon = clean_polygon(shape(raw_reader.shape(raw_fid).__geo_interface__))
            prediction_polygon = clean_polygon(shape(prediction_reader.shape(prediction_index).__geo_interface__))
            metrics = {"valid_geometry": 0.0} if raw_polygon is None or prediction_polygon is None else unmatched_geometry_metrics(raw_polygon, prediction_polygon)
            for key, value in metrics.items():
                if math.isfinite(value):
                    sums[key] = sums.get(key, 0.0) + value
                    metric_counts[key] = metric_counts.get(key, 0) + 1
            matched += 1
    report = {
        "requested": len(requested),
        "matched_predictions": matched,
        "missing_predictions": missing_prediction,
        "failure_categories": failure_categories,
        "metrics": {key: sums[key] / metric_counts[key] for key in sorted(sums)},
    }
    report["metrics"]["valid_geometry"] = sums.get("valid_geometry", 0.0) / max(len(requested), 1)
    output_path = Path(output_path)
    _write_json_atomic(output_path, report)
    return report
