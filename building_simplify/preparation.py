from __future__ import annotations

import hashlib
import json
import math
import heapq
from dataclasses import asdict, dataclass
from collections import defaultdict
from pathlib import Path

import shapefile
from pyproj import CRS
from pyproj import Transformer
from shapely.geometry import Polygon, shape
from shapely.ops import transform

from .evaluation import tokenization_geometry_metrics, vertex_bucket
from .geometry import clean_polygon, decode_polygon, make_source_tokens, make_target_tokens


@dataclass(frozen=True)
class PreparationReport:
    scale: int
    paired: int
    train_written: int
    validation_written: int
    rectangles_dropped: int
    unmatched: int
    failures: int
    pairing_failures: int
    preprocessing_failures: int
    simple_rectangles: int
    complex_samples: int
    all_jsonl: str
    train_jsonl: str
    validation_jsonl: str
    unmatched_jsonl: str
    failures_jsonl: str
    group_stats: dict
    split_stats: dict


class PairingError(ValueError):
    pass


def _stable_fraction(value: str, seed: int, namespace: str) -> float:
    payload = f"{seed}:{namespace}:{value}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(1 << 64)


def split_for_osm_id(osm_id: str, seed: int = 20260713, train_fraction: float = 0.8) -> str:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    return "train" if _stable_fraction(str(osm_id), seed, "split") < train_fraction else "validation"


def keep_training_sample(
    osm_id: str,
    difficulty: str,
    seed: int = 20260713,
    rectangle_keep_fraction: float = 0.1,
) -> bool:
    if difficulty != "simple_rectangle":
        return True
    if not 0.0 <= rectangle_keep_fraction <= 1.0:
        raise ValueError("rectangle_keep_fraction must be between 0 and 1")
    return _stable_fraction(str(osm_id), seed, "rectangle") < rectangle_keep_fraction


def _angle_degrees(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    first = (a[0] - b[0], a[1] - b[1])
    second = (c[0] - b[0], c[1] - b[1])
    denominator = math.hypot(*first) * math.hypot(*second)
    if denominator <= 1e-12:
        return 0.0
    cosine = max(-1.0, min(1.0, (first[0] * second[0] + first[1] * second[1]) / denominator))
    return math.degrees(math.acos(cosine))


def _parallel_error_degrees(first: tuple[float, float], second: tuple[float, float]) -> float:
    first_angle = math.degrees(math.atan2(first[1], first[0])) % 180.0
    second_angle = math.degrees(math.atan2(second[1], second[0])) % 180.0
    difference = abs(first_angle - second_angle)
    return min(difference, 180.0 - difference)


def is_simple_rectangle(
    polygon: Polygon,
    angle_tolerance_deg: float = 5.0,
    parallel_tolerance_deg: float = 5.0,
    min_fill_ratio: float = 0.98,
) -> bool:
    if polygon.is_empty or not polygon.is_valid or polygon.interiors:
        return False
    coords = list(polygon.exterior.coords)
    if len(coords) != 5:
        return False
    vertices = coords[:-1]
    if any(abs(_angle_degrees(vertices[i - 1], vertices[i], vertices[(i + 1) % 4]) - 90.0) > angle_tolerance_deg for i in range(4)):
        return False
    edges = [
        (vertices[(i + 1) % 4][0] - vertices[i][0], vertices[(i + 1) % 4][1] - vertices[i][1])
        for i in range(4)
    ]
    if _parallel_error_degrees(edges[0], edges[2]) > parallel_tolerance_deg:
        return False
    if _parallel_error_degrees(edges[1], edges[3]) > parallel_tolerance_deg:
        return False
    rectangle_area = polygon.minimum_rotated_rectangle.area
    return rectangle_area > 1e-12 and polygon.area / rectangle_area >= min_fill_ratio


def classify_pair(raw_polygon: Polygon, target_polygon: Polygon) -> str:
    return "simple_rectangle" if is_simple_rectangle(raw_polygon) and is_simple_rectangle(target_polygon) else "complex"


def projected_crs_from_shapefile(path: Path | str) -> CRS:
    path = Path(path)
    prj_path = path.with_suffix(".prj")
    if not prj_path.exists():
        raise ValueError(f"Missing CRS file: {prj_path}")
    crs = CRS.from_wkt(prj_path.read_text(encoding="utf-8", errors="ignore"))
    axis_units = {axis.unit_name.lower() for axis in crs.axis_info if axis.unit_name}
    if not crs.is_projected or not axis_units.intersection({"metre", "meter"}):
        raise ValueError(f"Shapefile CRS must be projected with metre units: {path}")
    return crs


def project_shapefile(input_path: Path | str, output_path: Path | str, target_epsg: int) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    source_prj = input_path.with_suffix(".prj")
    if not source_prj.exists():
        raise ValueError(f"Missing CRS file: {source_prj}")
    source_crs = CRS.from_wkt(source_prj.read_text(encoding="utf-8", errors="ignore"))
    target_crs = CRS.from_epsg(target_epsg)
    coordinate_transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with shapefile.Reader(str(input_path)) as reader:
        writer = shapefile.Writer(str(output_path), shapeType=reader.shapeType)
        writer.autoBalance = True
        for field in reader.fields[1:]:
            writer.field(*field)
        for shape_record in reader.iterShapeRecords():
            geometry = transform(coordinate_transformer.transform, shape(shape_record.shape.__geo_interface__))
            writer.shape(geometry.__geo_interface__)
            writer.record(*list(shape_record.record))
        writer.close()
    output_path.with_suffix(".prj").write_text(target_crs.to_wkt(version="WKT1_ESRI"), encoding="utf-8")
    return output_path


def _polygon_from_shape_record(shape_record) -> Polygon | None:
    return clean_polygon(shape(shape_record.shape.__geo_interface__))


def _vertex_count(polygon: Polygon) -> int:
    return max(0, len(polygon.exterior.coords) - 1) + sum(max(0, len(ring.coords) - 1) for ring in polygon.interiors)


def build_filtered_raw_jsonl(
    raw_path: Path | str,
    target_path: Path | str,
    output_dir: Path | str,
    scale: int,
    seed: int = 20260713,
    train_fraction: float = 0.8,
    rectangle_keep_fraction: float = 0.1,
    sample_prefix: str = "la",
    split_override: str | None = None,
    write_all: bool = True,
    write_split_files: bool = True,
) -> PreparationReport:
    raw_crs = projected_crs_from_shapefile(raw_path)
    target_crs = projected_crs_from_shapefile(target_path)
    if not raw_crs.equals(target_crs):
        raise ValueError("Raw and target shapefiles must use the same projected CRS")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train" / "atomic.jsonl"
    validation_path = output_dir / "validation" / "atomic.jsonl"
    all_path = output_dir / "all" / "atomic.jsonl"
    unmatched_path = output_dir / "unmatched" / "ids.jsonl"
    failures_path = output_dir / "audits" / "preparation_failures.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    all_path.parent.mkdir(parents=True, exist_ok=True)
    unmatched_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    train_work = train_path.with_suffix(train_path.suffix + ".tmp")
    validation_work = validation_path.with_suffix(validation_path.suffix + ".tmp")
    all_work = all_path.with_suffix(all_path.suffix + ".tmp")
    unmatched_work = unmatched_path.with_suffix(unmatched_path.suffix + ".tmp")
    failures_work = failures_path.with_suffix(failures_path.suffix + ".tmp")

    paired = train_written = validation_written = rectangles_dropped = failures = 0
    pairing_failures = preprocessing_failures = 0
    simple_rectangles = complex_samples = 0
    group_counts: dict[str, int] = defaultdict(int)
    group_metric_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    split_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with shapefile.Reader(str(raw_path)) as raw_reader, shapefile.Reader(str(target_path)) as target_reader:
        raw_names = [field[0] for field in raw_reader.fields[1:]]
        target_names = [field[0] for field in target_reader.fields[1:]]
        if "InBld_FID" not in target_names:
            raise ValueError("Target shapefile is missing InBld_FID")
        target_fid_index = target_names.index("InBld_FID")
        raw_osm_index = raw_names.index("osm_id") if "osm_id" in raw_names else None
        target_osm_index = target_names.index("osm_id") if "osm_id" in target_names else None
        matched = bytearray(len(raw_reader))

        with train_work.open("w", encoding="utf-8") as train_out, validation_work.open("w", encoding="utf-8") as validation_out, all_work.open("w", encoding="utf-8") as all_out, failures_work.open("w", encoding="utf-8") as failure_out:
            for target_fid, target_record in enumerate(target_reader.iterShapeRecords()):
                raw_fid = None
                try:
                    try:
                        raw_fid = int(target_record.record[target_fid_index])
                    except (TypeError, ValueError) as exc:
                        raise PairingError(f"Invalid InBld_FID: {target_record.record[target_fid_index]!r}") from exc
                    if raw_fid < 0 or raw_fid >= len(raw_reader):
                        raise PairingError(f"InBld_FID {raw_fid} is outside raw shapefile")
                    if matched[raw_fid]:
                        raise PairingError(f"Duplicate InBld_FID {raw_fid}")
                    matched[raw_fid] = 1
                    raw_record = raw_reader.shapeRecord(raw_fid)
                    raw_osm = str(raw_record.record[raw_osm_index]) if raw_osm_index is not None else str(raw_fid)
                    target_osm = str(target_record.record[target_osm_index]) if target_osm_index is not None else raw_osm
                    if raw_osm != target_osm:
                        raise PairingError(f"osm_id mismatch for InBld_FID {raw_fid}: {raw_osm} != {target_osm}")
                    raw_polygon = _polygon_from_shape_record(raw_record)
                    target_polygon = _polygon_from_shape_record(target_record)
                    if raw_polygon is None or target_polygon is None:
                        raise ValueError("Empty or unsupported polygon geometry")
                    difficulty = classify_pair(raw_polygon, target_polygon)
                    split = split_override or split_for_osm_id(raw_osm, seed=seed, train_fraction=train_fraction)
                    source_tokens, frame = make_source_tokens(raw_polygon, scale)
                    target_tokens = make_target_tokens(target_polygon, frame)
                    tokenized_target = decode_polygon(target_tokens, frame)
                    if tokenized_target.is_empty or not tokenized_target.is_valid:
                        raise ValueError("Target tokenization produced invalid geometry")
                    tokenization = tokenization_geometry_metrics(tokenized_target, target_polygon)
                    paired += 1
                    if difficulty == "simple_rectangle":
                        simple_rectangles += 1
                    else:
                        complex_samples += 1
                    groups = ["overall", difficulty, f"vertices:{vertex_bucket(_vertex_count(raw_polygon))}"]
                    for group in groups:
                        group_counts[group] += 1
                        for metric_name in ("outline_iou", "mean_boundary_distance", "perimeter_relative_error", "area_relative_error"):
                            group_metric_sums[group][metric_name] += tokenization[metric_name]
                    row = {
                        "sample_id": f"{sample_prefix}:{raw_fid}:{scale}",
                        "raw_fid": raw_fid,
                        "target_fid": target_fid,
                        "osm_id": raw_osm,
                        "scale": scale,
                        "split": split,
                        "difficulty": difficulty,
                        "source_tokens": source_tokens,
                        "target_tokens": target_tokens,
                        "source_frame": asdict(frame),
                        "source_vertex_count": _vertex_count(raw_polygon),
                        "target_vertex_count": _vertex_count(target_polygon),
                        "tokenization_metrics": tokenization,
                    }
                    if write_all:
                        all_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if not write_split_files:
                        continue
                    split_group = "train_before_filter" if split == "train" else "validation"
                    split_stats[split_group][difficulty] += 1
                    split_stats[split_group][f"vertices:{vertex_bucket(_vertex_count(raw_polygon))}"] += 1
                    if split == "train" and not keep_training_sample(
                        raw_osm, difficulty, seed=seed, rectangle_keep_fraction=rectangle_keep_fraction
                    ):
                        rectangles_dropped += 1
                        continue
                    if split == "train":
                        split_stats["train_after_filter"][difficulty] += 1
                        split_stats["train_after_filter"][f"vertices:{vertex_bucket(_vertex_count(raw_polygon))}"] += 1
                    output = train_out if split == "train" else validation_out
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if split == "train":
                        train_written += 1
                    else:
                        validation_written += 1
                except PairingError as exc:
                    failures += 1
                    pairing_failures += 1
                    failure_out.write(json.dumps({"target_fid": target_fid, "raw_fid": raw_fid, "failure_type": "pairing", "error": str(exc)}, ensure_ascii=False) + "\n")
                except Exception as exc:
                    failures += 1
                    preprocessing_failures += 1
                    failure_out.write(json.dumps({"target_fid": target_fid, "raw_fid": raw_fid, "failure_type": "preprocessing", "error": str(exc)}, ensure_ascii=False) + "\n")

        unmatched = 0
        with unmatched_work.open("w", encoding="utf-8") as unmatched_out:
            for raw_fid, is_matched in enumerate(matched):
                if is_matched:
                    continue
                record = raw_reader.record(raw_fid)
                osm_id = str(record[raw_osm_index]) if raw_osm_index is not None else str(raw_fid)
                unmatched_out.write(json.dumps({"raw_fid": raw_fid, "osm_id": osm_id, "scale": scale}, ensure_ascii=False) + "\n")
                unmatched += 1

    for temporary, final in (
        (train_work, train_path),
        (validation_work, validation_path),
        (all_work, all_path),
        (unmatched_work, unmatched_path),
        (failures_work, failures_path),
    ):
        temporary.replace(final)

    group_stats = {}
    for group in sorted(group_counts):
        group_stats[group] = {"count": group_counts[group]}
        for metric_name, total in group_metric_sums[group].items():
            group_stats[group][f"tokenization_{metric_name}"] = total / group_counts[group]
    report = PreparationReport(
        scale=scale,
        paired=paired,
        train_written=train_written,
        validation_written=validation_written,
        rectangles_dropped=rectangles_dropped,
        unmatched=unmatched,
        failures=failures,
        pairing_failures=pairing_failures,
        preprocessing_failures=preprocessing_failures,
        simple_rectangles=simple_rectangles,
        complex_samples=complex_samples,
        all_jsonl=str(all_path),
        train_jsonl=str(train_path),
        validation_jsonl=str(validation_path),
        unmatched_jsonl=str(unmatched_path),
        failures_jsonl=str(failures_path),
        group_stats=group_stats,
        split_stats={group: dict(values) for group, values in split_stats.items()},
    )
    audit_path = output_dir / "audits" / "preparation.json"
    audit_work = audit_path.with_suffix(audit_path.suffix + ".tmp")
    audit_work.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_work.replace(audit_path)
    return report


def select_complex_subset(
    input_jsonl: Path | str,
    output_jsonl: Path | str,
    count: int = 512,
    seed: int = 20260713,
) -> int:
    if count <= 0:
        raise ValueError("count must be positive")
    selected: list[tuple[int, str, dict]] = []
    with Path(input_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("difficulty") != "complex":
                continue
            osm_id = str(row.get("osm_id", row.get("raw_fid", "")))
            score = int.from_bytes(hashlib.sha256(f"{seed}:diagnostic:{osm_id}".encode("utf-8")).digest()[:8], "big")
            item = (-score, osm_id, row)
            if len(selected) < count:
                heapq.heappush(selected, item)
            elif item > selected[0]:
                heapq.heapreplace(selected, item)
    ordered = sorted(selected, key=lambda item: (-item[0], item[1]))
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for _, _, row in ordered:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(output_jsonl)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return len(ordered)


def export_unmatched_source_jsonl(
    raw_path: Path | str,
    ids_jsonl: Path | str,
    output_jsonl: Path | str,
    scale: int,
) -> int:
    projected_crs_from_shapefile(raw_path)
    requested: list[tuple[int, str]] = []
    with Path(ids_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                requested.append((int(row["raw_fid"]), str(row.get("osm_id", row["raw_fid"]))))
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    written = 0
    try:
        with shapefile.Reader(str(raw_path)) as reader, temporary.open("w", encoding="utf-8") as output:
            for raw_fid, osm_id in requested:
                if raw_fid < 0 or raw_fid >= len(reader):
                    continue
                polygon = _polygon_from_shape_record(reader.shapeRecord(raw_fid))
                if polygon is None:
                    continue
                source_tokens, frame = make_source_tokens(polygon, scale)
                output.write(json.dumps({
                    "sample_id": f"unmatched:{raw_fid}:{scale}",
                    "raw_fid": raw_fid,
                    "osm_id": osm_id,
                    "scale": scale,
                    "source_tokens": source_tokens,
                    "source_frame": asdict(frame),
                    "source_vertex_count": _vertex_count(polygon),
                }, ensure_ascii=False) + "\n")
                written += 1
        temporary.replace(output_jsonl)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return written
