import pytest
import json
from pathlib import Path
import shapefile
from shapely.geometry import Polygon

from building_simplify.evaluation import categorize_failure, evaluate_unmatched_shapefile, unmatched_geometry_metrics


def test_unmatched_metrics_compare_prediction_to_raw_without_ground_truth():
    raw = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])
    prediction = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])

    metrics = unmatched_geometry_metrics(raw, prediction)

    assert metrics["valid_geometry"] == 1.0
    assert metrics["vertex_compression_ratio"] == pytest.approx(0.0)
    assert metrics["area_ratio"] == pytest.approx(1.0)
    assert metrics["perimeter_ratio"] == pytest.approx(1.0)
    assert metrics["minimum_rectangle_iou"] == pytest.approx(1.0)


def test_missing_predictions_count_as_invalid(tmp_path: Path):
    raw_path = tmp_path / "raw.shp"
    pred_path = tmp_path / "pred.shp"
    polygon = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])
    raw_writer = shapefile.Writer(str(raw_path), shapeType=shapefile.POLYGON)
    raw_writer.field("osm_id", "C", 20)
    for osm_id in ("a", "b"):
        raw_writer.poly([list(polygon.exterior.coords)])
        raw_writer.record(osm_id)
    raw_writer.close()
    pred_writer = shapefile.Writer(str(pred_path), shapeType=shapefile.POLYGON)
    pred_writer.field("raw_fid", "N", 12)
    pred_writer.poly([list(polygon.exterior.coords)])
    pred_writer.record(0)
    pred_writer.close()
    unmatched = tmp_path / "unmatched.jsonl"
    unmatched.write_text(json.dumps({"raw_fid": 0}) + "\n" + json.dumps({"raw_fid": 1}) + "\n", encoding="utf-8")

    report = evaluate_unmatched_shapefile(raw_path, pred_path, unmatched, tmp_path / "report.json")

    assert report["requested"] == 2
    assert report["missing_predictions"] == 1
    assert report["metrics"]["valid_geometry"] == pytest.approx(0.5)


def test_failure_categories_are_stable():
    assert categorize_failure("Predicted polygon is empty or has zero area") == "empty_geometry"
    assert categorize_failure("Predicted geometry is not polygonal: LineString") == "non_polygon"
    assert categorize_failure("No outer ring found in token sequence") == "decode_error"
