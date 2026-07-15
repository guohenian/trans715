from pathlib import Path

import pytest
import shapefile
from shapely.geometry import Polygon

from building_simplify.preparation import (
    classify_pair,
    build_filtered_raw_jsonl,
    export_unmatched_source_jsonl,
    is_simple_rectangle,
    keep_training_sample,
    project_shapefile,
    projected_crs_from_shapefile,
    select_complex_subset,
    split_for_osm_id,
)


def test_simple_rectangle_requires_both_raw_and_target_to_be_rectangles():
    rectangle = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])
    complex_raw = Polygon([(0, 0), (5, 0), (10, 0), (10, 4), (0, 4)])

    assert is_simple_rectangle(rectangle)
    assert not is_simple_rectangle(complex_raw)
    assert classify_pair(rectangle, rectangle) == "simple_rectangle"
    assert classify_pair(complex_raw, rectangle) == "complex"


def test_split_and_rectangle_sampling_are_stable():
    assert split_for_osm_id("123", seed=20260713, train_fraction=0.8) == split_for_osm_id(
        "123", seed=20260713, train_fraction=0.8
    )
    decisions = [keep_training_sample(str(i), "simple_rectangle", 20260713, 0.1) for i in range(1000)]
    assert 70 <= sum(decisions) <= 130
    assert keep_training_sample("123", "complex", 20260713, 0.1)


def test_projected_crs_rejects_geographic_shapefile(tmp_path: Path):
    shp_path = tmp_path / "buildings.shp"
    shp_path.touch()
    shp_path.with_suffix(".prj").write_text(
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="projected.*metre"):
        projected_crs_from_shapefile(shp_path)


def test_project_shapefile_converts_wgs84_to_metre_crs(tmp_path: Path):
    source = tmp_path / "wgs84.shp"
    polygon = Polygon([(-118.25, 34.05), (-118.2499, 34.05), (-118.2499, 34.0501), (-118.25, 34.0501)])
    _write_polygon_shapefile(source, [polygon], [("a",)], [("osm_id", "C", 20)])
    source.with_suffix(".prj").write_text(
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]',
        encoding="utf-8",
    )
    output = tmp_path / "utm.shp"

    project_shapefile(source, output, 32611)

    assert projected_crs_from_shapefile(output).to_epsg() == 32611
    with shapefile.Reader(str(output)) as reader:
        assert len(reader) == 1
        x, y = reader.shape(0).points[0]
        assert 300000 < x < 500000
        assert 3_000_000 < y < 4_000_000


def _write_polygon_shapefile(path: Path, polygons: list[Polygon], records: list[tuple], fields: list[tuple]):
    writer = shapefile.Writer(str(path), shapeType=shapefile.POLYGON)
    for field in fields:
        writer.field(*field)
    for polygon, record in zip(polygons, records):
        writer.poly([list(polygon.exterior.coords)])
        writer.record(*record)
    writer.close()
    path.with_suffix(".prj").write_text(
        'PROJCS["WGS 84 / UTM zone 11N",GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
        'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-117],'
        'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
        'PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]',
        encoding="utf-8",
    )


def test_filtered_builder_writes_split_metadata_and_unmatched_rows(tmp_path: Path):
    raw = tmp_path / "raw.shp"
    target = tmp_path / "target.shp"
    rectangle = Polygon([(500000, 3760000), (500010, 3760000), (500010, 3760004), (500000, 3760004)])
    complex_polygon = Polygon([(500020, 3760000), (500030, 3760000), (500030, 3760004), (500025, 3760006), (500020, 3760004)])
    _write_polygon_shapefile(raw, [rectangle, complex_polygon], [("r",), ("c",)], [("osm_id", "C", 20)])
    _write_polygon_shapefile(
        target,
        [rectangle],
        [("r", 0)],
        [("osm_id", "C", 20), ("InBld_FID", "N", 12)],
    )

    report = build_filtered_raw_jsonl(
        raw,
        target,
        output_dir=tmp_path / "prepared",
        scale=5000,
        seed=20260713,
        train_fraction=0.8,
        rectangle_keep_fraction=1.0,
    )

    assert report.paired == 1
    assert report.unmatched == 1
    assert report.simple_rectangles == 1
    assert report.complex_samples == 0
    assert report.group_stats["overall"]["count"] == 1
    assert report.group_stats["simple_rectangle"]["tokenization_outline_iou"] > 0.98
    retained = sum(report.split_stats.get(group, {}).get(key, 0) for group in ("train_after_filter", "validation") for key in ("simple_rectangle", "complex"))
    assert retained == 1
    assert len(Path(report.all_jsonl).read_text(encoding="utf-8").splitlines()) == 1
    rows = []
    for path in (Path(report.train_jsonl), Path(report.validation_jsonl)):
        if path.exists():
            rows.extend(__import__("json").loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    assert rows[0]["difficulty"] == "simple_rectangle"
    assert rows[0]["split"] in {"train", "validation"}
    assert rows[0]["tokenization_metrics"]["outline_iou"] > 0.98
    unmatched = Path(report.unmatched_jsonl).read_text(encoding="utf-8")
    assert '"raw_fid": 1' in unmatched


def test_filtered_builder_rejects_different_projected_crs(tmp_path: Path):
    raw = tmp_path / "raw.shp"
    target = tmp_path / "target.shp"
    rectangle = Polygon([(500000, 3760000), (500010, 3760000), (500010, 3760004), (500000, 3760004)])
    _write_polygon_shapefile(raw, [rectangle], [("r",)], [("osm_id", "C", 20)])
    _write_polygon_shapefile(target, [rectangle], [("r", 0)], [("osm_id", "C", 20), ("InBld_FID", "N", 12)])
    target.with_suffix(".prj").write_text(
        target.with_suffix(".prj").read_text(encoding="utf-8").replace("zone 11N", "zone 18N").replace("-117", "-75"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="same projected CRS"):
        build_filtered_raw_jsonl(raw, target, tmp_path / "prepared", scale=5000)


def test_preprocessing_failure_is_not_reported_as_arcgis_unmatched(tmp_path: Path):
    raw = tmp_path / "raw.shp"
    target = tmp_path / "target.shp"
    rectangle = Polygon([(500000, 3760000), (500010, 3760000), (500010, 3760004), (500000, 3760004)])
    _write_polygon_shapefile(raw, [rectangle], [("raw",)], [("osm_id", "C", 20)])
    _write_polygon_shapefile(target, [rectangle], [("different", 0)], [("osm_id", "C", 20), ("InBld_FID", "N", 12)])

    report = build_filtered_raw_jsonl(raw, target, tmp_path / "prepared", scale=5000)

    assert report.failures == 1
    assert report.pairing_failures == 1
    assert report.preprocessing_failures == 0
    assert report.unmatched == 0
    assert Path(report.unmatched_jsonl).read_text(encoding="utf-8") == ""


def test_tokenization_failure_is_audited_without_becoming_unmatched(tmp_path: Path, monkeypatch):
    raw = tmp_path / "raw.shp"
    target = tmp_path / "target.shp"
    rectangle = Polygon([(500000, 3760000), (500010, 3760000), (500010, 3760004), (500000, 3760004)])
    _write_polygon_shapefile(raw, [rectangle], [("r",)], [("osm_id", "C", 20)])
    _write_polygon_shapefile(target, [rectangle], [("r", 0)], [("osm_id", "C", 20), ("InBld_FID", "N", 12)])

    from building_simplify import preparation

    monkeypatch.setattr(preparation, "make_source_tokens", lambda *_: (_ for _ in ()).throw(ValueError("bad tokenization")))
    report = build_filtered_raw_jsonl(raw, target, tmp_path / "prepared", scale=5000)

    assert report.failures == 1
    assert report.pairing_failures == 0
    assert report.preprocessing_failures == 1
    assert report.paired == 0
    assert report.unmatched == 0


def test_complex_subset_is_deterministic_and_excludes_rectangles(tmp_path: Path):
    source = tmp_path / "train.jsonl"
    rows = [
        {"osm_id": str(i), "difficulty": "complex" if i % 2 else "simple_rectangle", "value": i}
        for i in range(20)
    ]
    source.write_text("".join(__import__("json").dumps(row) + "\n" for row in rows), encoding="utf-8")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    select_complex_subset(source, first, count=5, seed=20260713)
    select_complex_subset(source, second, count=5, seed=20260713)

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    selected = [__import__("json").loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert len(selected) == 5
    assert all(row["difficulty"] == "complex" for row in selected)


def test_complex_subset_preserves_existing_output_on_write_failure(tmp_path: Path, monkeypatch):
    source = tmp_path / "train.jsonl"
    rows = [{"osm_id": str(i), "difficulty": "complex"} for i in range(3)]
    source.write_text("".join(__import__("json").dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "diagnostic.jsonl"
    output.write_text("previous complete output\n", encoding="utf-8")

    from building_simplify import preparation

    real_dumps = preparation.json.dumps
    calls = 0

    def fail_on_second_row(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated serialization failure")
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(preparation.json, "dumps", fail_on_second_row)

    with pytest.raises(RuntimeError, match="simulated serialization failure"):
        select_complex_subset(source, output, count=3)

    assert output.read_text(encoding="utf-8") == "previous complete output\n"
    assert not output.with_suffix(".jsonl.tmp").exists()


def test_unmatched_source_export_contains_no_target_tokens(tmp_path: Path):
    raw = tmp_path / "raw.shp"
    rectangle = Polygon([(500000, 3760000), (500010, 3760000), (500010, 3760004), (500000, 3760004)])
    _write_polygon_shapefile(raw, [rectangle], [("r",)], [("osm_id", "C", 20)])
    ids = tmp_path / "ids.jsonl"
    ids.write_text(__import__("json").dumps({"raw_fid": 0, "osm_id": "r", "scale": 5000}) + "\n", encoding="utf-8")
    output = tmp_path / "source_atomic.jsonl"

    written = export_unmatched_source_jsonl(raw, ids, output, 5000)

    row = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert written == 1
    assert row["source_tokens"][0] == 725
    assert "target_tokens" not in row


def test_unmatched_source_export_preserves_existing_output_on_failure(tmp_path: Path, monkeypatch):
    raw = tmp_path / "raw.shp"
    rectangle = Polygon([(500000, 3760000), (500010, 3760000), (500010, 3760004), (500000, 3760004)])
    shifted = Polygon([(500020, 3760000), (500030, 3760000), (500030, 3760004), (500020, 3760004)])
    _write_polygon_shapefile(raw, [rectangle, shifted], [("r",), ("s",)], [("osm_id", "C", 20)])
    ids = tmp_path / "ids.jsonl"
    ids.write_text(
        __import__("json").dumps({"raw_fid": 0, "osm_id": "r", "scale": 5000}) + "\n"
        + __import__("json").dumps({"raw_fid": 1, "osm_id": "s", "scale": 5000}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "source_atomic.jsonl"
    output.write_text("previous complete output\n", encoding="utf-8")

    from building_simplify import preparation

    real_make_source_tokens = preparation.make_source_tokens
    calls = 0

    def fail_on_second_polygon(polygon, scale):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated encoding failure")
        return real_make_source_tokens(polygon, scale)

    monkeypatch.setattr(preparation, "make_source_tokens", fail_on_second_polygon)

    with pytest.raises(RuntimeError, match="simulated encoding failure"):
        export_unmatched_source_jsonl(raw, ids, output, 5000)

    assert output.read_text(encoding="utf-8") == "previous complete output\n"
    assert not output.with_suffix(".jsonl.tmp").exists()
