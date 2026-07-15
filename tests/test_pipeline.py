from pathlib import Path

import shapefile
from shapely.geometry import Polygon

from building_simplify.pipeline import (
    DatasetLayout,
    _manifest_is_current,
    _validate_scale_outputs,
    build_parser,
    prepare_all,
    prepare_scale,
)
from building_simplify.preparation import split_for_osm_id


def test_dataset_layout_uses_purpose_directories(tmp_path: Path):
    layout = DatasetLayout(tmp_path / "datasets", 5000)

    assert layout.train_atomic == tmp_path / "datasets" / "scale5000" / "train" / "atomic.jsonl"
    assert layout.validation_bpe == tmp_path / "datasets" / "scale5000" / "validation" / "bpe.jsonl"
    assert layout.new_york_atomic == tmp_path / "datasets" / "scale5000" / "test_new_york" / "atomic.jsonl"
    assert layout.unmatched_ids == tmp_path / "datasets" / "scale5000" / "unmatched" / "ids.jsonl"
    assert layout.diagnostic_bpe == tmp_path / "datasets" / "scale5000" / "diagnostic_512" / "bpe.jsonl"
    assert layout.vocab == tmp_path / "datasets" / "scale5000" / "vocab" / "bpe_vocab.txt"


def test_pipeline_exposes_only_the_five_public_workflows():
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {"prepare-all", "train-baseline", "train-diagnostic", "evaluate", "infer-unmatched"}


def _write_projected(path: Path, osm_ids: list[str], include_target_fid: bool):
    writer = shapefile.Writer(str(path), shapeType=shapefile.POLYGON)
    writer.field("osm_id", "C", 20)
    if include_target_fid:
        writer.field("InBld_FID", "N", 12)
    for index, osm_id in enumerate(osm_ids):
        x = 500000 + index * 20
        polygon = Polygon([(x, 3760000), (x + 10, 3760000), (x + 10, 3760004), (x, 3760004)])
        writer.poly([list(polygon.exterior.coords)])
        writer.record(osm_id, index) if include_target_fid else writer.record(osm_id)
    writer.close()
    path.with_suffix(".prj").write_text(
        'PROJCS["WGS_1984_UTM_Zone_11N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
        'SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],'
        'UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
        'PARAMETER["False_Easting",500000],PARAMETER["False_Northing",0],'
        'PARAMETER["Central_Meridian",-117],PARAMETER["Scale_Factor",0.9996],'
        'PARAMETER["Latitude_Of_Origin",0],UNIT["Meter",1]]', encoding="utf-8")


def test_prepare_scale_exports_all_purpose_directories(tmp_path: Path):
    train_id = next(str(i) for i in range(100) if split_for_osm_id(str(i)) == "train")
    validation_id = next(str(i) for i in range(100) if split_for_osm_id(str(i)) == "validation")
    osm_ids = [train_id, validation_id, "extra"]
    la_raw, la_target = tmp_path / "la_raw.shp", tmp_path / "la_target.shp"
    ny_raw, ny_target = tmp_path / "ny_raw.shp", tmp_path / "ny_target.shp"
    _write_projected(la_raw, osm_ids, False)
    _write_projected(la_target, osm_ids[:2], True)
    _write_projected(ny_raw, osm_ids, False)
    _write_projected(ny_target, osm_ids, True)
    layout = DatasetLayout(tmp_path / "datasets", 5000)

    report = prepare_scale(la_raw, la_target, ny_raw, ny_target, layout, vocab_size=1502, rectangle_keep_fraction=1.0)

    for path in (layout.train_atomic, layout.train_bpe, layout.validation_atomic, layout.validation_bpe,
                 layout.new_york_atomic, layout.new_york_bpe, layout.unmatched_ids,
                 layout.unmatched_atomic, layout.diagnostic_atomic, layout.diagnostic_bpe, layout.vocab):
        assert path.exists(), path
    assert report["scale"] == 5000
    from building_simplify import config

    assert getattr(config, "TOKENIZATION_VERSION", None) == 2
    assert report["config"]["tokenization_version"] == 2
    assert (layout.audits_dir / "manifest.json").exists()
    assert (layout.audits_dir / "diagnostic_bpe.json").exists()
    assert (layout.audits_dir / "unmatched.json").exists()
    assert (layout.audits_dir / "new_york_preparation_failures.jsonl").exists()
    assert report["generated_at"]
    train_details = report["outputs"]["scale5000/train/atomic.jsonl"]
    assert train_details["lines"] == 1
    assert train_details["bytes"] == layout.train_atomic.stat().st_size
    assert len(train_details["sha256"]) == 64
    ny_row = __import__("json").loads(layout.new_york_atomic.read_text(encoding="utf-8").splitlines()[0])
    assert ny_row["sample_id"].startswith("ny:")
    assert ny_row["split"] == "test_new_york"


def _write_geographic(path: Path, osm_ids: list[str], include_target_fid: bool, lon: float, lat: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = shapefile.Writer(str(path), shapeType=shapefile.POLYGON)
    writer.field("osm_id", "C", 20)
    if include_target_fid:
        writer.field("InBld_FID", "N", 12)
    for index, osm_id in enumerate(osm_ids):
        x = lon + index * 0.001
        polygon = Polygon([(x, lat), (x + 0.0001, lat), (x + 0.0001, lat + 0.0001), (x, lat + 0.0001)])
        writer.poly([list(polygon.exterior.coords)])
        writer.record(osm_id, index) if include_target_fid else writer.record(osm_id)
    writer.close()
    path.with_suffix(".prj").write_text(
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]', encoding="utf-8")


def test_prepare_all_projects_both_cities_and_scales(tmp_path: Path):
    workspace = tmp_path / "workspace"
    train_id = next(str(i) for i in range(100) if split_for_osm_id(str(i)) == "train")
    validation_id = next(str(i) for i in range(100) if split_for_osm_id(str(i)) == "validation")
    osm_ids = [train_id, validation_id]
    _write_geographic(workspace / "洛杉矶建筑数据" / "los_angeles_bld_osm.shp", osm_ids, False, -118.25, 34.05)
    _write_geographic(workspace / "纽约建筑数据" / "ny_city_bld_osm.shp", osm_ids, False, -74.0, 40.7)
    for scale in (5000, 10000):
        _write_geographic(workspace / "洛杉矶建筑处理后" / f"los_angeles__bld_{scale}os_SimplifyBuild.shp", osm_ids, True, -118.25, 34.05)
        _write_geographic(workspace / "纽约建筑处理后" / f"ny_city_bld_{scale}os_SimplifyBuild.shp", osm_ids, True, -74.0, 40.7)

    report = prepare_all(
        workspace,
        workspace / "datasets",
        vocab_size=1502,
        rectangle_keep_fraction=1.0,
        enforce_known_counts=False,
    )

    assert set(report["scales"]) == {"5000", "10000"}
    assert (workspace / "datasets" / "projected" / "los_angeles" / "raw.shp").exists()
    assert (workspace / "datasets" / "projected" / "new_york" / "raw.shp").exists()
    assert DatasetLayout(workspace / "datasets", 10000).new_york_bpe.exists()


def test_prepare_all_enforces_known_unmatched_counts_by_default(tmp_path: Path):
    workspace = tmp_path / "workspace"
    train_id = next(str(i) for i in range(100) if split_for_osm_id(str(i)) == "train")
    validation_id = next(str(i) for i in range(100) if split_for_osm_id(str(i)) == "validation")
    osm_ids = [train_id, validation_id]
    _write_geographic(workspace / "洛杉矶建筑数据" / "los_angeles_bld_osm.shp", osm_ids, False, -118.25, 34.05)
    _write_geographic(workspace / "纽约建筑数据" / "ny_city_bld_osm.shp", osm_ids, False, -74.0, 40.7)
    for scale in (5000, 10000):
        _write_geographic(
            workspace / "洛杉矶建筑处理后" / f"los_angeles__bld_{scale}os_SimplifyBuild.shp",
            osm_ids,
            True,
            -118.25,
            34.05,
        )
        _write_geographic(
            workspace / "纽约建筑处理后" / f"ny_city_bld_{scale}os_SimplifyBuild.shp",
            osm_ids,
            True,
            -74.0,
            40.7,
        )

    import pytest

    with pytest.raises(RuntimeError, match="Unmatched count 0 != expected 107"):
        prepare_all(workspace, workspace / "datasets", vocab_size=1502, rectangle_keep_fraction=1.0)


def test_manifest_invalidates_when_sidecar_or_config_changes(tmp_path: Path):
    root = tmp_path / "datasets"
    manifest_path = root / "scale5000" / "audits" / "manifest.json"
    source = tmp_path / "source.dbf"
    output = root / "scale5000" / "train" / "atomic.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"dbf-v1")
    output.write_bytes(b"rows")
    import hashlib, json
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "config": {"seed": 1, "vocab_size": 1502},
        "inputs": {str(source): {"sha256": sha(source)}},
        "outputs": {"scale5000/train/atomic.jsonl": {"sha256": sha(output)}},
    }), encoding="utf-8")

    assert _manifest_is_current(manifest_path, {"seed": 1, "vocab_size": 1502})
    assert not _manifest_is_current(manifest_path, {"seed": 2, "vocab_size": 1502})
    source.write_bytes(b"dbf-v2")
    assert not _manifest_is_current(manifest_path, {"seed": 1, "vocab_size": 1502})


def test_scale_validation_rejects_bpe_row_count_mismatch(tmp_path: Path):
    from types import SimpleNamespace

    layout = DatasetLayout(tmp_path / "datasets", 5000)
    layout.train_atomic.parent.mkdir(parents=True, exist_ok=True)
    layout.validation_atomic.parent.mkdir(parents=True, exist_ok=True)
    layout.train_atomic.write_text('{"sample_id":"la:0:5000","osm_id":"train","split":"train"}\n', encoding="utf-8")
    layout.validation_atomic.write_text(
        '{"sample_id":"la:1:5000","osm_id":"validation","split":"validation"}\n', encoding="utf-8"
    )
    la_report = SimpleNamespace(
        train_written=1,
        validation_written=1,
        unmatched=0,
        failures=0,
        split_stats={"train_before_filter": {"complex": 1}, "train_after_filter": {"complex": 1}},
    )
    bpe_reports = {
        "train": SimpleNamespace(written=0, skipped=0),
        "validation": SimpleNamespace(written=1, skipped=0),
        "new_york": SimpleNamespace(written=1, skipped=0),
        "diagnostic": SimpleNamespace(written=1, skipped=0),
    }
    audits = {
        name: {"rows": 1, "roundtrip_failures": 0, "invalid_token_count": 0, "row_alignment_failures": 0}
        for name in ("train", "validation", "new_york")
    }

    import pytest

    with pytest.raises(RuntimeError, match="train BPE rows 0 != atomic rows 1"):
        _validate_scale_outputs(layout, la_report, bpe_reports, audits, 1, 0.1, None)

    bpe_reports["train"].written = 1
    ny_report = SimpleNamespace(paired=1, pairing_failures=1)
    with pytest.raises(RuntimeError, match="New York preparation had 1 pairing failures"):
        _validate_scale_outputs(layout, la_report, bpe_reports, audits, 1, 0.1, None, ny_report=ny_report)

    ny_report.pairing_failures = 0
    la_report.preprocessing_failures = 1
    with pytest.raises(RuntimeError, match="Los Angeles preparation had 1 preprocessing failures"):
        _validate_scale_outputs(layout, la_report, bpe_reports, audits, 1, 0.1, None, ny_report=ny_report)

    la_report.preprocessing_failures = 0
    ny_report.preprocessing_failures = 1
    with pytest.raises(RuntimeError, match="New York preparation had 1 preprocessing failures"):
        _validate_scale_outputs(layout, la_report, bpe_reports, audits, 1, 0.1, None, ny_report=ny_report)
