from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from .bpe import build_bpe_jsonl_from_raw_jsonl, select_bpe_training_subset, train_bpe_from_jsonl_corpus
from .config import TOKENIZATION_VERSION
from .evaluation import audit_bpe_jsonl
from .evaluation import evaluate_prediction_jsonl
from .infer import run_inference
from .preparation import build_filtered_raw_jsonl, export_unmatched_source_jsonl, project_shapefile, select_complex_subset
from .train import train_from_config


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    scale: int

    @property
    def scale_dir(self) -> Path:
        return self.root / f"scale{self.scale}"

    @property
    def train_atomic(self) -> Path:
        return self.scale_dir / "train" / "atomic.jsonl"

    @property
    def train_bpe(self) -> Path:
        return self.scale_dir / "train" / "bpe.jsonl"

    @property
    def validation_atomic(self) -> Path:
        return self.scale_dir / "validation" / "atomic.jsonl"

    @property
    def validation_bpe(self) -> Path:
        return self.scale_dir / "validation" / "bpe.jsonl"

    @property
    def new_york_atomic(self) -> Path:
        return self.scale_dir / "test_new_york" / "atomic.jsonl"

    @property
    def new_york_bpe(self) -> Path:
        return self.scale_dir / "test_new_york" / "bpe.jsonl"

    @property
    def unmatched_ids(self) -> Path:
        return self.scale_dir / "unmatched" / "ids.jsonl"

    @property
    def unmatched_atomic(self) -> Path:
        return self.scale_dir / "unmatched" / "source_atomic.jsonl"

    @property
    def diagnostic_atomic(self) -> Path:
        return self.scale_dir / "diagnostic_512" / "atomic.jsonl"

    @property
    def diagnostic_bpe(self) -> Path:
        return self.scale_dir / "diagnostic_512" / "bpe.jsonl"

    @property
    def audits_dir(self) -> Path:
        return self.scale_dir / "audits"

    @property
    def vocab(self) -> Path:
        return self.scale_dir / "vocab" / "bpe_vocab.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shapefile_family(path: Path) -> list[Path]:
    required = [path.with_suffix(extension) for extension in (".shp", ".shx", ".dbf", ".prj")]
    missing = [str(candidate) for candidate in required if not candidate.exists()]
    if missing:
        raise FileNotFoundError("Incomplete shapefile family:\n" + "\n".join(missing))
    optional = [path.with_suffix(".cpg")]
    return required + [candidate for candidate in optional if candidate.exists()]


def _file_details(paths: list[Path]) -> dict[str, dict[str, int | str]]:
    return {str(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in paths}


def _artifact_details(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            if path.suffix in {".jsonl", ".txt"}:
                lines += chunk.count(b"\n")
    details: dict[str, int | str] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    if path.suffix in {".jsonl", ".txt"}:
        details["lines"] = lines
    return details


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_split_identity(train_path: Path, validation_path: Path) -> None:
    train_ids: set[str] = set()
    with train_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise RuntimeError(f"Non-train row found in {train_path}: {row.get('sample_id')}")
            train_ids.add(str(row["osm_id"]))
    with validation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "validation":
                raise RuntimeError(f"Non-validation row found in {validation_path}: {row.get('sample_id')}")
            if str(row["osm_id"]) in train_ids:
                raise RuntimeError(f"Train/validation leakage for osm_id {row['osm_id']}")


def _validate_scale_outputs(
    layout: DatasetLayout,
    la_report,
    bpe_reports: dict,
    audits: dict,
    selected: int,
    rectangle_keep_fraction: float,
    expected_unmatched: int | None,
    ny_report=None,
) -> None:
    for name, report in bpe_reports.items():
        if report.skipped:
            raise RuntimeError(f"BPE encoding skipped {report.skipped} rows in {name}")
    expected_rows = {
        "train": la_report.train_written,
        "validation": la_report.validation_written,
        "diagnostic": selected,
    }
    for name, expected in expected_rows.items():
        actual = bpe_reports[name].written
        if actual != expected:
            raise RuntimeError(f"{name} BPE rows {actual} != atomic rows {expected}")
    for name, audit in audits.items():
        failure_count = sum(int(audit[key]) for key in ("roundtrip_failures", "invalid_token_count", "row_alignment_failures"))
        if failure_count:
            raise RuntimeError(f"BPE audit failed for {name}: {audit}")
        if audit["rows"] != bpe_reports[name].written:
            raise RuntimeError(f"{name} audit rows {audit['rows']} != BPE rows {bpe_reports[name].written}")
    pairing_failures = getattr(la_report, "pairing_failures", getattr(la_report, "failures", 0))
    if pairing_failures:
        raise RuntimeError(f"Los Angeles preparation had {pairing_failures} pairing failures")
    preprocessing_failures = getattr(la_report, "preprocessing_failures", 0)
    if preprocessing_failures:
        raise RuntimeError(f"Los Angeles preparation had {preprocessing_failures} preprocessing failures")
    if ny_report is not None:
        ny_pairing_failures = getattr(ny_report, "pairing_failures", getattr(ny_report, "failures", 0))
        if ny_pairing_failures:
            raise RuntimeError(f"New York preparation had {ny_pairing_failures} pairing failures")
        ny_preprocessing_failures = getattr(ny_report, "preprocessing_failures", 0)
        if ny_preprocessing_failures:
            raise RuntimeError(f"New York preparation had {ny_preprocessing_failures} preprocessing failures")
        if bpe_reports["new_york"].written != ny_report.paired:
            raise RuntimeError(
                f"new_york BPE rows {bpe_reports['new_york'].written} != atomic rows {ny_report.paired}"
            )
    _validate_split_identity(layout.train_atomic, layout.validation_atomic)
    before = la_report.split_stats.get("train_before_filter", {}).get("simple_rectangle", 0)
    after = la_report.split_stats.get("train_after_filter", {}).get("simple_rectangle", 0)
    if before >= 100 and abs((after / before) - rectangle_keep_fraction) > 0.02:
        raise RuntimeError(f"Simple rectangle retention {after / before:.4f} differs from {rectangle_keep_fraction:.4f}")
    complex_rows = la_report.split_stats.get("train_after_filter", {}).get("complex", 0)
    if selected != min(512, complex_rows):
        raise RuntimeError(f"Diagnostic row count {selected} does not match available complex rows {complex_rows}")
    if expected_unmatched is not None and la_report.unmatched != expected_unmatched:
        raise RuntimeError(f"Unmatched count {la_report.unmatched} != expected {expected_unmatched} for scale {layout.scale}")


def prepare_scale(
    la_raw: Path,
    la_target: Path,
    ny_raw: Path,
    ny_target: Path,
    layout: DatasetLayout,
    *,
    seed: int = 20260713,
    rectangle_keep_fraction: float = 0.1,
    vocab_size: int = 6000,
    bpe_training_rows: int = 10_000,
    expected_unmatched: int | None = None,
) -> dict:
    layout.scale_dir.mkdir(parents=True, exist_ok=True)
    la_report = build_filtered_raw_jsonl(
        la_raw,
        la_target,
        layout.scale_dir,
        layout.scale,
        seed=seed,
        train_fraction=0.8,
        rectangle_keep_fraction=rectangle_keep_fraction,
        write_all=False,
    )
    unmatched_written = export_unmatched_source_jsonl(
        la_raw, layout.unmatched_ids, layout.unmatched_atomic, layout.scale
    )

    layout.new_york_atomic.parent.mkdir(parents=True, exist_ok=True)
    ny_failures_path = layout.audits_dir / "new_york_preparation_failures.jsonl"
    ny_unmatched_path = layout.audits_dir / "new_york_unmatched_ids.jsonl"
    with tempfile.TemporaryDirectory(prefix=f"ny_scale{layout.scale}_", dir=layout.root) as temp_dir:
        ny_report = build_filtered_raw_jsonl(
            ny_raw,
            ny_target,
            Path(temp_dir),
            layout.scale,
            seed=seed,
            train_fraction=0.8,
            rectangle_keep_fraction=1.0,
            sample_prefix="ny",
            split_override="test_new_york",
            write_split_files=False,
        )
        shutil.copy2(ny_report.all_jsonl, layout.new_york_atomic)
        shutil.copy2(ny_report.failures_jsonl, ny_failures_path)
        shutil.copy2(ny_report.unmatched_jsonl, ny_unmatched_path)
        ny_report = replace(
            ny_report,
            all_jsonl=str(layout.new_york_atomic),
            train_jsonl="",
            validation_jsonl="",
            unmatched_jsonl=str(ny_unmatched_path),
            failures_jsonl=str(ny_failures_path),
        )
    _write_json(layout.audits_dir / "new_york_preparation.json", asdict(ny_report))

    all_dir = layout.scale_dir / "all"
    if all_dir.exists():
        shutil.rmtree(all_dir)

    layout.vocab.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"bpe_scale{layout.scale}_", dir=layout.root) as temp_dir:
        bpe_training_sample = Path(temp_dir) / "training_sample.jsonl"
        sampled_rows = select_bpe_training_subset(layout.train_atomic, bpe_training_sample, count=bpe_training_rows, seed=seed)
        vocab_report = train_bpe_from_jsonl_corpus(
            bpe_training_sample,
            layout.vocab,
            vocab_size=vocab_size,
            scale=layout.scale,
        )
    bpe_reports = {
        "train": build_bpe_jsonl_from_raw_jsonl(layout.train_atomic, layout.vocab, layout.train_bpe),
        "validation": build_bpe_jsonl_from_raw_jsonl(layout.validation_atomic, layout.vocab, layout.validation_bpe),
        "new_york": build_bpe_jsonl_from_raw_jsonl(layout.new_york_atomic, layout.vocab, layout.new_york_bpe),
    }
    selected = select_complex_subset(layout.train_atomic, layout.diagnostic_atomic, count=512, seed=seed)
    diagnostic_report = build_bpe_jsonl_from_raw_jsonl(layout.diagnostic_atomic, layout.vocab, layout.diagnostic_bpe)

    audits = {
        "train": audit_bpe_jsonl(layout.train_atomic, layout.train_bpe, layout.vocab),
        "validation": audit_bpe_jsonl(layout.validation_atomic, layout.validation_bpe, layout.vocab),
        "new_york": audit_bpe_jsonl(layout.new_york_atomic, layout.new_york_bpe, layout.vocab),
        "diagnostic": audit_bpe_jsonl(layout.diagnostic_atomic, layout.diagnostic_bpe, layout.vocab),
    }
    _validate_scale_outputs(
        layout,
        la_report,
        {**bpe_reports, "diagnostic": diagnostic_report},
        audits,
        selected,
        rectangle_keep_fraction,
        expected_unmatched,
        ny_report=ny_report,
    )
    for name, audit in audits.items():
        _write_json(layout.audits_dir / f"{name}_bpe.json", audit)
    _write_json(layout.audits_dir / "tokenization_metrics.json", la_report.group_stats)
    _write_json(layout.audits_dir / "unmatched.json", {
        "requested": la_report.unmatched,
        "source_atomic_written": unmatched_written,
        "unmatched_export_failures": la_report.unmatched - unmatched_written,
        "preprocessing_failures": la_report.preprocessing_failures,
    })

    output_paths = [
        layout.train_atomic, layout.train_bpe, layout.validation_atomic, layout.validation_bpe,
        layout.new_york_atomic, layout.new_york_bpe, layout.unmatched_ids, layout.unmatched_atomic,
        layout.diagnostic_atomic, layout.diagnostic_bpe, layout.vocab,
        layout.audits_dir / "preparation.json",
        layout.audits_dir / "preparation_failures.jsonl",
        layout.audits_dir / "new_york_preparation.json",
        ny_failures_path,
        ny_unmatched_path,
        layout.audits_dir / "train_bpe.json",
        layout.audits_dir / "validation_bpe.json",
        layout.audits_dir / "new_york_bpe.json",
        layout.audits_dir / "diagnostic_bpe.json",
        layout.audits_dir / "tokenization_metrics.json",
        layout.audits_dir / "unmatched.json",
    ]
    config = {
        "tokenization_version": TOKENIZATION_VERSION,
        "seed": seed,
        "rectangle_keep_fraction": rectangle_keep_fraction,
        "vocab_size": vocab_size,
        "bpe_training_rows": bpe_training_rows,
        "expected_unmatched": expected_unmatched,
    }
    input_files = []
    for shapefile_path in (la_raw, la_target, ny_raw, ny_target):
        input_files.extend(_shapefile_family(shapefile_path))
    manifest = {
        "scale": layout.scale,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "bpe_sampled_rows": sampled_rows,
        "diagnostic_count": selected,
        "inputs": _file_details(input_files),
        "outputs": {path.relative_to(layout.root).as_posix(): _artifact_details(path) for path in output_paths},
        "la_preparation": asdict(la_report),
        "ny_preparation": asdict(ny_report),
        "vocab": asdict(vocab_report),
        "bpe": {name: asdict(report) for name, report in bpe_reports.items()},
        "diagnostic_bpe": asdict(diagnostic_report),
    }
    _write_json(layout.audits_dir / "manifest.json", manifest)
    return manifest


def _manifest_is_current(path: Path, expected_config: dict | None = None) -> bool:
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if expected_config is not None and manifest.get("config") != expected_config:
            return False
        for filename, details in manifest.get("inputs", {}).items():
            source = Path(filename)
            if not source.exists() or _sha256(source) != details["sha256"]:
                return False
        for relative, details in manifest.get("outputs", {}).items():
            output = path.parents[2] / relative
            if not output.exists() or _sha256(output) != details["sha256"]:
                return False
        return True
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _project_atomic(source: Path, output: Path, epsg: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="projection_", dir=output.parent) as temp_dir:
        temporary = Path(temp_dir) / output.name
        project_shapefile(source, temporary, epsg)
        for temporary_sidecar in temporary.parent.glob(f"{temporary.stem}.*"):
            destination = output.parent / temporary_sidecar.name
            temporary_sidecar.replace(destination)


def prepare_all(
    workspace: Path,
    output_root: Path,
    *,
    seed: int = 20260713,
    force: bool = False,
    vocab_size: int = 6000,
    bpe_training_rows: int = 10_000,
    rectangle_keep_fraction: float = 0.1,
    min_free_gb: float = 20.0,
    enforce_known_counts: bool = True,
) -> dict:
    workspace = workspace.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output_root).free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(f"At least {min_free_gb:.1f} GB free space is required; found {free_gb:.1f} GB")

    sources = {
        "la_raw": workspace / "洛杉矶建筑数据" / "los_angeles_bld_osm.shp",
        "ny_raw": workspace / "纽约建筑数据" / "ny_city_bld_osm.shp",
    }
    for scale in (5000, 10000):
        sources[f"la_target_{scale}"] = workspace / "洛杉矶建筑处理后" / f"los_angeles__bld_{scale}os_SimplifyBuild.shp"
        sources[f"ny_target_{scale}"] = workspace / "纽约建筑处理后" / f"ny_city_bld_{scale}os_SimplifyBuild.shp"
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input shapefiles:\n" + "\n".join(missing))

    projected = {
        "la_raw": output_root / "projected" / "los_angeles" / "raw.shp",
        "ny_raw": output_root / "projected" / "new_york" / "raw.shp",
    }
    for scale in (5000, 10000):
        projected[f"la_target_{scale}"] = output_root / "projected" / "los_angeles" / f"target_{scale}.shp"
        projected[f"ny_target_{scale}"] = output_root / "projected" / "new_york" / f"target_{scale}.shp"

    projection_audits = output_root / "projected" / "audits"
    projection_audits.mkdir(parents=True, exist_ok=True)
    projection_report = {}
    for name, source in sources.items():
        output = projected[name]
        epsg = 32611 if name.startswith("la_") else 32618
        audit_path = projection_audits / f"{name}.json"
        source_details = _file_details(_shapefile_family(source))
        current = False
        if not force and audit_path.exists() and output.exists():
            try:
                previous = json.loads(audit_path.read_text(encoding="utf-8"))
                current = previous.get("source_family") == source_details and previous.get("epsg") == epsg
                if current:
                    current = previous.get("output_family") == _file_details(_shapefile_family(output))
            except (OSError, FileNotFoundError, json.JSONDecodeError):
                current = False
        if not current:
            _project_atomic(source, output, epsg)
        details = {
            "source": str(source),
            "source_family": source_details,
            "epsg": epsg,
            "output": str(output),
            "output_family": _file_details(_shapefile_family(output)),
        }
        _write_json(audit_path, details)
        projection_report[name] = details

    scale_reports = {}
    expected_unmatched_counts = {5000: 107, 10000: 38_465}
    for scale in (5000, 10000):
        layout = DatasetLayout(output_root, scale)
        manifest_path = layout.audits_dir / "manifest.json"
        expected_unmatched = expected_unmatched_counts[scale] if enforce_known_counts else None
        scale_config = {
            "tokenization_version": TOKENIZATION_VERSION,
            "seed": seed,
            "rectangle_keep_fraction": rectangle_keep_fraction,
            "vocab_size": vocab_size,
            "bpe_training_rows": bpe_training_rows,
            "expected_unmatched": expected_unmatched,
        }
        if not force and _manifest_is_current(manifest_path, scale_config):
            scale_reports[str(scale)] = json.loads(manifest_path.read_text(encoding="utf-8"))
            continue
        scale_reports[str(scale)] = prepare_scale(
            projected["la_raw"],
            projected[f"la_target_{scale}"],
            projected["ny_raw"],
            projected[f"ny_target_{scale}"],
            layout,
            seed=seed,
            rectangle_keep_fraction=rectangle_keep_fraction,
            vocab_size=vocab_size,
            bpe_training_rows=bpe_training_rows,
            expected_unmatched=expected_unmatched,
        )

    report = {"workspace": str(workspace), "output_root": str(output_root), "free_gb_before": free_gb, "projection": projection_report, "scales": scale_reports}
    _write_json(output_root / "manifest.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified building simplification workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-all")
    prepare.add_argument("--workspace", type=Path, default=Path.cwd())
    prepare.add_argument("--output-root", type=Path, default=Path("datasets"))
    prepare.add_argument("--seed", type=int, default=20260713)
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--vocab-size", type=int, default=6000)
    prepare.add_argument("--bpe-training-rows", type=int, default=10_000)
    prepare.add_argument("--min-free-gb", type=float, default=20.0)

    for command in ("train-baseline", "train-diagnostic"):
        train = subparsers.add_parser(command)
        train.add_argument("--datasets-root", type=Path, default=Path("datasets"))
        train.add_argument("--scale", type=int, choices=[5000, 10000], required=True)
        train.add_argument("--output-dir", type=Path, required=True)
        train.add_argument("--device", default=None)
        train.add_argument("--epochs", type=int, default=30 if command == "train-baseline" else 100)
        train.add_argument("--batch-size", type=int, default=32)
        train.add_argument("--eval-batch-size", type=int, default=32)
        train.add_argument("--seed", type=int, default=20260713)
        train.add_argument("--d-model", type=int, default=256)
        train.add_argument("--nhead", type=int, default=8)
        train.add_argument("--num-layers", type=int, default=4)
        train.add_argument("--dim-feedforward", type=int, default=1024)
        train.add_argument("--dropout", type=float, default=0.1 if command == "train-baseline" else 0.0)
        train.add_argument("--pre-ln", action="store_true")
        train.add_argument("--max-steps", type=int, default=None if command == "train-baseline" else 3000)
        train.add_argument("--no-amp", dest="use_amp", action="store_false", help="Disable automatic mixed precision")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--vocab", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    unmatched = subparsers.add_parser("infer-unmatched")
    unmatched.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    unmatched.add_argument("--scale", type=int, choices=[5000, 10000], required=True)
    unmatched.add_argument("--checkpoint", type=Path, required=True)
    unmatched.add_argument("--output", type=Path, required=True)
    unmatched.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-all":
        root = args.output_root if args.output_root.is_absolute() else args.workspace / args.output_root
        report = prepare_all(
            args.workspace,
            root,
            seed=args.seed,
            force=args.force,
            vocab_size=args.vocab_size,
            bpe_training_rows=args.bpe_training_rows,
            min_free_gb=args.min_free_gb,
        )
        print(json.dumps({"output_root": report["output_root"], "scales": sorted(report["scales"])}, ensure_ascii=False, indent=2))
        return
    if args.command in {"train-baseline", "train-diagnostic"}:
        layout = DatasetLayout(args.datasets_root.resolve(), args.scale)
        diagnostic = args.command == "train-diagnostic"
        dataset = layout.diagnostic_bpe if diagnostic else layout.train_bpe
        validation = layout.diagnostic_bpe if diagnostic else layout.validation_bpe
        checkpoint = train_from_config(
            dataset,
            args.output_dir,
            vocab_path=layout.vocab,
            test_dataset_path=validation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            device=args.device,
            max_steps=args.max_steps,
            seed=args.seed,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            pre_ln=args.pre_ln,
            weight_decay=0.0 if diagnostic else 0.01,
            scheduled_sampling_max=0.0 if diagnostic else 0.25,
            use_amp=args.use_amp,
        )
        print(checkpoint)
        return
    if args.command == "evaluate":
        report = evaluate_prediction_jsonl(args.predictions, args.vocab, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    layout = DatasetLayout(args.datasets_root.resolve(), args.scale)
    raw_path = args.datasets_root.resolve() / "projected" / "los_angeles" / "raw.shp"
    report = run_inference(
        raw_path,
        args.checkpoint,
        args.output,
        args.scale,
        device=args.device,
        vocab_path=layout.vocab,
        fid_filter_path=layout.unmatched_ids,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
