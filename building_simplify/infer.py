from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import shapefile
import torch
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .bpe import BPEVocabulary
from .config import (
    BPE_VOCAB_10000_PATH,
    BPE_VOCAB_5000_PATH,
    DEFAULT_MODEL_D_MODEL,
    DEFAULT_MODEL_DIM_FEEDFORWARD,
    DEFAULT_MODEL_DROPOUT,
    DEFAULT_MODEL_NHEAD,
    DEFAULT_MODEL_NUM_LAYERS,
    TOKEN_BOS,
    TOKEN_EOS,
)
from .evaluation import vertex_count
from .geometry import BuildingRecord, TokenFrame, _record_from_shape_record, decode_polygon, make_source_tokens
from .model import BuildingTransformer, generate_beam_search, generate_greedy


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_fid_filter(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    fids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            fids.add(int(row["raw_fid"]))
    return fids


def frame_from_dict(data: dict) -> TokenFrame:
    payload = dict(data)
    payload["inner_starts"] = tuple(tuple(pair) for pair in payload.get("inner_starts", ()))
    return TokenFrame(**payload)


def tokens_to_frame(data: dict) -> TokenFrame:
    return frame_from_dict(data)


def polygon_to_writer_parts(polygon: Polygon) -> list[list[tuple[float, float]]]:
    exterior = list(polygon.exterior.coords)
    if not exterior:
        raise ValueError("Predicted polygon has empty exterior")
    return [exterior, *[list(ring.coords) for ring in polygon.interiors if list(ring.coords)]]


def _largest_polygon(geometry: BaseGeometry) -> Polygon:
    if isinstance(geometry, Polygon):
        polygon = geometry
    elif isinstance(geometry, MultiPolygon):
        polygon = max(geometry.geoms, key=lambda item: item.area)
    elif isinstance(geometry, GeometryCollection):
        polygons = [item for item in geometry.geoms if isinstance(item, Polygon)]
        if not polygons:
            raise ValueError("Predicted geometry collection has no polygon")
        polygon = max(polygons, key=lambda item: item.area)
    else:
        raise ValueError(f"Predicted geometry is not polygonal: {geometry.geom_type}")
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
        if not isinstance(polygon, Polygon):
            return _largest_polygon(polygon)
    if polygon.is_empty or polygon.area <= 1e-12:
        raise ValueError("Predicted polygon is empty or has zero area")
    if len(polygon.exterior.coords) < 4:
        raise ValueError("Predicted polygon has too few exterior coordinates")
    return polygon


def target_compression_band(scale: int) -> tuple[float, float]:
    if scale == 5000:
        return 0.16, 0.20
    if scale == 10000:
        return 0.20, 0.23
    raise ValueError(f"Unsupported scale: {scale}")


def _distance_to_band(value: float, band: tuple[float, float]) -> float:
    low, high = band
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _geometry_score(
    model_score: float,
    raw_polygon: Polygon,
    predicted: Polygon,
    scale: int,
) -> tuple[float, dict[str, float]]:
    raw_vertices = vertex_count(raw_polygon)
    predicted_vertices = vertex_count(predicted)
    compression = (raw_vertices - predicted_vertices) / raw_vertices if raw_vertices else 0.0
    compression_penalty = _distance_to_band(compression, target_compression_band(scale))
    area_ratio = predicted.area / max(raw_polygon.area, 1e-12)
    area_penalty = abs(math.log(max(area_ratio, 1e-6)))
    vertex_floor_penalty = 1.0 if predicted_vertices < 4 else 0.0
    final_score = model_score - (6.0 * compression_penalty) - (0.8 * area_penalty) - vertex_floor_penalty
    return final_score, {
        "model_score": model_score,
        "compression": compression,
        "compression_penalty": compression_penalty,
        "area_ratio": area_ratio,
        "area_penalty": area_penalty,
        "predicted_vertices": float(predicted_vertices),
    }


def _default_vocab_path_for_scale(scale: int) -> Path:
    if scale == 10000:
        return BPE_VOCAB_10000_PATH
    return BPE_VOCAB_5000_PATH


def load_model(checkpoint: Path, device: str, vocab_path: Path | None = None, scale: int | None = None) -> tuple[BuildingTransformer, BPEVocabulary]:
    payload = torch.load(checkpoint, map_location=device)
    bpe_vocab = BPEVocabulary.load(vocab_path or _default_vocab_path_for_scale(scale or 5000))
    model_config = payload.get(
        "model_config",
        {
            "vocab_size": bpe_vocab.vocab_size,
            "d_model": DEFAULT_MODEL_D_MODEL,
            "nhead": DEFAULT_MODEL_NHEAD,
            "num_layers": DEFAULT_MODEL_NUM_LAYERS,
            "dim_feedforward": DEFAULT_MODEL_DIM_FEEDFORWARD,
            "dropout": DEFAULT_MODEL_DROPOUT,
        },
    )
    model = BuildingTransformer(**model_config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, bpe_vocab


@torch.no_grad()
def predict_tokens(
    model: BuildingTransformer,
    bpe_vocab: BPEVocabulary,
    source_tokens: list[int],
    device: str,
    max_len: int = 4096,
    decode: str = "greedy",
    beam_size: int = 5,
    length_penalty: float = 0.8,
) -> list[int]:
    source = torch.tensor([source_tokens], dtype=torch.long, device=device)
    if decode == "greedy":
        generated = generate_greedy(model, source, TOKEN_BOS, TOKEN_EOS, max_len=max_len)
        return bpe_vocab.decode(generated[0].tolist())
    if decode == "beam":
        candidates = generate_beam_search(
            model,
            source,
            TOKEN_BOS,
            TOKEN_EOS,
            max_len=max_len,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )
        if not candidates:
            raise ValueError("Beam search produced no candidates")
        return bpe_vocab.decode(candidates[0][0])
    raise ValueError(f"Unsupported decode mode: {decode}")


@torch.no_grad()
def predict_polygon_candidates(
    model: BuildingTransformer,
    bpe_vocab: BPEVocabulary,
    raw_polygon: Polygon,
    scale: int,
    device: str,
    max_len: int = 1024,
    decode: str = "beam",
    beam_size: int = 5,
    length_penalty: float = 0.8,
    geometry_rerank: bool = True,
) -> tuple[Polygon, dict]:
    source_tokens, frame = make_source_tokens(raw_polygon, scale)
    source = torch.tensor([bpe_vocab.encode(source_tokens)], dtype=torch.long, device=device)
    if decode == "greedy":
        generated = generate_greedy(model, source, TOKEN_BOS, TOKEN_EOS, max_len=max_len)
        candidates = [(bpe_vocab.decode(generated[0].tolist()), 0.0)]
    elif decode == "beam":
        candidates = [
            (bpe_vocab.decode(tokens), score)
            for tokens, score in generate_beam_search(
                model,
                source,
                TOKEN_BOS,
                TOKEN_EOS,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
            )
        ]
    else:
        raise ValueError(f"Unsupported decode mode: {decode}")

    best: tuple[float, Polygon, dict] | None = None
    errors: list[str] = []
    for rank, (tokens, model_score) in enumerate(candidates):
        try:
            polygon = _largest_polygon(decode_polygon(tokens, frame))
            if geometry_rerank:
                score, details = _geometry_score(model_score, raw_polygon, polygon, scale)
            else:
                score, details = model_score, {"model_score": model_score}
            details.update({"candidate_rank": float(rank), "token_count": float(len(tokens))})
            if best is None or score > best[0]:
                best = (score, polygon, details)
        except Exception as exc:
            errors.append(f"candidate {rank}: {exc}")

    if best is None:
        raise ValueError("; ".join(errors[:5]) if errors else "No valid polygon candidate")
    metadata = dict(best[2])
    metadata["rerank_score"] = best[0]
    metadata["candidate_count"] = len(candidates)
    metadata["invalid_candidate_count"] = len(errors)
    return best[1], metadata


@torch.no_grad()
def predict_polygon(
    model: BuildingTransformer,
    bpe_vocab: BPEVocabulary,
    raw_polygon: Polygon,
    scale: int,
    device: str,
    max_len: int = 4096,
    decode: str = "greedy",
    beam_size: int = 5,
    length_penalty: float = 0.8,
    geometry_rerank: bool = True,
) -> Polygon:
    polygon, _ = predict_polygon_candidates(
        model,
        bpe_vocab,
        raw_polygon,
        scale,
        device,
        max_len=max_len,
        decode=decode,
        beam_size=beam_size,
        length_penalty=length_penalty,
        geometry_rerank=geometry_rerank,
    )
    return polygon


def iter_shapefile_records(path: Path | str) -> Sequence[BuildingRecord]:
    sf = shapefile.Reader(str(path))
    field_names = [field[0] for field in sf.fields[1:]]
    raw_fid_index = field_names.index("RAW_FID") if "RAW_FID" in field_names else None
    try:
        for fid, shape_record in enumerate(sf.iterShapeRecords()):
            record_fid = int(shape_record.record[raw_fid_index]) if raw_fid_index is not None else fid
            record = _record_from_shape_record(record_fid, field_names, shape_record)
            if record is not None:
                yield record
    finally:
        sf.close()


def run_inference(
    raw_path: Path,
    checkpoint: Path,
    output_path: Path,
    scale: int,
    limit: int | None = None,
    device: str | None = None,
    max_len: int = 1024,
    progress_every: int = 1000,
    decode: str = "beam",
    beam_size: int = 5,
    length_penalty: float = 0.8,
    geometry_rerank: bool = True,
    vocab_path: Path | None = None,
    fid_filter_path: Path | None = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, bpe_vocab = load_model(checkpoint, device, vocab_path=vocab_path, scale=scale)
    fid_filter = load_fid_filter(fid_filter_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path = output_path.with_suffix(".failures.jsonl")
    if failures_path.exists():
        failures_path.unlink()
    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POLYGON)
    writer.autoBalance = True
    writer.field("raw_fid", "N", 12)
    writer.field("scale", "N", 8)
    writer.field("vtx_cmp", "F", 12, 6)
    writer.field("area_rat", "F", 12, 6)
    writer.field("beam_rank", "N", 8)
    written = 0
    failures = 0
    processed = 0
    fail_handle = None

    def write_failure(payload: dict) -> None:
        nonlocal fail_handle
        if fail_handle is None:
            fail_handle = failures_path.open("w", encoding="utf-8")
        fail_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    try:
        for record in iter_shapefile_records(raw_path):
            if fid_filter is not None and record.fid not in fid_filter:
                continue
            if limit is not None and processed >= limit:
                break
            processed += 1
            try:
                polygon, metadata = predict_polygon_candidates(
                    model,
                    bpe_vocab,
                    record.geometry,
                    scale,
                    device,
                    max_len=max_len,
                    decode=decode,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                    geometry_rerank=geometry_rerank,
                )
                writer.poly(polygon_to_writer_parts(polygon))
                writer.record(
                    record.fid,
                    scale,
                    float(metadata.get("compression", 0.0)),
                    float(metadata.get("area_ratio", 0.0)),
                    int(metadata.get("candidate_rank", 0)),
                )
                written += 1
                if written % progress_every == 0:
                    _log(f"Inference wrote {written} features")
            except Exception as exc:
                failures += 1
                write_failure({"raw_fid": record.fid, "scale": scale, "error": str(exc)})
            if processed % progress_every == 0:
                _log(f"Inference processed {processed} features")
    finally:
        if fail_handle is not None:
            fail_handle.close()
        writer.close()
        # 复制源数据的 .prj 坐标系文件
        src_prj = Path(raw_path).with_suffix(".prj")
        dst_prj = output_path.with_suffix(".prj")
        if src_prj.exists() and not dst_prj.exists():
            import shutil
            shutil.copy2(src_prj, dst_prj)
    return {
        "written": written,
        "failures": failures,
        "output": str(output_path),
        "failures_path": str(failures_path) if fail_handle is not None else None,
        "decode": decode,
        "beam_size": beam_size,
        "length_penalty": length_penalty,
        "geometry_rerank": geometry_rerank,
        "fid_filter_path": str(fid_filter_path) if fid_filter_path is not None else None,
    }


def predictions_jsonl_to_shapefile(
    predictions_path: Path | str,
    vocab_path: Path | str,
    output_path: Path | str,
    source_prj: Path | str | None = None,
) -> int:
    vocab = BPEVocabulary.load(vocab_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POLYGON)
    writer.autoBalance = True
    writer.field("raw_fid", "N", 12)
    writer.field("osm_id", "C", 40)
    writer.field("scale", "N", 8)
    written = 0
    with Path(predictions_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                polygon = _largest_polygon(decode_polygon(vocab.decode(row["pred_tokens"]), frame_from_dict(row["source_frame"])))
            except Exception:
                continue
            writer.poly(polygon_to_writer_parts(polygon))
            writer.record(row.get("raw_fid", written), row.get("osm_id", ""), row.get("scale", 0))
            written += 1
    writer.close()
    if source_prj is not None:
        source_prj = Path(source_prj)
        if source_prj.exists():
            import shutil
            shutil.copy2(source_prj, output_path.with_suffix(".prj"))
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=int, choices=[5000, 10000], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--length-penalty", type=float, default=0.8)
    parser.add_argument("--vocab-path", type=Path, default=None)
    parser.add_argument("--no-geometry-rerank", action="store_true")
    parser.add_argument("--fid-filter", type=Path, default=None, help="JSONL containing raw_fid values to process")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_inference(
                args.raw,
                args.checkpoint,
                args.output,
                args.scale,
                args.limit,
                device=args.device,
                max_len=args.max_len,
                progress_every=args.progress_every,
                decode=args.decode,
                beam_size=args.beam_size,
                length_penalty=args.length_penalty,
                vocab_path=args.vocab_path,
                geometry_rerank=not args.no_geometry_rerank,
                fid_filter_path=args.fid_filter,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
