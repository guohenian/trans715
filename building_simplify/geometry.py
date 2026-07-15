from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Sequence

import shapefile
from shapely import affinity
from shapely.geometry import LinearRing, Polygon, shape
from shapely.geometry.polygon import orient

from .config import (
    DEFAULT_CONFIG,
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_INNER,
    TOKEN_SCALE_10000,
    TOKEN_SCALE_5000,
    TOKEN_SEP,
    TokenConfig,
)


@dataclass(frozen=True)
class TokenFrame:
    start_x: float
    start_y: float
    ref_x: float
    ref_y: float
    rotation_deg: float
    longest_edge_m: float
    start_vertex_index: int
    inner_starts: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class BuildingRecord:
    fid: int
    geometry: Polygon
    properties: dict


def _half_up(value: float) -> int:
    return int(Decimal(str(value)).to_integral_value(rounding=ROUND_HALF_UP))


def quantize_direction(angle_rad: float, cfg: TokenConfig = DEFAULT_CONFIG) -> int:
    return _half_up(angle_rad / cfg.bin_width_rad) % cfg.directions


def quantize_length(length_m: float, cfg: TokenConfig = DEFAULT_CONFIG) -> int:
    return max(1, _half_up(length_m / cfg.token_length_m))


def direction_to_radians(token_id: int, cfg: TokenConfig = DEFAULT_CONFIG) -> float:
    return token_id * cfg.bin_width_rad


def scale_to_token(scale: int) -> int:
    if scale == 5000:
        return TOKEN_SCALE_5000
    if scale == 10000:
        return TOKEN_SCALE_10000
    raise ValueError(f"Unsupported scale: {scale}")


def clean_polygon(geom) -> Polygon | None:
    if geom.is_empty:
        return None
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda part: part.area)
    if geom.geom_type != "Polygon":
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty or geom.geom_type != "Polygon":
        return None
    return geom


def representative_point(polygon: Polygon) -> tuple[float, float]:
    pt = polygon.representative_point()
    return pt.x, pt.y


def find_longest_edge(polygon: Polygon) -> tuple[float, float, int]:
    coords = list(polygon.exterior.coords)
    best_len = -1.0
    best_angle = 0.0
    best_idx = 0
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if length > best_len:
            best_len = length
            best_angle = angle
            best_idx = i
    return best_angle, best_len, best_idx


def _start_from_edge(coords: Sequence[tuple[float, float]], edge_idx: int) -> int:
    a = coords[edge_idx]
    b = coords[(edge_idx + 1) % len(coords)]
    return edge_idx if (a[0], a[1]) <= (b[0], b[1]) else (edge_idx + 1) % len(coords)


def _ccw_outer_coords(polygon: Polygon) -> list[tuple[float, float]]:
    oriented = orient(polygon, sign=1.0)
    return list(oriented.exterior.coords)[:-1]


def _ccw_longest_edge_start(coords: Sequence[tuple[float, float]]) -> tuple[float, int]:
    candidates: list[tuple[float, float, int]] = []
    best_len = -1.0
    n = len(coords)
    for edge_idx in range(n):
        x1, y1 = coords[edge_idx]
        x2, y2 = coords[(edge_idx + 1) % n]
        length = math.hypot(x2 - x1, y2 - y1)
        if length <= 1e-12:
            continue
        start_idx = _start_from_edge(coords, edge_idx)
        sx, sy = coords[start_idx]
        nx, ny = coords[(start_idx + 1) % n]
        forward_dx = nx - sx
        if length > best_len + 1e-9:
            best_len = length
            candidates = [(forward_dx, sy, start_idx)]
        elif abs(length - best_len) <= 1e-9:
            candidates.append((forward_dx, sy, start_idx))
    if not candidates:
        raise ValueError("No valid exterior edge found")
    candidates.sort(key=lambda item: (item[0] <= 0, item[1], item[2]))
    return best_len, candidates[0][2]


def _closest_vertex(coords: Sequence[tuple[float, float]], anchor: tuple[float, float]) -> int:
    ax, ay = anchor
    return min(range(len(coords)), key=lambda i: math.hypot(coords[i][0] - ax, coords[i][1] - ay))


def _traverse_ring(
    coords: Sequence[tuple[float, float]],
    start_idx: int,
    cfg: TokenConfig,
) -> list[int]:
    tokens: list[int] = []
    n = len(coords)
    for j in range(n):
        p1 = coords[(start_idx + j) % n]
        p2 = coords[(start_idx + j + 1) % n]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            continue
        direction = quantize_direction(math.atan2(dy, dx), cfg)
        count = quantize_length(length, cfg)
        tokens.extend([direction] * count)
    return tokens


def encode_polygon(
    polygon: Polygon,
    cfg: TokenConfig = DEFAULT_CONFIG,
    frame: TokenFrame | None = None,
) -> tuple[list[int], TokenFrame]:
    polygon = clean_polygon(polygon)
    if polygon is None:
        raise ValueError("Cannot encode empty or non-polygon geometry")

    if frame is None:
        ref_x, ref_y = representative_point(polygon)
        rotation_deg, _, _ = find_longest_edge(polygon)
        local = affinity.rotate(polygon, -rotation_deg, origin=(ref_x, ref_y))
        local = orient(local, sign=1.0)
        outer = list(local.exterior.coords)[:-1]
        longest_edge_m, start_idx = _ccw_longest_edge_start(outer)
        inner_starts = tuple((ring.coords[0][0], ring.coords[0][1]) for ring in local.interiors)
        frame = TokenFrame(
            start_x=outer[start_idx][0],
            start_y=outer[start_idx][1],
            ref_x=ref_x,
            ref_y=ref_y,
            rotation_deg=rotation_deg,
            longest_edge_m=longest_edge_m,
            start_vertex_index=start_idx,
            inner_starts=inner_starts,
        )
    else:
        local = affinity.rotate(polygon, -frame.rotation_deg, origin=(frame.ref_x, frame.ref_y))
        local = orient(local, sign=1.0)
        outer = list(local.exterior.coords)[:-1]
        start_idx = _closest_vertex(outer, (frame.start_x, frame.start_y))
        inner_starts = frame.inner_starts

    tokens = [TOKEN_BOS]
    tokens.extend(_traverse_ring(outer, start_idx, cfg))
    for ri, interior in enumerate(local.interiors):
        inner = list(interior.coords)[:-1]
        if not inner:
            continue
        anchor = inner_starts[ri] if ri < len(inner_starts) else inner[0]
        tokens.extend([TOKEN_SEP, TOKEN_INNER])
        tokens.extend(_traverse_ring(inner, _closest_vertex(inner, anchor), cfg))
    tokens.append(TOKEN_EOS)
    return tokens, frame


def _merge_edges(edges: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not edges:
        return []
    merged = [edges[0]]
    for direction, length in edges[1:]:
        prev_direction, prev_length = merged[-1]
        if direction == prev_direction:
            merged[-1] = (prev_direction, prev_length + length)
        else:
            merged.append((direction, length))
    return merged


def _parse_token_rings(tokens: Sequence[int], cfg: TokenConfig) -> list[list[tuple[int, float]]]:
    rings: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == TOKEN_BOS:
            i += 1
            continue
        if token == TOKEN_EOS:
            break
        if token == TOKEN_SEP:
            rings.append(_merge_edges(current))
            current = []
            i += 1
            if i < len(tokens) and tokens[i] == TOKEN_INNER:
                i += 1
            continue
        if 0 <= token < cfg.directions:
            direction = token
            count = 0
            while i < len(tokens) and tokens[i] == direction:
                count += 1
                i += 1
            current.append((direction, count * cfg.token_length_m))
            continue
        i += 1
    rings.append(_merge_edges(current))
    return rings


def _walk_ring(start: tuple[float, float], edges: Sequence[tuple[int, float]], cfg: TokenConfig) -> list[tuple[float, float]]:
    x, y = start
    points = [(x, y)]
    for direction, length in edges:
        angle = direction_to_radians(direction, cfg)
        x += math.cos(angle) * length
        y += math.sin(angle) * length
        points.append((x, y))
    if math.hypot(points[-1][0] - start[0], points[-1][1] - start[1]) <= cfg.token_length_m * 0.5:
        points[-1] = start
    else:
        points.append(start)
    return points


def decode_polygon(tokens: Sequence[int], frame: TokenFrame, cfg: TokenConfig = DEFAULT_CONFIG) -> Polygon:
    rings = _parse_token_rings(tokens, cfg)
    if not rings or not rings[0]:
        raise ValueError("No outer ring found in token sequence")
    outer = _walk_ring((frame.start_x, frame.start_y), rings[0], cfg)
    holes = []
    for i, ring_edges in enumerate(rings[1:]):
        if not ring_edges:
            continue
        start = frame.inner_starts[i] if i < len(frame.inner_starts) else outer[0]
        holes.append(_walk_ring(start, ring_edges, cfg))
    poly = Polygon(LinearRing(outer), [LinearRing(hole) for hole in holes])
    if not poly.is_valid:
        poly = poly.buffer(0)
    rotated = affinity.rotate(poly, frame.rotation_deg, origin=(frame.ref_x, frame.ref_y))
    if rotated.geom_type == "Polygon" and not rotated.is_empty and rotated.is_valid:
        return rotated
    repaired = clean_polygon(rotated)
    if repaired is None or repaired.is_empty or not repaired.is_valid:
        raise ValueError("Decoded polygon is invalid after final-coordinate topology repair")
    return repaired


def make_source_tokens(raw_polygon: Polygon, scale: int, cfg: TokenConfig = DEFAULT_CONFIG) -> tuple[list[int], TokenFrame]:
    tokens, frame = encode_polygon(raw_polygon, cfg)
    return [scale_to_token(scale)] + tokens, frame


def make_target_tokens(target_polygon: Polygon, source_frame: TokenFrame, cfg: TokenConfig = DEFAULT_CONFIG) -> list[int]:
    tokens, _ = encode_polygon(target_polygon, cfg, frame=source_frame)
    return tokens


def _record_from_shape_record(fid: int, field_names: Sequence[str], shape_record) -> BuildingRecord | None:
    geom = clean_polygon(shape(shape_record.shape.__geo_interface__))
    if geom is None:
        return None
    return BuildingRecord(fid=fid, geometry=geom, properties=dict(zip(field_names, shape_record.record)))


def iter_paired_records_from_shapefiles(
    raw_path: Path | str,
    target_path: Path | str,
) -> Sequence[tuple[BuildingRecord, BuildingRecord]]:
    raw_sf = shapefile.Reader(str(raw_path))
    target_sf = shapefile.Reader(str(target_path))
    raw_field_names = [field[0] for field in raw_sf.fields[1:]]
    target_field_names = [field[0] for field in target_sf.fields[1:]]
    raw_count = len(raw_sf)
    raw_has_original_fids = "RAW_FID" in raw_field_names
    raw_by_original_fid: dict[int, int] = {}
    raw_by_osm_id: dict[str, int] = {}
    raw_osm_idx = raw_field_names.index("osm_id") if "osm_id" in raw_field_names else None
    target_osm_idx = target_field_names.index("osm_id") if "osm_id" in target_field_names else None
    if raw_osm_idx is not None:
        for local_idx, record in enumerate(raw_sf.records()):
            raw_by_osm_id[str(record[raw_osm_idx])] = local_idx
    if raw_has_original_fids:
        raw_fid_idx = raw_field_names.index("RAW_FID")
        for local_idx, record in enumerate(raw_sf.records()):
            raw_by_original_fid[int(record[raw_fid_idx])] = local_idx
    try:
        for target_fid, target_shape_record in enumerate(target_sf.iterShapeRecords()):
            target = _record_from_shape_record(target_fid, target_field_names, target_shape_record)
            if target is None:
                continue
            raw = None
            target_osm_id = str(target_shape_record.record[target_osm_idx]) if target_osm_idx is not None else None
            if target_osm_id is not None and target_osm_id in raw_by_osm_id:
                raw_shape_record = raw_sf.shapeRecord(raw_by_osm_id[target_osm_id])
                raw = _record_from_shape_record(raw_by_osm_id[target_osm_id], raw_field_names, raw_shape_record)
            if "InBld_FID" not in target.properties:
                if raw is None:
                    raise ValueError("Target record is missing InBld_FID and cannot be matched by osm_id")
                source_raw_fid = raw.fid
            else:
                source_raw_fid = int(target.properties["InBld_FID"])
                if raw is None:
                    if raw_has_original_fids:
                        if source_raw_fid not in raw_by_original_fid:
                            raise ValueError(f"InBld_FID {source_raw_fid} does not exist in filtered raw records")
                        raw_shape_record = raw_sf.shapeRecord(raw_by_original_fid[source_raw_fid])
                        raw = _record_from_shape_record(source_raw_fid, raw_field_names, raw_shape_record)
                    else:
                        if source_raw_fid < 0 or source_raw_fid >= raw_count:
                            raise ValueError(f"InBld_FID {source_raw_fid} does not exist in raw records")
                        raw = _record_from_shape_record(source_raw_fid, raw_field_names, raw_sf.shapeRecord(source_raw_fid))
            if raw is None:
                raise ValueError(f"Raw geometry at InBld_FID {source_raw_fid} is empty or invalid")
            if "osm_id" in raw.properties and "osm_id" in target.properties:
                if str(raw.properties["osm_id"]) != str(target.properties["osm_id"]):
                    raise ValueError(f"osm_id mismatch for InBld_FID {source_raw_fid}")
            yield raw, target
    finally:
        raw_sf.close()
        target_sf.close()
