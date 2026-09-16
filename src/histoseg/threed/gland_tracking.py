"""Cross-slice tracking for HistoSeg gland instance outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Any, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from shapely.geometry import shape


PathLike = Union[str, Path]


@dataclass(frozen=True)
class GlandInstanceTrackingConfig:
    """Configuration for assigning stable gland IDs across neighboring slices."""

    segmentation_result_dir: PathLike
    out_dir: PathLike
    max_centroid_distance_um: float = 300.0
    min_overlap_ratio: float = 0.15
    min_area_ratio: float = 0.25
    max_area_ratio: float = 4.0
    lumen_center_weight: float = 0.4
    marker_profile_weight: float = 0.3
    use_one_to_one: bool = True


@dataclass
class GlandInstanceTrackingResult:
    out_dir: Path
    gland_instance_tracks_csv: Path
    gland_instance_qc_index_csv: Path
    gland_count: int


@dataclass(frozen=True)
class _Instance:
    slice_instance_id: str
    slice_order: int
    sample_id: str
    z_um: float
    semantic_structure: str
    geometry: Any
    area_um2: float
    centroid_x_um: float
    centroid_y_um: float
    lumen_area_um2: float
    lumen_centroid_x_um: float
    lumen_centroid_y_um: float
    ring_support_score: float
    epithelial_marker_score: float
    stromal_immune_contamination_score: float
    qc_flags: str
    marker_profile: Mapping[str, float]
    row: Mapping[str, Any]


@dataclass(frozen=True)
class _Candidate:
    source: str
    target: str
    score: float
    overlap_iou: float
    centroid_distance_um: float
    area_ratio: float
    lumen_center_distance_um: float
    marker_profile_similarity: float


def track_gland_instances(
    cfg: GlandInstanceTrackingConfig,
) -> GlandInstanceTrackingResult:
    """Track per-slice gland instances across neighboring z slices."""

    _validate_config(cfg)
    segmentation_dir = Path(cfg.segmentation_result_dir).expanduser().resolve()
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = _load_instances(segmentation_dir)
    candidates, accepted = _link_neighboring_slices(instances, cfg)
    gland_ids = _assign_gland_ids(instances, accepted)
    tracks = _build_tracks(instances, candidates, accepted, gland_ids)
    index = _build_gland_index(tracks)

    tracks_csv = out_dir / "gland_instance_tracks.csv"
    index_csv = out_dir / "gland_instance_qc_index.csv"
    tracks.to_csv(tracks_csv, index=False)
    index.to_csv(index_csv, index=False)

    return GlandInstanceTrackingResult(
        out_dir=out_dir,
        gland_instance_tracks_csv=tracks_csv,
        gland_instance_qc_index_csv=index_csv,
        gland_count=int(index["gland_id"].nunique()) if not index.empty else 0,
    )


def _load_instances(segmentation_dir: Path) -> list[_Instance]:
    geojson_path = segmentation_dir / "slice_gland_instances.geojson"
    qc_path = segmentation_dir / "slice_gland_instances_qc.csv"
    if not geojson_path.exists():
        raise FileNotFoundError(geojson_path)
    if not qc_path.exists():
        raise FileNotFoundError(qc_path)
    qc = pd.read_csv(qc_path)
    row_lookup = {
        str(row["slice_instance_id"]): row.to_dict()
        for _, row in qc.iterrows()
    }
    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    instances: list[_Instance] = []
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        slice_instance_id = str(props.get("slice_instance_id", ""))
        if not slice_instance_id:
            continue
        row = dict(row_lookup.get(slice_instance_id, {}))
        merged = {**row, **props}
        geom = shape(feature.get("geometry"))
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        marker_profile = _parse_marker_profile(merged)
        instances.append(
            _Instance(
                slice_instance_id=slice_instance_id,
                slice_order=int(merged.get("slice_order", 0)),
                sample_id=str(merged.get("sample_id", "")),
                z_um=float(merged.get("z_um", math.nan)),
                semantic_structure=str(merged.get("semantic_structure", merged.get("assigned_structure", ""))),
                geometry=geom,
                area_um2=float(merged.get("area_um2", geom.area)),
                centroid_x_um=float(merged.get("centroid_x_um", geom.centroid.x)),
                centroid_y_um=float(merged.get("centroid_y_um", geom.centroid.y)),
                lumen_area_um2=float(merged.get("lumen_area_um2", 0.0)),
                lumen_centroid_x_um=float(merged.get("lumen_centroid_x_um", math.nan)),
                lumen_centroid_y_um=float(merged.get("lumen_centroid_y_um", math.nan)),
                ring_support_score=float(merged.get("ring_support_score", math.nan)),
                epithelial_marker_score=float(merged.get("epithelial_marker_score", math.nan)),
                stromal_immune_contamination_score=float(
                    merged.get("stromal_immune_contamination_score", math.nan)
                ),
                qc_flags=str(merged.get("qc_flags", "")) if not pd.isna(merged.get("qc_flags", "")) else "",
                marker_profile=marker_profile,
                row=merged,
            )
        )
    return sorted(instances, key=lambda item: (item.slice_order, item.semantic_structure, item.slice_instance_id))


def _link_neighboring_slices(
    instances: Sequence[_Instance],
    cfg: GlandInstanceTrackingConfig,
) -> tuple[list[_Candidate], list[_Candidate]]:
    by_order_structure: dict[tuple[int, str], list[_Instance]] = {}
    for instance in instances:
        by_order_structure.setdefault((instance.slice_order, instance.semantic_structure), []).append(instance)
    orders = sorted({instance.slice_order for instance in instances})
    all_candidates: list[_Candidate] = []
    accepted: list[_Candidate] = []
    for order in orders:
        next_order = order + 1
        structures = sorted(
            {
                structure
                for slice_order, structure in by_order_structure
                if slice_order in {order, next_order}
            }
        )
        for structure in structures:
            left = by_order_structure.get((order, structure), [])
            right = by_order_structure.get((next_order, structure), [])
            if not left or not right:
                continue
            pair_candidates = [
                candidate
                for source in left
                for target in right
                if (candidate := _candidate_link(source, target, cfg)) is not None
            ]
            all_candidates.extend(pair_candidates)
            if cfg.use_one_to_one:
                accepted.extend(_hungarian_accept(left, right, pair_candidates))
            else:
                accepted.extend(_mutual_best_accept(pair_candidates))
    return all_candidates, accepted


def _candidate_link(
    source: _Instance,
    target: _Instance,
    cfg: GlandInstanceTrackingConfig,
) -> _Candidate | None:
    intersection_area = float(source.geometry.intersection(target.geometry).area)
    union_area = float(source.geometry.union(target.geometry).area)
    overlap_iou = intersection_area / union_area if union_area > 0 else 0.0
    centroid_distance = _distance(
        source.centroid_x_um,
        source.centroid_y_um,
        target.centroid_x_um,
        target.centroid_y_um,
    )
    area_ratio = max(source.area_um2, target.area_um2) / max(min(source.area_um2, target.area_um2), 1e-9)
    lumen_distance = _lumen_distance(source, target, fallback=centroid_distance)
    marker_similarity = _marker_cosine(source.marker_profile, target.marker_profile)
    if (
        overlap_iou < cfg.min_overlap_ratio
        and centroid_distance > cfg.max_centroid_distance_um
    ):
        return None
    if area_ratio < cfg.min_area_ratio or area_ratio > cfg.max_area_ratio:
        return None

    centroid_score = max(0.0, 1.0 - centroid_distance / max(cfg.max_centroid_distance_um, 1e-9))
    lumen_score = max(0.0, 1.0 - lumen_distance / max(cfg.max_centroid_distance_um, 1e-9))
    area_score = min(source.area_um2, target.area_um2) / max(source.area_um2, target.area_um2, 1e-9)
    residual_weight = max(0.0, 1.0 - float(cfg.lumen_center_weight) - float(cfg.marker_profile_weight))
    score = (
        residual_weight * (0.5 * overlap_iou + 0.3 * centroid_score + 0.2 * area_score)
        + float(cfg.lumen_center_weight) * lumen_score
        + float(cfg.marker_profile_weight) * marker_similarity
    )
    return _Candidate(
        source=source.slice_instance_id,
        target=target.slice_instance_id,
        score=float(score),
        overlap_iou=float(overlap_iou),
        centroid_distance_um=float(centroid_distance),
        area_ratio=float(area_ratio),
        lumen_center_distance_um=float(lumen_distance),
        marker_profile_similarity=float(marker_similarity),
    )


def _hungarian_accept(
    left: Sequence[_Instance],
    right: Sequence[_Instance],
    candidates: Sequence[_Candidate],
) -> list[_Candidate]:
    if not candidates:
        return []
    left_ids = [instance.slice_instance_id for instance in left]
    right_ids = [instance.slice_instance_id for instance in right]
    left_index = {value: index for index, value in enumerate(left_ids)}
    right_index = {value: index for index, value in enumerate(right_ids)}
    score_matrix = np.full((len(left_ids), len(right_ids)), -np.inf, dtype=float)
    candidate_lookup: dict[tuple[int, int], _Candidate] = {}
    for candidate in candidates:
        i = left_index[candidate.source]
        j = right_index[candidate.target]
        if candidate.score > score_matrix[i, j]:
            score_matrix[i, j] = candidate.score
            candidate_lookup[(i, j)] = candidate
    finite = np.isfinite(score_matrix)
    if not finite.any():
        return []
    min_score = float(np.nanmin(score_matrix[finite]))
    cost = np.where(finite, -score_matrix, -min_score + 1e6)
    row_ind, col_ind = linear_sum_assignment(cost)
    accepted: list[_Candidate] = []
    for i, j in zip(row_ind, col_ind):
        candidate = candidate_lookup.get((int(i), int(j)))
        if candidate is not None and candidate.score > 0:
            accepted.append(candidate)
    return accepted


def _mutual_best_accept(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    best_out: dict[str, _Candidate] = {}
    best_in: dict[str, _Candidate] = {}
    for candidate in candidates:
        if candidate.score > best_out.get(candidate.source, candidate).score or candidate.source not in best_out:
            best_out[candidate.source] = candidate
        if candidate.score > best_in.get(candidate.target, candidate).score or candidate.target not in best_in:
            best_in[candidate.target] = candidate
    return [
        candidate
        for candidate in candidates
        if best_out.get(candidate.source) == candidate
        and best_in.get(candidate.target) == candidate
    ]


def _assign_gland_ids(
    instances: Sequence[_Instance],
    accepted: Sequence[_Candidate],
) -> dict[str, str]:
    parent = {instance.slice_instance_id: instance.slice_instance_id for instance in instances}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for link in accepted:
        union(link.source, link.target)

    by_root: dict[str, list[_Instance]] = {}
    by_id = {instance.slice_instance_id: instance for instance in instances}
    for instance in instances:
        by_root.setdefault(find(instance.slice_instance_id), []).append(instance)
    sorted_roots = sorted(
        by_root,
        key=lambda root: (
            min(item.slice_order for item in by_root[root]),
            by_root[root][0].semantic_structure,
            np.median([item.centroid_x_um for item in by_root[root]]),
            np.median([item.centroid_y_um for item in by_root[root]]),
        ),
    )
    root_ids = {root: f"gland_{index:04d}" for index, root in enumerate(sorted_roots, start=1)}
    return {node_id: root_ids[find(node_id)] for node_id in by_id}


def _build_tracks(
    instances: Sequence[_Instance],
    candidates: Sequence[_Candidate],
    accepted: Sequence[_Candidate],
    gland_ids: Mapping[str, str],
) -> pd.DataFrame:
    accepted_next = {link.source: link for link in accepted}
    accepted_prev = {link.target: link for link in accepted}
    accepted_pairs = {(link.source, link.target) for link in accepted}
    candidate_by_node: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        candidate_by_node.setdefault(candidate.source, []).append(candidate)
        candidate_by_node.setdefault(candidate.target, []).append(candidate)

    rows: list[dict[str, Any]] = []
    for instance in instances:
        next_link = accepted_next.get(instance.slice_instance_id)
        prev_link = accepted_prev.get(instance.slice_instance_id)
        branch_candidates = [
            _candidate_label(candidate, perspective=instance.slice_instance_id)
            for candidate in candidate_by_node.get(instance.slice_instance_id, [])
            if (candidate.source, candidate.target) not in accepted_pairs
        ]
        row = dict(instance.row)
        row.update(
            {
                "gland_id": gland_ids[instance.slice_instance_id],
                "slice_instance_id": instance.slice_instance_id,
                "slice_order": instance.slice_order,
                "sample_id": instance.sample_id,
                "z_um": instance.z_um,
                "semantic_structure": instance.semantic_structure,
                "area_um2": instance.area_um2,
                "centroid_x_um": instance.centroid_x_um,
                "centroid_y_um": instance.centroid_y_um,
                "lumen_area_um2": instance.lumen_area_um2,
                "ring_support_score": instance.ring_support_score,
                "epithelial_marker_score": instance.epithelial_marker_score,
                "stromal_immune_contamination_score": instance.stromal_immune_contamination_score,
                "prev_slice_instance_id": prev_link.source if prev_link else "",
                "next_slice_instance_id": next_link.target if next_link else "",
                "prev_link_score": prev_link.score if prev_link else math.nan,
                "next_link_score": next_link.score if next_link else math.nan,
                "link_score": next_link.score if next_link else (prev_link.score if prev_link else math.nan),
                "branch_merge_candidates": ";".join(branch_candidates),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["gland_id", "slice_order", "slice_instance_id"])


def _build_gland_index(tracks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if tracks.empty:
        return pd.DataFrame()
    for gland_id, group in tracks.groupby("gland_id", sort=True):
        group = group.sort_values("slice_order")
        areas = group["area_um2"].to_numpy(dtype=float)
        rows.append(
            {
                "gland_id": gland_id,
                "semantic_structure": str(group["semantic_structure"].mode().iloc[0]),
                "slice_count": int(group["slice_order"].nunique()),
                "component_count": int(len(group)),
                "first_slice_order": int(group["slice_order"].min()),
                "last_slice_order": int(group["slice_order"].max()),
                "z_min_um": float(group["z_um"].min()),
                "z_max_um": float(group["z_um"].max()),
                "z_span_um": float(group["z_um"].max() - group["z_um"].min()),
                "area_median_um2": float(np.median(areas)) if len(areas) else math.nan,
                "area_cv": float(np.std(areas) / np.mean(areas)) if len(areas) and np.mean(areas) > 0 else math.nan,
                "median_ring_support_score": float(group["ring_support_score"].median()),
                "median_epithelial_marker_score": float(group["epithelial_marker_score"].median()),
                "max_stromal_immune_contamination_score": float(
                    group["stromal_immune_contamination_score"].max()
                ),
                "branch_merge_candidate_count": int(
                    group["branch_merge_candidates"].fillna("").astype(str).map(bool).sum()
                ),
                "qc_flags": ";".join(
                    sorted(
                        {
                            flag
                            for value in group["qc_flags"].fillna("").astype(str)
                            for flag in value.split(";")
                            if flag
                        }
                    )
                ),
                "page": f"glands/{gland_id}.html",
            }
        )
    return pd.DataFrame(rows)


def _candidate_label(candidate: _Candidate, *, perspective: str) -> str:
    other = candidate.target if candidate.source == perspective else candidate.source
    return f"{other}:{candidate.score:.3f}"


def _parse_marker_profile(row: Mapping[str, Any]) -> dict[str, float]:
    raw = row.get("marker_profile_json")
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw)
            return {str(key): float(value) for key, value in payload.items()}
        except Exception:
            pass
    result: dict[str, float] = {}
    for key, value in row.items():
        if str(key).startswith("marker_"):
            try:
                result[str(key)] = float(value)
            except Exception:
                continue
    return result


def _marker_cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(set(left).union(right))
    if not keys:
        return 1.0
    a = np.array([float(left.get(key, 0.0)) for key in keys], dtype=float)
    b = np.array([float(right.get(key, 0.0)) for key in keys], dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 1.0
    return float(np.dot(a, b) / denom)


def _lumen_distance(source: _Instance, target: _Instance, *, fallback: float) -> float:
    if all(
        math.isfinite(value)
        for value in [
            source.lumen_centroid_x_um,
            source.lumen_centroid_y_um,
            target.lumen_centroid_x_um,
            target.lumen_centroid_y_um,
        ]
    ):
        return _distance(
            source.lumen_centroid_x_um,
            source.lumen_centroid_y_um,
            target.lumen_centroid_x_um,
            target.lumen_centroid_y_um,
        )
    return fallback


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = float(x1) - float(x2)
    dy = float(y1) - float(y2)
    return math.sqrt(dx * dx + dy * dy)


def _validate_config(cfg: GlandInstanceTrackingConfig) -> None:
    if cfg.max_centroid_distance_um <= 0:
        raise ValueError("`max_centroid_distance_um` must be positive.")
    if cfg.min_overlap_ratio < 0:
        raise ValueError("`min_overlap_ratio` must be non-negative.")
    if cfg.min_area_ratio <= 0 or cfg.max_area_ratio <= 0:
        raise ValueError("Area ratio thresholds must be positive.")
    if cfg.min_area_ratio > cfg.max_area_ratio:
        raise ValueError("`min_area_ratio` cannot exceed `max_area_ratio`.")


__all__ = [
    "GlandInstanceTrackingConfig",
    "GlandInstanceTrackingResult",
    "track_gland_instances",
]
