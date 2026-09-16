from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ContourTearClosureConfig:
    """Configuration for contour-guided local tear closure.

    The moving contours and cells are expected to already be in the fixed
    coordinate system after a hard or dense pre-alignment. The closure stage is
    deliberately not a TPS fit: it estimates local translations from
    same-celltype reciprocal contour correspondences and applies a capped,
    distance-weighted translation field to moving cells and contours.
    """

    fixed_geojson: PathLike
    moving_aligned_geojson: PathLike
    moving_cells_csv: PathLike
    out_dir: PathLike
    fixed_cells_csv: PathLike | None = None
    group_property: str = "assigned_structure"
    moving_x_column: str = "x_aligned_um"
    moving_y_column: str = "y_aligned_um"
    fixed_x_column: str = "x_centroid"
    fixed_y_column: str = "y_centroid"
    cluster_column: str = "cluster"
    output_coordinate_prefix: str = "contour_tear_closure"
    centroid_search_radius_um: float = 650.0
    area_ratio_min: float = 0.10
    area_ratio_max: float = 10.0
    min_pair_score: float = 0.0
    min_motion_um: float = 0.0
    tear_motion_threshold_um: float = 25.0
    influence_radius_um: float = 650.0
    max_neighbors: int = 12
    max_displacement_um: float = 200.0
    save_preview_png: bool = True
    overwrite: bool = False
    dpi: int = 180


@dataclass
class ContourTearClosureResult:
    """Artifacts produced by contour-guided local tear closure."""

    out_dir: Path
    closed_cells_csv: Path
    closed_contours_geojson: Path
    correspondence_csv: Path
    landmarks_csv: Path
    summary_json: Path
    review_png: Path | None = None


@dataclass(frozen=True)
class _ContourRecord:
    feature_index: int
    group: str
    name: str
    geometry: Any
    area: float
    centroid_x: float
    centroid_y: float


@dataclass(frozen=True)
class _LocalTranslationField:
    anchor_xy: np.ndarray
    displacement_xy: np.ndarray
    influence_radius_um: float
    max_neighbors: int
    max_displacement_um: float

    def warp(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if xy.ndim == 1:
            xy = xy.reshape(1, 2)
        return xy + self.displacement(xy)

    def displacement(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if xy.ndim == 1:
            xy = xy.reshape(1, 2)
        if len(self.anchor_xy) == 0:
            return np.zeros_like(xy)
        k = min(max(int(self.max_neighbors), 1), len(self.anchor_xy))
        tree = cKDTree(self.anchor_xy)
        distances, indices = tree.query(
            xy,
            k=k,
            distance_upper_bound=float(self.influence_radius_um),
        )
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        valid = np.isfinite(distances) & (indices < len(self.anchor_xy))
        out = np.zeros((len(xy), 2), dtype=float)
        if not valid.any():
            return out
        sigma = max(float(self.influence_radius_um) / 2.0, 1e-6)
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        weights[~valid] = 0.0
        weight_sum = weights.sum(axis=1)
        nonzero = weight_sum > 0
        if not nonzero.any():
            return out
        disp = self.displacement_xy[np.clip(indices[nonzero], 0, len(self.anchor_xy) - 1)]
        weighted = (disp * weights[nonzero, :, None]).sum(axis=1) / weight_sum[nonzero, None]
        out[nonzero] = weighted
        norms = np.linalg.norm(out, axis=1)
        too_large = norms > float(self.max_displacement_um)
        if too_large.any():
            out[too_large] *= (float(self.max_displacement_um) / norms[too_large])[:, None]
        return out


def run_contour_tear_closure(cfg: ContourTearClosureConfig) -> ContourTearClosureResult:
    """Close tear-like moving-slice gaps using contour correspondences.

    The method first finds same-group reciprocal contour correspondences between
    the fixed slice and the pre-aligned moving slice. Corresponding contour
    centroids define local translation anchors. A capped inverse-distance-like
    translation field then moves only the moving cells and contours. Unmatched
    fixed contours are reported as possible missing tissue; the method does not
    synthesize new cells or polygons for missing tissue.
    """

    _validate_config(cfg)
    out_dir = Path(cfg.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _result_paths(out_dir)
    if (
        not cfg.overwrite
        and paths["closed_cells_csv"].exists()
        and paths["closed_contours_geojson"].exists()
        and paths["summary_json"].exists()
    ):
        return ContourTearClosureResult(
            out_dir=out_dir,
            closed_cells_csv=paths["closed_cells_csv"],
            closed_contours_geojson=paths["closed_contours_geojson"],
            correspondence_csv=paths["correspondence_csv"],
            landmarks_csv=paths["landmarks_csv"],
            summary_json=paths["summary_json"],
            review_png=paths["review_png"] if paths["review_png"].exists() else None,
        )

    fixed_payload = _read_geojson(cfg.fixed_geojson)
    moving_payload = _read_geojson(cfg.moving_aligned_geojson)
    fixed_records = _records_from_geojson(fixed_payload, cfg.group_property)
    moving_records = _records_from_geojson(moving_payload, cfg.group_property)

    correspondences = _find_reciprocal_correspondences(fixed_records, moving_records, cfg)
    correspondences.to_csv(paths["correspondence_csv"], index=False)

    landmarks = _landmarks_from_correspondences(correspondences, cfg)
    landmarks.to_csv(paths["landmarks_csv"], index=False)
    field = _field_from_landmarks(landmarks, cfg)

    closed_payload = _warp_geojson(moving_payload, field, cfg)
    paths["closed_contours_geojson"].write_text(
        json.dumps(closed_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    closed_cells = _warp_cells(cfg.moving_cells_csv, field, cfg)
    closed_cells.to_csv(paths["closed_cells_csv"], index=False)

    review_png = None
    if cfg.save_preview_png:
        review_png = paths["review_png"]
        _write_landmark_review_png(
            fixed_payload=fixed_payload,
            moving_payload=moving_payload,
            closed_payload=closed_payload,
            landmarks=landmarks,
            out_png=review_png,
            dpi=cfg.dpi,
        )

    summary = _build_summary(
        cfg=cfg,
        fixed_records=fixed_records,
        moving_records=moving_records,
        correspondences=correspondences,
        landmarks=landmarks,
        closed_cells=closed_cells,
        outputs=paths,
        review_png=review_png,
    )
    paths["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    return ContourTearClosureResult(
        out_dir=out_dir,
        closed_cells_csv=paths["closed_cells_csv"],
        closed_contours_geojson=paths["closed_contours_geojson"],
        correspondence_csv=paths["correspondence_csv"],
        landmarks_csv=paths["landmarks_csv"],
        summary_json=paths["summary_json"],
        review_png=review_png,
    )


def _validate_config(cfg: ContourTearClosureConfig) -> None:
    if cfg.centroid_search_radius_um <= 0:
        raise ValueError("centroid_search_radius_um must be greater than 0.")
    if cfg.influence_radius_um <= 0:
        raise ValueError("influence_radius_um must be greater than 0.")
    if cfg.max_neighbors < 1:
        raise ValueError("max_neighbors must be at least 1.")
    if cfg.max_displacement_um <= 0:
        raise ValueError("max_displacement_um must be greater than 0.")
    if cfg.area_ratio_min <= 0 or cfg.area_ratio_max < cfg.area_ratio_min:
        raise ValueError("area_ratio_min/max must define a positive ordered range.")
    if cfg.min_pair_score < 0:
        raise ValueError("min_pair_score must be non-negative.")
    if cfg.min_motion_um < 0 or cfg.tear_motion_threshold_um < 0:
        raise ValueError("motion thresholds must be non-negative.")


def _result_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "closed_cells_csv": out_dir / "moving_contour_tear_closed_cells.csv",
        "closed_contours_geojson": out_dir / "moving_contour_tear_closed_contours.geojson",
        "correspondence_csv": out_dir / "contour_tear_closure_correspondences.csv",
        "landmarks_csv": out_dir / "contour_tear_closure_landmarks.csv",
        "summary_json": out_dir / "contour_tear_closure_summary.json",
        "review_png": out_dir / "contour_tear_closure_landmarks.png",
    }


def _read_geojson(path: PathLike) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _records_from_geojson(payload: dict[str, Any], group_property: str) -> list[_ContourRecord]:
    records: list[_ContourRecord] = []
    for index, feature in enumerate(payload.get("features", [])):
        props = feature.get("properties") or {}
        try:
            geom = make_valid(shape(feature.get("geometry")))
        except Exception:
            continue
        if geom.is_empty or geom.area <= 0:
            continue
        group = (
            props.get(group_property)
            or props.get("assigned_structure")
            or props.get("structure")
            or (props.get("classification") or {}).get("name")
            or "Unknown"
        )
        centroid = geom.centroid
        records.append(
            _ContourRecord(
                feature_index=index,
                group=str(group),
                name=str(props.get("name", f"feature_{index}")),
                geometry=geom,
                area=float(geom.area),
                centroid_x=float(centroid.x),
                centroid_y=float(centroid.y),
            )
        )
    return records


def _find_reciprocal_correspondences(
    fixed_records: list[_ContourRecord],
    moving_records: list[_ContourRecord],
    cfg: ContourTearClosureConfig,
) -> pd.DataFrame:
    fixed_by_group: dict[str, list[_ContourRecord]] = {}
    for record in fixed_records:
        fixed_by_group.setdefault(record.group, []).append(record)

    group_indexes: dict[str, tuple[STRtree, list[_ContourRecord], cKDTree]] = {}
    for group, records in fixed_by_group.items():
        geoms = [record.geometry for record in records]
        centroids = np.asarray([[r.centroid_x, r.centroid_y] for r in records], dtype=float)
        group_indexes[group] = (STRtree(geoms), records, cKDTree(centroids))

    pair_rows: list[dict[str, Any]] = []
    for moving in moving_records:
        if moving.group not in group_indexes:
            continue
        tree, fixed_group_records, centroid_tree = group_indexes[moving.group]
        candidate_indices: set[int] = set(
            int(item) for item in tree.query(moving.geometry, predicate="intersects")
        )
        centroid_hits = centroid_tree.query_ball_point(
            [moving.centroid_x, moving.centroid_y],
            r=float(cfg.centroid_search_radius_um),
        )
        candidate_indices.update(int(item) for item in centroid_hits)
        for candidate_index in sorted(candidate_indices):
            fixed = fixed_group_records[candidate_index]
            area_ratio = moving.area / fixed.area if fixed.area else math.inf
            if area_ratio < cfg.area_ratio_min or area_ratio > cfg.area_ratio_max:
                continue
            intersection_area = float(moving.geometry.intersection(fixed.geometry).area)
            union_area = moving.area + fixed.area - intersection_area
            dice = (
                2.0 * intersection_area / (moving.area + fixed.area)
                if moving.area + fixed.area
                else 0.0
            )
            iou = intersection_area / union_area if union_area else 0.0
            centroid_distance = math.hypot(
                fixed.centroid_x - moving.centroid_x,
                fixed.centroid_y - moving.centroid_y,
            )
            proximity = math.exp(
                -centroid_distance / max(float(cfg.centroid_search_radius_um), 1e-6)
            )
            area_similarity = math.exp(abs(math.log(max(area_ratio, 1e-12))) * -1.0)
            score = dice + 0.20 * proximity * area_similarity
            if score < cfg.min_pair_score:
                continue
            dx = fixed.centroid_x - moving.centroid_x
            dy = fixed.centroid_y - moving.centroid_y
            motion = math.hypot(dx, dy)
            pair_rows.append(
                {
                    "group": moving.group,
                    "fixed_feature_index": fixed.feature_index,
                    "moving_feature_index": moving.feature_index,
                    "fixed_name": fixed.name,
                    "moving_name": moving.name,
                    "fixed_centroid_x": fixed.centroid_x,
                    "fixed_centroid_y": fixed.centroid_y,
                    "moving_centroid_x": moving.centroid_x,
                    "moving_centroid_y": moving.centroid_y,
                    "dx_um": dx,
                    "dy_um": dy,
                    "motion_um": motion,
                    "fixed_area_um2": fixed.area,
                    "moving_area_um2": moving.area,
                    "area_ratio_moving_over_fixed": area_ratio,
                    "intersection_area_um2": intersection_area,
                    "dice": dice,
                    "iou": iou,
                    "centroid_distance_um": centroid_distance,
                    "score": score,
                }
            )

    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        return _empty_correspondence_frame()

    best_for_moving: dict[int, int] = {}
    best_for_fixed: dict[int, int] = {}
    for idx, row in pairs.iterrows():
        moving_index = int(row["moving_feature_index"])
        fixed_index = int(row["fixed_feature_index"])
        key = (float(row["score"]), float(row["dice"]), float(row["intersection_area_um2"]))
        if moving_index not in best_for_moving:
            best_for_moving[moving_index] = int(idx)
        else:
            current = pairs.loc[best_for_moving[moving_index]]
            current_key = (
                float(current["score"]),
                float(current["dice"]),
                float(current["intersection_area_um2"]),
            )
            if key > current_key:
                best_for_moving[moving_index] = int(idx)
        if fixed_index not in best_for_fixed:
            best_for_fixed[fixed_index] = int(idx)
        else:
            current = pairs.loc[best_for_fixed[fixed_index]]
            current_key = (
                float(current["score"]),
                float(current["dice"]),
                float(current["intersection_area_um2"]),
            )
            if key > current_key:
                best_for_fixed[fixed_index] = int(idx)

    reciprocal_indices = []
    for moving_index, idx in best_for_moving.items():
        fixed_index = int(pairs.loc[idx, "fixed_feature_index"])
        if best_for_fixed.get(fixed_index) == idx:
            reciprocal_indices.append(idx)
    out = pairs.loc[sorted(reciprocal_indices)].copy().reset_index(drop=True)
    if out.empty:
        return _empty_correspondence_frame()
    out["used_for_closure"] = out["motion_um"] >= float(cfg.min_motion_um)
    out["tear_like_motion"] = out["motion_um"] >= float(cfg.tear_motion_threshold_um)
    return out


def _empty_correspondence_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "group",
            "fixed_feature_index",
            "moving_feature_index",
            "fixed_name",
            "moving_name",
            "fixed_centroid_x",
            "fixed_centroid_y",
            "moving_centroid_x",
            "moving_centroid_y",
            "dx_um",
            "dy_um",
            "motion_um",
            "fixed_area_um2",
            "moving_area_um2",
            "area_ratio_moving_over_fixed",
            "intersection_area_um2",
            "dice",
            "iou",
            "centroid_distance_um",
            "score",
            "used_for_closure",
            "tear_like_motion",
        ]
    )


def _landmarks_from_correspondences(
    correspondences: pd.DataFrame,
    cfg: ContourTearClosureConfig,
) -> pd.DataFrame:
    if correspondences.empty:
        return pd.DataFrame(
            columns=[
                "landmark_kind",
                "group",
                "fixed_feature_index",
                "moving_feature_index",
                "src_x",
                "src_y",
                "dst_x",
                "dst_y",
                "dx",
                "dy",
                "motion_um",
                "used_for_transform",
            ]
        )
    rows = []
    for row in correspondences.itertuples(index=False):
        rows.append(
            {
                "landmark_kind": "reciprocal_contour_centroid",
                "group": row.group,
                "fixed_feature_index": int(row.fixed_feature_index),
                "moving_feature_index": int(row.moving_feature_index),
                "src_x": float(row.moving_centroid_x),
                "src_y": float(row.moving_centroid_y),
                "dst_x": float(row.fixed_centroid_x),
                "dst_y": float(row.fixed_centroid_y),
                "dx": float(row.dx_um),
                "dy": float(row.dy_um),
                "motion_um": float(row.motion_um),
                "used_for_transform": bool(row.used_for_closure),
                "tear_like_motion": bool(row.tear_like_motion),
            }
        )
    landmarks = pd.DataFrame(rows)
    return landmarks.loc[landmarks["used_for_transform"].astype(bool)].reset_index(drop=True)


def _field_from_landmarks(
    landmarks: pd.DataFrame,
    cfg: ContourTearClosureConfig,
) -> _LocalTranslationField:
    if landmarks.empty:
        return _LocalTranslationField(
            anchor_xy=np.zeros((0, 2), dtype=float),
            displacement_xy=np.zeros((0, 2), dtype=float),
            influence_radius_um=cfg.influence_radius_um,
            max_neighbors=cfg.max_neighbors,
            max_displacement_um=cfg.max_displacement_um,
        )
    anchor_xy = landmarks[["src_x", "src_y"]].to_numpy(dtype=float)
    displacement = landmarks[["dx", "dy"]].to_numpy(dtype=float)
    norms = np.linalg.norm(displacement, axis=1)
    too_large = norms > float(cfg.max_displacement_um)
    if too_large.any():
        displacement[too_large] *= (float(cfg.max_displacement_um) / norms[too_large])[:, None]
    return _LocalTranslationField(
        anchor_xy=anchor_xy,
        displacement_xy=displacement,
        influence_radius_um=cfg.influence_radius_um,
        max_neighbors=cfg.max_neighbors,
        max_displacement_um=cfg.max_displacement_um,
    )


def _warp_cells(
    moving_cells_csv: PathLike,
    field: _LocalTranslationField,
    cfg: ContourTearClosureConfig,
) -> pd.DataFrame:
    cells = pd.read_csv(Path(moving_cells_csv).expanduser())
    missing = {cfg.moving_x_column, cfg.moving_y_column} - set(cells.columns)
    if missing:
        raise ValueError(f"moving_cells_csv missing coordinate columns: {sorted(missing)}")
    xy = cells[[cfg.moving_x_column, cfg.moving_y_column]].to_numpy(dtype=float)
    disp = field.displacement(xy)
    prefix = cfg.output_coordinate_prefix
    cells[f"x_{prefix}_um"] = xy[:, 0] + disp[:, 0]
    cells[f"y_{prefix}_um"] = xy[:, 1] + disp[:, 1]
    cells[f"dx_{prefix}_um"] = disp[:, 0]
    cells[f"dy_{prefix}_um"] = disp[:, 1]
    return cells


def _warp_geojson(
    payload: dict[str, Any],
    field: _LocalTranslationField,
    cfg: ContourTearClosureConfig,
) -> dict[str, Any]:
    out = copy.deepcopy(payload)

    def _warp(x, y, z=None):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        flat = np.column_stack([x_arr.reshape(-1), y_arr.reshape(-1)])
        warped = field.warp(flat)
        wx = warped[:, 0].reshape(x_arr.shape)
        wy = warped[:, 1].reshape(y_arr.shape)
        if wx.ndim == 0:
            if z is None:
                return float(wx), float(wy)
            return float(wx), float(wy), z
        if z is None:
            return wx, wy
        return wx, wy, z

    for feature in out.get("features", []):
        try:
            geom = shape(feature.get("geometry"))
            warped = make_valid(shapely_transform(_warp, geom))
        except Exception:
            continue
        feature["geometry"] = mapping(warped)
        props = feature.setdefault("properties", {})
        props["tear_closure_method"] = "contour_weighted_local_translation"
        props["tear_closure_coordinate_prefix"] = cfg.output_coordinate_prefix
    return out


def _build_summary(
    *,
    cfg: ContourTearClosureConfig,
    fixed_records: list[_ContourRecord],
    moving_records: list[_ContourRecord],
    correspondences: pd.DataFrame,
    landmarks: pd.DataFrame,
    closed_cells: pd.DataFrame,
    outputs: dict[str, Path],
    review_png: Path | None,
) -> dict[str, Any]:
    matched_fixed = (
        set(correspondences["fixed_feature_index"].astype(int).tolist())
        if not correspondences.empty
        else set()
    )
    matched_moving = (
        set(correspondences["moving_feature_index"].astype(int).tolist())
        if not correspondences.empty
        else set()
    )
    prefix = cfg.output_coordinate_prefix
    dx_col = f"dx_{prefix}_um"
    dy_col = f"dy_{prefix}_um"
    displacement = (
        np.hypot(closed_cells[dx_col].to_numpy(dtype=float), closed_cells[dy_col].to_numpy(dtype=float))
        if dx_col in closed_cells.columns and dy_col in closed_cells.columns
        else np.asarray([], dtype=float)
    )
    return {
        "method": "contour_tear_closure",
        "method_family": "contour_guided_local_translation_no_tps",
        "description": (
            "Same-group reciprocal contour correspondences estimate local "
            "translations for tear-like moving-slice gaps. Unmatched fixed "
            "contours are reported as possible missing tissue and are not synthesized."
        ),
        "config": _jsonable_config(cfg),
        "inputs": {
            "fixed_geojson": str(Path(cfg.fixed_geojson)),
            "moving_aligned_geojson": str(Path(cfg.moving_aligned_geojson)),
            "moving_cells_csv": str(Path(cfg.moving_cells_csv)),
            "fixed_cells_csv": str(Path(cfg.fixed_cells_csv)) if cfg.fixed_cells_csv else None,
        },
        "contours": {
            "fixed_total": len(fixed_records),
            "moving_total": len(moving_records),
            "reciprocal_correspondence_pairs": int(len(correspondences)),
            "fixed_matched": int(len(matched_fixed)),
            "moving_matched": int(len(matched_moving)),
            "fixed_unmatched_possible_missing": int(len(fixed_records) - len(matched_fixed)),
            "moving_unmatched_passive": int(len(moving_records) - len(matched_moving)),
            "tear_like_motion_pair_count": int(
                correspondences["tear_like_motion"].astype(bool).sum()
                if "tear_like_motion" in correspondences
                else 0
            ),
        },
        "landmarks": {
            "used_local_translation_anchor_count": int(len(landmarks)),
            "median_anchor_motion_um": _safe_median(landmarks.get("motion_um")),
            "p90_anchor_motion_um": _safe_quantile(landmarks.get("motion_um"), 0.90),
        },
        "cells": {
            "moving_cell_count": int(len(closed_cells)),
            "median_cell_displacement_um": float(np.median(displacement)) if len(displacement) else 0.0,
            "p90_cell_displacement_um": float(np.quantile(displacement, 0.90))
            if len(displacement)
            else 0.0,
            "moved_gt_10um_fraction": float((displacement > 10.0).mean())
            if len(displacement)
            else 0.0,
        },
        "outputs": {
            "closed_cells_csv": str(outputs["closed_cells_csv"]),
            "closed_contours_geojson": str(outputs["closed_contours_geojson"]),
            "correspondence_csv": str(outputs["correspondence_csv"]),
            "landmarks_csv": str(outputs["landmarks_csv"]),
            "summary_json": str(outputs["summary_json"]),
            "review_png": str(review_png) if review_png is not None else None,
        },
    }


def _safe_median(values: pd.Series | None) -> float:
    if values is None or len(values) == 0:
        return 0.0
    return float(np.median(pd.to_numeric(values, errors="coerce").dropna()))


def _safe_quantile(values: pd.Series | None, q: float) -> float:
    if values is None or len(values) == 0:
        return 0.0
    return float(np.quantile(pd.to_numeric(values, errors="coerce").dropna(), q))


def _jsonable_config(cfg: ContourTearClosureConfig) -> dict[str, Any]:
    data = asdict(cfg)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def _write_landmark_review_png(
    *,
    fixed_payload: dict[str, Any],
    moving_payload: dict[str, Any],
    closed_payload: dict[str, Any],
    landmarks: pd.DataFrame,
    out_png: Path,
    dpi: int,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    _plot_geojson(axes[0], fixed_payload, "#111111", 0.65, 0.85)
    _plot_geojson(axes[0], moving_payload, "#e53935", 0.65, 0.75)
    _plot_landmark_links(axes[0], landmarks, source_cols=("src_x", "src_y"), target_cols=("dst_x", "dst_y"))
    axes[0].set_title("Before tear closure")

    _plot_geojson(axes[1], fixed_payload, "#111111", 0.65, 0.85)
    _plot_geojson(axes[1], closed_payload, "#1b9e77", 0.65, 0.75)
    _plot_landmark_links(axes[1], landmarks, source_cols=("dst_x", "dst_y"), target_cols=("dst_x", "dst_y"))
    axes[1].set_title("After contour-guided local translation")

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, color="#eeeeee", linewidth=0.5, alpha=0.7)
    _set_shared_limits(axes, [fixed_payload, moving_payload, closed_payload])
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _iter_polygons(payload: dict[str, Any]) -> Iterable[Polygon]:
    for feature in payload.get("features", []):
        try:
            geom = shape(feature.get("geometry"))
        except Exception:
            continue
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            yield geom
        elif isinstance(geom, MultiPolygon):
            yield from geom.geoms
        elif isinstance(geom, GeometryCollection):
            for part in geom.geoms:
                if isinstance(part, Polygon):
                    yield part
                elif isinstance(part, MultiPolygon):
                    yield from part.geoms


def _plot_geojson(
    ax: plt.Axes,
    payload: dict[str, Any],
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    for polygon in _iter_polygons(payload):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)
        for interior in polygon.interiors:
            hx, hy = interior.xy
            ax.plot(hx, hy, color=color, linewidth=max(linewidth * 0.65, 0.25), alpha=alpha * 0.7)


def _plot_landmark_links(
    ax: plt.Axes,
    landmarks: pd.DataFrame,
    *,
    source_cols: tuple[str, str],
    target_cols: tuple[str, str],
) -> None:
    if landmarks.empty:
        return
    sx, sy = source_cols
    tx, ty = target_cols
    if not {sx, sy, tx, ty}.issubset(landmarks.columns):
        return
    for row in landmarks.itertuples(index=False):
        ax.plot(
            [float(getattr(row, sx)), float(getattr(row, tx))],
            [float(getattr(row, sy)), float(getattr(row, ty))],
            color="#7c3aed",
            linewidth=0.55,
            alpha=0.45,
        )


def _set_shared_limits(axes: Iterable[plt.Axes], payloads: Iterable[dict[str, Any]]) -> None:
    union = unary_union([polygon for payload in payloads for polygon in _iter_polygons(payload)])
    if union.is_empty:
        return
    minx, miny, maxx, maxy = union.bounds
    pad_x = max((maxx - minx) * 0.04, 1.0)
    pad_y = max((maxy - miny) * 0.04, 1.0)
    for ax in axes:
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(maxy + pad_y, miny - pad_y)
