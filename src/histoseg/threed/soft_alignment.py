from __future__ import annotations

import copy
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

PathLike = Union[str, Path]


class ThreeDFeatureUnavailableError(NotImplementedError):
    """Raised when a reserved 3D Analysis feature is invoked before release."""


@dataclass(frozen=True)
class ThreeDContourReconstructionConfig:
    """Configuration for pairwise 3D contour soft alignment.

    The moving GeoJSON is expected to have already been hard-aligned to the
    fixed sample with a similarity transform.
    """

    fixed_geojson: PathLike
    moving_hard_aligned_geojson: PathLike
    out_dir: PathLike = "outputs/3d_soft_alignment"
    group_property: str = "structure"
    sampling_distance_um: float = 50.0
    max_landmark_distance_um: float = 180.0
    landmarks_per_structure: int | None = 260
    diagnostic_structure_landmarks: int | None = 620
    landmark_candidate_count: int = 8
    landmark_candidate_spacing_um: float | None = None
    landmark_normal_weight_um: float = 0.0
    landmark_normal_step_um: float | None = None
    rbf_kernel: str = "thin_plate_spline"
    rbf_neighbors: int | None = 96
    rbf_smoothing: float | str = 1e-4
    rbf_smoothing_candidates: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2)
    topology_grid_size: int = 24
    topology_min_area_ratio: float = 0.5
    topology_max_area_ratio: float = 2.0
    diagnostic_structure: str | None = "Structure 5"
    dpi: int = 200
    save_preview_png: bool = True
    # Curvature-weighted landmark sampling: fraction of boundary length sampled at high-curvature
    # regions. 0.0 disables, values between 0 and 1 control how much extra density high-curvature
    # areas receive.
    curvature_landmark_weight: float = 0.5
    # Mutual nearest-neighbour consistency check to remove cross-matched landmarks.
    mutual_nn_check: bool = True
    # Number of ICP-style iterative refinement passes after the initial TPS fit (1 = no ICP).
    icp_iterations: int = 2
    # Number of zero-displacement anchors placed on the padded convex-hull perimeter.
    zero_anchor_count: int = 16
    # Reject boundary landmarks whose displacement deviates by more than this many MAD units from
    # their neighbourhood median. None disables outlier rejection.
    landmark_outlier_mad_threshold: float | None = 3.5
    # Accept soft alignment per structure independently instead of globally.
    per_structure_soft_acceptance: bool = True


@dataclass
class ThreeDContourReconstructionResult:
    """Artifacts produced by pairwise 3D contour soft alignment."""

    out_dir: Path
    soft_aligned_geojson: Path
    metrics_csv: Path
    landmarks_csv: Path
    summary_json: Path
    diagnostic_report_png: Path | None = None
    residuals_csv: Path | None = None
    overlay_before_png: Path | None = None
    overlay_after_png: Path | None = None
    landmarks_qc_png: Path | None = None


@dataclass(frozen=True)
class _FeatureRecord:
    feature: dict[str, Any]
    group: str
    geometry: Any


@dataclass(frozen=True)
class _TPSModel:
    interpolator: RBFInterpolator
    center_xy: np.ndarray
    scale: float

    def displacement(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if xy.ndim == 1:
            xy = xy.reshape(1, 2)
        normalized = (xy - self.center_xy) / self.scale
        return np.asarray(self.interpolator(normalized), dtype=float) * self.scale

    def warp(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, dtype=float) + self.displacement(xy)


@dataclass(frozen=True)
class _BoundaryCandidates:
    xy: np.ndarray
    normals: np.ndarray
    tree: cKDTree


def run_3d_contour_reconstruction(
    cfg: ThreeDContourReconstructionConfig,
) -> ThreeDContourReconstructionResult:
    """Run conservative TPS soft alignment for a hard-aligned contour pair."""

    _validate_config(cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_geojson = _read_geojson(Path(cfg.fixed_geojson))
    moving_geojson = _read_geojson(Path(cfg.moving_hard_aligned_geojson))
    fixed_records = _records_from_geojson(fixed_geojson, cfg.group_property, "fixed")
    moving_records = _records_from_geojson(moving_geojson, cfg.group_property, "moving")

    landmarks = _generate_landmarks(fixed_records, moving_records, cfg)
    if landmarks.empty:
        raise ValueError(
            "No TPS landmarks were generated. Check group_property, sampling distance, "
            "and max_landmark_distance_um."
        )

    boundary_landmarks = landmarks.loc[landmarks["kind"] == "boundary"].copy()
    if len(boundary_landmarks) < 3:
        raise ValueError("At least three boundary landmarks are required for TPS alignment.")

    anchors = _zero_anchor_landmarks(fixed_records + moving_records, cfg.zero_anchor_count)
    model_landmarks = pd.concat([boundary_landmarks, anchors], ignore_index=True)
    model = _fit_tps_model(model_landmarks, cfg)

    soft_geojson, soft_records, geometry_status = _warp_geojson(
        moving_geojson,
        moving_records,
        model,
        cfg.group_property,
    )

    # ICP-style iterative refinement: re-generate landmarks from warped positions and apply
    # a correction TPS model, stopping early if IoU stops improving.
    icp_iterations_executed = 1
    if cfg.icp_iterations > 1:
        soft_geojson, soft_records, geometry_status, icp_iterations_executed = _icp_refine(
            fixed_records=fixed_records,
            initial_soft_geojson=soft_geojson,
            initial_soft_records=soft_records,
            cfg=cfg,
        )

    topology = _check_tps_topology(moving_records, model, cfg)

    soft_geojson_path = out_dir / "soft_aligned_contours.geojson"
    soft_geojson_path.write_text(json.dumps(soft_geojson, ensure_ascii=False), encoding="utf-8")

    landmarks_path = out_dir / "soft_tps_landmarks.csv"
    model_landmarks.to_csv(landmarks_path, index=False)

    residuals = _compute_residuals(boundary_landmarks, model)
    residuals_path = out_dir / "soft_tps_diagnostic_residuals.csv"
    residuals.to_csv(residuals_path, index=False)

    metrics = _compute_metrics(fixed_records, moving_records, soft_records)
    metrics_path = out_dir / "soft_tps_alignment_metrics.csv"
    metrics["per_structure"].to_csv(metrics_path, index=False)

    outputs: dict[str, Path | None] = {
        "overlay_before_png": None,
        "overlay_after_png": None,
        "landmarks_qc_png": None,
        "diagnostic_report_png": None,
    }
    if cfg.save_preview_png:
        outputs = _write_qc_figures(
            fixed_records=fixed_records,
            moving_records=moving_records,
            soft_records=soft_records,
            landmarks=boundary_landmarks,
            residuals=residuals,
            model=model,
            cfg=cfg,
            out_dir=out_dir,
        )

    summary = _build_summary(
        cfg=cfg,
        fixed_records=fixed_records,
        moving_records=moving_records,
        soft_records=soft_records,
        boundary_landmarks=boundary_landmarks,
        anchors=anchors,
        metrics=metrics,
        residuals=residuals,
        geometry_status=geometry_status,
        topology=topology,
        model=model,
        icp_iterations_executed=icp_iterations_executed,
        out_dir=out_dir,
        soft_geojson_path=soft_geojson_path,
        landmarks_path=landmarks_path,
        residuals_path=residuals_path,
        metrics_path=metrics_path,
        outputs=outputs,
    )
    summary_path = out_dir / "soft_tps_alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return ThreeDContourReconstructionResult(
        out_dir=out_dir,
        soft_aligned_geojson=soft_geojson_path,
        metrics_csv=metrics_path,
        landmarks_csv=landmarks_path,
        summary_json=summary_path,
        diagnostic_report_png=outputs["diagnostic_report_png"],
        residuals_csv=residuals_path,
        overlay_before_png=outputs["overlay_before_png"],
        overlay_after_png=outputs["overlay_after_png"],
        landmarks_qc_png=outputs["landmarks_qc_png"],
    )


def _validate_config(cfg: ThreeDContourReconstructionConfig) -> None:
    if cfg.sampling_distance_um <= 0:
        raise ValueError("sampling_distance_um must be greater than 0.")
    if cfg.max_landmark_distance_um <= 0:
        raise ValueError("max_landmark_distance_um must be greater than 0.")
    if cfg.landmarks_per_structure is not None and cfg.landmarks_per_structure < 3:
        raise ValueError("landmarks_per_structure must be at least 3 when provided.")
    if (
        cfg.diagnostic_structure_landmarks is not None
        and cfg.diagnostic_structure_landmarks < 3
    ):
        raise ValueError("diagnostic_structure_landmarks must be at least 3 when provided.")
    if cfg.rbf_neighbors is not None and cfg.rbf_neighbors < 3:
        raise ValueError("rbf_neighbors must be at least 3 when provided.")
    if isinstance(cfg.rbf_smoothing, str):
        if cfg.rbf_smoothing != "auto":
            raise ValueError("rbf_smoothing must be a non-negative float or 'auto'.")
    elif cfg.rbf_smoothing < 0:
        raise ValueError("rbf_smoothing must be non-negative.")
    if cfg.landmark_candidate_count < 1:
        raise ValueError("landmark_candidate_count must be at least 1.")
    if (
        cfg.landmark_candidate_spacing_um is not None
        and cfg.landmark_candidate_spacing_um <= 0
    ):
        raise ValueError("landmark_candidate_spacing_um must be positive when provided.")
    if cfg.landmark_normal_weight_um < 0:
        raise ValueError("landmark_normal_weight_um must be non-negative.")
    if cfg.landmark_normal_step_um is not None and cfg.landmark_normal_step_um <= 0:
        raise ValueError("landmark_normal_step_um must be positive when provided.")
    if cfg.topology_grid_size < 0:
        raise ValueError("topology_grid_size must be non-negative.")
    if cfg.topology_min_area_ratio <= 0:
        raise ValueError("topology_min_area_ratio must be greater than 0.")
    if cfg.topology_max_area_ratio <= cfg.topology_min_area_ratio:
        raise ValueError("topology_max_area_ratio must be greater than topology_min_area_ratio.")
    if cfg.curvature_landmark_weight < 0 or cfg.curvature_landmark_weight > 1:
        raise ValueError("curvature_landmark_weight must be between 0 and 1.")
    if cfg.icp_iterations < 1:
        raise ValueError("icp_iterations must be at least 1.")
    if cfg.zero_anchor_count < 4:
        raise ValueError("zero_anchor_count must be at least 4.")
    if cfg.landmark_outlier_mad_threshold is not None and cfg.landmark_outlier_mad_threshold <= 0:
        raise ValueError("landmark_outlier_mad_threshold must be positive when provided.")


def _read_geojson(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise ValueError(f"{path} must be a GeoJSON FeatureCollection.")
    return data


def _records_from_geojson(
    geojson: dict[str, Any],
    group_property: str,
    role: str,
) -> list[_FeatureRecord]:
    records: list[_FeatureRecord] = []
    for index, feature in enumerate(geojson["features"]):
        properties = feature.get("properties") or {}
        group = _feature_group(properties, group_property)
        if group is None:
            raise ValueError(
                f"{role} feature {index} is missing properties['{group_property}']."
            )
        geom = shape(feature["geometry"])
        if geom.is_empty:
            continue
        records.append(
            _FeatureRecord(
                feature=feature,
                group=str(group),
                geometry=geom,
            )
        )
    if not records:
        raise ValueError(f"{role} GeoJSON contains no non-empty features.")
    return records


def _generate_landmarks(
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    cfg: ThreeDContourReconstructionConfig,
) -> pd.DataFrame:
    if cfg.landmark_normal_weight_um > 0:
        landmarks = _generate_normal_aware_landmarks(fixed_records, moving_records, cfg)
        if len(landmarks) >= 3:
            landmarks = _postprocess_landmarks(landmarks, fixed_records, moving_records, cfg)
            if landmarks.empty:
                return landmarks
            return _limit_landmarks_per_group(landmarks, cfg)

    rows: list[dict[str, float | str]] = []
    fixed_by_group = _union_by_group(fixed_records)
    moving_by_group = _union_by_group(moving_records)
    for group in sorted(set(fixed_by_group) & set(moving_by_group)):
        fixed_boundary = fixed_by_group[group].boundary
        moving_boundary = moving_by_group[group].boundary
        if fixed_boundary.is_empty or moving_boundary.is_empty:
            continue
        distances = np.arange(0.0, float(moving_boundary.length), cfg.sampling_distance_um)
        for distance in distances:
            src = moving_boundary.interpolate(float(distance))
            dst = fixed_boundary.interpolate(fixed_boundary.project(src))
            source_distance = float(src.distance(dst))
            if source_distance > cfg.max_landmark_distance_um:
                continue
            rows.append(
                {
                    "kind": "boundary",
                    "structure": group,
                    "src_x": float(src.x),
                    "src_y": float(src.y),
                    "dst_x": float(dst.x),
                    "dst_y": float(dst.y),
                    "dx": float(dst.x - src.x),
                    "dy": float(dst.y - src.y),
                    "source_distance_um": source_distance,
                    "match_cost_um": source_distance,
                    "normal_dot_abs": math.nan,
                    "match_method": "nearest_projection",
                }
            )
    landmarks = pd.DataFrame(rows)
    if landmarks.empty:
        return landmarks
    landmarks = _postprocess_landmarks(landmarks, fixed_records, moving_records, cfg)
    if landmarks.empty:
        return landmarks
    return _limit_landmarks_per_group(landmarks, cfg)


def _generate_normal_aware_landmarks(
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    cfg: ThreeDContourReconstructionConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    fixed_by_group = _union_by_group(fixed_records)
    moving_by_group = _union_by_group(moving_records)
    candidate_spacing = cfg.landmark_candidate_spacing_um
    if candidate_spacing is None:
        candidate_spacing = max(min(cfg.sampling_distance_um / 2.0, 25.0), 1e-6)

    for group in sorted(set(fixed_by_group) & set(moving_by_group)):
        fixed_boundary = fixed_by_group[group].boundary
        moving_boundary = moving_by_group[group].boundary
        if fixed_boundary.is_empty or moving_boundary.is_empty:
            continue
        fixed_candidates = _sample_boundary_candidates(
            fixed_boundary,
            spacing_um=candidate_spacing,
            normal_step_um=cfg.landmark_normal_step_um or max(candidate_spacing / 2.0, 1e-6),
        )
        if fixed_candidates is None:
            continue
        for src_xy, src_normal in _iter_boundary_samples(
            moving_boundary,
            spacing_um=cfg.sampling_distance_um,
            normal_step_um=cfg.landmark_normal_step_um or max(cfg.sampling_distance_um / 2.0, 1e-6),
        ):
            match = _match_normal_aware_candidate(src_xy, src_normal, fixed_candidates, cfg)
            if match is None:
                continue
            dst_xy, distance, cost, normal_dot_abs, method = match
            if distance > cfg.max_landmark_distance_um:
                continue
            rows.append(
                {
                    "kind": "boundary",
                    "structure": group,
                    "src_x": float(src_xy[0]),
                    "src_y": float(src_xy[1]),
                    "dst_x": float(dst_xy[0]),
                    "dst_y": float(dst_xy[1]),
                    "dx": float(dst_xy[0] - src_xy[0]),
                    "dy": float(dst_xy[1] - src_xy[1]),
                    "source_distance_um": float(distance),
                    "match_cost_um": float(cost),
                    "normal_dot_abs": float(normal_dot_abs) if np.isfinite(normal_dot_abs) else math.nan,
                    "match_method": method,
                }
            )
    return pd.DataFrame(rows)


def _sample_boundary_candidates(
    boundary: Any,
    *,
    spacing_um: float,
    normal_step_um: float,
) -> _BoundaryCandidates | None:
    xy_values: list[np.ndarray] = []
    normal_values: list[np.ndarray] = []
    for xy, normal in _iter_boundary_samples(
        boundary,
        spacing_um=spacing_um,
        normal_step_um=normal_step_um,
    ):
        xy_values.append(xy)
        normal_values.append(normal)
    if not xy_values:
        return None
    xy = np.vstack(xy_values).astype(float)
    normals = np.vstack(normal_values).astype(float)
    return _BoundaryCandidates(xy=xy, normals=normals, tree=cKDTree(xy))


def _iter_boundary_samples(
    boundary: Any,
    *,
    spacing_um: float,
    normal_step_um: float,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for line in _iter_line_parts(boundary):
        length = float(line.length)
        if length <= 0:
            continue
        distances = np.arange(0.0, length, max(spacing_um, 1e-6))
        if len(distances) == 0:
            distances = np.array([0.0])
        for distance in distances:
            point = line.interpolate(float(distance))
            normal = _line_normal(line, float(distance), normal_step_um)
            yield np.array([float(point.x), float(point.y)], dtype=float), normal


def _iter_line_parts(geom: Any) -> Iterable[Any]:
    if geom.is_empty:
        return
    if geom.geom_type in {"LineString", "LinearRing"}:
        yield geom
        return
    if hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from _iter_line_parts(part)


def _line_normal(line: Any, distance: float, step: float) -> np.ndarray:
    length = float(line.length)
    if length <= 0:
        return np.array([math.nan, math.nan], dtype=float)
    coords = list(line.coords)
    closed = len(coords) > 2 and np.allclose(coords[0], coords[-1])
    if closed:
        d0 = (distance - step) % length
        d1 = (distance + step) % length
    else:
        d0 = min(max(distance - step, 0.0), length)
        d1 = min(max(distance + step, 0.0), length)
    p0 = line.interpolate(float(d0))
    p1 = line.interpolate(float(d1))
    tangent = np.array([float(p1.x - p0.x), float(p1.y - p0.y)], dtype=float)
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-12:
        return np.array([math.nan, math.nan], dtype=float)
    tangent /= norm
    return np.array([-tangent[1], tangent[0]], dtype=float)


def _match_normal_aware_candidate(
    src_xy: np.ndarray,
    src_normal: np.ndarray,
    fixed_candidates: _BoundaryCandidates,
    cfg: ThreeDContourReconstructionConfig,
) -> tuple[np.ndarray, float, float, float, str] | None:
    k = min(int(cfg.landmark_candidate_count), len(fixed_candidates.xy))
    distances, indexes = fixed_candidates.tree.query(src_xy, k=k)
    distances = np.atleast_1d(distances).astype(float)
    indexes = np.atleast_1d(indexes).astype(int)
    finite = np.isfinite(distances) & (indexes >= 0) & (indexes < len(fixed_candidates.xy))
    if not np.any(finite):
        return None

    candidate_distances = distances[finite]
    candidate_indexes = indexes[finite]
    candidate_normals = fixed_candidates.normals[candidate_indexes]
    costs = candidate_distances.copy()
    normal_dots = np.full(len(candidate_indexes), math.nan, dtype=float)
    if np.all(np.isfinite(src_normal)):
        valid_normals = np.isfinite(candidate_normals).all(axis=1)
        if np.any(valid_normals):
            dots = np.abs(candidate_normals[valid_normals] @ src_normal)
            dots = np.clip(dots, 0.0, 1.0)
            normal_dots[valid_normals] = dots
            costs[valid_normals] += cfg.landmark_normal_weight_um * (1.0 - dots)

    best_position = int(np.argmin(costs))
    best_index = int(candidate_indexes[best_position])
    return (
        fixed_candidates.xy[best_index],
        float(candidate_distances[best_position]),
        float(costs[best_position]),
        float(normal_dots[best_position]),
        "normal_aware_kdtree",
    )


def _limit_landmarks_per_group(
    landmarks: pd.DataFrame,
    cfg: ThreeDContourReconstructionConfig,
) -> pd.DataFrame:
    if cfg.landmarks_per_structure is None and cfg.diagnostic_structure_landmarks is None:
        return landmarks.reset_index(drop=True)

    limited: list[pd.DataFrame] = []
    for structure, group_df in landmarks.groupby("structure", sort=True):
        target = cfg.landmarks_per_structure
        if structure == cfg.diagnostic_structure and cfg.diagnostic_structure_landmarks is not None:
            target = cfg.diagnostic_structure_landmarks
        if target is None or len(group_df) <= target:
            limited.append(group_df)
            continue
        indexes = np.linspace(0, len(group_df) - 1, target, dtype=int)
        limited.append(group_df.iloc[np.unique(indexes)])
    return pd.concat(limited, ignore_index=True)


def _zero_anchor_landmarks(
    records: list[_FeatureRecord],
    zero_anchor_count: int = 16,
) -> pd.DataFrame:
    """Place zero-displacement anchor landmarks on the padded convex-hull perimeter.

    Anchors are evenly spaced along the exterior of the convex hull of all record
    geometries, buffered outward by 15 % of the characteristic span. This constrains
    the TPS to near-identity far from the data region, replacing the previous
    hard-coded 8-point bounding-box corners with a denser, shape-adaptive ring.
    """
    all_geom = unary_union([r.geometry for r in records])
    minx, miny, maxx, maxy = all_geom.bounds
    pad = 0.15 * max(maxx - minx, maxy - miny, 1.0)
    try:
        hull = all_geom.convex_hull
        boundary = hull.buffer(pad, join_style="mitre").exterior
    except Exception:
        from shapely.geometry import box as _box

        boundary = _box(minx - pad, miny - pad, maxx + pad, maxy + pad).exterior

    count = max(int(zero_anchor_count), 4)
    length = float(boundary.length)
    step = length / count
    points = []
    for i in range(count):
        pt = boundary.interpolate(i * step)
        points.append((float(pt.x), float(pt.y)))

    return pd.DataFrame(
        [
            {
                "kind": "anchor",
                "structure": "__fixed_anchor__",
                "src_x": float(x),
                "src_y": float(y),
                "dst_x": float(x),
                "dst_y": float(y),
                "dx": 0.0,
                "dy": 0.0,
                "source_distance_um": 0.0,
            }
            for x, y in points
        ]
    )


def _fit_tps_model(
    landmarks: pd.DataFrame,
    cfg: ThreeDContourReconstructionConfig,
) -> _TPSModel:
    src = landmarks[["src_x", "src_y"]].to_numpy(dtype=float)
    dst = landmarks[["dst_x", "dst_y"]].to_numpy(dtype=float)
    center = src.mean(axis=0)
    scale = float(np.ptp(src, axis=0).max())
    if scale <= 0:
        raise ValueError("TPS landmarks must span a non-zero coordinate range.")
    src_norm = (src - center) / scale
    displacement_norm = (dst - src) / scale
    neighbors = None
    if cfg.rbf_neighbors is not None and len(src_norm) > cfg.rbf_neighbors:
        neighbors = cfg.rbf_neighbors
    if isinstance(cfg.rbf_smoothing, str) and cfg.rbf_smoothing == "auto":
        smoothing = _select_rbf_smoothing_cv(src_norm, displacement_norm, cfg, neighbors)
    else:
        smoothing = float(cfg.rbf_smoothing)
    interpolator = RBFInterpolator(
        src_norm,
        displacement_norm,
        kernel=cfg.rbf_kernel,
        smoothing=smoothing,
        neighbors=neighbors,
    )
    return _TPSModel(interpolator=interpolator, center_xy=center, scale=scale)


def _check_tps_topology(
    moving_records: list[_FeatureRecord],
    model: Any,
    cfg: ThreeDContourReconstructionConfig,
) -> dict[str, Any]:
    if cfg.topology_grid_size <= 1:
        return {
            "enabled": False,
            "valid": True,
            "reason": "disabled",
        }
    moving_union = unary_union([record.geometry for record in moving_records])
    if moving_union.is_empty:
        return {
            "enabled": True,
            "valid": False,
            "reason": "empty_moving_geometry",
        }

    minx, miny, maxx, maxy = moving_union.bounds
    width = max(maxx - minx, 1e-9)
    height = max(maxy - miny, 1e-9)
    x_values = np.linspace(minx, maxx, int(cfg.topology_grid_size))
    y_values = np.linspace(miny, maxy, int(cfg.topology_grid_size))
    xx, yy = np.meshgrid(x_values, y_values)
    original = np.column_stack([xx.ravel(), yy.ravel()])
    warped = np.asarray(model.warp(original), dtype=float).reshape(xx.shape + (2,))
    warped_x = warped[:, :, 0]
    warped_y = warped[:, :, 1]

    base_area = (width / (len(x_values) - 1)) * (height / (len(y_values) - 1))
    ratios: list[float] = []
    folded = 0
    compressed = 0
    expanded = 0
    checked = 0
    for y_index in range(len(y_values) - 1):
        for x_index in range(len(x_values) - 1):
            center_x = float((x_values[x_index] + x_values[x_index + 1]) / 2.0)
            center_y = float((y_values[y_index] + y_values[y_index + 1]) / 2.0)
            if not _geometry_contains_xy(moving_union, center_x, center_y):
                continue
            polygon_xy = np.array(
                [
                    [warped_x[y_index, x_index], warped_y[y_index, x_index]],
                    [warped_x[y_index, x_index + 1], warped_y[y_index, x_index + 1]],
                    [warped_x[y_index + 1, x_index + 1], warped_y[y_index + 1, x_index + 1]],
                    [warped_x[y_index + 1, x_index], warped_y[y_index + 1, x_index]],
                ],
                dtype=float,
            )
            signed_area = _signed_polygon_area(polygon_xy)
            ratio = signed_area / base_area
            ratios.append(float(ratio))
            checked += 1
            if ratio <= 0:
                folded += 1
            if ratio < cfg.topology_min_area_ratio:
                compressed += 1
            if ratio > cfg.topology_max_area_ratio:
                expanded += 1

    if checked == 0:
        return {
            "enabled": True,
            "valid": True,
            "reason": "no_grid_cells_inside_geometry",
            "grid_size": int(cfg.topology_grid_size),
            "checked_cells": 0,
        }

    ratio_array = np.asarray(ratios, dtype=float)
    valid = folded == 0 and compressed == 0 and expanded == 0
    return {
        "enabled": True,
        "valid": bool(valid),
        "grid_size": int(cfg.topology_grid_size),
        "checked_cells": int(checked),
        "folded_cell_count": int(folded),
        "compressed_cell_count": int(compressed),
        "expanded_cell_count": int(expanded),
        "min_area_ratio": float(np.min(ratio_array)),
        "median_area_ratio": float(np.median(ratio_array)),
        "max_area_ratio": float(np.max(ratio_array)),
        "min_area_ratio_threshold": float(cfg.topology_min_area_ratio),
        "max_area_ratio_threshold": float(cfg.topology_max_area_ratio),
    }


def _geometry_contains_xy(geom: Any, x: float, y: float) -> bool:
    try:
        from shapely import contains_xy

        return bool(contains_xy(geom, x, y))
    except Exception:
        from shapely.geometry import Point

        return bool(geom.contains(Point(x, y)))


def _signed_polygon_area(xy: np.ndarray) -> float:
    x = xy[:, 0]
    y = xy[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _warp_geojson(
    moving_geojson: dict[str, Any],
    moving_records: list[_FeatureRecord],
    model: _TPSModel,
    group_property: str,
) -> tuple[dict[str, Any], list[_FeatureRecord], Counter[str]]:
    warped = copy.deepcopy(moving_geojson)
    status_counts: Counter[str] = Counter()
    warped_records: list[_FeatureRecord] = []
    for index, (feature, record) in enumerate(zip(warped["features"], moving_records)):
        new_geom = shapely_transform(_make_coordinate_warper(model), record.geometry)
        status = "valid"
        if not new_geom.is_valid:
            repaired = new_geom.buffer(0)
            if repaired.is_valid and not repaired.is_empty:
                new_geom = repaired
                status = "repaired_valid"
            else:
                status = "invalid"
        status_counts[status] += 1
        feature["geometry"] = mapping(new_geom)
        group = str(_feature_group(feature.get("properties") or {}, group_property))
        warped_records.append(_FeatureRecord(feature=feature, group=group, geometry=new_geom))

    if len(warped_records) != len(moving_records):
        raise ValueError("Warped feature count does not match moving feature count.")
    return warped, warped_records, status_counts


def _feature_group(properties: dict[str, Any], group_property: str) -> Any | None:
    if group_property in properties:
        return properties[group_property]
    if "." in group_property:
        value: Any = properties
        for part in group_property.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value
    if group_property == "structure" and "assigned_structure" in properties:
        return properties["assigned_structure"]
    return None


def _make_coordinate_warper(model: _TPSModel):
    def _warp(x, y, z=None):
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
        scalar = x_array.ndim == 0
        xy = np.column_stack([np.atleast_1d(x_array), np.atleast_1d(y_array)])
        warped = model.warp(xy)
        new_x = warped[:, 0]
        new_y = warped[:, 1]
        if scalar:
            if z is None:
                return float(new_x[0]), float(new_y[0])
            return float(new_x[0]), float(new_y[0]), z
        if z is None:
            return new_x, new_y
        return new_x, new_y, z

    return _warp


def _compute_residuals(landmarks: pd.DataFrame, model: _TPSModel) -> pd.DataFrame:
    src = landmarks[["src_x", "src_y"]].to_numpy(dtype=float)
    dst = landmarks[["dst_x", "dst_y"]].to_numpy(dtype=float)
    warped = model.warp(src)
    displacement = dst - src
    residual = dst - warped
    result = landmarks.copy()
    result["warped_x"] = warped[:, 0]
    result["warped_y"] = warped[:, 1]
    result["displacement_um"] = np.linalg.norm(displacement, axis=1)
    result["residual_um"] = np.linalg.norm(residual, axis=1)
    return result


def _compute_metrics(
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    soft_records: list[_FeatureRecord],
) -> dict[str, Any]:
    fixed_by_group = _union_by_group(fixed_records)
    moving_by_group = _union_by_group(moving_records)
    soft_by_group = _union_by_group(soft_records)
    labels = sorted(set(fixed_by_group) | set(moving_by_group) | set(soft_by_group))
    rows = []
    for label in labels:
        fixed = fixed_by_group.get(label, GeometryCollection())
        moving = moving_by_group.get(label, GeometryCollection())
        soft = soft_by_group.get(label, GeometryCollection())
        iou_before = _iou(fixed, moving)
        iou_after = _iou(fixed, soft)
        rows.append(
            {
                "structure": label,
                "iou_hard_before_soft": iou_before,
                "iou_soft_after": iou_after,
                "delta_iou_soft": iou_after - iou_before,
                "area_fixed": float(fixed.area),
                "area_hard_aligned": float(moving.area),
                "area_soft_aligned": float(soft.area),
            }
        )
    fixed_union = unary_union([record.geometry for record in fixed_records])
    moving_union = unary_union([record.geometry for record in moving_records])
    soft_union = unary_union([record.geometry for record in soft_records])
    union_iou_before = _iou(fixed_union, moving_union)
    union_iou_after = _iou(fixed_union, soft_union)
    return {
        "per_structure": pd.DataFrame(rows),
        "union_iou_hard_before_soft": union_iou_before,
        "union_iou_soft_after": union_iou_after,
        "delta_union_iou_soft": union_iou_after - union_iou_before,
    }


def _write_qc_figures(
    *,
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    soft_records: list[_FeatureRecord],
    landmarks: pd.DataFrame,
    residuals: pd.DataFrame,
    model: _TPSModel,
    cfg: ThreeDContourReconstructionConfig,
    out_dir: Path,
) -> dict[str, Path | None]:
    overlay_before = out_dir / "soft_tps_overlay_hard_before.png"
    overlay_after = out_dir / "soft_tps_overlay_soft_after.png"
    landmarks_qc = out_dir / "soft_tps_landmarks_qc.png"
    diagnostic_report = out_dir / "soft_tps_diagnostic_report.png"

    _plot_overlay(
        fixed_records,
        moving_records,
        overlay_before,
        title="Fixed vs hard-aligned moving contours",
        moving_label="Hard-aligned moving",
        moving_color="#ff6b6b",
        dpi=cfg.dpi,
    )
    _plot_overlay(
        fixed_records,
        soft_records,
        overlay_after,
        title="Fixed vs TPS soft-aligned moving contours",
        moving_label="Soft-aligned moving",
        moving_color="#2ecc71",
        dpi=cfg.dpi,
    )
    _plot_landmarks_qc(fixed_records, landmarks, landmarks_qc, cfg.dpi)
    _plot_diagnostic_report(
        fixed_records=fixed_records,
        moving_records=moving_records,
        soft_records=soft_records,
        landmarks=landmarks,
        residuals=residuals,
        model=model,
        cfg=cfg,
        output_path=diagnostic_report,
    )
    return {
        "overlay_before_png": overlay_before,
        "overlay_after_png": overlay_after,
        "landmarks_qc_png": landmarks_qc,
        "diagnostic_report_png": diagnostic_report,
    }


def _plot_overlay(
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    output_path: Path,
    *,
    title: str,
    moving_label: str,
    moving_color: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    for geom in _iter_geometries(fixed_records):
        _plot_geometry_outline(ax, geom, color="#d9dee7", linewidth=1.1, alpha=0.85)
    for geom in _iter_geometries(moving_records):
        _plot_geometry_outline(ax, geom, color=moving_color, linewidth=0.9, alpha=0.75)
    ax.plot([], [], color="#d9dee7", label="Fixed 01")
    ax.plot([], [], color=moving_color, label=moving_label)
    _style_spatial_axis(ax, title)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_landmarks_qc(
    fixed_records: list[_FeatureRecord],
    landmarks: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    for geom in _iter_geometries(fixed_records):
        _plot_geometry_outline(ax, geom, color="#b0b6bf", linewidth=0.8, alpha=0.45)
    magnitude = np.hypot(landmarks["dx"], landmarks["dy"])
    scatter = ax.scatter(
        landmarks["src_x"],
        landmarks["src_y"],
        c=magnitude,
        cmap="viridis",
        s=8,
        alpha=0.85,
    )
    ax.quiver(
        landmarks["src_x"],
        landmarks["src_y"],
        landmarks["dx"],
        landmarks["dy"],
        magnitude,
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0015,
        alpha=0.45,
    )
    fig.colorbar(scatter, ax=ax, label="Landmark displacement (um)")
    _style_spatial_axis(ax, "TPS landmark displacement QC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_diagnostic_report(
    *,
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    soft_records: list[_FeatureRecord],
    landmarks: pd.DataFrame,
    residuals: pd.DataFrame,
    model: _TPSModel,
    cfg: ThreeDContourReconstructionConfig,
    output_path: Path,
) -> None:
    fixed_union = unary_union([record.geometry for record in fixed_records])
    magnitude = residuals["displacement_um"].to_numpy(dtype=float)
    residual_um = residuals["residual_um"].to_numpy(dtype=float)
    norm_vmax = max(float(np.nanpercentile(magnitude, 98)), 1.0)

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.16)

    ax_a = fig.add_subplot(gs[0, 0])
    _plot_geometry_outline(ax_a, fixed_union, color="#9aa4b2", linewidth=1.0, alpha=0.35)
    q = ax_a.quiver(
        landmarks["src_x"],
        landmarks["src_y"],
        landmarks["dx"],
        landmarks["dy"],
        magnitude,
        cmap="viridis",
        clim=(0, norm_vmax),
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0013,
        alpha=0.85,
    )
    fig.colorbar(q, ax=ax_a, label="Displacement magnitude (um)")
    _style_spatial_axis(ax_a, "A. TPS displacement vector field")

    ax_b = fig.add_subplot(gs[0, 1])
    _plot_warped_grid(ax_b, fixed_records, model)
    _plot_geometry_outline(ax_b, fixed_union, color="#e3e6eb", linewidth=1.1, alpha=0.7)
    _style_spatial_axis(ax_b, "B. Warped grid deformation")

    ax_c = fig.add_subplot(gs[1, 0])
    focus_bounds = _diagnostic_bounds(
        fixed_records,
        moving_records,
        soft_records,
        cfg.diagnostic_structure,
    )
    _plot_group_records(ax_c, fixed_records, cfg.diagnostic_structure, "#f1f5f9", 1.6, "-", 0.95)
    _plot_group_records(ax_c, moving_records, cfg.diagnostic_structure, "#ff6b6b", 1.1, "--", 0.75)
    _plot_group_records(ax_c, soft_records, cfg.diagnostic_structure, "#2ecc71", 1.4, "-", 0.9)
    ax_c.plot([], [], color="#f1f5f9", label="Fixed 01")
    ax_c.plot([], [], color="#ff6b6b", linestyle="--", label="Hard-aligned 02")
    ax_c.plot([], [], color="#2ecc71", label="Soft-aligned 02")
    ax_c.set_xlim(focus_bounds[0], focus_bounds[2])
    ax_c.set_ylim(focus_bounds[1], focus_bounds[3])
    _style_spatial_axis(
        ax_c,
        f"C. {cfg.diagnostic_structure or 'All structures'} local alignment",
    )
    ax_c.legend(loc="upper right", frameon=True)

    inner = gs[1, 1].subgridspec(1, 2, wspace=0.28)
    ax_d1 = fig.add_subplot(inner[0, 0])
    ax_d2 = fig.add_subplot(inner[0, 1])
    ax_d1.hist(residual_um, bins=40, color="#43a2ca", alpha=0.75, edgecolor="#17212f")
    ax_d1.axvline(float(np.mean(residual_um)), color="#f97316", linestyle="--", label="Mean")
    ax_d1.axvline(float(np.median(residual_um)), color="#22c55e", linestyle=":", label="Median")
    ax_d1.set_title("D1. Post-warp landmark residuals")
    ax_d1.set_xlabel("Residual (um)")
    ax_d1.set_ylabel("Landmarks")
    ax_d1.legend(frameon=False)

    _plot_geometry_outline(ax_d2, fixed_union, color="#9aa4b2", linewidth=0.8, alpha=0.3)
    scatter = ax_d2.scatter(
        residuals["warped_x"],
        residuals["warped_y"],
        c=residual_um,
        cmap="magma",
        s=12,
        alpha=0.9,
    )
    fig.colorbar(scatter, ax=ax_d2, label="Residual (um)")
    _style_spatial_axis(ax_d2, "D2. Spatial residual map")

    fig.suptitle("TPS contour soft-alignment diagnostic report", fontsize=18)
    fig.savefig(output_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_warped_grid(
    ax,
    fixed_records: list[_FeatureRecord],
    model: _TPSModel,
    grid_n: int = 24,
) -> None:
    minx, miny, maxx, maxy = _combined_bounds(fixed_records)
    xs = np.linspace(minx, maxx, grid_n)
    ys = np.linspace(miny, maxy, grid_n)
    for y in ys:
        points = np.column_stack([xs, np.full_like(xs, y)])
        warped = model.warp(points)
        ax.plot(points[:, 0], points[:, 1], color="#475569", linewidth=0.35, alpha=0.22)
        ax.plot(warped[:, 0], warped[:, 1], color="#38bdf8", linewidth=0.55, alpha=0.7)
    for x in xs:
        points = np.column_stack([np.full_like(ys, x), ys])
        warped = model.warp(points)
        ax.plot(points[:, 0], points[:, 1], color="#475569", linewidth=0.35, alpha=0.22)
        ax.plot(warped[:, 0], warped[:, 1], color="#38bdf8", linewidth=0.55, alpha=0.7)


def _plot_group_records(
    ax,
    records: list[_FeatureRecord],
    group: str | None,
    color: str,
    linewidth: float,
    linestyle: str,
    alpha: float,
) -> None:
    selected = records if group is None else [record for record in records if record.group == group]
    for geom in _iter_geometries(selected):
        _plot_geometry_outline(
            ax,
            geom,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )


def _plot_geometry_outline(
    ax,
    geom,
    *,
    color: str,
    linewidth: float,
    alpha: float,
    linestyle: str = "-",
) -> None:
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        ax.plot(*geom.exterior.xy, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)
        return
    if isinstance(geom, MultiPolygon):
        for polygon in geom.geoms:
            _plot_geometry_outline(
                ax,
                polygon,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
            )
        return
    if hasattr(geom, "geoms"):
        for part in geom.geoms:
            _plot_geometry_outline(
                ax,
                part,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
            )


def _style_spatial_axis(ax, title: str) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.grid(False)


def _diagnostic_bounds(
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    soft_records: list[_FeatureRecord],
    diagnostic_structure: str | None,
) -> tuple[float, float, float, float]:
    records = fixed_records + moving_records + soft_records
    if diagnostic_structure is not None:
        selected = [record for record in records if record.group == diagnostic_structure]
        if selected:
            records = selected
    minx, miny, maxx, maxy = _combined_bounds(records)
    pad = 0.08 * max(maxx - minx, maxy - miny, 1.0)
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def _build_summary(
    *,
    cfg: ThreeDContourReconstructionConfig,
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    soft_records: list[_FeatureRecord],
    boundary_landmarks: pd.DataFrame,
    anchors: pd.DataFrame,
    metrics: dict[str, Any],
    residuals: pd.DataFrame,
    geometry_status: Counter[str],
    topology: dict[str, Any],
    model: _TPSModel,
    icp_iterations_executed: int,
    out_dir: Path,
    soft_geojson_path: Path,
    landmarks_path: Path,
    residuals_path: Path,
    metrics_path: Path,
    outputs: dict[str, Path | None],
) -> dict[str, Any]:
    displacement = residuals["displacement_um"].to_numpy(dtype=float)
    residual_um = residuals["residual_um"].to_numpy(dtype=float)
    per_structure = {
        str(row["structure"]): {
            "iou_hard_before_soft": float(row["iou_hard_before_soft"]),
            "iou_soft_after": float(row["iou_soft_after"]),
            "delta_iou_soft": float(row["delta_iou_soft"]),
        }
        for _, row in metrics["per_structure"].iterrows()
    }
    return {
        "fixed_sample_path": str(Path(cfg.fixed_geojson)),
        "moving_hard_aligned_sample_path": str(Path(cfg.moving_hard_aligned_geojson)),
        "method": {
            "type": "local_tps_rbf_displacement_field",
            "source": "hard-aligned moving boundary landmarks",
            "target": (
                "normal-aware fixed boundary candidates by matching structure"
                if cfg.landmark_normal_weight_um > 0
                else "nearest fixed boundary landmarks by matching structure"
            ),
            "group_property": cfg.group_property,
            "landmark_matching": (
                "normal_aware_kdtree"
                if cfg.landmark_normal_weight_um > 0
                else "nearest_projection"
            ),
            "landmark_candidate_count": cfg.landmark_candidate_count,
            "landmark_candidate_spacing_um": cfg.landmark_candidate_spacing_um,
            "landmark_normal_weight_um": cfg.landmark_normal_weight_um,
            "landmark_normal_step_um": cfg.landmark_normal_step_um,
            "rbf_kernel": cfg.rbf_kernel,
            "rbf_neighbors": cfg.rbf_neighbors,
            "rbf_smoothing": cfg.rbf_smoothing,
            "topology_grid_size": cfg.topology_grid_size,
            "topology_min_area_ratio": cfg.topology_min_area_ratio,
            "topology_max_area_ratio": cfg.topology_max_area_ratio,
            "landmarks_per_structure": cfg.landmarks_per_structure,
            "diagnostic_structure": cfg.diagnostic_structure,
            "diagnostic_structure_landmarks": cfg.diagnostic_structure_landmarks,
            "normalization_center_xy": [float(v) for v in model.center_xy],
            "normalization_scale": float(model.scale),
            "icp_iterations_configured": cfg.icp_iterations,
            "icp_iterations_executed": icp_iterations_executed,
            "curvature_landmark_weight": cfg.curvature_landmark_weight,
            "mutual_nn_check": cfg.mutual_nn_check,
            "zero_anchor_count": cfg.zero_anchor_count,
            "landmark_outlier_mad_threshold": cfg.landmark_outlier_mad_threshold,
        },
        "landmarks": {
            "boundary_landmark_count": int(len(boundary_landmarks)),
            "zero_anchor_count": int(len(anchors)),
            "total_landmark_count": int(len(boundary_landmarks) + len(anchors)),
            "distance_um_min": float(boundary_landmarks["source_distance_um"].min()),
            "distance_um_median": float(boundary_landmarks["source_distance_um"].median()),
            "distance_um_mean": float(boundary_landmarks["source_distance_um"].mean()),
            "distance_um_max": float(boundary_landmarks["source_distance_um"].max()),
        },
        "qc": {
            "fixed_feature_count": int(len(fixed_records)),
            "hard_aligned_feature_count": int(len(moving_records)),
            "soft_aligned_feature_count": int(len(soft_records)),
            "union_iou_hard_before_soft": float(metrics["union_iou_hard_before_soft"]),
            "union_iou_soft_after": float(metrics["union_iou_soft_after"]),
            "delta_union_iou_soft": float(metrics["delta_union_iou_soft"]),
            "geometry_status_counts": dict(geometry_status),
            "topology_check": topology,
            "displacement_magnitude_um": _distribution_summary(displacement),
            "post_warp_residual_um": _distribution_summary(residual_um),
            "per_structure": per_structure,
        },
        "outputs": {
            "out_dir": str(out_dir),
            "soft_geojson": str(soft_geojson_path),
            "metrics_csv": str(metrics_path),
            "landmarks_csv": str(landmarks_path),
            "residuals_csv": str(residuals_path),
            **{name: str(path) if path is not None else None for name, path in outputs.items()},
        },
    }


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _union_by_group(records: list[_FeatureRecord]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = {}
    for record in records:
        grouped.setdefault(record.group, []).append(record.geometry)
    return {group: unary_union(geometries) for group, geometries in grouped.items()}


def _iter_geometries(records: Iterable[_FeatureRecord]) -> Iterable[Any]:
    for record in records:
        yield record.geometry


def _combined_bounds(records: list[_FeatureRecord]) -> tuple[float, float, float, float]:
    if not records:
        raise ValueError("No geometries available for bounds calculation.")
    bounds = np.array([record.geometry.bounds for record in records], dtype=float)
    return (
        float(bounds[:, 0].min()),
        float(bounds[:, 1].min()),
        float(bounds[:, 2].max()),
        float(bounds[:, 3].max()),
    )


def _iou(a, b) -> float:
    if a.is_empty and b.is_empty:
        return math.nan
    union_area = float(a.union(b).area)
    if union_area <= 0:
        return math.nan
    return float(a.intersection(b).area / union_area)


# ---------------------------------------------------------------------------
# Landmark post-processing pipeline
# ---------------------------------------------------------------------------


def _postprocess_landmarks(
    landmarks: pd.DataFrame,
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    cfg: ThreeDContourReconstructionConfig,
) -> pd.DataFrame:
    """Apply curvature landmarks, mutual-NN filter, and outlier rejection."""
    # 1. Augment with extra landmarks at high-curvature boundary regions.
    if cfg.curvature_landmark_weight > 0:
        extra = _generate_curvature_landmarks(fixed_records, moving_records, cfg)
        if not extra.empty:
            landmarks = pd.concat([landmarks, extra], ignore_index=True)

    if landmarks.empty or len(landmarks) < 3:
        return landmarks

    # 2. Mutual nearest-neighbour consistency filter.
    if cfg.mutual_nn_check and len(landmarks) >= 4:
        filtered = _filter_mutual_nn_landmarks(landmarks, moving_records, cfg)
        if len(filtered) >= 3:
            landmarks = filtered

    # 3. MAD-based outlier rejection on displacement vectors.
    if cfg.landmark_outlier_mad_threshold is not None and len(landmarks) >= 8:
        filtered = _filter_landmark_outliers(
            landmarks, float(cfg.landmark_outlier_mad_threshold)
        )
        if len(filtered) >= 3:
            landmarks = filtered

    return landmarks


def _compute_boundary_curvature(
    boundary: Any,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (N, 2) points and (N,) absolute angle-change curvature values."""
    all_pts: list[list[float]] = []
    for line in _iter_line_parts(boundary):
        length = float(line.length)
        if length <= 0:
            continue
        distances = np.arange(0.0, length, max(spacing, 1e-6))
        for d in distances:
            pt = line.interpolate(float(d))
            all_pts.append([float(pt.x), float(pt.y)])

    if len(all_pts) < 3:
        pts = np.array(all_pts, dtype=float) if all_pts else np.empty((0, 2), dtype=float)
        return pts, np.zeros(len(pts), dtype=float)

    pts = np.array(all_pts, dtype=float)
    n = len(pts)
    curvatures = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 > 1e-12 and n2 > 1e-12:
            cos_angle = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
            curvatures[i] = abs(math.acos(cos_angle))
    return pts, curvatures


def _generate_curvature_landmarks(
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    cfg: ThreeDContourReconstructionConfig,
) -> pd.DataFrame:
    """Add extra landmarks concentrated at high-curvature boundary points."""
    from shapely.geometry import Point

    rows: list[dict[str, Any]] = []
    fixed_by_group = _union_by_group(fixed_records)
    moving_by_group = _union_by_group(moving_records)
    dense_spacing = max(cfg.sampling_distance_um / 5.0, 1e-6)

    for group in sorted(set(fixed_by_group) & set(moving_by_group)):
        moving_boundary = moving_by_group[group].boundary
        fixed_boundary = fixed_by_group[group].boundary
        if moving_boundary.is_empty or fixed_boundary.is_empty:
            continue

        pts, curvatures = _compute_boundary_curvature(moving_boundary, dense_spacing)
        if len(curvatures) < 4:
            continue

        # Select the top (curvature_landmark_weight) fraction of high-curvature points.
        # Use strict inequality so that zero-curvature flat segments are excluded.
        threshold = float(np.quantile(curvatures, 1.0 - cfg.curvature_landmark_weight))
        high_idx = np.where((curvatures > threshold) & (curvatures > 1e-6))[0]
        if len(high_idx) == 0:
            continue

        for idx in high_idx:
            src_xy = pts[idx]
            src_pt = Point(float(src_xy[0]), float(src_xy[1]))
            dst = fixed_boundary.interpolate(fixed_boundary.project(src_pt))
            dist = float(src_pt.distance(dst))
            if dist > cfg.max_landmark_distance_um:
                continue
            rows.append(
                {
                    "kind": "boundary",
                    "structure": group,
                    "src_x": float(src_xy[0]),
                    "src_y": float(src_xy[1]),
                    "dst_x": float(dst.x),
                    "dst_y": float(dst.y),
                    "dx": float(dst.x - src_xy[0]),
                    "dy": float(dst.y - src_xy[1]),
                    "source_distance_um": dist,
                    "match_cost_um": dist,
                    "normal_dot_abs": math.nan,
                    "match_method": "curvature_nearest_projection",
                }
            )
    return pd.DataFrame(rows)


def _filter_mutual_nn_landmarks(
    landmarks: pd.DataFrame,
    moving_records: list[_FeatureRecord],
    cfg: ThreeDContourReconstructionConfig,
) -> pd.DataFrame:
    """Keep only landmarks where the reverse projection is also approximately consistent.

    For each moving→fixed landmark pair, we project the fixed (target) point back
    onto the moving boundary and check that the resulting point is within
    ``2 × sampling_distance_um`` of the original source point.  This removes
    non-bijective correspondences (e.g. landmarks that cross contour features) that
    tend to cause TPS fold artefacts.
    """
    from shapely.geometry import Point

    moving_by_group = _union_by_group(moving_records)
    keep_mask = []
    # Use a spacing-based absolute tolerance so that the check is independent
    # of the forward displacement magnitude.
    abs_tolerance = 2.0 * cfg.sampling_distance_um

    for _, row in landmarks.iterrows():
        group = str(row["structure"])
        if group not in moving_by_group:
            keep_mask.append(True)
            continue
        moving_boundary = moving_by_group[group].boundary
        if moving_boundary.is_empty:
            keep_mask.append(True)
            continue
        dst_pt = Point(float(row["dst_x"]), float(row["dst_y"]))
        rev = moving_boundary.interpolate(moving_boundary.project(dst_pt))
        rev_dist = float(
            np.linalg.norm(
                [
                    float(rev.x) - float(row["src_x"]),
                    float(rev.y) - float(row["src_y"]),
                ]
            )
        )
        keep_mask.append(rev_dist <= abs_tolerance)

    return landmarks.loc[keep_mask].reset_index(drop=True)


def _filter_landmark_outliers(
    landmarks: pd.DataFrame,
    mad_threshold: float,
) -> pd.DataFrame:
    """Remove landmarks whose displacement vector deviates beyond *mad_threshold* × MAD."""
    dx = landmarks["dx"].to_numpy(dtype=float)
    dy = landmarks["dy"].to_numpy(dtype=float)

    median_dx = float(np.median(dx))
    median_dy = float(np.median(dy))
    mad_dx = float(np.median(np.abs(dx - median_dx)))
    mad_dy = float(np.median(np.abs(dy - median_dy)))

    # Gaussian-consistency scale factor (1.4826 ≈ 1/Φ⁻¹(0.75)).
    scale_dx = max(mad_dx * 1.4826, 1e-9)
    scale_dy = max(mad_dy * 1.4826, 1e-9)

    z_score = np.sqrt(((dx - median_dx) / scale_dx) ** 2 + ((dy - median_dy) / scale_dy) ** 2)
    keep = z_score <= mad_threshold
    return landmarks.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Adaptive RBF smoothing selection via k-fold cross-validation
# ---------------------------------------------------------------------------


def _select_rbf_smoothing_cv(
    src_norm: np.ndarray,
    displacement_norm: np.ndarray,
    cfg: ThreeDContourReconstructionConfig,
    neighbors: int | None,
    max_cv_points: int = 200,
    n_folds: int = 5,
) -> float:
    """Choose RBF smoothing via k-fold CV on a random subset of landmarks."""
    candidates = list(cfg.rbf_smoothing_candidates) if cfg.rbf_smoothing_candidates else [1e-4]
    n = len(src_norm)
    if n < 8 or len(candidates) == 0:
        return 1e-4

    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(n, size=min(n, max_cv_points), replace=False))
    X = src_norm[idx]
    Y = displacement_norm[idx]
    n_cv = len(X)
    actual_folds = min(n_folds, n_cv)
    fold_boundaries = np.array_split(np.arange(n_cv), actual_folds)

    best_s = float(candidates[0])
    best_err = math.inf

    for s in candidates:
        cv_errors: list[float] = []
        for fold_i, val_idx in enumerate(fold_boundaries):
            train_idx = np.concatenate(
                [fold_boundaries[j] for j in range(actual_folds) if j != fold_i]
            )
            if len(train_idx) < 3:
                continue
            nb = neighbors if neighbors is not None and len(train_idx) > neighbors else None
            try:
                interp = RBFInterpolator(
                    X[train_idx],
                    Y[train_idx],
                    kernel=cfg.rbf_kernel,
                    smoothing=float(s),
                    neighbors=nb,
                )
                pred = interp(X[val_idx])
                cv_errors.append(
                    float(np.mean(np.linalg.norm(pred - Y[val_idx], axis=1)))
                )
            except Exception:
                pass
        if cv_errors:
            err = float(np.mean(cv_errors))
            if err < best_err:
                best_err = err
                best_s = float(s)

    return best_s


# ---------------------------------------------------------------------------
# ICP-style iterative TPS refinement
# ---------------------------------------------------------------------------


def _icp_refine(
    *,
    fixed_records: list[_FeatureRecord],
    initial_soft_geojson: dict[str, Any],
    initial_soft_records: list[_FeatureRecord],
    cfg: ThreeDContourReconstructionConfig,
) -> tuple[dict[str, Any], list[_FeatureRecord], Counter[str], int]:
    """Iteratively refine TPS alignment by re-generating landmarks from warped positions.

    Each iteration fits a *correction* TPS from the currently warped contour to the
    fixed contour, composes with the previous warp, and accepts only if IoU improves.
    Returns ``(best_geojson, best_records, best_geometry_status, iterations_executed)``.
    """
    fixed_union = unary_union([r.geometry for r in fixed_records])
    best_geojson = initial_soft_geojson
    best_records = initial_soft_records
    best_geometry_status: Counter[str] = Counter()
    best_iou = _iou(fixed_union, unary_union([r.geometry for r in initial_soft_records]))

    current_geojson: dict[str, Any] = initial_soft_geojson
    current_records: list[_FeatureRecord] = initial_soft_records
    icp_executed = 1

    for _ in range(cfg.icp_iterations - 1):
        new_landmarks = _generate_landmarks(fixed_records, current_records, cfg)
        if new_landmarks.empty:
            break
        new_bnd = new_landmarks.loc[new_landmarks["kind"] == "boundary"].copy()
        if len(new_bnd) < 3:
            break
        new_anchors = _zero_anchor_landmarks(fixed_records + current_records, cfg.zero_anchor_count)
        new_model_lms = pd.concat([new_bnd, new_anchors], ignore_index=True)
        try:
            correction = _fit_tps_model(new_model_lms, cfg)
        except Exception:
            break

        topo = _check_tps_topology(current_records, correction, cfg)
        if not topo.get("valid", True):
            break

        new_geojson, new_records, new_status = _warp_geojson(
            current_geojson, current_records, correction, cfg.group_property
        )
        new_iou = _iou(fixed_union, unary_union([r.geometry for r in new_records]))
        icp_executed += 1
        if new_iou >= best_iou:
            best_geojson = new_geojson
            best_records = new_records
            best_geometry_status = new_status
            best_iou = new_iou
            current_geojson = new_geojson
            current_records = new_records
        else:
            break

    return best_geojson, best_records, best_geometry_status, icp_executed
