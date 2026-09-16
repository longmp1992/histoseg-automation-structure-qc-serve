from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPoint, MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from .multislice import _apply_affine_to_geojson, _apply_similarity_to_geojson, _iou
from .multislice import _SimilarityTransform as _SimilarityTransform
from .soft_alignment import _FeatureRecord, _check_tps_topology, _read_geojson

PathLike = Union[str, Path]


@dataclass(frozen=True)
class LabelFreeContourAlignmentConfig:
    """Configuration for structure-agnostic contour layout alignment."""

    fixed_geojson: PathLike
    moving_geojson: PathLike
    out_dir: PathLike
    maxiter: int = 80
    multistart: bool = True
    affine_fallback_iou_threshold: float = 0.15
    run_soft_tps: bool = True
    sampling_distance_um: float = 80.0
    max_landmark_distance_um: float = 250.0
    landmark_candidate_count: int = 8
    landmark_candidate_spacing_um: float | None = None
    landmark_normal_weight_um: float = 0.0
    landmark_normal_step_um: float | None = None
    rbf_kernel: str = "thin_plate_spline"
    rbf_neighbors: int | None = 96
    rbf_smoothing: float = 1e-4
    topology_grid_size: int = 24
    topology_min_area_ratio: float = 0.5
    topology_max_area_ratio: float = 2.0
    min_component_area_um2: float = 0.0
    max_component_weight: float = 0.08
    boundary_sample_count: int = 6000
    component_sample_count: int = 800
    partial_correspondence: bool = False
    diagnostic_only: bool = False
    search_window: float = 800.0
    knn_neighbors: int = 6
    min_anchor_score: float = 0.72
    min_review_score: float = 0.55
    min_anchor_count: int = 8
    area_ratio_min: float = 0.25
    area_ratio_max: float = 4.0
    envelope_area_fraction: float = 0.35
    envelope_bbox_fraction: float = 0.85
    overlap_ransac: bool = False
    overlap_candidate_count: int = 8
    overlap_ransac_trials: int = 20000
    overlap_min_descriptor_score: float = 0.42
    overlap_allow_scale: bool = False
    group_correspondence: bool = False
    group_candidate_count: int = 12
    group_ransac_trials: int = 15000
    group_min_descriptor_score: float = 0.35
    group_residual_limit_um: float = 900.0
    group_min_component_area_um2: float = 5000.0
    save_preview_png: bool = True
    overwrite: bool = False
    dpi: int = 180


@dataclass
class LabelFreeContourAlignmentResult:
    out_dir: Path
    aligned_geojson: Path | None
    summary_json: Path
    landmarks_csv: Path | None
    overlay_html: Path | None = None
    overlay_before_png: Path | None = None
    overlay_after_png: Path | None = None
    component_qc_csv: Path | None = None
    partial_nodes_csv: Path | None = None
    partial_matches_csv: Path | None = None
    partial_matrix_html: Path | None = None
    group_matrix_csv: Path | None = None
    group_matrix_html: Path | None = None


@dataclass(frozen=True)
class AnchorOnlyResidualTPSConfig:
    """Configuration for anchor-only residual TPS soft alignment.

    The moving GeoJSON is expected to be the hard-aligned output from the
    label-free group/partial-anchor hard seed. Only the anchor rows marked
    ``used_for_transform`` fit the residual field; all other geometry follows
    passively.
    """

    fixed_geojson: PathLike
    moving_hard_aligned_geojson: PathLike
    anchor_landmarks_csv: PathLike
    out_dir: PathLike
    group_property: str = "structure"
    min_anchor_count: int = 8
    residual_limit_um: float = 900.0
    bbox_padding_fraction: float = 0.10
    identity_padding_count: int = 16
    rbf_kernel: str = "thin_plate_spline"
    rbf_neighbors: int | None = 96
    rbf_smoothing: float = 1e-4
    jacobian_grid_size: int = 50
    max_negative_jacobian_ratio: float = 0.001
    save_preview_png: bool = True
    dpi: int = 180
    overwrite: bool = False


@dataclass
class AnchorOnlyResidualTPSResult:
    """Artifacts produced by anchor-only residual TPS soft alignment."""

    out_dir: Path
    soft_aligned_geojson: Path
    landmarks_csv: Path
    summary_json: Path
    review_html: Path | None = None
    overlay_before_png: Path | None = None
    overlay_after_png: Path | None = None


@dataclass(frozen=True)
class IterativeContourRefinementConfig:
    """Configuration for post-soft contour-level refinement.

    The input moving GeoJSON should already be hard aligned and, usually,
    anchor-only soft aligned. The refinement searches for additional local
    contour correspondences in this nearly aligned coordinate frame, then fits
    one conservative residual TPS pass from those correspondences.
    """

    fixed_geojson: PathLike
    moving_aligned_geojson: PathLike
    out_dir: PathLike
    group_property: str = "structure"
    min_component_area_um2: float = 1500.0
    centroid_search_radius_um: float = 350.0
    accepted_centroid_radius_um: float = 260.0
    area_ratio_min: float = 0.25
    area_ratio_max: float = 4.0
    relaxed_area_ratio_min: float = 0.55
    relaxed_area_ratio_max: float = 1.8
    min_overlap: float = 0.12
    relaxed_centroid_radius_um: float = 90.0
    min_pair_score: float = 0.52
    boundary_sample_spacing_um: float = 65.0
    fixed_boundary_sample_spacing_um: float = 45.0
    max_moving_boundary_points_per_pair: int = 70
    max_fixed_boundary_points_per_pair: int = 140
    max_anchor_distance_um: float = 180.0
    max_anchors_per_pair: int = 80
    max_total_anchors: int = 1800
    min_pair_count: int = 3
    min_anchor_count: int = 30
    residual_limit_um: float = 220.0
    min_delta_union_iou: float = 0.0
    bbox_padding_fraction: float = 0.08
    identity_padding_count: int = 32
    rbf_kernel: str = "thin_plate_spline"
    rbf_neighbors: int | None = 96
    rbf_smoothing: float = 2e-4
    jacobian_grid_size: int = 45
    max_negative_jacobian_ratio: float = 0.002
    save_preview_png: bool = True
    dpi: int = 180
    overwrite: bool = False


@dataclass
class IterativeContourRefinementResult:
    """Artifacts produced by one contour-level refinement pass."""

    out_dir: Path
    refined_geojson: Path
    candidate_pairs_csv: Path
    mutual_pairs_csv: Path
    accepted_pairs_csv: Path
    landmarks_csv: Path
    summary_json: Path
    tps_result: AnchorOnlyResidualTPSResult | None = None


@dataclass(frozen=True)
class _PointTransform:
    kind: str
    similarity: _SimilarityTransform | None = None
    affine_params: tuple[float, float, float, float, float, float] | None = None


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
class _ContourNode:
    role: str
    node_id: str
    feature_index: int
    source_label: str
    geometry: Any
    centroid_x: float
    centroid_y: float
    area: float
    perimeter: float
    compactness: float
    aspect_ratio: float
    orientation_degrees: float
    bbox_width: float
    bbox_height: float
    area_fraction: float
    bbox_fraction_x: float
    bbox_fraction_y: float
    envelope_only: bool
    exclude_reason: str
    knn_distances: tuple[float, ...]
    knn_angle_gaps: tuple[float, ...]
    knn_log_area_ratios: tuple[float, ...]


def align_contours_label_free(
    cfg: LabelFreeContourAlignmentConfig,
) -> LabelFreeContourAlignmentResult:
    """Align two contour GeoJSON files without using structure labels."""

    _validate_config(cfg)
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    aligned_path = out_dir / "moving_label_free_aligned.geojson"
    summary_path = out_dir / "label_free_alignment_summary.json"
    landmarks_path = out_dir / "label_free_landmarks.csv"
    component_qc_path = out_dir / "label_free_component_qc.csv"
    label_overlap_path = out_dir / "label_free_structure_overlap.csv"
    overlay_html_path = out_dir / "label_free_alignment_overlay.html"
    before_png = out_dir / "label_free_overlay_before.png"
    after_png = out_dir / "label_free_overlay_after.png"

    if (
        aligned_path.exists()
        and summary_path.exists()
        and landmarks_path.exists()
        and not cfg.overwrite
    ):
        return LabelFreeContourAlignmentResult(
            out_dir=out_dir,
            aligned_geojson=aligned_path,
            summary_json=summary_path,
            landmarks_csv=landmarks_path,
            overlay_html=overlay_html_path if overlay_html_path.exists() else None,
            overlay_before_png=before_png if before_png.exists() else None,
            overlay_after_png=after_png if after_png.exists() else None,
            component_qc_csv=component_qc_path if component_qc_path.exists() else None,
        )

    fixed_payload = _read_geojson(Path(cfg.fixed_geojson).expanduser())
    moving_payload = _read_geojson(Path(cfg.moving_geojson).expanduser())
    fixed_records = _records_from_geojson_label_free(fixed_payload, cfg, role="fixed")
    moving_records = _records_from_geojson_label_free(moving_payload, cfg, role="moving")
    fixed_union = unary_union([record.geometry for record in fixed_records])
    moving_union = unary_union([record.geometry for record in moving_records])
    if fixed_union.is_empty or moving_union.is_empty:
        raise ValueError("Fixed and moving contour sets must contain non-empty geometry.")

    if cfg.partial_correspondence:
        return _run_partial_correspondence_diagnostic(
            cfg=cfg,
            out_dir=out_dir,
            moving_payload=moving_payload,
            fixed_records=fixed_records,
            moving_records=moving_records,
            fixed_union=fixed_union,
            moving_union=moving_union,
        )

    fixed_points = _sample_boundary_points(fixed_union.boundary, cfg.boundary_sample_count)
    moving_points = _sample_boundary_points(moving_union.boundary, cfg.boundary_sample_count)
    fixed_components = _component_layout_table(fixed_records, cfg)
    moving_components = _component_layout_table(moving_records, cfg)
    component_qc = _write_component_qc(
        fixed_components,
        moving_components,
        component_qc_path,
    )

    score_inputs = {
        "fixed_union": fixed_union,
        "moving_union": moving_union,
        "fixed_points": fixed_points,
        "moving_points": moving_points,
        "fixed_components": fixed_components,
        "moving_components": moving_components,
    }
    before_score = _label_free_score(_PointTransform(kind="identity"), **score_inputs)
    hard_transform, hard_meta = _estimate_label_free_similarity(score_inputs, cfg)
    hard_score = _label_free_score(hard_transform, **score_inputs)
    selected_transform = hard_transform
    transform_type = "similarity"

    affine_meta: dict[str, Any] | None = None
    if hard_score["iou"] < cfg.affine_fallback_iou_threshold:
        affine_transform, affine_meta = _try_label_free_affine(score_inputs, hard_score, cfg)
        if affine_transform is not None:
            affine_score = _label_free_score(affine_transform, **score_inputs)
            if affine_score["score"] >= hard_score["score"]:
                selected_transform = affine_transform
                hard_score = affine_score
                transform_type = "affine"

    hard_accepted = hard_score["score"] >= before_score["score"]
    if not hard_accepted:
        selected_transform = _PointTransform(kind="identity")
        transform_type = "identity"
        hard_score = before_score
        hard_meta = {
            **hard_meta,
            "accepted": False,
            "accepted_reason": "identity_kept_because_label_free_score_worsened",
        }
    else:
        hard_meta = {**hard_meta, "accepted": True}

    hard_payload = _apply_point_transform_to_geojson(moving_payload, selected_transform)
    hard_records = _records_from_geojson_label_free(hard_payload, cfg, role="hard_aligned")

    soft_payload = hard_payload
    final_records = hard_records
    soft_summary: dict[str, Any] = {
        "attempted": False,
        "accepted": False,
        "reason": "disabled",
        "landmark_count": 0,
    }
    landmarks = pd.DataFrame(
        columns=[
            "kind",
            "structure",
            "src_x",
            "src_y",
            "dst_x",
            "dst_y",
            "dx",
            "dy",
            "source_distance_um",
            "match_cost_um",
            "normal_dot_abs",
            "match_method",
            "accepted_for_tps",
        ]
    )
    if cfg.run_soft_tps:
        soft_payload, final_records, landmarks, soft_summary = _run_label_free_soft_tps(
            fixed_records=fixed_records,
            hard_payload=hard_payload,
            hard_records=hard_records,
            score_inputs=score_inputs,
            hard_score=hard_score,
            cfg=cfg,
        )

    final_score = _score_geojson_against_fixed(
        fixed_union=fixed_union,
        moving_payload=soft_payload,
        fixed_points=fixed_points,
        fixed_components=fixed_components,
        cfg=cfg,
    )
    aligned_path.write_text(json.dumps(soft_payload, ensure_ascii=False), encoding="utf-8")
    landmarks.to_csv(landmarks_path, index=False)

    overlay_paths = {"before": None, "after": None}
    if cfg.save_preview_png:
        overlay_paths = _write_overlays(
            fixed_records=fixed_records,
            moving_records=moving_records,
            final_records=final_records,
            before_png=before_png,
            after_png=after_png,
            dpi=cfg.dpi,
        )
    correspondence = _compute_correspondence_diagnostics(
        fixed_records=fixed_records,
        moving_records=final_records,
        output_csv=label_overlap_path,
    )
    _write_overlay_html(
        fixed_records=fixed_records,
        moving_records=moving_records,
        final_records=final_records,
        output_html=overlay_html_path,
        title="Label-free contour alignment",
        warning=correspondence["warning"],
    )

    summary = {
        "fixed_geojson": str(Path(cfg.fixed_geojson)),
        "moving_geojson": str(Path(cfg.moving_geojson)),
        "output_geojson": str(aligned_path),
        "method": "label_free_contour_layout_alignment",
        "objective_note": (
            "Structure labels are ignored. The hard-alignment objective combines union IoU, "
            "a symmetric boundary nearest-distance loss, and an area-capped component layout loss."
        ),
        "labels_used_for_matching": False,
        "semantic_harmonization_performed": False,
        "transform_type": transform_type,
        "transform": _transform_payload(selected_transform),
        "hard_alignment": hard_meta,
        "affine_fallback": affine_meta,
        "scores": {
            "before": before_score,
            "after_hard": hard_score,
            "after_final": final_score,
        },
        "soft_tps": soft_summary,
        "fixed_feature_count": len(fixed_records),
        "moving_feature_count": len(moving_records),
        "component_qc_csv": str(component_qc_path),
        "component_qc": component_qc,
        "label_overlap_csv": str(label_overlap_path),
        "correspondence_diagnostics": correspondence,
        "overlay_html": str(overlay_html_path),
        "overlay_before_png": str(overlay_paths["before"]) if overlay_paths["before"] else None,
        "overlay_after_png": str(overlay_paths["after"]) if overlay_paths["after"] else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return LabelFreeContourAlignmentResult(
        out_dir=out_dir,
        aligned_geojson=aligned_path,
        summary_json=summary_path,
        landmarks_csv=landmarks_path,
        overlay_html=overlay_html_path,
        overlay_before_png=overlay_paths["before"],
        overlay_after_png=overlay_paths["after"],
        component_qc_csv=component_qc_path,
    )


def run_anchor_only_residual_tps(
    cfg: AnchorOnlyResidualTPSConfig,
) -> AnchorOnlyResidualTPSResult:
    """Fit a residual TPS from trusted label-free anchors only.

    This is the soft-alignment companion for partial-anchor hard seeds. The
    input moving GeoJSON is already hard-aligned. The TPS model receives
    residual landmarks from used anchors plus zero-residual identity padding,
    so unmatched contours remain passive geometry instead of attracting the
    moving slice toward unrelated fixed boundaries.
    """

    _validate_anchor_only_residual_tps_config(cfg)
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    soft_geojson_path = out_dir / "anchor_only_soft_aligned_contours.geojson"
    landmarks_path = out_dir / "anchor_only_tps_landmarks.csv"
    summary_path = out_dir / "anchor_only_tps_summary.json"
    review_html_path = out_dir / "anchor_only_tps_review.html"
    before_png = out_dir / "anchor_only_tps_before.png"
    after_png = out_dir / "anchor_only_tps_after.png"

    if (
        soft_geojson_path.exists()
        and landmarks_path.exists()
        and summary_path.exists()
        and not cfg.overwrite
    ):
        return AnchorOnlyResidualTPSResult(
            out_dir=out_dir,
            soft_aligned_geojson=soft_geojson_path,
            landmarks_csv=landmarks_path,
            summary_json=summary_path,
            review_html=review_html_path if review_html_path.exists() else None,
            overlay_before_png=before_png if before_png.exists() else None,
            overlay_after_png=after_png if after_png.exists() else None,
        )

    fixed_payload = _read_geojson(Path(cfg.fixed_geojson).expanduser())
    hard_payload = _read_geojson(Path(cfg.moving_hard_aligned_geojson).expanduser())
    record_cfg = LabelFreeContourAlignmentConfig(
        fixed_geojson=cfg.fixed_geojson,
        moving_geojson=cfg.moving_hard_aligned_geojson,
        out_dir=out_dir,
        min_component_area_um2=0.0,
    )
    fixed_records = _records_from_geojson_label_free(fixed_payload, record_cfg, role="fixed")
    hard_records = _records_from_geojson_label_free(
        hard_payload, record_cfg, role="hard_aligned"
    )
    fixed_union = unary_union([record.geometry for record in fixed_records])
    hard_union = unary_union([record.geometry for record in hard_records])

    failure_reasons: list[str] = []
    anchor_rows = pd.DataFrame()
    try:
        anchor_rows = pd.read_csv(Path(cfg.anchor_landmarks_csv).expanduser())
    except FileNotFoundError:
        failure_reasons.append("missing_anchor_landmarks_csv")
    except Exception as exc:
        failure_reasons.append(f"invalid_anchor_landmarks_csv:{type(exc).__name__}")

    model_landmarks = pd.DataFrame()
    landmarks = pd.DataFrame()
    used_anchor_count = 0
    model: _TPSModel | None = None
    soft_payload = copy.deepcopy(hard_payload)
    soft_records = hard_records
    geometry_status: dict[str, int] = {"hard_only": int(len(hard_records))}
    input_residual_um = np.array([], dtype=float)
    post_residual_um = np.array([], dtype=float)
    jacobian = {
        "enabled": False,
        "valid": True,
        "reason": "not_attempted",
        "negative_jacobian_ratio": 0.0,
    }

    if not failure_reasons:
        try:
            landmarks, model_landmarks, input_residual_um = _anchor_only_tps_landmarks(
                anchor_rows,
                fixed_records=fixed_records,
                hard_records=hard_records,
                cfg=cfg,
            )
            used_anchor_count = int(
                (landmarks["landmark_kind"] == "matched_anchor")
                .where(landmarks["used_for_tps"].astype(bool), False)
                .sum()
            )
        except ValueError as exc:
            failure_reasons.append(str(exc))

    if not failure_reasons:
        if used_anchor_count < int(cfg.min_anchor_count):
            failure_reasons.append("too_few_anchor_only_tps_anchors")
        if len(model_landmarks) < 3:
            failure_reasons.append("fewer_than_three_tps_landmarks")

    if not failure_reasons:
        try:
            model = _fit_tps_model(model_landmarks, cfg)
            soft_payload, soft_records, geometry_status = _warp_geojson_label_free(
                hard_payload,
                hard_records,
                model,
            )
            used = model_landmarks.loc[
                model_landmarks["landmark_kind"] == "matched_anchor"
            ].copy()
            warped_anchor_xy = model.warp(used[["src_x", "src_y"]].to_numpy(dtype=float))
            target_anchor_xy = used[["dst_x", "dst_y"]].to_numpy(dtype=float)
            post_residual_um = np.linalg.norm(warped_anchor_xy - target_anchor_xy, axis=1)
            jacobian = _anchor_only_jacobian_summary(
                model,
                fixed_records=fixed_records,
                hard_records=hard_records,
                cfg=cfg,
            )
        except Exception as exc:
            failure_reasons.append(f"tps_fit_or_warp_failed:{type(exc).__name__}")
            soft_payload = copy.deepcopy(hard_payload)
            soft_records = hard_records
            geometry_status = {"hard_only": int(len(hard_records))}

    invalid_geometry_count = int(geometry_status.get("invalid", 0))
    post_p90 = (
        float(np.percentile(post_residual_um, 90)) if len(post_residual_um) else math.inf
    )
    accepted = (
        not failure_reasons
        and used_anchor_count >= int(cfg.min_anchor_count)
        and math.isfinite(post_p90)
        and post_p90 <= float(cfg.residual_limit_um)
        and float(jacobian.get("negative_jacobian_ratio", 0.0))
        <= float(cfg.max_negative_jacobian_ratio)
        and invalid_geometry_count == 0
    )
    if not accepted and not failure_reasons:
        if not math.isfinite(post_p90) or post_p90 > float(cfg.residual_limit_um):
            failure_reasons.append("anchor_only_post_residual_p90_above_limit")
        if (
            float(jacobian.get("negative_jacobian_ratio", 0.0))
            > float(cfg.max_negative_jacobian_ratio)
        ):
            failure_reasons.append("negative_jacobian_ratio_above_limit")
        if invalid_geometry_count:
            failure_reasons.append("invalid_soft_geometry")

    soft_union = unary_union([record.geometry for record in soft_records])
    soft_geojson_path.write_text(json.dumps(soft_payload, ensure_ascii=False), encoding="utf-8")
    if landmarks.empty:
        landmarks = _empty_anchor_only_landmarks()
    landmarks.to_csv(landmarks_path, index=False)

    overlay_paths = {"before": None, "after": None}
    if cfg.save_preview_png:
        overlay_paths = _write_overlays(
            fixed_records=fixed_records,
            moving_records=hard_records,
            final_records=soft_records,
            before_png=before_png,
            after_png=after_png,
            dpi=cfg.dpi,
        )
    _write_anchor_only_residual_tps_html(
        fixed_records=fixed_records,
        hard_records=hard_records,
        soft_records=soft_records,
        landmarks=landmarks,
        output_html=review_html_path,
        accepted=accepted,
        failure_reasons=failure_reasons,
        jacobian=jacobian,
    )

    input_summary = _distribution_summary_or_nan(input_residual_um)
    post_summary = _distribution_summary_or_nan(post_residual_um)
    summary = {
        "fixed_geojson": str(Path(cfg.fixed_geojson)),
        "moving_hard_aligned_geojson": str(Path(cfg.moving_hard_aligned_geojson)),
        "anchor_landmarks_csv": str(Path(cfg.anchor_landmarks_csv)),
        "output_geojson": str(soft_geojson_path),
        "method": "anchor_only_residual_tps",
        "objective_note": (
            "The residual TPS is fitted only from label-free anchors already used by "
            "the hard transform. Non-anchor contours are passive geometry and do not "
            "pull the soft field toward unrelated boundaries."
        ),
        "attempted": bool(len(model_landmarks) >= 3),
        "accepted": bool(accepted),
        "reason": (
            "accepted_anchor_only_residual_tps"
            if accepted
            else ";".join(failure_reasons) if failure_reasons else "rejected"
        ),
        "config": {
            "min_anchor_count": int(cfg.min_anchor_count),
            "residual_limit_um": float(cfg.residual_limit_um),
            "bbox_padding_fraction": float(cfg.bbox_padding_fraction),
            "identity_padding_count": int(cfg.identity_padding_count),
            "rbf_kernel": cfg.rbf_kernel,
            "rbf_neighbors": cfg.rbf_neighbors,
            "rbf_smoothing": float(cfg.rbf_smoothing),
            "jacobian_grid_size": int(cfg.jacobian_grid_size),
            "max_negative_jacobian_ratio": float(cfg.max_negative_jacobian_ratio),
        },
        "landmarks": {
            "boundary_landmark_count": int(used_anchor_count),
            "anchor_landmark_count": int(used_anchor_count),
            "identity_padding_count": int(
                (landmarks["landmark_kind"] == "identity_padding").sum()
            ),
            "zero_anchor_count": int((landmarks["landmark_kind"] == "identity_padding").sum()),
            "total_landmark_count": int(len(model_landmarks)),
            "used_for_tps_count": int(landmarks["used_for_tps"].astype(bool).sum()),
            "source": "label_free_anchor_landmarks_csv",
            "input_residual_um": input_summary,
            "post_warp_residual_um": post_summary,
        },
        "qc": {
            "fixed_feature_count": int(len(fixed_records)),
            "hard_aligned_feature_count": int(len(hard_records)),
            "soft_aligned_feature_count": int(len(soft_records)),
            "union_iou_hard_before_soft": _iou(fixed_union, hard_union),
            "union_iou_soft_after": _iou(fixed_union, soft_union),
            "delta_union_iou_soft": _iou(fixed_union, soft_union) - _iou(fixed_union, hard_union),
            "geometry_status_counts": dict(geometry_status),
            "topology_check": {
                "enabled": bool(jacobian.get("enabled", False)),
                "valid": bool(jacobian.get("valid", True)),
                "grid_size": jacobian.get("grid_size"),
                "checked_cells": jacobian.get("checked_cells"),
                "folded_cell_count": jacobian.get("negative_cell_count"),
                "negative_jacobian_ratio": jacobian.get("negative_jacobian_ratio"),
                "min_jacobian_ratio": jacobian.get("min_jacobian_ratio"),
                "median_jacobian_ratio": jacobian.get("median_jacobian_ratio"),
                "max_jacobian_ratio": jacobian.get("max_jacobian_ratio"),
            },
            "jacobian_check": jacobian,
            "post_warp_residual_um": post_summary,
        },
        "outputs": {
            "out_dir": str(out_dir),
            "soft_geojson": str(soft_geojson_path),
            "landmarks_csv": str(landmarks_path),
            "review_html": str(review_html_path),
            "overlay_before_png": (
                str(overlay_paths["before"]) if overlay_paths["before"] else None
            ),
            "overlay_after_png": (
                str(overlay_paths["after"]) if overlay_paths["after"] else None
            ),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return AnchorOnlyResidualTPSResult(
        out_dir=out_dir,
        soft_aligned_geojson=soft_geojson_path,
        landmarks_csv=landmarks_path,
        summary_json=summary_path,
        review_html=review_html_path,
        overlay_before_png=overlay_paths["before"],
        overlay_after_png=overlay_paths["after"],
    )


def run_iterative_contour_refinement(
    cfg: IterativeContourRefinementConfig,
) -> IterativeContourRefinementResult:
    """Run one conservative contour-correspondence refinement pass.

    This is intended after a reliable hard/anchor-only soft seed. It does not
    infer semantic identity. It only uses post-alignment local overlap, centroid
    proximity, and area compatibility to discover additional contour pairs that
    are safe enough to act as residual TPS landmarks.
    """

    _validate_iterative_contour_refinement_config(cfg)
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_pairs_csv = out_dir / "iterative_contour_candidate_pairs.csv"
    mutual_pairs_csv = out_dir / "iterative_contour_mutual_pairs.csv"
    accepted_pairs_csv = out_dir / "iterative_contour_accepted_pairs.csv"
    landmarks_csv = out_dir / "iterative_contour_landmarks.csv"
    refined_geojson = out_dir / "iterative_refined_contours.geojson"
    summary_json = out_dir / "iterative_contour_refinement_summary.json"

    if (
        refined_geojson.exists()
        and candidate_pairs_csv.exists()
        and mutual_pairs_csv.exists()
        and accepted_pairs_csv.exists()
        and landmarks_csv.exists()
        and summary_json.exists()
        and not cfg.overwrite
    ):
        return IterativeContourRefinementResult(
            out_dir=out_dir,
            refined_geojson=refined_geojson,
            candidate_pairs_csv=candidate_pairs_csv,
            mutual_pairs_csv=mutual_pairs_csv,
            accepted_pairs_csv=accepted_pairs_csv,
            landmarks_csv=landmarks_csv,
            summary_json=summary_json,
            tps_result=None,
        )

    fixed_payload = _read_geojson(Path(cfg.fixed_geojson).expanduser())
    moving_payload = _read_geojson(Path(cfg.moving_aligned_geojson).expanduser())
    record_cfg = LabelFreeContourAlignmentConfig(
        fixed_geojson=cfg.fixed_geojson,
        moving_geojson=cfg.moving_aligned_geojson,
        out_dir=out_dir,
        min_component_area_um2=0.0,
    )
    fixed_records = _records_from_geojson_label_free(fixed_payload, record_cfg, role="fixed")
    moving_records = _records_from_geojson_label_free(
        moving_payload, record_cfg, role="moving_aligned"
    )
    fixed_union = unary_union([record.geometry for record in fixed_records])
    moving_union = unary_union([record.geometry for record in moving_records])

    candidates, mutual_pairs, accepted_pairs = _iterative_contour_pair_tables(
        fixed_records,
        moving_records,
        cfg,
    )
    candidates.to_csv(candidate_pairs_csv, index=False)
    mutual_pairs.to_csv(mutual_pairs_csv, index=False)
    accepted_pairs.to_csv(accepted_pairs_csv, index=False)

    landmarks = _iterative_contour_landmarks(
        fixed_records,
        moving_records,
        accepted_pairs,
        cfg,
    )
    if landmarks.empty:
        landmarks = _empty_anchor_only_landmarks()
    landmarks.to_csv(landmarks_csv, index=False)

    failure_reasons: list[str] = []
    if len(accepted_pairs) < int(cfg.min_pair_count):
        failure_reasons.append("too_few_iterative_contour_pairs")
    used_anchor_count = int(
        _coerce_bool_series(landmarks.get("used_for_transform", pd.Series(dtype=bool))).sum()
    )
    if used_anchor_count < int(cfg.min_anchor_count):
        failure_reasons.append("too_few_iterative_contour_anchors")

    tps_result: AnchorOnlyResidualTPSResult | None = None
    tps_summary: dict[str, Any] | None = None
    accepted = False
    if not failure_reasons:
        tps_result = run_anchor_only_residual_tps(
            AnchorOnlyResidualTPSConfig(
                fixed_geojson=cfg.fixed_geojson,
                moving_hard_aligned_geojson=cfg.moving_aligned_geojson,
                anchor_landmarks_csv=landmarks_csv,
                out_dir=out_dir / "iterative_contour_tps",
                group_property=cfg.group_property,
                min_anchor_count=cfg.min_anchor_count,
                residual_limit_um=cfg.residual_limit_um,
                bbox_padding_fraction=cfg.bbox_padding_fraction,
                identity_padding_count=cfg.identity_padding_count,
                rbf_kernel=cfg.rbf_kernel,
                rbf_neighbors=cfg.rbf_neighbors,
                rbf_smoothing=cfg.rbf_smoothing,
                jacobian_grid_size=cfg.jacobian_grid_size,
                max_negative_jacobian_ratio=cfg.max_negative_jacobian_ratio,
                save_preview_png=cfg.save_preview_png,
                dpi=cfg.dpi,
                overwrite=cfg.overwrite,
            )
        )
        tps_summary = json.loads(tps_result.summary_json.read_text(encoding="utf-8"))
        delta_iou = float(tps_summary.get("qc", {}).get("delta_union_iou_soft", math.nan))
        if not bool(tps_summary.get("accepted", False)):
            failure_reasons.append(
                str(tps_summary.get("reason") or "iterative_contour_tps_rejected")
            )
        if not math.isfinite(delta_iou) or delta_iou < float(cfg.min_delta_union_iou):
            failure_reasons.append("iterative_contour_delta_iou_below_limit")
        accepted = not failure_reasons

    if accepted and tps_result is not None:
        refined_payload = _read_geojson(tps_result.soft_aligned_geojson)
        refined_records = _records_from_geojson_label_free(
            refined_payload, record_cfg, role="iterative_refined"
        )
        refined_geojson.write_text(
            json.dumps(refined_payload, ensure_ascii=False), encoding="utf-8"
        )
    else:
        refined_records = moving_records
        refined_geojson.write_text(
            json.dumps(moving_payload, ensure_ascii=False), encoding="utf-8"
        )

    refined_union = unary_union([record.geometry for record in refined_records])
    summary = {
        "fixed_geojson": str(Path(cfg.fixed_geojson)),
        "moving_aligned_geojson": str(Path(cfg.moving_aligned_geojson)),
        "output_geojson": str(refined_geojson),
        "method": {
            "name": "iterative_contour_refinement",
            "rbf_kernel": cfg.rbf_kernel,
            "rbf_neighbors": cfg.rbf_neighbors,
            "rbf_smoothing": float(cfg.rbf_smoothing),
        },
        "objective_note": (
            "After an initial alignment, HistoSeg finds additional nearby contour "
            "pairs by overlap, centroid distance, and area compatibility, then fits "
            "one residual TPS pass from those contour-level correspondences."
        ),
        "attempted": bool(len(accepted_pairs) >= int(cfg.min_pair_count)),
        "accepted": bool(accepted),
        "reason": (
            "accepted_iterative_contour_refinement"
            if accepted
            else ";".join(dict.fromkeys(failure_reasons)) if failure_reasons else "rejected"
        ),
        "config": {
            "min_component_area_um2": float(cfg.min_component_area_um2),
            "centroid_search_radius_um": float(cfg.centroid_search_radius_um),
            "accepted_centroid_radius_um": float(cfg.accepted_centroid_radius_um),
            "min_overlap": float(cfg.min_overlap),
            "min_pair_score": float(cfg.min_pair_score),
            "max_anchor_distance_um": float(cfg.max_anchor_distance_um),
            "max_anchors_per_pair": int(cfg.max_anchors_per_pair),
            "max_total_anchors": int(cfg.max_total_anchors),
            "min_pair_count": int(cfg.min_pair_count),
            "min_anchor_count": int(cfg.min_anchor_count),
            "residual_limit_um": float(cfg.residual_limit_um),
            "min_delta_union_iou": float(cfg.min_delta_union_iou),
            "bbox_padding_fraction": float(cfg.bbox_padding_fraction),
            "identity_padding_count": int(cfg.identity_padding_count),
            "jacobian_grid_size": int(cfg.jacobian_grid_size),
            "max_negative_jacobian_ratio": float(cfg.max_negative_jacobian_ratio),
        },
        "pairs": {
            "candidate_pair_count": int(len(candidates)),
            "mutual_pair_count": int(len(mutual_pairs)),
            "accepted_pair_count": int(len(accepted_pairs)),
        },
        "landmarks": {
            "boundary_landmark_count": int(used_anchor_count),
            "anchor_landmark_count": int(used_anchor_count),
            "identity_padding_count": int(
                (landmarks.get("landmark_kind", pd.Series(dtype=str)) == "identity_padding").sum()
            ),
            "total_landmark_count": int(len(landmarks)),
            "used_for_tps_count": int(used_anchor_count),
            "input_residual_um": _distribution_summary_or_nan(
                landmarks.loc[
                    _coerce_bool_series(
                        landmarks.get("used_for_transform", pd.Series(dtype=bool))
                    ),
                    "input_residual_um",
                ].to_numpy(dtype=float)
                if "input_residual_um" in landmarks.columns
                else np.array([], dtype=float)
            ),
        },
        "qc": {
            "fixed_feature_count": int(len(fixed_records)),
            "moving_feature_count": int(len(moving_records)),
            "refined_feature_count": int(len(refined_records)),
            "union_iou_before_refinement": _iou(fixed_union, moving_union),
            "union_iou_after_refinement": _iou(fixed_union, refined_union),
            "delta_union_iou_refinement": _iou(fixed_union, refined_union)
            - _iou(fixed_union, moving_union),
            "geometry_status_counts": (
                (tps_summary or {}).get("qc", {}).get("geometry_status_counts")
                if tps_summary is not None
                else {"unchanged": int(len(moving_records))}
            ),
            "topology_check": (
                (tps_summary or {}).get("qc", {}).get("topology_check")
                if tps_summary is not None
                else {"enabled": False, "valid": True, "reason": "not_attempted"}
            ),
            "jacobian_check": (
                (tps_summary or {}).get("qc", {}).get("jacobian_check")
                if tps_summary is not None
                else {"enabled": False, "valid": True, "reason": "not_attempted"}
            ),
            "post_warp_residual_um": (
                (tps_summary or {}).get("qc", {}).get("post_warp_residual_um")
                if tps_summary is not None
                else _distribution_summary_or_nan(np.array([], dtype=float))
            ),
        },
        "outputs": {
            "out_dir": str(out_dir),
            "refined_geojson": str(refined_geojson),
            "candidate_pairs_csv": str(candidate_pairs_csv),
            "mutual_pairs_csv": str(mutual_pairs_csv),
            "accepted_pairs_csv": str(accepted_pairs_csv),
            "landmarks_csv": str(landmarks_csv),
            "tps_summary_json": str(tps_result.summary_json) if tps_result else None,
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return IterativeContourRefinementResult(
        out_dir=out_dir,
        refined_geojson=refined_geojson,
        candidate_pairs_csv=candidate_pairs_csv,
        mutual_pairs_csv=mutual_pairs_csv,
        accepted_pairs_csv=accepted_pairs_csv,
        landmarks_csv=landmarks_csv,
        summary_json=summary_json,
        tps_result=tps_result,
    )


def _validate_anchor_only_residual_tps_config(cfg: AnchorOnlyResidualTPSConfig) -> None:
    if cfg.min_anchor_count < 1:
        raise ValueError("min_anchor_count must be at least 1.")
    if cfg.residual_limit_um <= 0:
        raise ValueError("residual_limit_um must be greater than 0.")
    if cfg.bbox_padding_fraction < 0:
        raise ValueError("bbox_padding_fraction must be non-negative.")
    if cfg.identity_padding_count < 4:
        raise ValueError("identity_padding_count must be at least 4.")
    if cfg.rbf_neighbors is not None and cfg.rbf_neighbors < 3:
        raise ValueError("rbf_neighbors must be at least 3 when provided.")
    if cfg.rbf_smoothing < 0:
        raise ValueError("rbf_smoothing must be non-negative.")
    if cfg.jacobian_grid_size < 2:
        raise ValueError("jacobian_grid_size must be at least 2.")
    if not (0.0 <= cfg.max_negative_jacobian_ratio <= 1.0):
        raise ValueError("max_negative_jacobian_ratio must be in [0, 1].")


def _validate_iterative_contour_refinement_config(
    cfg: IterativeContourRefinementConfig,
) -> None:
    if cfg.min_component_area_um2 < 0:
        raise ValueError("min_component_area_um2 must be non-negative.")
    if cfg.centroid_search_radius_um <= 0:
        raise ValueError("centroid_search_radius_um must be positive.")
    if cfg.accepted_centroid_radius_um <= 0:
        raise ValueError("accepted_centroid_radius_um must be positive.")
    if cfg.area_ratio_min <= 0 or cfg.area_ratio_max <= cfg.area_ratio_min:
        raise ValueError("area_ratio_min/max must be positive and increasing.")
    if (
        cfg.relaxed_area_ratio_min <= 0
        or cfg.relaxed_area_ratio_max <= cfg.relaxed_area_ratio_min
    ):
        raise ValueError("relaxed_area_ratio_min/max must be positive and increasing.")
    if not (0.0 <= cfg.min_overlap <= 1.0):
        raise ValueError("min_overlap must be in [0, 1].")
    if not (0.0 <= cfg.min_pair_score <= 1.0):
        raise ValueError("min_pair_score must be in [0, 1].")
    if cfg.boundary_sample_spacing_um <= 0 or cfg.fixed_boundary_sample_spacing_um <= 0:
        raise ValueError("boundary sample spacing must be positive.")
    if cfg.max_moving_boundary_points_per_pair < 1:
        raise ValueError("max_moving_boundary_points_per_pair must be at least 1.")
    if cfg.max_fixed_boundary_points_per_pair < 1:
        raise ValueError("max_fixed_boundary_points_per_pair must be at least 1.")
    if cfg.max_anchor_distance_um <= 0:
        raise ValueError("max_anchor_distance_um must be positive.")
    if cfg.max_anchors_per_pair < 1:
        raise ValueError("max_anchors_per_pair must be at least 1.")
    if cfg.max_total_anchors < 1:
        raise ValueError("max_total_anchors must be at least 1.")
    if cfg.min_pair_count < 1:
        raise ValueError("min_pair_count must be at least 1.")
    if cfg.min_anchor_count < 1:
        raise ValueError("min_anchor_count must be at least 1.")
    if cfg.residual_limit_um <= 0:
        raise ValueError("residual_limit_um must be positive.")
    if cfg.identity_padding_count < 4:
        raise ValueError("identity_padding_count must be at least 4.")
    if cfg.rbf_neighbors is not None and cfg.rbf_neighbors < 3:
        raise ValueError("rbf_neighbors must be at least 3 when provided.")
    if cfg.rbf_smoothing < 0:
        raise ValueError("rbf_smoothing must be non-negative.")
    if cfg.jacobian_grid_size < 2:
        raise ValueError("jacobian_grid_size must be at least 2.")
    if not (0.0 <= cfg.max_negative_jacobian_ratio <= 1.0):
        raise ValueError("max_negative_jacobian_ratio must be in [0, 1].")


def _iterative_contour_pair_tables(
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    cfg: IterativeContourRefinementConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_table = _iterative_contour_record_table(
        fixed_records, min_area_um2=cfg.min_component_area_um2
    )
    moving_table = _iterative_contour_record_table(
        moving_records, min_area_um2=cfg.min_component_area_um2
    )
    columns = [
        "fixed_feature_index",
        "moving_feature_index",
        "fixed_label",
        "moving_label",
        "fixed_area",
        "moving_area",
        "area_ratio",
        "centroid_distance_um",
        "overlap_coefficient",
        "iou",
        "position_score",
        "area_score",
        "total_score",
        "is_mutual_best",
        "accepted",
        "acceptance_reason",
    ]
    if fixed_table.empty or moving_table.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, empty.copy(), empty.copy()

    moving_tree = cKDTree(moving_table[["centroid_x", "centroid_y"]].to_numpy(dtype=float))
    candidate_rows: list[dict[str, Any]] = []
    for _, fixed in fixed_table.iterrows():
        fixed_xy = np.array([float(fixed["centroid_x"]), float(fixed["centroid_y"])])
        moving_indexes = moving_tree.query_ball_point(
            fixed_xy, r=float(cfg.centroid_search_radius_um)
        )
        for moving_idx in moving_indexes:
            moving = moving_table.iloc[int(moving_idx)]
            fixed_area = float(fixed["area"])
            moving_area = float(moving["area"])
            if fixed_area <= 0 or moving_area <= 0:
                continue
            area_ratio = moving_area / fixed_area
            if area_ratio < cfg.area_ratio_min or area_ratio > cfg.area_ratio_max:
                continue
            fixed_geom = fixed_records[int(fixed["feature_index"])].geometry
            moving_geom = moving_records[int(moving["feature_index"])].geometry
            intersection_area = float(fixed_geom.intersection(moving_geom).area)
            union_area = float(fixed_geom.union(moving_geom).area)
            overlap = intersection_area / max(min(fixed_area, moving_area), 1e-12)
            iou = intersection_area / max(union_area, 1e-12)
            distance = float(
                math.hypot(
                    float(fixed["centroid_x"]) - float(moving["centroid_x"]),
                    float(fixed["centroid_y"]) - float(moving["centroid_y"]),
                )
            )
            position_score = math.exp(-distance / 180.0)
            area_score = max(
                0.0,
                1.0
                - abs(math.log(max(area_ratio, 1e-12)))
                / max(math.log(float(cfg.area_ratio_max)), 1e-12),
            )
            total_score = (
                0.38 * position_score
                + 0.32 * overlap
                + 0.20 * area_score
                + 0.10 * iou
            )
            candidate_rows.append(
                {
                    "fixed_feature_index": int(fixed["feature_index"]),
                    "moving_feature_index": int(moving["feature_index"]),
                    "fixed_label": str(fixed["label"]),
                    "moving_label": str(moving["label"]),
                    "fixed_area": fixed_area,
                    "moving_area": moving_area,
                    "area_ratio": float(area_ratio),
                    "centroid_distance_um": distance,
                    "overlap_coefficient": float(overlap),
                    "iou": float(iou),
                    "position_score": float(position_score),
                    "area_score": float(area_score),
                    "total_score": float(total_score),
                    "is_mutual_best": False,
                    "accepted": False,
                    "acceptance_reason": "",
                }
            )

    if not candidate_rows:
        empty = pd.DataFrame(columns=columns)
        return empty, empty.copy(), empty.copy()
    candidates = pd.DataFrame(candidate_rows, columns=columns)
    fixed_best = (
        candidates.sort_values("total_score", ascending=False)
        .drop_duplicates("fixed_feature_index")
        .set_index("fixed_feature_index")["moving_feature_index"]
        .to_dict()
    )
    moving_best = (
        candidates.sort_values("total_score", ascending=False)
        .drop_duplicates("moving_feature_index")
        .set_index("moving_feature_index")["fixed_feature_index"]
        .to_dict()
    )
    is_mutual = []
    accepted = []
    reasons = []
    for _, row in candidates.iterrows():
        fixed_idx = int(row["fixed_feature_index"])
        moving_idx = int(row["moving_feature_index"])
        mutual = fixed_best.get(fixed_idx) == moving_idx and moving_best.get(moving_idx) == fixed_idx
        is_mutual.append(bool(mutual))
        strict_overlap = float(row["overlap_coefficient"]) >= float(cfg.min_overlap)
        relaxed_close = (
            float(row["centroid_distance_um"]) <= float(cfg.relaxed_centroid_radius_um)
            and cfg.relaxed_area_ratio_min
            <= float(row["area_ratio"])
            <= cfg.relaxed_area_ratio_max
        )
        keep = (
            mutual
            and float(row["total_score"]) >= float(cfg.min_pair_score)
            and float(row["centroid_distance_um"]) <= float(cfg.accepted_centroid_radius_um)
            and (strict_overlap or relaxed_close)
        )
        accepted.append(bool(keep))
        if keep:
            reasons.append("mutual_high_score_local_overlap")
        elif not mutual:
            reasons.append("not_mutual_best")
        elif float(row["total_score"]) < float(cfg.min_pair_score):
            reasons.append("score_below_threshold")
        elif float(row["centroid_distance_um"]) > float(cfg.accepted_centroid_radius_um):
            reasons.append("centroid_distance_above_threshold")
        else:
            reasons.append("insufficient_overlap")
    candidates["is_mutual_best"] = is_mutual
    candidates["accepted"] = accepted
    candidates["acceptance_reason"] = reasons
    mutual_pairs = candidates.loc[candidates["is_mutual_best"].astype(bool)].copy()
    accepted_pairs = candidates.loc[candidates["accepted"].astype(bool)].copy()
    return (
        candidates.sort_values("total_score", ascending=False).reset_index(drop=True),
        mutual_pairs.sort_values("total_score", ascending=False).reset_index(drop=True),
        accepted_pairs.sort_values("total_score", ascending=False).reset_index(drop=True),
    )


def _iterative_contour_record_table(
    records: list[_FeatureRecord],
    *,
    min_area_um2: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        area = float(record.geometry.area)
        if area < float(min_area_um2):
            continue
        centroid = record.geometry.centroid
        rows.append(
            {
                "feature_index": int(index),
                "label": _original_label(record.feature),
                "area": area,
                "centroid_x": float(centroid.x),
                "centroid_y": float(centroid.y),
            }
        )
    return pd.DataFrame(rows)


def _iterative_contour_landmarks(
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    accepted_pairs: pd.DataFrame,
    cfg: IterativeContourRefinementConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if accepted_pairs.empty:
        return _empty_anchor_only_landmarks()
    for _, pair in accepted_pairs.iterrows():
        fixed_index = int(pair["fixed_feature_index"])
        moving_index = int(pair["moving_feature_index"])
        fixed_record = fixed_records[fixed_index]
        moving_record = moving_records[moving_index]
        fixed_points = _sample_geometry_boundary_by_spacing(
            fixed_record.geometry,
            spacing_um=cfg.fixed_boundary_sample_spacing_um,
            max_points=cfg.max_fixed_boundary_points_per_pair,
        )
        moving_points = _sample_geometry_boundary_by_spacing(
            moving_record.geometry,
            spacing_um=cfg.boundary_sample_spacing_um,
            max_points=cfg.max_moving_boundary_points_per_pair,
        )
        if len(fixed_points) == 0 or len(moving_points) == 0:
            continue
        tree = cKDTree(fixed_points)
        distances, indexes = tree.query(moving_points, k=1)
        pair_rows: list[dict[str, Any]] = []
        for src, distance, fixed_point_index in zip(moving_points, distances, indexes):
            if not math.isfinite(float(distance)) or float(distance) > cfg.max_anchor_distance_um:
                continue
            dst = fixed_points[int(fixed_point_index)]
            dx = float(dst[0] - src[0])
            dy = float(dst[1] - src[1])
            pair_rows.append(
                {
                    "kind": "matched_anchor",
                    "landmark_kind": "matched_anchor",
                    "used_for_tps": True,
                    "used_for_transform": True,
                    "accepted_for_tps": True,
                    "source": "iterative_contour_refinement",
                    "fixed_node_id": f"fixed:{fixed_index}",
                    "moving_node_id": f"moving:{moving_index}",
                    "fixed_feature_index": fixed_index,
                    "moving_feature_index": moving_index,
                    "fixed_status": "accepted_iterative_contour_pair",
                    "moving_status": "accepted_iterative_contour_pair",
                    "fixed_label": str(pair.get("fixed_label", "")),
                    "moving_label": str(pair.get("moving_label", "")),
                    "src_x": float(src[0]),
                    "src_y": float(src[1]),
                    "dst_x": float(dst[0]),
                    "dst_y": float(dst[1]),
                    "aligned_moving_centroid_x": float(src[0]),
                    "aligned_moving_centroid_y": float(src[1]),
                    "fixed_centroid_x": float(dst[0]),
                    "fixed_centroid_y": float(dst[1]),
                    "dx": dx,
                    "dy": dy,
                    "input_residual_um": float(distance),
                    "source_distance_um": float(distance),
                    "match_cost_um": float(distance),
                    "total_score": float(pair.get("total_score", math.nan)),
                    "pair_overlap_coefficient": float(
                        pair.get("overlap_coefficient", math.nan)
                    ),
                    "pair_iou": float(pair.get("iou", math.nan)),
                    "pair_centroid_distance_um": float(
                        pair.get("centroid_distance_um", math.nan)
                    ),
                }
            )
        if len(pair_rows) > int(cfg.max_anchors_per_pair):
            order = np.linspace(
                0, len(pair_rows) - 1, int(cfg.max_anchors_per_pair), dtype=int
            )
            pair_rows = [pair_rows[int(i)] for i in order]
        rows.extend(pair_rows)
    if not rows:
        return _empty_anchor_only_landmarks()
    if len(rows) > int(cfg.max_total_anchors):
        order = np.linspace(0, len(rows) - 1, int(cfg.max_total_anchors), dtype=int)
        rows = [rows[int(i)] for i in order]
    return pd.DataFrame(rows)


def _sample_geometry_boundary_by_spacing(
    geom: Any,
    *,
    spacing_um: float,
    max_points: int,
) -> np.ndarray:
    points: list[tuple[float, float]] = []
    for line in _iter_line_parts(geom.boundary):
        length = float(line.length)
        if length <= 0:
            continue
        distances = np.arange(0.0, length, float(spacing_um))
        if len(distances) == 0:
            distances = np.array([0.0])
        for distance in distances:
            point = line.interpolate(float(distance))
            points.append((float(point.x), float(point.y)))
    if len(points) > int(max_points):
        order = np.linspace(0, len(points) - 1, int(max_points), dtype=int)
        points = [points[int(i)] for i in order]
    return np.asarray(points, dtype=float)


def _anchor_only_tps_landmarks(
    anchor_rows: pd.DataFrame,
    *,
    fixed_records: list[_FeatureRecord],
    hard_records: list[_FeatureRecord],
    cfg: AnchorOnlyResidualTPSConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    required = {
        "used_for_transform",
        "aligned_moving_centroid_x",
        "aligned_moving_centroid_y",
        "fixed_centroid_x",
        "fixed_centroid_y",
    }
    missing = sorted(required - set(anchor_rows.columns))
    if missing:
        raise ValueError(f"anchor_landmarks_missing_columns:{','.join(missing)}")
    if anchor_rows.empty:
        raise ValueError("empty_anchor_landmarks_csv")

    used_mask = _coerce_bool_series(anchor_rows["used_for_transform"])
    rows: list[dict[str, Any]] = []
    input_residuals: list[float] = []
    for index, row in anchor_rows.iterrows():
        src_x = float(row["aligned_moving_centroid_x"])
        src_y = float(row["aligned_moving_centroid_y"])
        dst_x = float(row["fixed_centroid_x"])
        dst_y = float(row["fixed_centroid_y"])
        dx = dst_x - src_x
        dy = dst_y - src_y
        residual = float(math.hypot(dx, dy))
        used_for_tps = bool(used_mask.loc[index])
        if used_for_tps:
            input_residuals.append(residual)
        rows.append(
            {
                "kind": "matched_anchor",
                "landmark_kind": "matched_anchor",
                "used_for_tps": used_for_tps,
                "accepted_for_tps": used_for_tps,
                "source": "label_free_anchor_transform",
                "fixed_node_id": row.get("fixed_node_id"),
                "moving_node_id": row.get("moving_node_id"),
                "fixed_feature_index": row.get("fixed_feature_index"),
                "moving_feature_index": row.get("moving_feature_index"),
                "fixed_status": row.get("fixed_status"),
                "moving_status": row.get("moving_status"),
                "fixed_label": row.get("fixed_label"),
                "moving_label": row.get("moving_label"),
                "src_x": src_x,
                "src_y": src_y,
                "dst_x": dst_x,
                "dst_y": dst_y,
                "dx": dx,
                "dy": dy,
                "input_residual_um": residual,
                "source_distance_um": residual,
                "match_cost_um": row.get("match_cost_um", residual),
                "total_score": row.get("total_score"),
            }
        )
    padding = _anchor_only_identity_padding_landmarks(
        fixed_records=fixed_records,
        hard_records=hard_records,
        cfg=cfg,
    )
    landmarks = pd.DataFrame([*rows, *padding.to_dict("records")])
    model_landmarks = landmarks.loc[landmarks["used_for_tps"].astype(bool)].copy()
    return landmarks, model_landmarks, np.asarray(input_residuals, dtype=float)


def _anchor_only_identity_padding_landmarks(
    *,
    fixed_records: list[_FeatureRecord],
    hard_records: list[_FeatureRecord],
    cfg: AnchorOnlyResidualTPSConfig,
) -> pd.DataFrame:
    points = _anchor_only_identity_padding_points(
        fixed_records=fixed_records,
        hard_records=hard_records,
        padding_fraction=cfg.bbox_padding_fraction,
        count=cfg.identity_padding_count,
    )
    return pd.DataFrame(
        [
            {
                "kind": "identity_padding",
                "landmark_kind": "identity_padding",
                "used_for_tps": True,
                "accepted_for_tps": True,
                "source": "padded_union_bbox",
                "fixed_node_id": None,
                "moving_node_id": None,
                "fixed_feature_index": None,
                "moving_feature_index": None,
                "fixed_status": None,
                "moving_status": None,
                "fixed_label": None,
                "moving_label": None,
                "src_x": float(x),
                "src_y": float(y),
                "dst_x": float(x),
                "dst_y": float(y),
                "dx": 0.0,
                "dy": 0.0,
                "input_residual_um": 0.0,
                "source_distance_um": 0.0,
                "match_cost_um": 0.0,
                "total_score": None,
            }
            for x, y in points
        ]
    )


def _anchor_only_identity_padding_points(
    *,
    fixed_records: list[_FeatureRecord],
    hard_records: list[_FeatureRecord],
    padding_fraction: float,
    count: int,
) -> list[tuple[float, float]]:
    union = unary_union([record.geometry for record in [*fixed_records, *hard_records]])
    minx, miny, maxx, maxy = union.bounds
    span = max(maxx - minx, maxy - miny, 1.0)
    pad = float(padding_fraction) * span
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad
    corners = np.array(
        [
            [minx, miny],
            [maxx, miny],
            [maxx, maxy],
            [minx, maxy],
        ],
        dtype=float,
    )
    count = max(int(count), 4)
    perimeter = 2.0 * ((maxx - minx) + (maxy - miny))
    distances = np.linspace(0.0, perimeter, count, endpoint=False)
    points: list[tuple[float, float]] = []
    width = maxx - minx
    height = maxy - miny
    for distance in distances:
        d = float(distance)
        if d <= width:
            points.append((minx + d, miny))
        elif d <= width + height:
            points.append((maxx, miny + (d - width)))
        elif d <= 2.0 * width + height:
            points.append((maxx - (d - width - height), maxy))
        else:
            points.append((minx, maxy - (d - 2.0 * width - height)))
    if len(points) < 4:
        points.extend((float(x), float(y)) for x, y in corners)
    return points


def _anchor_only_jacobian_summary(
    model: _TPSModel,
    *,
    fixed_records: list[_FeatureRecord],
    hard_records: list[_FeatureRecord],
    cfg: AnchorOnlyResidualTPSConfig,
) -> dict[str, Any]:
    grid_size = int(cfg.jacobian_grid_size)
    union = unary_union([record.geometry for record in [*fixed_records, *hard_records]])
    minx, miny, maxx, maxy = union.bounds
    span = max(maxx - minx, maxy - miny, 1.0)
    pad = float(cfg.bbox_padding_fraction) * span
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad
    if maxx <= minx or maxy <= miny:
        return {"enabled": True, "valid": False, "reason": "degenerate_bbox"}

    xs = np.linspace(minx, maxx, grid_size)
    ys = np.linspace(miny, maxy, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    original = np.column_stack([xx.ravel(), yy.ravel()])
    warped = model.warp(original).reshape((grid_size, grid_size, 2))
    right = warped[:, 1:, :] - warped[:, :-1, :]
    down = warped[1:, :, :] - warped[:-1, :, :]
    right_cells = right[:-1, :, :]
    down_cells = down[:, :-1, :]
    det = (
        right_cells[:, :, 0] * down_cells[:, :, 1]
        - right_cells[:, :, 1] * down_cells[:, :, 0]
    )
    base_area = max((xs[1] - xs[0]) * (ys[1] - ys[0]), 1e-12)
    ratios = det / base_area
    checked = int(ratios.size)
    negative = int(np.sum(ratios <= 0.0))
    negative_ratio = float(negative / checked) if checked else 0.0
    valid = negative_ratio <= float(cfg.max_negative_jacobian_ratio)
    return {
        "enabled": True,
        "valid": bool(valid),
        "grid_size": grid_size,
        "checked_cells": checked,
        "negative_cell_count": negative,
        "negative_jacobian_ratio": negative_ratio,
        "max_negative_jacobian_ratio": float(cfg.max_negative_jacobian_ratio),
        "min_jacobian_ratio": float(np.min(ratios)) if checked else math.nan,
        "median_jacobian_ratio": float(np.median(ratios)) if checked else math.nan,
        "max_jacobian_ratio": float(np.max(ratios)) if checked else math.nan,
    }


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}
    ).fillna(False)


def _distribution_summary_or_nan(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "min": math.nan,
            "median": math.nan,
            "mean": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "max": math.nan,
        }
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _empty_anchor_only_landmarks() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "kind",
            "landmark_kind",
            "used_for_tps",
            "accepted_for_tps",
            "source",
            "fixed_node_id",
            "moving_node_id",
            "fixed_feature_index",
            "moving_feature_index",
            "fixed_status",
            "moving_status",
            "fixed_label",
            "moving_label",
            "src_x",
            "src_y",
            "dst_x",
            "dst_y",
            "dx",
            "dy",
            "input_residual_um",
            "source_distance_um",
            "match_cost_um",
            "total_score",
        ]
    )


def _validate_config(cfg: LabelFreeContourAlignmentConfig) -> None:
    if cfg.maxiter < 0:
        raise ValueError("maxiter must be non-negative.")
    if cfg.affine_fallback_iou_threshold < 0:
        raise ValueError("affine_fallback_iou_threshold must be non-negative.")
    if cfg.sampling_distance_um <= 0:
        raise ValueError("sampling_distance_um must be greater than 0.")
    if cfg.max_landmark_distance_um <= 0:
        raise ValueError("max_landmark_distance_um must be greater than 0.")
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
    if cfg.rbf_smoothing < 0:
        raise ValueError("rbf_smoothing must be non-negative.")
    if cfg.topology_grid_size < 0:
        raise ValueError("topology_grid_size must be non-negative.")
    if cfg.topology_min_area_ratio <= 0:
        raise ValueError("topology_min_area_ratio must be positive.")
    if cfg.topology_max_area_ratio <= cfg.topology_min_area_ratio:
        raise ValueError("topology_max_area_ratio must exceed topology_min_area_ratio.")
    if cfg.min_component_area_um2 < 0:
        raise ValueError("min_component_area_um2 must be non-negative.")
    if not (0 < cfg.max_component_weight <= 1):
        raise ValueError("max_component_weight must be in (0, 1].")
    if cfg.boundary_sample_count < 10:
        raise ValueError("boundary_sample_count must be at least 10.")
    if cfg.component_sample_count < 1:
        raise ValueError("component_sample_count must be at least 1.")
    if cfg.search_window <= 0:
        raise ValueError("search_window must be greater than 0.")
    if cfg.knn_neighbors < 1:
        raise ValueError("knn_neighbors must be at least 1.")
    if not (0 <= cfg.min_review_score <= cfg.min_anchor_score <= 1):
        raise ValueError("Require 0 <= min_review_score <= min_anchor_score <= 1.")
    if cfg.min_anchor_count < 1:
        raise ValueError("min_anchor_count must be at least 1.")
    if cfg.area_ratio_min <= 0 or cfg.area_ratio_max <= cfg.area_ratio_min:
        raise ValueError("area_ratio_min/max must be positive and increasing.")
    if not (0 < cfg.envelope_area_fraction <= 1):
        raise ValueError("envelope_area_fraction must be in (0, 1].")
    if not (0 < cfg.envelope_bbox_fraction <= 1):
        raise ValueError("envelope_bbox_fraction must be in (0, 1].")
    if cfg.overlap_candidate_count < 1:
        raise ValueError("overlap_candidate_count must be at least 1.")
    if cfg.overlap_ransac_trials < 1:
        raise ValueError("overlap_ransac_trials must be at least 1.")
    if not (0 <= cfg.overlap_min_descriptor_score <= 1):
        raise ValueError("overlap_min_descriptor_score must be in [0, 1].")
    if cfg.group_correspondence and not cfg.partial_correspondence:
        raise ValueError("group_correspondence requires partial_correspondence=True.")
    if cfg.group_candidate_count < 1:
        raise ValueError("group_candidate_count must be at least 1.")
    if cfg.group_ransac_trials < 1:
        raise ValueError("group_ransac_trials must be at least 1.")
    if not (0 <= cfg.group_min_descriptor_score <= 1):
        raise ValueError("group_min_descriptor_score must be in [0, 1].")
    if cfg.group_residual_limit_um <= 0:
        raise ValueError("group_residual_limit_um must be greater than 0.")
    if cfg.group_min_component_area_um2 < 0:
        raise ValueError("group_min_component_area_um2 must be non-negative.")


def _records_from_geojson_label_free(
    payload: dict[str, Any],
    cfg: LabelFreeContourAlignmentConfig,
    *,
    role: str,
) -> list[_FeatureRecord]:
    records: list[_FeatureRecord] = []
    for feature in payload.get("features", []):
        if feature.get("geometry") is None:
            continue
        geom = _valid_polygonal_geometry(shape(feature["geometry"]))
        if geom is None:
            continue
        if float(geom.area) < cfg.min_component_area_um2:
            continue
        records.append(_FeatureRecord(feature=feature, group="__label_free__", geometry=geom))
    if not records:
        raise ValueError(f"{role} GeoJSON contains no valid polygonal contour features.")
    return records


def _valid_polygonal_geometry(geom: Any) -> Any | None:
    if geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polygons = [part for part in geom.geoms if isinstance(part, (Polygon, MultiPolygon))]
        if not polygons:
            return None
        return unary_union(polygons)
    return None


def _sample_boundary_points(boundary: Any, max_points: int) -> np.ndarray:
    lines = list(_iter_line_parts(boundary))
    lengths = np.array([float(line.length) for line in lines], dtype=float)
    valid = lengths > 0
    lines = [line for line, keep in zip(lines, valid) if keep]
    lengths = lengths[valid]
    if not lines:
        return np.empty((0, 2), dtype=float)
    total = float(lengths.sum())
    spacing = max(total / max(int(max_points), 1), 1e-6)
    points: list[tuple[float, float]] = []
    for line in lines:
        distances = np.arange(0.0, float(line.length), spacing)
        if len(distances) == 0:
            distances = np.array([0.0])
        for distance in distances:
            point = line.interpolate(float(distance))
            points.append((float(point.x), float(point.y)))
    if len(points) > max_points:
        indexes = np.linspace(0, len(points) - 1, int(max_points), dtype=int)
        points = [points[int(index)] for index in indexes]
    return np.asarray(points, dtype=float)


def _iter_line_parts(geom: Any) -> Iterable[Any]:
    if geom.is_empty:
        return
    if geom.geom_type in {"LineString", "LinearRing"}:
        yield geom
        return
    if hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from _iter_line_parts(part)


def _component_layout_table(
    records: Sequence[_FeatureRecord],
    cfg: LabelFreeContourAlignmentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        geom = record.geometry
        area = float(geom.area)
        if area <= 0:
            continue
        centroid = geom.centroid
        rows.append(
            {
                "component_index": index,
                "area": area,
                "centroid_x": float(centroid.x),
                "centroid_y": float(centroid.y),
                "bounds_min_x": float(geom.bounds[0]),
                "bounds_min_y": float(geom.bounds[1]),
                "bounds_max_x": float(geom.bounds[2]),
                "bounds_max_y": float(geom.bounds[3]),
                "source_label": _original_label(record.feature),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("area", ascending=False).head(int(cfg.component_sample_count)).copy()
    raw = df["area"].to_numpy(dtype=float)
    weights = raw / max(float(raw.sum()), 1e-12)
    weights = np.minimum(weights, float(cfg.max_component_weight))
    df["layout_weight"] = weights
    return df.reset_index(drop=True)


def _original_label(feature: dict[str, Any]) -> str:
    props = feature.get("properties") or {}
    for key in ("assigned_structure", "structure", "name", "structure_id"):
        if key in props:
            return str(props[key])
    return ""


def _write_component_qc(
    fixed_components: pd.DataFrame,
    moving_components: pd.DataFrame,
    output_csv: Path,
) -> dict[str, Any]:
    rows = []
    for role, table in (("fixed", fixed_components), ("moving", moving_components)):
        for _, row in table.iterrows():
            rows.append(
                {
                    "role": role,
                    "component_index": int(row["component_index"]),
                    "source_label": str(row.get("source_label", "")),
                    "area": float(row["area"]),
                    "centroid_x": float(row["centroid_x"]),
                    "centroid_y": float(row["centroid_y"]),
                    "layout_weight": float(row["layout_weight"]),
                }
            )
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return {
        "fixed_component_count": int(len(fixed_components)),
        "moving_component_count": int(len(moving_components)),
        "max_component_weight": float(
            max(
                fixed_components["layout_weight"].max() if len(fixed_components) else 0.0,
                moving_components["layout_weight"].max() if len(moving_components) else 0.0,
            )
        ),
    }


def _run_partial_correspondence_diagnostic(
    *,
    cfg: LabelFreeContourAlignmentConfig,
    out_dir: Path,
    moving_payload: dict[str, Any],
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    fixed_union: Any,
    moving_union: Any,
) -> LabelFreeContourAlignmentResult:
    nodes_path = out_dir / "partial_correspondence_nodes.csv"
    matches_path = out_dir / "partial_correspondence_matches.csv"
    summary_path = out_dir / "partial_correspondence_summary.json"
    matrix_html_path = out_dir / "partial_correspondence_matrix.html"
    aligned_path = out_dir / "moving_partial_anchor_aligned.geojson"
    anchor_summary_path = out_dir / "partial_anchor_alignment_summary.json"
    anchor_landmarks_path = out_dir / "partial_anchor_alignment_anchors.csv"
    anchor_overlay_html_path = out_dir / "partial_anchor_alignment_overlay.html"
    anchor_before_png = out_dir / "partial_anchor_overlay_before.png"
    anchor_after_png = out_dir / "partial_anchor_overlay_after.png"
    group_matrix_csv_path = None
    group_matrix_html_path = None
    if cfg.group_correspondence:
        aligned_path = out_dir / "moving_group_overlap_aligned.geojson"
        anchor_summary_path = out_dir / "group_overlap_alignment_summary.json"
        anchor_landmarks_path = out_dir / "group_ransac_anchors.csv"
        anchor_overlay_html_path = out_dir / "group_overlap_alignment_overlay.html"
        anchor_before_png = out_dir / "group_overlap_overlay_before.png"
        anchor_after_png = out_dir / "group_overlap_overlay_after.png"
        group_matrix_csv_path = out_dir / "group_correspondence_matrix.csv"
        group_matrix_html_path = out_dir / "group_correspondence_matrix.html"
    elif cfg.overlap_ransac:
        aligned_path = out_dir / "moving_overlap_anchor_aligned.geojson"
        anchor_summary_path = out_dir / "overlap_anchor_alignment_summary.json"
        anchor_landmarks_path = out_dir / "overlap_anchor_alignment_anchors.csv"
        anchor_overlay_html_path = out_dir / "overlap_anchor_alignment_overlay.html"
        anchor_before_png = out_dir / "overlap_anchor_overlay_before.png"
        anchor_after_png = out_dir / "overlap_anchor_overlay_after.png"

    if (
        cfg.diagnostic_only
        and nodes_path.exists()
        and matches_path.exists()
        and summary_path.exists()
        and matrix_html_path.exists()
        and not cfg.overwrite
    ):
        return LabelFreeContourAlignmentResult(
            out_dir=out_dir,
            aligned_geojson=None,
            summary_json=summary_path,
            landmarks_csv=None,
            overlay_html=None,
            partial_nodes_csv=nodes_path,
            partial_matches_csv=matches_path,
            partial_matrix_html=matrix_html_path,
        )

    if (
        not cfg.diagnostic_only
        and aligned_path.exists()
        and anchor_summary_path.exists()
        and anchor_landmarks_path.exists()
        and anchor_overlay_html_path.exists()
        and not cfg.overwrite
    ):
        return LabelFreeContourAlignmentResult(
            out_dir=out_dir,
            aligned_geojson=aligned_path,
            summary_json=anchor_summary_path,
            landmarks_csv=anchor_landmarks_path,
            overlay_html=anchor_overlay_html_path,
            overlay_before_png=anchor_before_png if anchor_before_png.exists() else None,
            overlay_after_png=anchor_after_png if anchor_after_png.exists() else None,
            partial_nodes_csv=nodes_path if nodes_path.exists() else None,
            partial_matches_csv=matches_path if matches_path.exists() else None,
            partial_matrix_html=matrix_html_path if matrix_html_path.exists() else None,
            group_matrix_csv=group_matrix_csv_path if group_matrix_csv_path and group_matrix_csv_path.exists() else None,
            group_matrix_html=group_matrix_html_path if group_matrix_html_path and group_matrix_html_path.exists() else None,
        )

    fixed_nodes = _build_contour_nodes(
        fixed_records,
        role="fixed",
        tissue_union=fixed_union,
        cfg=cfg,
    )
    moving_nodes = _build_contour_nodes(
        moving_records,
        role="moving",
        tissue_union=moving_union,
        cfg=cfg,
    )
    matches = _build_partial_candidate_matches(fixed_nodes, moving_nodes, cfg)
    fixed_nodes, moving_nodes, matches = _classify_partial_correspondence(
        fixed_nodes,
        moving_nodes,
        matches,
        cfg,
    )
    nodes_df = _nodes_dataframe(fixed_nodes + moving_nodes, matches)
    matches_df = pd.DataFrame(matches)
    if matches_df.empty:
        matches_df = pd.DataFrame(
            columns=[
                "fixed_node_id",
                "moving_node_id",
                "distance",
                "area_ratio",
                "position_score",
                "topology_score",
                "area_score",
                "shape_score",
                "total_score",
                "fixed_rank",
                "moving_rank",
                "is_mutual_first",
                "is_mutual_top3",
                "classification",
                "rejection_codes",
            ]
        )

    nodes_df.to_csv(nodes_path, index=False)
    matches_df.to_csv(matches_path, index=False)

    summary = _partial_correspondence_summary(
        fixed_nodes=fixed_nodes,
        moving_nodes=moving_nodes,
        matches_df=matches_df,
        fixed_union=fixed_union,
        cfg=cfg,
    )
    summary.update(
        {
            "fixed_geojson": str(Path(cfg.fixed_geojson)),
            "moving_geojson": str(Path(cfg.moving_geojson)),
            "method": "label_free_partial_correspondence_diagnostic",
            "diagnostic_only": True,
            "coordinate_warp_performed": False,
            "partial_correspondence_nodes_csv": str(nodes_path),
            "partial_correspondence_matches_csv": str(matches_path),
            "partial_correspondence_matrix_html": str(matrix_html_path),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_partial_correspondence_html(
        fixed_nodes=fixed_nodes,
        moving_nodes=moving_nodes,
        matches_df=matches_df,
        output_html=matrix_html_path,
        summary=summary,
    )

    if not cfg.diagnostic_only:
        return _run_partial_anchor_alignment(
            cfg=cfg,
            out_dir=out_dir,
            moving_payload=moving_payload,
            fixed_records=fixed_records,
            moving_records=moving_records,
            fixed_union=fixed_union,
            moving_union=moving_union,
            matches_df=matches_df,
            partial_summary=summary,
            aligned_path=aligned_path,
            anchor_summary_path=anchor_summary_path,
            anchor_landmarks_path=anchor_landmarks_path,
            anchor_overlay_html_path=anchor_overlay_html_path,
            anchor_before_png=anchor_before_png,
            anchor_after_png=anchor_after_png,
            nodes_path=nodes_path,
            matches_path=matches_path,
            matrix_html_path=matrix_html_path,
            group_matrix_csv_path=group_matrix_csv_path,
            group_matrix_html_path=group_matrix_html_path,
        )

    return LabelFreeContourAlignmentResult(
        out_dir=out_dir,
        aligned_geojson=None,
        summary_json=summary_path,
        landmarks_csv=None,
        overlay_html=None,
        partial_nodes_csv=nodes_path,
        partial_matches_csv=matches_path,
        partial_matrix_html=matrix_html_path,
    )


def _run_partial_anchor_alignment(
    *,
    cfg: LabelFreeContourAlignmentConfig,
    out_dir: Path,
    moving_payload: dict[str, Any],
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    fixed_union: Any,
    moving_union: Any,
    matches_df: pd.DataFrame,
    partial_summary: dict[str, Any],
    aligned_path: Path,
    anchor_summary_path: Path,
    anchor_landmarks_path: Path,
    anchor_overlay_html_path: Path,
    anchor_before_png: Path,
    anchor_after_png: Path,
    nodes_path: Path,
    matches_path: Path,
    matrix_html_path: Path,
    group_matrix_csv_path: Path | None,
    group_matrix_html_path: Path | None,
) -> LabelFreeContourAlignmentResult:
    if cfg.group_correspondence:
        transform, transform_summary, anchor_rows, group_matrix = _estimate_group_correspondence_similarity(
            fixed_records=fixed_records,
            moving_records=moving_records,
            fixed_union=fixed_union,
            moving_union=moving_union,
            cfg=cfg,
        )
        if group_matrix_csv_path is not None:
            group_matrix.to_csv(group_matrix_csv_path, index=False)
        if group_matrix_html_path is not None:
            _write_group_correspondence_matrix_html(group_matrix, group_matrix_html_path)
        method = "label_free_group_correspondence_ransac_alignment"
        objective_note = (
            "Contours are first grouped by their within-slice assigned_structure labels. "
            "All fixed-group/moving-group combinations are evaluated, allowing a fixed group "
            "such as Structure 2 to match a differently named moving group such as Structure 3. "
            "Only RANSAC inliers from the best cross-group constellation estimate the rigid transform; "
            "labels are preserved and not harmonized."
        )
    elif cfg.overlap_ransac:
        transform, transform_summary, anchor_rows = _estimate_overlap_ransac_similarity(
            fixed_records=fixed_records,
            moving_records=moving_records,
            fixed_union=fixed_union,
            moving_union=moving_union,
            cfg=cfg,
        )
        method = "label_free_overlap_ransac_similarity_alignment"
        objective_note = (
            "A descriptor-first RANSAC search estimates the transform from a subset of contours "
            "that behave like a shared overlap region. Candidate pairs are generated without a "
            "current-coordinate distance prefilter, so non-overlapping contours do not force the fit."
        )
    else:
        anchor_rows = _select_transform_anchor_rows(matches_df)
        transform, transform_summary, anchor_rows = _estimate_anchor_only_similarity(
            anchor_rows,
            cfg,
        )
        method = "label_free_partial_anchor_similarity_alignment"
        objective_note = (
            "Only one-to-one matched_anchor contour centroid pairs from the v2 partial-correspondence "
            "diagnostic are used to estimate this transform. matched_review and no_counterpart "
            "contours follow the resulting background transform but do not contribute to fitting."
        )
    aligned_payload = _apply_point_transform_to_geojson(moving_payload, transform)
    final_records = _records_from_geojson_label_free(aligned_payload, cfg, role="anchor_aligned")

    aligned_path.write_text(json.dumps(aligned_payload, ensure_ascii=False), encoding="utf-8")
    anchor_rows.to_csv(anchor_landmarks_path, index=False)

    overlay_paths = {"before": None, "after": None}
    if cfg.save_preview_png:
        overlay_paths = _write_overlays(
            fixed_records=fixed_records,
            moving_records=moving_records,
            final_records=final_records,
            before_png=anchor_before_png,
            after_png=anchor_after_png,
            dpi=cfg.dpi,
        )
    _write_partial_anchor_alignment_html(
        fixed_records=fixed_records,
        moving_records=moving_records,
        final_records=final_records,
        anchor_rows=anchor_rows,
        output_html=anchor_overlay_html_path,
        transform_summary=transform_summary,
    )

    transformed_union = unary_union([record.geometry for record in final_records])
    summary = {
        "fixed_geojson": str(Path(cfg.fixed_geojson)),
        "moving_geojson": str(Path(cfg.moving_geojson)),
        "output_geojson": str(aligned_path),
        "method": method,
        "objective_note": objective_note,
        "diagnostic_only": False,
        "coordinate_warp_performed": True,
        "labels_used_for_matching": False,
        "semantic_harmonization_performed": False,
        "transform_type": transform.kind,
        "transform": _transform_payload(transform),
        "anchor_transform": transform_summary,
        "global_context_scores_not_used_for_fitting": {
            "union_iou_before": _iou(fixed_union, moving_union),
            "union_iou_after": _iou(fixed_union, transformed_union),
        },
        "partial_correspondence_summary": partial_summary,
        "partial_correspondence_nodes_csv": str(nodes_path),
        "partial_correspondence_matches_csv": str(matches_path),
        "partial_correspondence_matrix_html": str(matrix_html_path),
        "group_correspondence_matrix_csv": str(group_matrix_csv_path) if group_matrix_csv_path else None,
        "group_correspondence_matrix_html": str(group_matrix_html_path) if group_matrix_html_path else None,
        "anchor_landmarks_csv": str(anchor_landmarks_path),
        "overlay_html": str(anchor_overlay_html_path),
        "overlay_before_png": str(overlay_paths["before"]) if overlay_paths["before"] else None,
        "overlay_after_png": str(overlay_paths["after"]) if overlay_paths["after"] else None,
    }
    anchor_summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return LabelFreeContourAlignmentResult(
        out_dir=out_dir,
        aligned_geojson=aligned_path,
        summary_json=anchor_summary_path,
        landmarks_csv=anchor_landmarks_path,
        overlay_html=anchor_overlay_html_path,
        overlay_before_png=overlay_paths["before"],
        overlay_after_png=overlay_paths["after"],
        partial_nodes_csv=nodes_path,
        partial_matches_csv=matches_path,
        partial_matrix_html=matrix_html_path,
        group_matrix_csv=group_matrix_csv_path,
        group_matrix_html=group_matrix_html_path,
    )


def _select_transform_anchor_rows(matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df.empty or "classification" not in matches_df.columns:
        return pd.DataFrame()
    anchors = matches_df.loc[matches_df["classification"] == "matched_anchor"].copy()
    if anchors.empty:
        return anchors
    anchors["_is_mutual_first_sort"] = anchors["is_mutual_first"].astype(bool).astype(int)
    anchors = anchors.sort_values(
        ["_is_mutual_first_sort", "total_score", "topology_score", "distance"],
        ascending=[False, False, False, True],
    )
    used_fixed: set[str] = set()
    used_moving: set[str] = set()
    selected: list[dict[str, Any]] = []
    for row in anchors.to_dict("records"):
        fixed_id = str(row["fixed_node_id"])
        moving_id = str(row["moving_node_id"])
        if fixed_id in used_fixed or moving_id in used_moving:
            continue
        used_fixed.add(fixed_id)
        used_moving.add(moving_id)
        selected.append(row)
    result = pd.DataFrame(selected)
    if "_is_mutual_first_sort" in result.columns:
        result = result.drop(columns=["_is_mutual_first_sort"])
    return result.reset_index(drop=True)


def _estimate_anchor_only_similarity(
    anchor_rows: pd.DataFrame,
    cfg: LabelFreeContourAlignmentConfig,
    *,
    allow_scale: bool = True,
) -> tuple[_PointTransform, dict[str, Any], pd.DataFrame]:
    if anchor_rows.empty:
        transform = _PointTransform(kind="identity")
        return transform, {
            "accepted": False,
            "reason": "no_matched_anchor_pairs",
            "anchor_pair_count": 0,
            "used_anchor_pair_count": 0,
        }, anchor_rows.copy()

    rows = anchor_rows.copy()
    source = rows[["moving_centroid_x", "moving_centroid_y"]].to_numpy(dtype=float)
    target = rows[["fixed_centroid_x", "fixed_centroid_y"]].to_numpy(dtype=float)
    weights = rows["total_score"].to_numpy(dtype=float)
    weights = np.clip(weights, 1e-6, None)

    if len(rows) == 1:
        dx, dy = target[0] - source[0]
        transform = _PointTransform(
            kind="similarity",
            similarity=_SimilarityTransform(
                origin_x=float(source[0, 0]),
                origin_y=float(source[0, 1]),
                rotation_degrees=0.0,
                scale=1.0,
                translate_x=float(dx),
                translate_y=float(dy),
            ),
        )
        used_mask = np.array([True])
    else:
        used_mask = np.ones(len(rows), dtype=bool)
        for _ in range(3):
            candidate = _fit_weighted_similarity_from_points(
                source[used_mask],
                target[used_mask],
                weights[used_mask],
                allow_scale=allow_scale,
            )
            residuals = np.linalg.norm(
                _apply_point_transform_to_points(source, candidate) - target,
                axis=1,
            )
            median = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - median)))
            cutoff = min(
                max(80.0, median + 3.0 * 1.4826 * mad),
                max(80.0, float(cfg.search_window) * 0.75),
            )
            next_mask = residuals <= cutoff
            if int(next_mask.sum()) < max(2, min(8, len(rows) // 2)):
                break
            if np.array_equal(next_mask, used_mask):
                break
            used_mask = next_mask
        transform = _fit_weighted_similarity_from_points(
            source[used_mask],
            target[used_mask],
            weights[used_mask],
            allow_scale=allow_scale,
        )

    aligned_source = _apply_point_transform_to_points(source, transform)
    residuals = np.linalg.norm(aligned_source - target, axis=1)
    rows["used_for_transform"] = used_mask
    rows["aligned_moving_centroid_x"] = aligned_source[:, 0]
    rows["aligned_moving_centroid_y"] = aligned_source[:, 1]
    rows["anchor_residual"] = residuals
    used_residuals = residuals[used_mask]
    warnings: list[str] = []
    if int(used_mask.sum()) < cfg.min_anchor_count:
        warnings.append("too_few_transform_anchors")
    return transform, {
        "accepted": bool(int(used_mask.sum()) >= 1),
        "reason": "fit_from_matched_anchor_pairs",
        "anchor_pair_count": int(len(rows)),
        "used_anchor_pair_count": int(used_mask.sum()),
        "discarded_anchor_pair_count": int((~used_mask).sum()),
        "residual_median": float(np.median(used_residuals)) if len(used_residuals) else math.inf,
        "residual_p90": float(np.percentile(used_residuals, 90)) if len(used_residuals) else math.inf,
        "residual_max": float(np.max(used_residuals)) if len(used_residuals) else math.inf,
        "warnings": warnings,
    }, rows


def _fit_weighted_similarity_from_points(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    allow_scale: bool = True,
) -> _PointTransform:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(source) != len(target) or len(source) != len(weights):
        raise ValueError("source, target, and weights must have the same length.")
    if len(source) < 1:
        return _PointTransform(kind="identity")
    weights = np.clip(weights, 1e-9, None)
    weights = weights / float(weights.sum())
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    if len(source) < 2 or float(np.sum(weights * np.sum(source_centered**2, axis=1))) <= 1e-12:
        translate = target_center - source_center
        similarity = _SimilarityTransform(
            origin_x=float(source_center[0]),
            origin_y=float(source_center[1]),
            rotation_degrees=0.0,
            scale=1.0,
            translate_x=float(translate[0]),
            translate_y=float(translate[1]),
        )
        return _PointTransform(kind="similarity", similarity=similarity)

    covariance = source_centered.T @ (target_centered * weights[:, None])
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    row_rotation = u_matrix @ vt_matrix
    if np.linalg.det(row_rotation) < 0:
        vt_matrix[-1, :] *= -1
        row_rotation = u_matrix @ vt_matrix
    variance = float(np.sum(weights * np.sum(source_centered**2, axis=1)))
    scale = float(np.sum(singular_values) / max(variance, 1e-12)) if allow_scale else 1.0
    scale = max(scale, 1e-9)
    rotation_degrees = math.degrees(math.atan2(row_rotation[0, 1], row_rotation[0, 0]))
    similarity = _SimilarityTransform(
        origin_x=float(source_center[0]),
        origin_y=float(source_center[1]),
        rotation_degrees=float(rotation_degrees),
        scale=float(scale),
        translate_x=float(target_center[0] - source_center[0]),
        translate_y=float(target_center[1] - source_center[1]),
    )
    return _PointTransform(kind="similarity", similarity=similarity)


def _estimate_overlap_ransac_similarity(
    *,
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    fixed_union: Any,
    moving_union: Any,
    cfg: LabelFreeContourAlignmentConfig,
) -> tuple[_PointTransform, dict[str, Any], pd.DataFrame]:
    fixed_nodes = _build_contour_nodes(
        fixed_records,
        role="fixed",
        tissue_union=fixed_union,
        cfg=cfg,
    )
    moving_nodes = _build_contour_nodes(
        moving_records,
        role="moving",
        tissue_union=moving_union,
        cfg=cfg,
    )
    candidates = _build_overlap_descriptor_candidates(fixed_nodes, moving_nodes, cfg)
    if candidates.empty:
        return _PointTransform(kind="identity"), {
            "accepted": False,
            "reason": "no_descriptor_candidates",
            "candidate_pair_count": 0,
            "anchor_pair_count": 0,
            "used_anchor_pair_count": 0,
        }, pd.DataFrame()

    candidate_records = candidates.to_dict("records")
    source = candidates[["moving_centroid_x", "moving_centroid_y"]].to_numpy(dtype=float)
    target = candidates[["fixed_centroid_x", "fixed_centroid_y"]].to_numpy(dtype=float)
    candidate_weights = candidates["descriptor_score"].to_numpy(dtype=float)
    trial_pairs = _overlap_ransac_trial_indices(candidates, cfg)
    best: tuple[float, _PointTransform, pd.DataFrame] | None = None
    for first_idx, second_idx in trial_pairs:
        transform = _similarity_from_candidate_pair(
            candidate_records[first_idx],
            candidate_records[second_idx],
            cfg,
        )
        if transform is None:
            continue
        selected = _score_overlap_transform_candidates(
            candidates,
            source,
            target,
            candidate_weights,
            transform,
            cfg,
        )
        if selected.empty:
            continue
        score = _overlap_transform_score(selected, fixed_union)
        if best is None or score > best[0]:
            best = (score, transform, selected)

    if best is None:
        return _PointTransform(kind="identity"), {
            "accepted": False,
            "reason": "no_valid_ransac_transform",
            "candidate_pair_count": int(len(candidates)),
            "anchor_pair_count": 0,
            "used_anchor_pair_count": 0,
        }, pd.DataFrame()

    _, _, selected = best
    transform, transform_summary, selected = _refine_overlap_transform(selected, cfg)
    transform_summary.update(
        {
            "accepted": True,
            "reason": "fit_from_overlap_ransac_descriptor_pairs",
            "candidate_pair_count": int(len(candidates)),
            "ransac_trial_count": int(len(trial_pairs)),
        }
    )
    return transform, transform_summary, selected


def _estimate_group_correspondence_similarity(
    *,
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    fixed_union: Any,
    moving_union: Any,
    cfg: LabelFreeContourAlignmentConfig,
) -> tuple[_PointTransform, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    fixed_nodes = _nodes_with_group_context(
        [
            node
            for node in _build_contour_nodes(
                fixed_records,
                role="fixed",
                tissue_union=fixed_union,
                cfg=cfg,
            )
            if node.area >= cfg.group_min_component_area_um2
        ],
        cfg,
    )
    moving_nodes = _nodes_with_group_context(
        [
            node
            for node in _build_contour_nodes(
                moving_records,
                role="moving",
                tissue_union=moving_union,
                cfg=cfg,
            )
            if node.area >= cfg.group_min_component_area_um2
        ],
        cfg,
    )
    fixed_groups = _nodes_by_label(fixed_nodes)
    moving_groups = _nodes_by_label(moving_nodes)
    matrix_rows: list[dict[str, Any]] = []
    best: tuple[float, _PointTransform, pd.DataFrame, dict[str, Any]] | None = None
    for fixed_group, fixed_group_nodes in sorted(fixed_groups.items()):
        for moving_group, moving_group_nodes in sorted(moving_groups.items()):
            if len(fixed_group_nodes) < 3 or len(moving_group_nodes) < 3:
                continue
            candidates = _build_group_descriptor_candidates(
                fixed_group_nodes,
                moving_group_nodes,
                fixed_group=fixed_group,
                moving_group=moving_group,
                cfg=cfg,
            )
            pair_summary: dict[str, Any] = {
                "fixed_group": fixed_group,
                "moving_group": moving_group,
                "candidate_count": int(len(candidates)),
                "trial_count": 0,
                "inlier_count": 0,
                "score": -math.inf,
                "median_residual": math.inf,
                "rotation_degrees": math.nan,
                "scale": math.nan,
                "translate_x": math.nan,
                "translate_y": math.nan,
            }
            if len(candidates) >= 2:
                pair_best = _run_group_pair_ransac(candidates, fixed_union, cfg)
                if pair_best is not None:
                    score, transform, selected, trial_count = pair_best
                    residuals = selected["anchor_residual"].to_numpy(dtype=float)
                    pair_summary.update(
                        {
                            "trial_count": int(trial_count),
                            "inlier_count": int(len(selected)),
                            "score": float(score),
                            "median_residual": float(np.median(residuals)) if len(residuals) else math.inf,
                            "rotation_degrees": (
                                float(transform.similarity.rotation_degrees)
                                if transform.similarity is not None
                                else math.nan
                            ),
                            "scale": (
                                float(transform.similarity.scale)
                                if transform.similarity is not None
                                else math.nan
                            ),
                            "translate_x": (
                                float(transform.similarity.translate_x)
                                if transform.similarity is not None
                                else math.nan
                            ),
                            "translate_y": (
                                float(transform.similarity.translate_y)
                                if transform.similarity is not None
                                else math.nan
                            ),
                        }
                    )
                    if best is None or score > best[0]:
                        best = (score, transform, selected, pair_summary.copy())
            matrix_rows.append(pair_summary)

    matrix = pd.DataFrame(matrix_rows)
    if not matrix.empty:
        matrix = matrix.sort_values(
            ["score", "inlier_count", "median_residual"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        matrix.insert(0, "rank", np.arange(1, len(matrix) + 1, dtype=int))

    if best is None:
        return _PointTransform(kind="identity"), {
            "accepted": False,
            "reason": "no_valid_group_correspondence",
            "anchor_pair_count": 0,
            "used_anchor_pair_count": 0,
        }, pd.DataFrame(), matrix

    _, _, selected, pair_summary = best
    transform, transform_summary, selected = _refine_overlap_transform(selected, cfg)
    transform_summary.update(
        {
            "accepted": True,
            "reason": "fit_from_group_correspondence_ransac_pairs",
            "fixed_group": pair_summary["fixed_group"],
            "moving_group": pair_summary["moving_group"],
            "group_pair_score": float(pair_summary["score"]),
            "candidate_pair_count": int(pair_summary["candidate_count"]),
            "ransac_trial_count": int(pair_summary["trial_count"]),
        }
    )
    return transform, transform_summary, selected, matrix


def _nodes_with_group_context(
    nodes: list[_ContourNode],
    cfg: LabelFreeContourAlignmentConfig,
) -> list[_ContourNode]:
    grouped = _nodes_by_label(nodes, include_envelopes=False)
    result: list[_ContourNode] = []
    for _, group_nodes in sorted(grouped.items()):
        partial = [
            {
                "centroid_x": node.centroid_x,
                "centroid_y": node.centroid_y,
                "area": node.area,
            }
            for node in group_nodes
        ]
        descriptors = _knn_context_descriptors(partial, int(cfg.knn_neighbors))
        for node, descriptor in zip(group_nodes, descriptors):
            result.append(
                replace(
                    node,
                    knn_distances=tuple(float(x) for x in descriptor["distances"]),
                    knn_angle_gaps=tuple(float(x) for x in descriptor["angle_gaps"]),
                    knn_log_area_ratios=tuple(float(x) for x in descriptor["log_area_ratios"]),
                )
            )
    return result


def _nodes_by_label(
    nodes: list[_ContourNode],
    *,
    include_envelopes: bool = False,
) -> dict[str, list[_ContourNode]]:
    grouped: dict[str, list[_ContourNode]] = {}
    for node in nodes:
        if node.envelope_only and not include_envelopes:
            continue
        grouped.setdefault(str(node.source_label), []).append(node)
    return grouped


def _build_group_descriptor_candidates(
    fixed_nodes: list[_ContourNode],
    moving_nodes: list[_ContourNode],
    *,
    fixed_group: str,
    moving_group: str,
    cfg: LabelFreeContourAlignmentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fixed in fixed_nodes:
        local_rows: list[dict[str, Any]] = []
        for moving in moving_nodes:
            area_ratio = moving.area / max(fixed.area, 1e-12)
            area_score = _wide_area_similarity(area_ratio)
            if area_score <= 0:
                continue
            topology_score = _topology_similarity(fixed, moving)
            shape_score = _shape_similarity(fixed, moving)
            descriptor_score = (
                0.45 * topology_score
                + 0.25 * shape_score
                + 0.30 * area_score
            )
            if descriptor_score < cfg.group_min_descriptor_score:
                continue
            local_rows.append(
                {
                    "fixed_group": fixed_group,
                    "moving_group": moving_group,
                    "fixed_node_id": fixed.node_id,
                    "moving_node_id": moving.node_id,
                    "fixed_feature_index": int(fixed.feature_index),
                    "moving_feature_index": int(moving.feature_index),
                    "fixed_label": fixed.source_label,
                    "moving_label": moving.source_label,
                    "fixed_centroid_x": fixed.centroid_x,
                    "fixed_centroid_y": fixed.centroid_y,
                    "moving_centroid_x": moving.centroid_x,
                    "moving_centroid_y": moving.centroid_y,
                    "fixed_area": fixed.area,
                    "moving_area": moving.area,
                    "area_ratio": area_ratio,
                    "topology_score": topology_score,
                    "area_score": area_score,
                    "shape_score": shape_score,
                    "descriptor_score": descriptor_score,
                }
            )
        local_rows.sort(key=lambda row: float(row["descriptor_score"]), reverse=True)
        rows.extend(local_rows[: int(cfg.group_candidate_count)])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["descriptor_score", "topology_score", "shape_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _wide_area_similarity(area_ratio: float) -> float:
    if area_ratio <= 0 or area_ratio < 0.1 or area_ratio > 10.0:
        return 0.0
    return float(np.clip(1.0 - abs(math.log(area_ratio)) / math.log(10.0), 0.0, 1.0))


def _run_group_pair_ransac(
    candidates: pd.DataFrame,
    fixed_union: Any,
    cfg: LabelFreeContourAlignmentConfig,
) -> tuple[float, _PointTransform, pd.DataFrame, int] | None:
    candidate_records = candidates.to_dict("records")
    source = candidates[["moving_centroid_x", "moving_centroid_y"]].to_numpy(dtype=float)
    target = candidates[["fixed_centroid_x", "fixed_centroid_y"]].to_numpy(dtype=float)
    weights = candidates["descriptor_score"].to_numpy(dtype=float)
    top_count = min(len(candidates), 350)
    best: tuple[float, _PointTransform, pd.DataFrame, int] | None = None
    trial_count = 0
    for first_idx in range(top_count):
        for second_idx in range(first_idx + 1, top_count):
            first = candidate_records[first_idx]
            second = candidate_records[second_idx]
            if (
                str(first["fixed_node_id"]) == str(second["fixed_node_id"])
                or str(first["moving_node_id"]) == str(second["moving_node_id"])
            ):
                continue
            transform = _similarity_from_candidate_pair(first, second, cfg)
            if transform is None:
                continue
            selected = _score_overlap_transform_candidates(
                candidates,
                source,
                target,
                weights,
                transform,
                cfg,
                residual_limit=float(cfg.group_residual_limit_um),
            )
            trial_count += 1
            if selected.empty:
                continue
            score = _overlap_transform_score(selected, fixed_union)
            if best is None or score > best[0]:
                best = (score, transform, selected, trial_count)
            if trial_count >= cfg.group_ransac_trials:
                return best
    return best


def _build_overlap_descriptor_candidates(
    fixed_nodes: list[_ContourNode],
    moving_nodes: list[_ContourNode],
    cfg: LabelFreeContourAlignmentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fixed_matchable = [node for node in fixed_nodes if not node.envelope_only]
    moving_matchable = [node for node in moving_nodes if not node.envelope_only]
    for fixed in fixed_matchable:
        local_rows: list[dict[str, Any]] = []
        for moving in moving_matchable:
            area_ratio = moving.area / max(fixed.area, 1e-12)
            if area_ratio < cfg.area_ratio_min or area_ratio > cfg.area_ratio_max:
                continue
            topology_score = _topology_similarity(fixed, moving)
            area_score = _area_similarity(area_ratio, cfg)
            shape_score = _shape_similarity(fixed, moving)
            descriptor_score = (
                0.45 * topology_score
                + 0.35 * shape_score
                + 0.20 * area_score
            )
            if descriptor_score < cfg.overlap_min_descriptor_score:
                continue
            local_rows.append(
                {
                    "fixed_node_id": fixed.node_id,
                    "moving_node_id": moving.node_id,
                    "fixed_label": fixed.source_label,
                    "moving_label": moving.source_label,
                    "fixed_centroid_x": fixed.centroid_x,
                    "fixed_centroid_y": fixed.centroid_y,
                    "moving_centroid_x": moving.centroid_x,
                    "moving_centroid_y": moving.centroid_y,
                    "fixed_area": fixed.area,
                    "moving_area": moving.area,
                    "area_ratio": area_ratio,
                    "topology_score": topology_score,
                    "area_score": area_score,
                    "shape_score": shape_score,
                    "descriptor_score": descriptor_score,
                }
            )
        local_rows.sort(key=lambda row: float(row["descriptor_score"]), reverse=True)
        rows.extend(local_rows[: int(cfg.overlap_candidate_count)])
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["descriptor_score", "topology_score", "shape_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _overlap_ransac_trial_indices(
    candidates: pd.DataFrame,
    cfg: LabelFreeContourAlignmentConfig,
) -> list[tuple[int, int]]:
    candidate_count = len(candidates)
    if candidate_count < 2:
        return []
    top_count = min(candidate_count, 260)
    trials: list[tuple[int, int]] = []
    for first in range(top_count):
        for second in range(first + 1, top_count):
            trials.append((first, second))
            if len(trials) >= cfg.overlap_ransac_trials:
                return trials
    if len(trials) >= cfg.overlap_ransac_trials:
        return trials
    rng = np.random.default_rng(20260507)
    max_index = min(candidate_count, 1200)
    seen = set(trials)
    while len(trials) < cfg.overlap_ransac_trials and len(seen) < max_index * (max_index - 1) // 2:
        first, second = sorted(rng.choice(max_index, size=2, replace=False).tolist())
        pair = (int(first), int(second))
        if pair in seen:
            continue
        seen.add(pair)
        trials.append(pair)
    return trials


def _similarity_from_candidate_pair(
    first: dict[str, Any],
    second: dict[str, Any],
    cfg: LabelFreeContourAlignmentConfig,
) -> _PointTransform | None:
    source = np.array(
        [
            [float(first["moving_centroid_x"]), float(first["moving_centroid_y"])],
            [float(second["moving_centroid_x"]), float(second["moving_centroid_y"])],
        ],
        dtype=float,
    )
    target = np.array(
        [
            [float(first["fixed_centroid_x"]), float(first["fixed_centroid_y"])],
            [float(second["fixed_centroid_x"]), float(second["fixed_centroid_y"])],
        ],
        dtype=float,
    )
    source_delta = source[1] - source[0]
    target_delta = target[1] - target[0]
    source_dist = float(np.linalg.norm(source_delta))
    target_dist = float(np.linalg.norm(target_delta))
    if source_dist < 250.0 or target_dist < 250.0:
        return None
    scale = target_dist / source_dist if cfg.overlap_allow_scale else 1.0
    if cfg.overlap_allow_scale and (scale < 0.5 or scale > 2.0):
        return None
    source_angle = math.atan2(float(source_delta[1]), float(source_delta[0]))
    target_angle = math.atan2(float(target_delta[1]), float(target_delta[0]))
    rotation = math.degrees(target_angle - source_angle)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    radians = math.radians(rotation)
    rot = np.array(
        [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]],
        dtype=float,
    )
    moved_center = source_center @ rot.T * scale
    translate = target_center - moved_center
    return _PointTransform(
        kind="similarity",
        similarity=_SimilarityTransform(
            origin_x=0.0,
            origin_y=0.0,
            rotation_degrees=float(rotation),
            scale=float(scale),
            translate_x=float(translate[0]),
            translate_y=float(translate[1]),
        ),
    )


def _score_overlap_transform_candidates(
    candidates: pd.DataFrame,
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    transform: _PointTransform,
    cfg: LabelFreeContourAlignmentConfig,
    *,
    residual_limit: float | None = None,
) -> pd.DataFrame:
    aligned = _apply_point_transform_to_points(source, transform)
    residuals = np.linalg.norm(aligned - target, axis=1)
    work = candidates.copy()
    work["aligned_moving_centroid_x"] = aligned[:, 0]
    work["aligned_moving_centroid_y"] = aligned[:, 1]
    work["anchor_residual"] = residuals
    residual_limit = (
        float(residual_limit)
        if residual_limit is not None
        else max(120.0, min(float(cfg.search_window), 650.0))
    )
    work = work.loc[work["anchor_residual"] <= residual_limit].copy()
    if work.empty:
        return work
    work["support_score"] = (
        work["descriptor_score"].astype(float)
        * np.exp(-work["anchor_residual"].astype(float) / max(residual_limit / 3.0, 1e-6))
    )
    work = work.sort_values(
        ["support_score", "descriptor_score", "anchor_residual"],
        ascending=[False, False, True],
    )
    selected: list[dict[str, Any]] = []
    used_fixed: set[str] = set()
    used_moving: set[str] = set()
    for row in work.to_dict("records"):
        fixed_id = str(row["fixed_node_id"])
        moving_id = str(row["moving_node_id"])
        if fixed_id in used_fixed or moving_id in used_moving:
            continue
        used_fixed.add(fixed_id)
        used_moving.add(moving_id)
        selected.append(row)
    result = pd.DataFrame(selected)
    if not result.empty:
        result["used_for_transform"] = True
    return result


def _overlap_transform_score(selected: pd.DataFrame, fixed_union: Any) -> float:
    if selected.empty:
        return -math.inf
    support = float(selected["support_score"].sum())
    count_bonus = float(len(selected)) * 0.35
    residual_penalty = float(selected["anchor_residual"].median()) / 250.0
    coverage = _anchor_coverage_ratio(
        selected.rename(
            columns={
                "fixed_centroid_x": "fixed_centroid_x",
                "fixed_centroid_y": "fixed_centroid_y",
            }
        ),
        fixed_union,
    )
    return support + count_bonus + 1.5 * math.sqrt(max(coverage, 0.0)) - residual_penalty


def _refine_overlap_transform(
    selected: pd.DataFrame,
    cfg: LabelFreeContourAlignmentConfig,
) -> tuple[_PointTransform, dict[str, Any], pd.DataFrame]:
    if selected.empty:
        return _PointTransform(kind="identity"), {
            "accepted": False,
            "reason": "no_selected_overlap_anchors",
            "anchor_pair_count": 0,
            "used_anchor_pair_count": 0,
        }, selected.copy()
    rows = selected.copy()
    rows["total_score"] = rows["support_score"].astype(float)
    transform, summary, rows = _estimate_anchor_only_similarity(
        rows,
        cfg,
        allow_scale=cfg.overlap_allow_scale,
    )
    summary["anchor_pair_count"] = int(len(selected))
    return transform, summary, rows


def _build_contour_nodes(
    records: list[_FeatureRecord],
    *,
    role: str,
    tissue_union: Any,
    cfg: LabelFreeContourAlignmentConfig,
) -> list[_ContourNode]:
    total_area = max(sum(float(record.geometry.area) for record in records), 1e-12)
    tissue_minx, tissue_miny, tissue_maxx, tissue_maxy = tissue_union.bounds
    tissue_width = max(float(tissue_maxx - tissue_minx), 1e-12)
    tissue_height = max(float(tissue_maxy - tissue_miny), 1e-12)
    partial: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        geom = record.geometry
        area = float(geom.area)
        if area <= 0:
            continue
        minx, miny, maxx, maxy = geom.bounds
        width = float(maxx - minx)
        height = float(maxy - miny)
        area_fraction = area / total_area
        bbox_fraction_x = width / tissue_width
        bbox_fraction_y = height / tissue_height
        exclude_reasons = []
        if area_fraction > cfg.envelope_area_fraction:
            exclude_reasons.append("area_fraction")
        if (
            bbox_fraction_x >= cfg.envelope_bbox_fraction
            and bbox_fraction_y >= cfg.envelope_bbox_fraction
        ):
            exclude_reasons.append("bbox_span")
        perimeter = float(geom.length)
        compactness = float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-12))
        aspect_ratio = float(max(width, height) / max(min(width, height), 1e-12))
        centroid = geom.centroid
        partial.append(
            {
                "role": role,
                "node_id": f"{role}_{index:05d}",
                "feature_index": index,
                "source_label": _original_label(record.feature),
                "geometry": geom,
                "centroid_x": float(centroid.x),
                "centroid_y": float(centroid.y),
                "area": area,
                "perimeter": perimeter,
                "compactness": compactness,
                "aspect_ratio": aspect_ratio,
                "orientation_degrees": _geometry_orientation_degrees(geom),
                "bbox_width": width,
                "bbox_height": height,
                "area_fraction": area_fraction,
                "bbox_fraction_x": bbox_fraction_x,
                "bbox_fraction_y": bbox_fraction_y,
                "envelope_only": bool(exclude_reasons),
                "exclude_reason": ";".join(exclude_reasons),
            }
        )

    descriptors = _knn_context_descriptors(partial, int(cfg.knn_neighbors))
    nodes: list[_ContourNode] = []
    for row, descriptor in zip(partial, descriptors):
        nodes.append(
            _ContourNode(
                role=str(row["role"]),
                node_id=str(row["node_id"]),
                feature_index=int(row["feature_index"]),
                source_label=str(row["source_label"]),
                geometry=row["geometry"],
                centroid_x=float(row["centroid_x"]),
                centroid_y=float(row["centroid_y"]),
                area=float(row["area"]),
                perimeter=float(row["perimeter"]),
                compactness=float(row["compactness"]),
                aspect_ratio=float(row["aspect_ratio"]),
                orientation_degrees=float(row["orientation_degrees"]),
                bbox_width=float(row["bbox_width"]),
                bbox_height=float(row["bbox_height"]),
                area_fraction=float(row["area_fraction"]),
                bbox_fraction_x=float(row["bbox_fraction_x"]),
                bbox_fraction_y=float(row["bbox_fraction_y"]),
                envelope_only=bool(row["envelope_only"]),
                exclude_reason=str(row["exclude_reason"]),
                knn_distances=tuple(float(x) for x in descriptor["distances"]),
                knn_angle_gaps=tuple(float(x) for x in descriptor["angle_gaps"]),
                knn_log_area_ratios=tuple(float(x) for x in descriptor["log_area_ratios"]),
            )
        )
    return nodes


def _geometry_orientation_degrees(geom: Any) -> float:
    try:
        rect = geom.minimum_rotated_rectangle
        coords = np.asarray(rect.exterior.coords, dtype=float)
        if len(coords) < 2:
            return 0.0
        edges = np.diff(coords[:4], axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        edge = edges[int(np.argmax(lengths))]
        return float(math.degrees(math.atan2(edge[1], edge[0])))
    except Exception:
        return 0.0


def _knn_context_descriptors(rows: list[dict[str, Any]], k_neighbors: int) -> list[dict[str, np.ndarray]]:
    if not rows:
        return []
    xy = np.array([[float(row["centroid_x"]), float(row["centroid_y"])] for row in rows])
    areas = np.array([float(row["area"]) for row in rows])
    tree = cKDTree(xy)
    k = min(max(int(k_neighbors), 1) + 1, len(rows))
    distances, indexes = tree.query(xy, k=k)
    distances = np.atleast_2d(distances)
    indexes = np.atleast_2d(indexes)
    descriptors: list[dict[str, np.ndarray]] = []
    for row_index, (dist_row, index_row) in enumerate(zip(distances, indexes)):
        neighbor_indexes = [int(idx) for idx in index_row if int(idx) != row_index]
        neighbor_indexes = neighbor_indexes[: int(k_neighbors)]
        if not neighbor_indexes:
            descriptors.append(
                {
                    "distances": np.empty(0, dtype=float),
                    "angle_gaps": np.empty(0, dtype=float),
                    "log_area_ratios": np.empty(0, dtype=float),
                }
            )
            continue
        vectors = xy[neighbor_indexes] - xy[row_index]
        neighbor_distances = np.linalg.norm(vectors, axis=1)
        angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2.0 * math.pi)
        order = np.argsort(angles)
        angles = angles[order]
        neighbor_distances = neighbor_distances[order]
        neighbor_areas = areas[neighbor_indexes][order]
        median_distance = max(float(np.median(neighbor_distances)), 1e-12)
        distance_profile = neighbor_distances / median_distance
        if len(angles) >= 2:
            angle_gaps = np.diff(np.r_[angles, angles[0] + 2.0 * math.pi]) / (2.0 * math.pi)
        else:
            angle_gaps = np.array([1.0], dtype=float)
        log_area_ratios = np.log(np.maximum(neighbor_areas, 1e-12) / max(areas[row_index], 1e-12))
        descriptors.append(
            {
                "distances": distance_profile.astype(float),
                "angle_gaps": angle_gaps.astype(float),
                "log_area_ratios": log_area_ratios.astype(float),
            }
        )
    return descriptors


def _build_partial_candidate_matches(
    fixed_nodes: list[_ContourNode],
    moving_nodes: list[_ContourNode],
    cfg: LabelFreeContourAlignmentConfig,
) -> list[dict[str, Any]]:
    fixed_matchable = [node for node in fixed_nodes if not node.envelope_only]
    moving_matchable = [node for node in moving_nodes if not node.envelope_only]
    if not fixed_matchable or not moving_matchable:
        return []
    moving_xy = np.array([[node.centroid_x, node.centroid_y] for node in moving_matchable])
    tree = cKDTree(moving_xy)
    k = min(max(10, int(cfg.knn_neighbors) * 2), len(moving_matchable))
    rows: list[dict[str, Any]] = []
    for fixed in fixed_matchable:
        distances, indexes = tree.query([fixed.centroid_x, fixed.centroid_y], k=k)
        distances = np.atleast_1d(distances).astype(float)
        indexes = np.atleast_1d(indexes).astype(int)
        for distance, moving_index in zip(distances, indexes):
            if moving_index < 0 or moving_index >= len(moving_matchable):
                continue
            moving = moving_matchable[int(moving_index)]
            area_ratio = moving.area / max(fixed.area, 1e-12)
            position_score = _position_similarity(float(distance), float(cfg.search_window))
            area_score = _area_similarity(area_ratio, cfg)
            topology_score = _topology_similarity(fixed, moving)
            shape_score = _shape_similarity(fixed, moving)
            total_score = (
                0.35 * position_score
                + 0.35 * topology_score
                + 0.20 * area_score
                + 0.10 * shape_score
            )
            rejection_codes = _candidate_rejection_codes(
                distance=float(distance),
                area_ratio=area_ratio,
                topology_score=topology_score,
                cfg=cfg,
            )
            rows.append(
                {
                    "fixed_node_id": fixed.node_id,
                    "moving_node_id": moving.node_id,
                    "fixed_feature_index": int(fixed.feature_index),
                    "moving_feature_index": int(moving.feature_index),
                    "fixed_label": fixed.source_label,
                    "moving_label": moving.source_label,
                    "fixed_centroid_x": fixed.centroid_x,
                    "fixed_centroid_y": fixed.centroid_y,
                    "moving_centroid_x": moving.centroid_x,
                    "moving_centroid_y": moving.centroid_y,
                    "distance": float(distance),
                    "area_ratio": float(area_ratio),
                    "position_score": float(position_score),
                    "topology_score": float(topology_score),
                    "area_score": float(area_score),
                    "shape_score": float(shape_score),
                    "total_score": float(total_score),
                    "rejection_codes": ";".join(rejection_codes),
                    "classification": "rejected",
                }
            )
    _rank_candidate_rows(rows)
    return rows


def _position_similarity(distance: float, search_window: float) -> float:
    return float(math.exp(-max(distance, 0.0) / max(search_window / 4.0, 1e-12)))


def _area_similarity(area_ratio: float, cfg: LabelFreeContourAlignmentConfig) -> float:
    if area_ratio <= 0:
        return 0.0
    score = 1.0 - abs(math.log(area_ratio) / math.log(float(cfg.area_ratio_max)))
    return float(np.clip(score, 0.0, 1.0))


def _shape_similarity(a: _ContourNode, b: _ContourNode) -> float:
    compactness_score = math.exp(-abs(a.compactness - b.compactness) / 0.25)
    aspect_score = math.exp(-abs(math.log(max(a.aspect_ratio, 1e-12) / max(b.aspect_ratio, 1e-12))))
    return float(np.clip(0.6 * compactness_score + 0.4 * aspect_score, 0.0, 1.0))


def _topology_similarity(a: _ContourNode, b: _ContourNode) -> float:
    dist_a = np.asarray(a.knn_distances, dtype=float)
    dist_b = np.asarray(b.knn_distances, dtype=float)
    gaps_a = np.asarray(a.knn_angle_gaps, dtype=float)
    gaps_b = np.asarray(b.knn_angle_gaps, dtype=float)
    area_a = np.asarray(a.knn_log_area_ratios, dtype=float)
    area_b = np.asarray(b.knn_log_area_ratios, dtype=float)
    n = min(len(dist_a), len(dist_b), len(gaps_a), len(gaps_b), len(area_a), len(area_b))
    if n < 3:
        return 0.0
    dist_a = dist_a[:n]
    gaps_a = gaps_a[:n]
    area_a = area_a[:n]
    best = 0.0
    for shift in range(n):
        dist_shift = np.roll(dist_b[:n], shift)
        gaps_shift = np.roll(gaps_b[:n], shift)
        area_shift = np.roll(area_b[:n], shift)
        distance_score = math.exp(float(-np.mean(np.abs(np.log(np.maximum(dist_a, 1e-9) / np.maximum(dist_shift, 1e-9))))))
        angle_score = 1.0 - float(np.clip(np.mean(np.abs(gaps_a - gaps_shift)) / 0.25, 0.0, 1.0))
        area_score = 1.0 - float(np.clip(np.mean(np.abs(area_a - area_shift)) / math.log(4.0), 0.0, 1.0))
        best = max(best, 0.45 * distance_score + 0.35 * angle_score + 0.20 * area_score)
    return float(np.clip(best, 0.0, 1.0))


def _candidate_rejection_codes(
    *,
    distance: float,
    area_ratio: float,
    topology_score: float,
    cfg: LabelFreeContourAlignmentConfig,
) -> list[str]:
    codes: list[str] = []
    if distance > cfg.search_window:
        codes.append("ERR_DIST")
    if area_ratio < cfg.area_ratio_min or area_ratio > cfg.area_ratio_max:
        codes.append("ERR_AREA")
    if topology_score < 0.25:
        codes.append("ERR_TOPO")
    return codes


def _rank_candidate_rows(rows: list[dict[str, Any]]) -> None:
    fixed_groups: dict[str, list[int]] = {}
    moving_groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        fixed_groups.setdefault(str(row["fixed_node_id"]), []).append(index)
        moving_groups.setdefault(str(row["moving_node_id"]), []).append(index)
    for group in fixed_groups.values():
        for rank, index in enumerate(
            sorted(group, key=lambda i: float(rows[i]["total_score"]), reverse=True),
            start=1,
        ):
            rows[index]["fixed_rank"] = rank
    for group in moving_groups.values():
        for rank, index in enumerate(
            sorted(group, key=lambda i: float(rows[i]["total_score"]), reverse=True),
            start=1,
        ):
            rows[index]["moving_rank"] = rank
    for row in rows:
        fixed_rank = int(row.get("fixed_rank", 10**9))
        moving_rank = int(row.get("moving_rank", 10**9))
        row["is_mutual_first"] = fixed_rank == 1 and moving_rank == 1
        row["is_mutual_top3"] = fixed_rank <= 3 and moving_rank <= 3


def _classify_partial_correspondence(
    fixed_nodes: list[_ContourNode],
    moving_nodes: list[_ContourNode],
    matches: list[dict[str, Any]],
    cfg: LabelFreeContourAlignmentConfig,
) -> tuple[list[_ContourNode], list[_ContourNode], list[dict[str, Any]]]:
    for row in matches:
        hard_rejected = any(
            code in str(row["rejection_codes"]).split(";")
            for code in ("ERR_DIST", "ERR_AREA", "ERR_TOPO")
        )
        if (
            not hard_rejected
            and float(row["total_score"]) >= cfg.min_anchor_score
            and bool(row["is_mutual_top3"])
        ):
            row["classification"] = "matched_anchor"
        elif not hard_rejected and float(row["total_score"]) >= cfg.min_review_score:
            row["classification"] = "matched_review"
            if not bool(row["is_mutual_top3"]):
                codes = [code for code in str(row["rejection_codes"]).split(";") if code]
                if "ERR_NOT_MUTUAL" not in codes:
                    codes.append("ERR_NOT_MUTUAL")
                row["rejection_codes"] = ";".join(codes)
        else:
            row["classification"] = "rejected"

    node_status: dict[str, str] = {}
    for node in fixed_nodes + moving_nodes:
        node_status[node.node_id] = "envelope_only" if node.envelope_only else "no_counterpart"
    for classification in ("matched_review", "matched_anchor"):
        for row in matches:
            if row["classification"] != classification:
                continue
            for node_id in (str(row["fixed_node_id"]), str(row["moving_node_id"])):
                if node_status.get(node_id) != "matched_anchor":
                    node_status[node_id] = classification
    for row in matches:
        row["fixed_status"] = node_status.get(str(row["fixed_node_id"]), "no_counterpart")
        row["moving_status"] = node_status.get(str(row["moving_node_id"]), "no_counterpart")

    return fixed_nodes, moving_nodes, matches


def _nodes_dataframe(nodes: list[_ContourNode], matches: list[dict[str, Any]]) -> pd.DataFrame:
    top_candidates = _top_candidate_summaries(matches)
    status_by_node: dict[str, str] = {}
    for node in nodes:
        status_by_node[node.node_id] = "envelope_only" if node.envelope_only else "no_counterpart"
    for classification in ("matched_review", "matched_anchor"):
        for row in matches:
            if row["classification"] != classification:
                continue
            for node_id in (str(row["fixed_node_id"]), str(row["moving_node_id"])):
                if status_by_node.get(node_id) != "matched_anchor":
                    status_by_node[node_id] = classification
    rows = []
    for node in nodes:
        rows.append(
            {
                "role": node.role,
                "node_id": node.node_id,
                "feature_index": node.feature_index,
                "source_label": node.source_label,
                "status": status_by_node.get(node.node_id, "no_counterpart"),
                "envelope_only": node.envelope_only,
                "exclude_reason": node.exclude_reason,
                "centroid_x": node.centroid_x,
                "centroid_y": node.centroid_y,
                "area": node.area,
                "perimeter": node.perimeter,
                "compactness": node.compactness,
                "aspect_ratio": node.aspect_ratio,
                "orientation_degrees": node.orientation_degrees,
                "area_fraction": node.area_fraction,
                "bbox_fraction_x": node.bbox_fraction_x,
                "bbox_fraction_y": node.bbox_fraction_y,
                "knn_distances": json.dumps([round(x, 6) for x in node.knn_distances]),
                "knn_angle_gaps": json.dumps([round(x, 6) for x in node.knn_angle_gaps]),
                "knn_log_area_ratios": json.dumps([round(x, 6) for x in node.knn_log_area_ratios]),
                "top3_candidates": top_candidates.get(node.node_id, ""),
            }
        )
    return pd.DataFrame(rows)


def _top_candidate_summaries(matches: list[dict[str, Any]], limit: int = 3) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in matches:
        grouped.setdefault(str(row["fixed_node_id"]), []).append(row)
        grouped.setdefault(str(row["moving_node_id"]), []).append(row)
    summaries: dict[str, str] = {}
    for node_id, rows in grouped.items():
        parts = []
        for row in sorted(rows, key=lambda item: float(item["total_score"]), reverse=True)[:limit]:
            other = (
                row["moving_node_id"]
                if node_id == str(row["fixed_node_id"])
                else row["fixed_node_id"]
            )
            reason = str(row["rejection_codes"]) or "OK"
            parts.append(
                f"{other}: score={float(row['total_score']):.3f}, "
                f"d={float(row['distance']):.1f}, reason={reason}"
            )
        summaries[node_id] = " | ".join(parts)
    return summaries


def _partial_correspondence_summary(
    *,
    fixed_nodes: list[_ContourNode],
    moving_nodes: list[_ContourNode],
    matches_df: pd.DataFrame,
    fixed_union: Any,
    cfg: LabelFreeContourAlignmentConfig,
) -> dict[str, Any]:
    node_df = _nodes_dataframe(fixed_nodes + moving_nodes, matches_df.to_dict("records"))
    status_counts = (
        node_df.groupby(["role", "status"]).size().unstack(fill_value=0).to_dict(orient="index")
        if not node_df.empty
        else {}
    )
    anchor_rows = matches_df.loc[matches_df["classification"] == "matched_anchor"].copy()
    anchor_count = int(len(anchor_rows))
    entropy_payload = _anchor_spatial_entropy(anchor_rows, fixed_union)
    coverage_ratio = _anchor_coverage_ratio(anchor_rows, fixed_union)
    warnings = []
    if anchor_count < cfg.min_anchor_count:
        warnings.append("too_few_anchors")
    if entropy_payload["occupied_quadrants"] < 3 or entropy_payload["entropy_normalized"] < 0.6:
        warnings.append("clustered_anchors")
    if coverage_ratio < 0.2:
        warnings.append("low_anchor_coverage")
    return {
        "fixed_node_count": int(len(fixed_nodes)),
        "moving_node_count": int(len(moving_nodes)),
        "match_candidate_count": int(len(matches_df)),
        "anchor_count": anchor_count,
        "review_match_count": int((matches_df["classification"] == "matched_review").sum()) if len(matches_df) else 0,
        "status_counts": status_counts,
        "anchor_spatial_entropy": entropy_payload,
        "anchor_convex_hull_coverage_ratio": float(coverage_ratio),
        "warnings": warnings,
        "parameters": {
            "search_window": float(cfg.search_window),
            "knn_neighbors": int(cfg.knn_neighbors),
            "min_anchor_score": float(cfg.min_anchor_score),
            "min_review_score": float(cfg.min_review_score),
            "area_ratio_min": float(cfg.area_ratio_min),
            "area_ratio_max": float(cfg.area_ratio_max),
        },
    }


def _anchor_spatial_entropy(anchor_rows: pd.DataFrame, fixed_union: Any) -> dict[str, Any]:
    if anchor_rows.empty:
        return {
            "entropy": 0.0,
            "entropy_normalized": 0.0,
            "occupied_quadrants": 0,
            "quadrant_counts": [0, 0, 0, 0],
        }
    minx, miny, maxx, maxy = fixed_union.bounds
    cx = (float(minx) + float(maxx)) / 2.0
    cy = (float(miny) + float(maxy)) / 2.0
    counts = [0, 0, 0, 0]
    for _, row in anchor_rows.iterrows():
        x = float(row["fixed_centroid_x"])
        y = float(row["fixed_centroid_y"])
        quadrant = (1 if x >= cx else 0) + (2 if y >= cy else 0)
        counts[quadrant] += 1
    total = max(sum(counts), 1)
    probs = np.array([count / total for count in counts if count > 0], dtype=float)
    entropy = float(-np.sum(probs * np.log(probs))) if len(probs) else 0.0
    return {
        "entropy": entropy,
        "entropy_normalized": float(entropy / math.log(4.0)) if entropy > 0 else 0.0,
        "occupied_quadrants": int(sum(count > 0 for count in counts)),
        "quadrant_counts": counts,
    }


def _anchor_coverage_ratio(anchor_rows: pd.DataFrame, fixed_union: Any) -> float:
    if len(anchor_rows) < 3:
        return 0.0
    points = [
        (float(row["fixed_centroid_x"]), float(row["fixed_centroid_y"]))
        for _, row in anchor_rows.iterrows()
    ]
    hull_area = float(MultiPoint(points).convex_hull.area)
    envelope_area = max(float(fixed_union.convex_hull.area), 1e-12)
    return float(np.clip(hull_area / envelope_area, 0.0, 1.0))


def _write_partial_correspondence_html(
    *,
    fixed_nodes: list[_ContourNode],
    moving_nodes: list[_ContourNode],
    matches_df: pd.DataFrame,
    output_html: Path,
    summary: dict[str, Any],
) -> Path:
    node_status = {
        row["node_id"]: row["status"]
        for row in _nodes_dataframe(fixed_nodes + moving_nodes, matches_df.to_dict("records")).to_dict("records")
    }
    traces = []
    for role, nodes in (("fixed", fixed_nodes), ("moving", moving_nodes)):
        for status, color in (
            ("matched_anchor", "#19a974"),
            ("matched_review", "#d6a11d"),
            ("no_counterpart", "#e45756"),
            ("envelope_only", "#9aa4b2"),
        ):
            selected = [node for node in nodes if node_status.get(node.node_id) == status]
            if selected:
                traces.append(_nodes_to_boundary_trace(selected, name=f"{role} {status}", color=color, role=role))
    anchor_rows = matches_df.loc[matches_df["classification"] == "matched_anchor"] if len(matches_df) else pd.DataFrame()
    review_rows = matches_df.loc[matches_df["classification"] == "matched_review"] if len(matches_df) else pd.DataFrame()
    if len(anchor_rows):
        traces.append(_matches_to_link_trace(anchor_rows, "matched anchors", "#19a974", width=2.4))
    if len(review_rows):
        traces.append(_matches_to_link_trace(review_rows, "matched review", "#d6a11d", width=1.6))

    warning = "; ".join(summary.get("warnings", [])) or "none"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Partial correspondence candidate matrix</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f9fc; color: #172033; }}
    header {{ padding: 16px 22px 8px 22px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    p {{ margin: 6px 0 0 0; color: #536273; }}
    .metrics {{ display: flex; gap: 10px; flex-wrap: wrap; padding: 8px 22px; }}
    .metric {{ background: white; border: 1px solid #d8e0ea; border-radius: 6px; padding: 8px 10px; }}
    #plot {{ width: 100vw; height: calc(100vh - 138px); min-height: 580px; }}
  </style>
</head>
<body>
  <header>
    <h1>Partial correspondence candidate matrix</h1>
    <p>Green = anchor, yellow = review, red = no counterpart, gray = envelope only. No coordinate warp is performed.</p>
  </header>
  <div class="metrics">
    <div class="metric">anchors: {int(summary.get("anchor_count", 0))}</div>
    <div class="metric">review: {int(summary.get("review_match_count", 0))}</div>
    <div class="metric">coverage: {float(summary.get("anchor_convex_hull_coverage_ratio", 0.0)):.3f}</div>
    <div class="metric">warnings: {warning}</div>
  </div>
  <div id="plot"></div>
  <script>
    const data = {json.dumps(traces)};
    const layout = {{
      margin: {{l: 55, r: 20, t: 20, b: 50}},
      xaxis: {{title: "x", scaleanchor: "y", scaleratio: 1, zeroline: false}},
      yaxis: {{title: "y", autorange: "reversed", zeroline: false}},
      legend: {{orientation: "h", y: 1.05}},
      hovermode: "closest"
    }};
    Plotly.newPlot("plot", data, layout, {{responsive: true, scrollZoom: true}});
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")
    return output_html


def _nodes_to_boundary_trace(
    nodes: list[_ContourNode],
    *,
    name: str,
    color: str,
    role: str,
) -> dict[str, Any]:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    text_values: list[str | None] = []
    dash = "solid" if role == "fixed" else "dot"
    for node in nodes:
        hover = (
            f"{node.node_id}<br>{node.source_label}<br>"
            f"area={node.area:.1f}<br>status={name}"
        )
        for polygon in _iter_polygons(node.geometry):
            coords = list(polygon.exterior.coords)
            for x, y in coords:
                x_values.append(round(float(x), 3))
                y_values.append(round(float(y), 3))
                text_values.append(hover)
            x_values.append(None)
            y_values.append(None)
            text_values.append(None)
    return {
        "type": "scattergl",
        "mode": "lines",
        "name": name,
        "x": x_values,
        "y": y_values,
        "text": text_values,
        "hovertemplate": "%{text}<extra></extra>",
        "line": {"color": color, "width": 1.15, "dash": dash},
        "opacity": 0.86,
    }


def _matches_to_link_trace(
    rows: pd.DataFrame,
    name: str,
    color: str,
    *,
    width: float,
) -> dict[str, Any]:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    text_values: list[str | None] = []
    for _, row in rows.iterrows():
        x_values.extend([float(row["fixed_centroid_x"]), float(row["moving_centroid_x"]), None])
        y_values.extend([float(row["fixed_centroid_y"]), float(row["moving_centroid_y"]), None])
        hover = (
            f"{row['fixed_node_id']} -> {row['moving_node_id']}<br>"
            f"score={float(row['total_score']):.3f}<br>"
            f"dist={float(row['distance']):.1f}<br>"
            f"reason={row['rejection_codes'] or 'OK'}"
        )
        text_values.extend([hover, hover, None])
    return {
        "type": "scattergl",
        "mode": "lines",
        "name": name,
        "x": x_values,
        "y": y_values,
        "text": text_values,
        "hovertemplate": "%{text}<extra></extra>",
        "line": {"color": color, "width": width},
        "opacity": 0.75,
    }


def _estimate_label_free_similarity(
    score_inputs: dict[str, Any],
    cfg: LabelFreeContourAlignmentConfig,
) -> tuple[_PointTransform, dict[str, Any]]:
    fixed_union = score_inputs["fixed_union"]
    moving_union = score_inputs["moving_union"]
    origin = moving_union.centroid
    scale0 = math.sqrt(max(fixed_union.area, 1e-9) / max(moving_union.area, 1e-9))
    rotation0 = _principal_axis_angle(fixed_union) - _principal_axis_angle(moving_union)
    moving_centroid = np.array([float(moving_union.centroid.x), float(moving_union.centroid.y)])
    fixed_centroid = np.array([float(fixed_union.centroid.x), float(fixed_union.centroid.y)])

    starts = [float(rotation0)]
    if cfg.multistart:
        for extra in (0.0, 90.0, 180.0, 270.0):
            candidate = float(rotation0 + extra)
            if not any(abs(candidate - existing) < 5.0 for existing in starts):
                starts.append(candidate)

    best_params: np.ndarray | None = None
    best_score = -math.inf
    best_meta: dict[str, Any] = {}

    for seed in starts:
        seed_transform = _SimilarityTransform(
            origin_x=float(origin.x),
            origin_y=float(origin.y),
            rotation_degrees=float(seed),
            scale=float(scale0),
            translate_x=0.0,
            translate_y=0.0,
        )
        transformed_centroid = _apply_similarity_to_points(moving_centroid.reshape(1, 2), seed_transform)[0]
        shift = fixed_centroid - transformed_centroid
        params0 = np.array([float(seed), math.log(max(scale0, 1e-9)), shift[0], shift[1]])

        def objective(params: np.ndarray) -> float:
            transform = _PointTransform(
                kind="similarity",
                similarity=_SimilarityTransform(
                    origin_x=float(origin.x),
                    origin_y=float(origin.y),
                    rotation_degrees=float(params[0]),
                    scale=float(math.exp(params[1])),
                    translate_x=float(params[2]),
                    translate_y=float(params[3]),
                ),
            )
            score = _label_free_score(transform, **score_inputs)
            return -float(score["score"])

        if cfg.maxiter > 0:
            result = minimize(
                objective,
                params0,
                method="Nelder-Mead",
                options={"maxiter": int(cfg.maxiter), "xatol": 1e-3, "fatol": 1e-5},
            )
            params = result.x if objective(result.x) <= objective(params0) else params0
            meta = {
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(getattr(result, "nit", 0)),
                "rotation_seed_degrees": float(seed),
            }
        else:
            params = params0
            meta = {
                "success": True,
                "message": "maxiter=0",
                "iterations": 0,
                "rotation_seed_degrees": float(seed),
            }
        score_value = -objective(params)
        if score_value > best_score:
            best_score = score_value
            best_params = params
            best_meta = meta

    if best_params is None:
        raise ValueError("Similarity alignment did not produce a valid transform.")
    transform = _PointTransform(
        kind="similarity",
        similarity=_SimilarityTransform(
            origin_x=float(origin.x),
            origin_y=float(origin.y),
            rotation_degrees=float(best_params[0]),
            scale=float(math.exp(best_params[1])),
            translate_x=float(best_params[2]),
            translate_y=float(best_params[3]),
        ),
    )
    return transform, {
        "method": "label_free_hybrid_multistart_nelder_mead",
        "multistart_seeds": len(starts),
        **best_meta,
    }


def _try_label_free_affine(
    score_inputs: dict[str, Any],
    similarity_score: dict[str, Any],
    cfg: LabelFreeContourAlignmentConfig,
) -> tuple[_PointTransform | None, dict[str, Any]]:
    fixed_union = score_inputs["fixed_union"]
    moving_union = score_inputs["moving_union"]
    tx0 = float(fixed_union.centroid.x) - float(moving_union.centroid.x)
    ty0 = float(fixed_union.centroid.y) - float(moving_union.centroid.y)
    params0 = np.array([1.0, 0.0, 0.0, 1.0, tx0, ty0], dtype=float)

    def objective(params: np.ndarray) -> float:
        transform = _PointTransform(kind="affine", affine_params=tuple(float(x) for x in params))
        score = _label_free_score(transform, **score_inputs)
        return -float(score["score"])

    result = minimize(
        objective,
        params0,
        method="Nelder-Mead",
        options={"maxiter": int(cfg.maxiter), "xatol": 1e-4, "fatol": 1e-5},
    )
    params = result.x if objective(result.x) <= objective(params0) else params0
    transform = _PointTransform(kind="affine", affine_params=tuple(float(x) for x in params))
    score = _label_free_score(transform, **score_inputs)
    accepted = score["score"] >= similarity_score["score"]
    return (
        transform if accepted else None,
        {
            "attempted": True,
            "accepted": bool(accepted),
            "score": score,
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
        },
    )


def _label_free_score(
    transform: _PointTransform,
    *,
    fixed_union: Any,
    moving_union: Any,
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    fixed_components: pd.DataFrame,
    moving_components: pd.DataFrame,
) -> dict[str, float]:
    transformed_union = _apply_point_transform_to_geometry(moving_union, transform)
    iou = _iou(fixed_union, transformed_union)
    transformed_points = _apply_point_transform_to_points(moving_points, transform)
    boundary_loss = _symmetric_nearest_loss(fixed_points, transformed_points)
    transformed_components = _transform_component_table(moving_components, transform)
    layout_loss = _layout_loss(fixed_components, transformed_components)
    span = _characteristic_span(fixed_union, moving_union)
    boundary_similarity = math.exp(-boundary_loss / max(span * 0.015, 1e-6))
    layout_similarity = math.exp(-layout_loss / max(span * 0.02, 1e-6))
    score = 0.45 * float(iou) + 0.40 * boundary_similarity + 0.15 * layout_similarity
    return {
        "score": float(score),
        "iou": float(iou),
        "boundary_chamfer_um": float(boundary_loss),
        "boundary_similarity": float(boundary_similarity),
        "layout_loss_um": float(layout_loss),
        "layout_similarity": float(layout_similarity),
    }


def _score_geojson_against_fixed(
    *,
    fixed_union: Any,
    moving_payload: dict[str, Any],
    fixed_points: np.ndarray,
    fixed_components: pd.DataFrame,
    cfg: LabelFreeContourAlignmentConfig,
) -> dict[str, float]:
    moving_records = _records_from_geojson_label_free(moving_payload, cfg, role="final")
    moving_union = unary_union([record.geometry for record in moving_records])
    moving_points = _sample_boundary_points(moving_union.boundary, cfg.boundary_sample_count)
    moving_components = _component_layout_table(moving_records, cfg)
    return _label_free_score(
        _PointTransform(kind="identity"),
        fixed_union=fixed_union,
        moving_union=moving_union,
        fixed_points=fixed_points,
        moving_points=moving_points,
        fixed_components=fixed_components,
        moving_components=moving_components,
    )


def _symmetric_nearest_loss(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return math.inf
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    dist_ab = tree_b.query(a, k=1)[0]
    dist_ba = tree_a.query(b, k=1)[0]
    return float((np.median(dist_ab) + np.median(dist_ba)) / 2.0)


def _layout_loss(fixed: pd.DataFrame, moving: pd.DataFrame) -> float:
    if fixed.empty or moving.empty:
        return math.inf
    fixed_xy = fixed[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    moving_xy = moving[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    tree_fixed = cKDTree(fixed_xy)
    tree_moving = cKDTree(moving_xy)
    d_mf = tree_fixed.query(moving_xy, k=1)[0]
    d_fm = tree_moving.query(fixed_xy, k=1)[0]
    w_m = moving["layout_weight"].to_numpy(dtype=float)
    w_f = fixed["layout_weight"].to_numpy(dtype=float)
    return float((np.sum(w_m * d_mf) + np.sum(w_f * d_fm)) / 2.0)


def _transform_component_table(table: pd.DataFrame, transform: _PointTransform) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    xy = table[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    transformed = _apply_point_transform_to_points(xy, transform)
    result = table.copy()
    result["centroid_x"] = transformed[:, 0]
    result["centroid_y"] = transformed[:, 1]
    return result


def _principal_axis_angle(geom: Any) -> float:
    hull = geom.convex_hull
    if hull.is_empty or not hasattr(hull, "exterior"):
        return 0.0
    coords = np.asarray(hull.exterior.coords, dtype=float)
    if len(coords) < 3:
        return 0.0
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    return float(math.degrees(math.atan2(axis[1], axis[0])))


def _characteristic_span(*geoms: Any) -> float:
    union = unary_union(list(geoms))
    minx, miny, maxx, maxy = union.bounds
    return float(math.hypot(maxx - minx, maxy - miny))


def _apply_point_transform_to_geojson(
    payload: dict[str, Any],
    transform: _PointTransform,
) -> dict[str, Any]:
    if transform.kind == "similarity":
        if transform.similarity is None:
            raise ValueError("Similarity transform payload is missing.")
        return _apply_similarity_to_geojson(payload, transform.similarity)
    if transform.kind == "affine":
        if transform.affine_params is None:
            raise ValueError("Affine transform payload is missing.")
        return _apply_affine_to_geojson(payload, np.asarray(transform.affine_params, dtype=float))
    return copy.deepcopy(payload)


def _apply_point_transform_to_geometry(geom: Any, transform: _PointTransform) -> Any:
    if transform.kind == "similarity":
        if transform.similarity is None:
            raise ValueError("Similarity transform payload is missing.")
        origin = (transform.similarity.origin_x, transform.similarity.origin_y)
        result = affinity.rotate(geom, transform.similarity.rotation_degrees, origin=origin)
        result = affinity.scale(
            result,
            xfact=transform.similarity.scale,
            yfact=transform.similarity.scale,
            origin=origin,
        )
        return affinity.translate(
            result,
            xoff=transform.similarity.translate_x,
            yoff=transform.similarity.translate_y,
        )
    if transform.kind == "affine":
        if transform.affine_params is None:
            raise ValueError("Affine transform payload is missing.")
        return affinity.affine_transform(geom, list(transform.affine_params))
    return geom


def _apply_point_transform_to_points(points: np.ndarray, transform: _PointTransform) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points.reshape(0, 2)
    if transform.kind == "similarity":
        if transform.similarity is None:
            raise ValueError("Similarity transform payload is missing.")
        return _apply_similarity_to_points(points, transform.similarity)
    if transform.kind == "affine":
        if transform.affine_params is None:
            raise ValueError("Affine transform payload is missing.")
        a, b, d, e, xoff, yoff = transform.affine_params
        x = points[:, 0]
        y = points[:, 1]
        return np.column_stack([a * x + b * y + xoff, d * x + e * y + yoff])
    return points.copy()


def _apply_similarity_to_points(points: np.ndarray, transform: _SimilarityTransform) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    origin = np.array([transform.origin_x, transform.origin_y], dtype=float)
    radians = math.radians(transform.rotation_degrees)
    rot = np.array(
        [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]],
        dtype=float,
    )
    centered = points - origin
    transformed = centered @ rot.T
    transformed = transformed * float(transform.scale)
    transformed = transformed + origin
    transformed[:, 0] += float(transform.translate_x)
    transformed[:, 1] += float(transform.translate_y)
    return transformed


def _transform_payload(transform: _PointTransform) -> dict[str, Any]:
    if transform.kind == "similarity" and transform.similarity is not None:
        return {"kind": "similarity", **asdict(transform.similarity)}
    if transform.kind == "affine" and transform.affine_params is not None:
        return {"kind": "affine", "params": list(transform.affine_params)}
    return {"kind": "identity"}


def _run_label_free_soft_tps(
    *,
    fixed_records: list[_FeatureRecord],
    hard_payload: dict[str, Any],
    hard_records: list[_FeatureRecord],
    score_inputs: dict[str, Any],
    hard_score: dict[str, float],
    cfg: LabelFreeContourAlignmentConfig,
) -> tuple[dict[str, Any], list[_FeatureRecord], pd.DataFrame, dict[str, Any]]:
    landmarks = _generate_label_free_landmarks(fixed_records, hard_records, cfg)
    if len(landmarks) < 3:
        landmarks["accepted_for_tps"] = False
        return hard_payload, hard_records, landmarks, {
            "attempted": True,
            "accepted": False,
            "reason": "fewer_than_three_landmarks",
            "landmark_count": int(len(landmarks)),
        }

    landmarks["accepted_for_tps"] = True
    model = _fit_tps_model(landmarks, cfg)
    soft_payload, soft_records, geometry_status = _warp_geojson_label_free(
        hard_payload,
        hard_records,
        model,
    )
    topology = _check_tps_topology(hard_records, model, cfg)
    soft_score = _score_geojson_against_fixed(
        fixed_union=score_inputs["fixed_union"],
        moving_payload=soft_payload,
        fixed_points=score_inputs["fixed_points"],
        fixed_components=score_inputs["fixed_components"],
        cfg=cfg,
    )
    geometry_valid = int(geometry_status.get("invalid", 0)) == 0
    accepted = (
        soft_score["score"] >= hard_score["score"]
        and bool(topology.get("valid", True))
        and geometry_valid
    )
    summary = {
        "attempted": True,
        "accepted": bool(accepted),
        "reason": (
            "soft_tps_improved_or_preserved_label_free_score"
            if accepted
            else "hard_alignment_kept_after_soft_tps_qc"
        ),
        "landmark_count": int(len(landmarks)),
        "score_after_soft": soft_score,
        "topology": topology,
        "geometry_status": dict(geometry_status),
    }
    if accepted:
        return soft_payload, soft_records, landmarks, summary
    return hard_payload, hard_records, landmarks, summary


def _generate_label_free_landmarks(
    fixed_records: list[_FeatureRecord],
    hard_records: list[_FeatureRecord],
    cfg: LabelFreeContourAlignmentConfig,
) -> pd.DataFrame:
    fixed_union = unary_union([record.geometry for record in fixed_records])
    moving_union = unary_union([record.geometry for record in hard_records])
    candidate_spacing = cfg.landmark_candidate_spacing_um
    if candidate_spacing is None:
        candidate_spacing = max(min(cfg.sampling_distance_um / 2.0, 25.0), 1e-6)
    fixed_candidates = _sample_candidates_with_normals(
        fixed_union.boundary,
        spacing_um=candidate_spacing,
        normal_step_um=cfg.landmark_normal_step_um or max(candidate_spacing / 2.0, 1e-6),
    )
    moving_candidates = _sample_candidates_with_normals(
        moving_union.boundary,
        spacing_um=cfg.sampling_distance_um,
        normal_step_um=cfg.landmark_normal_step_um or max(cfg.sampling_distance_um / 2.0, 1e-6),
    )
    if fixed_candidates is None or moving_candidates is None:
        return pd.DataFrame()

    fixed_tree = cKDTree(fixed_candidates["xy"])
    moving_tree = cKDTree(moving_candidates["xy"])
    rows: list[dict[str, Any]] = []
    k = min(int(cfg.landmark_candidate_count), len(fixed_candidates["xy"]))
    for index, (src_xy, src_normal) in enumerate(
        zip(moving_candidates["xy"], moving_candidates["normals"])
    ):
        distances, indexes = fixed_tree.query(src_xy, k=k)
        distances = np.atleast_1d(distances).astype(float)
        indexes = np.atleast_1d(indexes).astype(int)
        valid = np.isfinite(distances) & (indexes >= 0) & (indexes < len(fixed_candidates["xy"]))
        if not np.any(valid):
            continue
        candidate_distances = distances[valid]
        candidate_indexes = indexes[valid]
        candidate_normals = fixed_candidates["normals"][candidate_indexes]
        costs = candidate_distances.copy()
        normal_dots = np.full(len(candidate_indexes), math.nan, dtype=float)
        if cfg.landmark_normal_weight_um > 0 and np.all(np.isfinite(src_normal)):
            valid_normals = np.isfinite(candidate_normals).all(axis=1)
            if np.any(valid_normals):
                dots = np.abs(candidate_normals[valid_normals] @ src_normal)
                dots = np.clip(dots, 0.0, 1.0)
                normal_dots[valid_normals] = dots
                costs[valid_normals] += cfg.landmark_normal_weight_um * (1.0 - dots)
        best_position = int(np.argmin(costs))
        best_index = int(candidate_indexes[best_position])
        best_distance = float(candidate_distances[best_position])
        if best_distance > cfg.max_landmark_distance_um:
            continue

        # Mutual nearest-neighbour consistency in unlabeled boundary space.
        reverse_distance, reverse_index = moving_tree.query(fixed_candidates["xy"][best_index], k=1)
        reverse_ok = int(reverse_index) == index or float(reverse_distance) <= 2.0 * cfg.sampling_distance_um
        if not reverse_ok:
            continue
        dst_xy = fixed_candidates["xy"][best_index]
        rows.append(
            {
                "kind": "boundary",
                "structure": "__label_free__",
                "src_x": float(src_xy[0]),
                "src_y": float(src_xy[1]),
                "dst_x": float(dst_xy[0]),
                "dst_y": float(dst_xy[1]),
                "dx": float(dst_xy[0] - src_xy[0]),
                "dy": float(dst_xy[1] - src_xy[1]),
                "source_distance_um": best_distance,
                "match_cost_um": float(costs[best_position]),
                "normal_dot_abs": (
                    float(normal_dots[best_position])
                    if np.isfinite(normal_dots[best_position])
                    else math.nan
                ),
                "match_method": "label_free_mutual_kdtree",
            }
        )
    landmarks = pd.DataFrame(rows)
    if landmarks.empty:
        return landmarks
    return _filter_landmark_outliers(landmarks)


def _sample_candidates_with_normals(
    boundary: Any,
    *,
    spacing_um: float,
    normal_step_um: float,
) -> dict[str, np.ndarray] | None:
    xy_values: list[np.ndarray] = []
    normal_values: list[np.ndarray] = []
    for line in _iter_line_parts(boundary):
        length = float(line.length)
        if length <= 0:
            continue
        distances = np.arange(0.0, length, max(spacing_um, 1e-6))
        if len(distances) == 0:
            distances = np.array([0.0])
        for distance in distances:
            point = line.interpolate(float(distance))
            xy_values.append(np.array([float(point.x), float(point.y)], dtype=float))
            normal_values.append(_line_normal(line, float(distance), normal_step_um))
    if not xy_values:
        return None
    return {
        "xy": np.vstack(xy_values).astype(float),
        "normals": np.vstack(normal_values).astype(float),
    }


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


def _filter_landmark_outliers(landmarks: pd.DataFrame) -> pd.DataFrame:
    if len(landmarks) < 8:
        return landmarks.reset_index(drop=True)
    dx = landmarks["dx"].to_numpy(dtype=float)
    dy = landmarks["dy"].to_numpy(dtype=float)
    median_dx = float(np.median(dx))
    median_dy = float(np.median(dy))
    mad_dx = max(float(np.median(np.abs(dx - median_dx))) * 1.4826, 1e-9)
    mad_dy = max(float(np.median(np.abs(dy - median_dy))) * 1.4826, 1e-9)
    z_score = np.sqrt(((dx - median_dx) / mad_dx) ** 2 + ((dy - median_dy) / mad_dy) ** 2)
    return landmarks.loc[z_score <= 3.5].reset_index(drop=True)


def _fit_tps_model(landmarks: pd.DataFrame, cfg: LabelFreeContourAlignmentConfig) -> _TPSModel:
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
    interpolator = RBFInterpolator(
        src_norm,
        displacement_norm,
        kernel=cfg.rbf_kernel,
        smoothing=float(cfg.rbf_smoothing),
        neighbors=neighbors,
    )
    return _TPSModel(interpolator=interpolator, center_xy=center, scale=scale)


def _warp_geojson_label_free(
    moving_geojson: dict[str, Any],
    moving_records: list[_FeatureRecord],
    model: _TPSModel,
) -> tuple[dict[str, Any], list[_FeatureRecord], dict[str, int]]:
    from collections import Counter

    warped = copy.deepcopy(moving_geojson)
    status_counts: Counter[str] = Counter()
    warped_records: list[_FeatureRecord] = []
    for feature, record in zip(warped["features"], moving_records):
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
        warped_records.append(_FeatureRecord(feature=feature, group="__label_free__", geometry=new_geom))
    return warped, warped_records, dict(status_counts)


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


def _write_overlays(
    *,
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    final_records: list[_FeatureRecord],
    before_png: Path,
    after_png: Path,
    dpi: int,
) -> dict[str, Path | None]:
    _plot_overlay(fixed_records, moving_records, before_png, "Before label-free alignment", dpi)
    _plot_overlay(fixed_records, final_records, after_png, "After label-free alignment", dpi)
    return {"before": before_png, "after": after_png}


def _plot_overlay(
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    output_png: Path,
    title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    for record in fixed_records:
        _plot_geometry_boundary(ax, record.geometry, color="#1f77b4", linewidth=0.7, alpha=0.65)
    for record in moving_records:
        _plot_geometry_boundary(ax, record.geometry, color="#d62728", linewidth=0.7, alpha=0.55)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot([], [], color="#1f77b4", label="fixed")
    ax.plot([], [], color="#d62728", label="moving")
    ax.legend(loc="best")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def _plot_geometry_boundary(ax: Any, geom: Any, *, color: str, linewidth: float, alpha: float) -> None:
    for polygon in _iter_polygons(geom):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)


def _write_overlay_html(
    *,
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    final_records: list[_FeatureRecord],
    output_html: Path,
    title: str,
    warning: str | None,
) -> Path:
    before_traces = [
        _records_to_plotly_trace(fixed_records, name="fixed", color="#1f77b4"),
        _records_to_plotly_trace(moving_records, name="moving before", color="#d62728"),
    ]
    after_traces = [
        _records_to_plotly_trace(fixed_records, name="fixed", color="#1f77b4"),
        _records_to_plotly_trace(final_records, name="moving aligned", color="#2ca02c"),
    ]
    warning_html = ""
    if warning:
        warning_html = f"""
    <div class="warning">
      <strong>Alignment warning:</strong> {warning}
    </div>"""
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f9fc; color: #172033; }}
    header {{ padding: 16px 24px 8px 24px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    p {{ margin: 6px 0 0 0; color: #536273; }}
    .warning {{ margin: 12px 24px 0 24px; padding: 12px 14px; border: 1px solid #f2bf65; border-radius: 6px; background: #fff7e8; color: #6b4300; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px 16px 16px 16px; }}
    .panel {{ height: calc(100vh - 96px); min-height: 520px; border: 1px solid #d8e0ea; background: white; }}
    @media (max-width: 1000px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .panel {{ height: 70vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>Structure labels are preserved but ignored for matching. Blue is fixed; red is moving before alignment; green is moving after alignment.</p>
  </header>
  {warning_html}
  <div class="grid">
    <div id="before" class="panel"></div>
    <div id="after" class="panel"></div>
  </div>
  <script>
    const beforeData = {json.dumps(before_traces)};
    const afterData = {json.dumps(after_traces)};
    const baseLayout = {{
      margin: {{l: 50, r: 20, t: 38, b: 45}},
      xaxis: {{title: "x", scaleanchor: "y", scaleratio: 1, zeroline: false}},
      yaxis: {{title: "y", autorange: "reversed", zeroline: false}},
      legend: {{orientation: "h", y: 1.08}},
      hovermode: "closest"
    }};
    Plotly.newPlot("before", beforeData, {{...baseLayout, title: "Before"}}, {{responsive: true, scrollZoom: true}});
    Plotly.newPlot("after", afterData, {{...baseLayout, title: "After"}}, {{responsive: true, scrollZoom: true}});
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html


def _write_partial_anchor_alignment_html(
    *,
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    final_records: list[_FeatureRecord],
    anchor_rows: pd.DataFrame,
    output_html: Path,
    transform_summary: dict[str, Any],
) -> Path:
    before_traces = [
        _records_to_plotly_trace(fixed_records, name="fixed", color="#1f77b4"),
        _records_to_plotly_trace(moving_records, name="moving before", color="#d62728"),
    ]
    after_traces = [
        _records_to_plotly_trace(fixed_records, name="fixed", color="#1f77b4"),
        _records_to_plotly_trace(final_records, name="moving anchor-only aligned", color="#2ca02c"),
    ]
    if not anchor_rows.empty:
        used = anchor_rows.loc[anchor_rows["used_for_transform"].astype(bool)].copy()
        if not used.empty:
            before_traces.append(
                _anchor_alignment_link_trace(
                    used,
                    name="transform anchors before",
                    color="#7b2cbf",
                    aligned=False,
                )
            )
            after_traces.append(
                _anchor_alignment_link_trace(
                    used,
                    name="transform anchors after",
                    color="#7b2cbf",
                    aligned=True,
                )
            )
    metrics = (
        f"anchor pairs: {transform_summary.get('used_anchor_pair_count', 0)} used / "
        f"{transform_summary.get('anchor_pair_count', 0)} selected; "
        f"median residual: {float(transform_summary.get('residual_median', math.inf)):.2f}; "
        f"p90 residual: {float(transform_summary.get('residual_p90', math.inf)):.2f}"
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Partial-anchor contour alignment</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f9fc; color: #172033; }}
    header {{ padding: 16px 24px 8px 24px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    p {{ margin: 6px 0 0 0; color: #536273; }}
    .note {{ margin: 12px 24px 0 24px; padding: 12px 14px; border: 1px solid #b8d5ff; border-radius: 6px; background: #eef6ff; color: #17324d; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px 16px 16px 16px; }}
    .panel {{ height: calc(100vh - 124px); min-height: 560px; border: 1px solid #d8e0ea; background: white; }}
    @media (max-width: 1000px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .panel {{ height: 70vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Partial-anchor contour alignment</h1>
    <p>Blue is fixed, red is moving before alignment, green is moving after anchor-only alignment. Purple links are the matched-anchor pairs used to estimate the transform.</p>
  </header>
  <div class="note">
    This view does not force all contours to overlap. Only high-confidence matched_anchor pairs fit the transform; matched_review and no_counterpart contours are transformed only as passive geometry.
    <br><strong>{metrics}</strong>
  </div>
  <div class="grid">
    <div id="before" class="panel"></div>
    <div id="after" class="panel"></div>
  </div>
  <script>
    const beforeData = {json.dumps(before_traces)};
    const afterData = {json.dumps(after_traces)};
    const baseLayout = {{
      margin: {{l: 50, r: 20, t: 38, b: 45}},
      xaxis: {{title: "x", scaleanchor: "y", scaleratio: 1, zeroline: false}},
      yaxis: {{title: "y", autorange: "reversed", zeroline: false}},
      legend: {{orientation: "h", y: 1.08}},
      hovermode: "closest"
    }};
    Plotly.newPlot("before", beforeData, {{...baseLayout, title: "Before"}}, {{responsive: true, scrollZoom: true}});
    Plotly.newPlot("after", afterData, {{...baseLayout, title: "After: anchor-only transform"}}, {{responsive: true, scrollZoom: true}});
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html


def _write_anchor_only_residual_tps_html(
    *,
    fixed_records: list[_FeatureRecord],
    hard_records: list[_FeatureRecord],
    soft_records: list[_FeatureRecord],
    landmarks: pd.DataFrame,
    output_html: Path,
    accepted: bool,
    failure_reasons: Sequence[str],
    jacobian: dict[str, Any],
) -> Path:
    used = (
        landmarks.loc[
            (landmarks.get("landmark_kind") == "matched_anchor")
            & landmarks.get("used_for_tps", pd.Series(False, index=landmarks.index)).astype(bool)
        ].copy()
        if not landmarks.empty
        else pd.DataFrame()
    )
    before_traces = [
        _records_to_plotly_trace(fixed_records, name="fixed", color="#1f77b4"),
        _records_to_plotly_trace(hard_records, name="moving hard-aligned", color="#d62728"),
    ]
    after_traces = [
        _records_to_plotly_trace(fixed_records, name="fixed", color="#1f77b4"),
        _records_to_plotly_trace(soft_records, name="moving residual TPS", color="#2ca02c"),
    ]
    if not used.empty:
        before_traces.append(
            _anchor_only_residual_link_trace(
                used,
                name="TPS source-to-target anchors",
                color="#7b2cbf",
                warped=False,
            )
        )
        after_traces.append(
            _anchor_only_residual_link_trace(
                used,
                name="post-TPS residual vectors",
                color="#f97316",
                warped=True,
            )
        )
    reason_text = "; ".join(failure_reasons) if failure_reasons else "accepted"
    jacobian_text = (
        f"negative Jacobian ratio: "
        f"{float(jacobian.get('negative_jacobian_ratio', 0.0)):.4f}; "
        f"checked cells: {jacobian.get('checked_cells', 0)}"
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Anchor-only residual TPS alignment</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f9fc; color: #172033; }}
    header {{ padding: 16px 24px 8px 24px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    p {{ margin: 6px 0 0 0; color: #536273; }}
    .note {{ margin: 12px 24px 0 24px; padding: 12px 14px; border: 1px solid #b8d5ff; border-radius: 6px; background: #eef6ff; color: #17324d; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px 16px 16px 16px; }}
    .panel {{ height: calc(100vh - 132px); min-height: 560px; border: 1px solid #d8e0ea; background: white; }}
    @media (max-width: 1000px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .panel {{ height: 70vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Anchor-only residual TPS alignment</h1>
    <p>Blue is fixed, red is hard-aligned moving before residual TPS, green is moving after residual TPS. Purple links are active evidence anchors; orange links are post-TPS residuals.</p>
  </header>
  <div class="note">
    Anchor-only mode uses only hard-stage matched anchors as TPS control points. Passive/no-counterpart contours follow the field but do not attract boundaries.
    <br><strong>accepted: {str(bool(accepted)).lower()}; active anchors: {len(used)}; {jacobian_text}; reason: {reason_text}</strong>
  </div>
  <div class="grid">
    <div id="before" class="panel"></div>
    <div id="after" class="panel"></div>
  </div>
  <script>
    const beforeData = {json.dumps(before_traces)};
    const afterData = {json.dumps(after_traces)};
    const baseLayout = {{
      margin: {{l: 50, r: 20, t: 38, b: 45}},
      xaxis: {{title: "x", scaleanchor: "y", scaleratio: 1, zeroline: false}},
      yaxis: {{title: "y", autorange: "reversed", zeroline: false}},
      legend: {{orientation: "h", y: 1.08}},
      hovermode: "closest"
    }};
    Plotly.newPlot("before", beforeData, {{...baseLayout, title: "Before residual TPS"}}, {{responsive: true, scrollZoom: true}});
    Plotly.newPlot("after", afterData, {{...baseLayout, title: "After residual TPS"}}, {{responsive: true, scrollZoom: true}});
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html


def _write_group_correspondence_matrix_html(
    matrix: pd.DataFrame,
    output_html: Path,
) -> Path:
    display = matrix.copy()
    if not display.empty:
        for column in (
            "score",
            "median_residual",
            "rotation_degrees",
            "scale",
            "translate_x",
            "translate_y",
        ):
            if column in display.columns:
                display[column] = display[column].map(
                    lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
                )
    table_html = (
        display.to_html(index=False, escape=True, classes="matrix")
        if not display.empty
        else "<p>No valid group correspondences were found.</p>"
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Group correspondence matrix</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #172033; }}
    h1 {{ margin-bottom: 4px; }}
    p {{ color: #536273; }}
    table.matrix {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
    .matrix th, .matrix td {{ border: 1px solid #d8e0ea; padding: 6px 8px; text-align: right; }}
    .matrix th {{ background: #eef3f8; position: sticky; top: 0; }}
    .matrix td:nth-child(2), .matrix td:nth-child(3),
    .matrix th:nth-child(2), .matrix th:nth-child(3) {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Label-free group correspondence matrix</h1>
  <p>Each row is a fixed assigned_structure group matched against a moving assigned_structure group. The top row is used for group-overlap alignment.</p>
  {table_html}
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html


def _anchor_alignment_link_trace(
    rows: pd.DataFrame,
    *,
    name: str,
    color: str,
    aligned: bool,
) -> dict[str, Any]:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    text_values: list[str | None] = []
    for _, row in rows.iterrows():
        moving_x = (
            float(row["aligned_moving_centroid_x"])
            if aligned
            else float(row["moving_centroid_x"])
        )
        moving_y = (
            float(row["aligned_moving_centroid_y"])
            if aligned
            else float(row["moving_centroid_y"])
        )
        fixed_x = float(row["fixed_centroid_x"])
        fixed_y = float(row["fixed_centroid_y"])
        x_values.extend([fixed_x, moving_x, None])
        y_values.extend([fixed_y, moving_y, None])
        hover = (
            f"{row['fixed_node_id']} -> {row['moving_node_id']}<br>"
            f"score={float(row['total_score']):.3f}<br>"
            f"residual={float(row['anchor_residual']):.2f}"
        )
        text_values.extend([hover, hover, None])
    return {
        "type": "scattergl",
        "mode": "lines",
        "name": name,
        "x": x_values,
        "y": y_values,
        "text": text_values,
        "hovertemplate": "%{text}<extra></extra>",
        "line": {"color": color, "width": 1.25},
        "opacity": 0.65,
    }


def _anchor_only_residual_link_trace(
    rows: pd.DataFrame,
    *,
    name: str,
    color: str,
    warped: bool,
) -> dict[str, Any]:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    text_values: list[str | None] = []
    for _, row in rows.iterrows():
        src_x = float(row["src_x"])
        src_y = float(row["src_y"])
        dst_x = float(row["dst_x"])
        dst_y = float(row["dst_y"])
        x_values.extend([src_x, dst_x, None])
        y_values.extend([src_y, dst_y, None])
        residual = float(row.get("input_residual_um", math.hypot(dst_x - src_x, dst_y - src_y)))
        hover = (
            f"{row.get('fixed_node_id')} <- {row.get('moving_node_id')}<br>"
            f"residual={residual:.2f}<br>"
            f"used_for_tps={bool(row.get('used_for_tps', False))}"
        )
        text_values.extend([hover, hover, None])
    return {
        "type": "scattergl",
        "mode": "lines",
        "name": name,
        "x": x_values,
        "y": y_values,
        "text": text_values,
        "hovertemplate": "%{text}<extra></extra>",
        "line": {"color": color, "width": 1.25 if not warped else 1.0},
        "opacity": 0.70 if not warped else 0.55,
    }


def _compute_correspondence_diagnostics(
    *,
    fixed_records: list[_FeatureRecord],
    moving_records: list[_FeatureRecord],
    output_csv: Path,
    dominant_area_fraction: float = 0.05,
    min_internal_area: float = 1e3,
) -> dict[str, Any]:
    fixed_by_label = _records_by_original_label(fixed_records)
    moving_by_label = _records_by_original_label(moving_records)
    rows: list[dict[str, Any]] = []
    best_iou = 0.0
    for fixed_label, fixed_geom in sorted(fixed_by_label.items()):
        for moving_label, moving_geom in sorted(moving_by_label.items()):
            iou = _iou(fixed_geom, moving_geom)
            rows.append(
                {
                    "fixed_label": fixed_label,
                    "moving_label": moving_label,
                    "iou": iou,
                    "fixed_area": float(fixed_geom.area),
                    "moving_area": float(moving_geom.area),
                }
            )
            best_iou = max(best_iou, float(iou))
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    fixed_total = max(sum(float(record.geometry.area) for record in fixed_records), 1e-12)
    moving_total = max(sum(float(record.geometry.area) for record in moving_records), 1e-12)
    fixed_internal = [
        record.geometry
        for record in fixed_records
        if float(record.geometry.area) >= min_internal_area
        and float(record.geometry.area) / fixed_total < dominant_area_fraction
    ]
    moving_internal = [
        record.geometry
        for record in moving_records
        if float(record.geometry.area) >= min_internal_area
        and float(record.geometry.area) / moving_total < dominant_area_fraction
    ]
    internal_chamfer = math.inf
    internal_similarity = 0.0
    if fixed_internal and moving_internal:
        fixed_union = unary_union(fixed_internal)
        moving_union = unary_union(moving_internal)
        fixed_points = _sample_boundary_points(fixed_union.boundary, 4000)
        moving_points = _sample_boundary_points(moving_union.boundary, 4000)
        internal_chamfer = _symmetric_nearest_loss(fixed_points, moving_points)
        all_union = unary_union([fixed_union, moving_union])
        span = _characteristic_span(all_union)
        internal_similarity = float(math.exp(-internal_chamfer / max(span * 0.015, 1e-6)))

    warning: str | None = None
    if fixed_internal and moving_internal and internal_similarity < 0.55:
        warning = (
            "The tissue/envelope coordinate frame is comparable, but internal contours have low "
            "label-free correspondence. These independently generated Leiden/StructureMap contours "
            "should not be treated as homologous landmarks; use this output as an envelope/coordinate "
            "diagnostic or rerun with harmonized labels/raw cell-density registration."
        )

    return {
        "label_overlap_csv": str(output_csv),
        "best_label_pair_iou": float(best_iou),
        "internal_component_count_fixed": int(len(fixed_internal)),
        "internal_component_count_moving": int(len(moving_internal)),
        "internal_boundary_chamfer_um": float(internal_chamfer),
        "internal_boundary_similarity": float(internal_similarity),
        "warning": warning,
    }


def _records_by_original_label(records: list[_FeatureRecord]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = {}
    for record in records:
        label = _original_label(record.feature) or "__unlabeled__"
        grouped.setdefault(str(label), []).append(record.geometry)
    return {label: unary_union(geoms) for label, geoms in grouped.items()}


def _records_to_plotly_trace(
    records: list[_FeatureRecord],
    *,
    name: str,
    color: str,
    max_points: int = 80000,
) -> dict[str, Any]:
    lines = []
    total_length = 0.0
    for record in records:
        for polygon in _iter_polygons(record.geometry):
            lines.append(polygon.exterior)
            total_length += float(polygon.exterior.length)
            for interior in polygon.interiors:
                lines.append(interior)
                total_length += float(interior.length)
    spacing = max(total_length / max(max_points, 1), 1e-6) if total_length > 0 else 1.0
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    for line in lines:
        length = float(line.length)
        if length <= 0:
            continue
        distances = np.arange(0.0, length, spacing)
        if len(distances) == 0:
            distances = np.array([0.0])
        for distance in distances:
            point = line.interpolate(float(distance))
            x_values.append(round(float(point.x), 3))
            y_values.append(round(float(point.y), 3))
        x_values.append(None)
        y_values.append(None)
    return {
        "type": "scattergl",
        "mode": "lines",
        "name": name,
        "x": x_values,
        "y": y_values,
        "line": {"color": color, "width": 1.1},
        "opacity": 0.82,
    }


def _iter_polygons(geom: Any) -> Iterable[Polygon]:
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        yield from geom.geoms
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from _iter_polygons(part)
