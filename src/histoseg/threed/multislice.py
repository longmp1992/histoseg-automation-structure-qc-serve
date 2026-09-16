from __future__ import annotations

import copy
import html
import json
import math
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label as nd_label
from scipy.optimize import minimize
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union
from skimage import measure

from histoseg.contour import (
    MultiStructureContourConfig,
    MultiStructureSpec,
    run_multi_structure_contours,
)

from .soft_alignment import (
    ThreeDContourReconstructionConfig,
    _feature_group,
    _iou,
    _read_geojson,
    _records_from_geojson,
    run_3d_contour_reconstruction,
)
from .image_registration import (
    CODA_METHOD_CREDIT,
    CODA_METHOD_REFERENCE_DOI,
    CODA_METHOD_REFERENCE_URL,
    CODA_METHODOLOGY_URL,
    CodaImageRegistrationConfig,
    estimate_coda_image_registration,
)
try:  # Shapely 2.x fast vectorized containment.
    from shapely import contains_xy as _contains_xy
except Exception:  # pragma: no cover - only used on older Shapely versions.
    _contains_xy = None

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ThreeDStackReconstructionConfig:
    """Configuration for same-sample multi-slice Xenium contour reconstruction.

    This workflow reads Xenium output folders through GitHub pyXenium's
    ``XeniumSlide`` IO path, builds 2D structure contours for each slice, aligns
    slices in a fixed order, and exports a 3D contour stack plus surface meshes.
    """

    xenium_root: PathLike | None = None
    out_dir: PathLike = "outputs/3d_stack_reconstruction"
    contour_manifest: PathLike | None = None
    segmentation_strategy: PathLike | None = None
    structures: Sequence[MultiStructureSpec | Mapping[str, Any]] | None = None
    slice_dirs: Sequence[PathLike] | None = None
    sample_glob: str = "*"
    reference_slice_index: int = 0
    z_spacing_um: float = 5.0
    xenium_pixel_size_um: float = 0.2125
    group_property: str = "structure"
    cluster_column_name: str = "cluster"
    clusters_relpath: str = "analysis/clustering/gene_expression_graphclust/clusters.csv"
    merged_h5ad: PathLike | None = None
    merged_cluster_column: str | None = None
    merged_sample_column: str = "sample_id"
    merged_barcode_column: str = "barcode"
    bins_x: int = 900
    bins_y: int = 700
    gaussian_sigma: float = 2.25
    density_scale_quantile: float = 0.98
    support_quantile: float = 0.18
    tissue_quantile: float = 0.06
    min_dominance: float = 0.34
    min_cells: int = 500
    min_component_pixels: int = 180
    save_slice_preview_png: bool = False
    registration_backend: str = "auto"
    hard_alignment_maxiter: int = 80
    # Enable multi-start similarity search (tries PCA rotation + 0/90/180/270° seeds).
    hard_alignment_multistart: bool = True
    # When similarity IoU after hard alignment falls below this threshold, try a 6-DOF affine
    # fallback and use it if it improves IoU.  Set to 0.0 to disable.
    affine_fallback_iou_threshold: float = 0.0
    coda_raster_size: int = 512
    coda_angle_step: float = 1.0
    coda_phase_upsample_factor: int = 1
    coda_mask_padding_fraction: float = 0.05
    label_free_search_window: float = 800.0
    label_free_knn_neighbors: int = 6
    label_free_min_anchor_count: int = 8
    label_free_group_candidate_count: int = 12
    label_free_group_ransac_trials: int = 15000
    label_free_group_min_descriptor_score: float = 0.35
    label_free_group_residual_limit_um: float = 900.0
    label_free_group_min_component_area_um2: float = 5000.0
    label_free_guarded_alignment: bool = True
    label_free_min_similarity_anchor_count: int = 6
    label_free_max_rotation_degrees: float = 15.0
    label_free_high_rotation_min_anchor_count: int = 8
    label_free_high_rotation_residual_limit_um: float = 120.0
    label_free_min_anchor_coverage: float = 0.05
    label_free_min_anchor_quadrants: int = 2
    label_free_near_180_rotation_degrees: float = 150.0
    # Apply a linear drift correction after the full pairwise alignment chain to reduce
    # cumulative translation drift across the stack.
    global_drift_correction: bool = False
    run_soft_alignment: bool = True
    soft_alignment_mode: str = "auto"
    anchor_only_bbox_padding_fraction: float = 0.10
    anchor_only_identity_padding_count: int = 16
    anchor_only_rbf_smoothing: float = 1e-4
    anchor_only_jacobian_grid_size: int = 50
    anchor_only_max_negative_jacobian_ratio: float = 0.001
    iterative_contour_refinement: bool = True
    iterative_contour_refinement_rounds: int = 1
    iterative_contour_min_pair_count: int = 3
    iterative_contour_min_anchor_count: int = 30
    iterative_contour_min_pair_score: float = 0.52
    iterative_contour_min_overlap: float = 0.12
    iterative_contour_centroid_search_radius_um: float = 350.0
    iterative_contour_accepted_centroid_radius_um: float = 260.0
    iterative_contour_max_anchor_distance_um: float = 180.0
    iterative_contour_residual_limit_um: float = 220.0
    iterative_contour_max_negative_jacobian_ratio: float = 0.002
    sampling_distance_um: float = 50.0
    max_landmark_distance_um: float = 180.0
    landmarks_per_structure: int | None = 260
    diagnostic_structure_landmarks: int | None = 620
    landmark_candidate_count: int = 8
    landmark_candidate_spacing_um: float | None = None
    landmark_normal_weight_um: float = 0.0
    landmark_normal_step_um: float | None = None
    rbf_neighbors: int | None = 96
    rbf_smoothing: float | str = 1e-4
    rbf_smoothing_candidates: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2)
    topology_grid_size: int = 24
    topology_min_area_ratio: float = 0.5
    topology_max_area_ratio: float = 2.0
    diagnostic_structure: str | None = "Structure 5"
    save_alignment_preview_png: bool = True
    curvature_landmark_weight: float = 0.5
    mutual_nn_check: bool = True
    icp_iterations: int = 2
    zero_anchor_count: int = 16
    landmark_outlier_mad_threshold: float | None = 3.5
    # Accept soft alignment per structure independently: structures where soft IoU ≥ hard IoU
    # use the soft-aligned geometry; others fall back to hard-aligned geometry.
    per_structure_soft_acceptance: bool = True
    point_sample_distance_um: float = 80.0
    voxel_size_um: float = 80.0
    mesh_method: str = "marching_cubes"
    mesh_smoothing_sigma_um: float | None = 40.0
    mesh_level: float = 0.5
    mesh_export_formats: Sequence[str] = ("ply", "obj")
    mesh_cleanup: bool = True
    min_mesh_component_volume_um3: float | None = None
    mesh_max_faces_for_html: int = 25000
    local_z_orientation: str = "off"
    vertical_qc_backend: str = "none"
    apply_local_z_flip: bool = False
    transcript_relpath: str = "transcripts.parquet"
    orientation_spatial_unit: str = "auto"
    contour_group_property: str = "structure"
    contour_min_transcripts: int = 50
    contour_match_min_iou: float = 0.01
    contour_match_max_distance_um: float = 120.0
    orientation_bootstrap_iterations: int = 100
    orientation_bootstrap_seed: int = 0
    ovrlpy_n_components: int = 20
    ovrlpy_n_workers: int = 1
    ovrlpy_fit_umap: bool = True
    ovrlpy_min_transcripts: float = 10.0
    overwrite: bool = False
    dpi: int = 180


@dataclass
class ThreeDStackReconstructionResult:
    out_dir: Path
    slice_manifest_csv: Path
    pairwise_metrics_csv: Path
    aligned_manifest_csv: Path
    contour_points_csv: Path
    summary_json: Path
    visualization_html: Path
    mesh_dir: Path
    guarded_alignment_audit_csv: Path | None = None
    guarded_alignment_audit_html: Path | None = None
    local_z_orientation_manifest_csv: Path | None = None
    aligned_transcripts_parquet: Path | None = None
    biological_z_report_html: Path | None = None


@dataclass(frozen=True)
class _SliceInput:
    order: int
    sample_id: str
    sample_dir: Path
    xenium_dir: Path
    z_um: float | None = None
    precomputed_geojson: Path | None = None


@dataclass(frozen=True)
class _SimilarityTransform:
    origin_x: float
    origin_y: float
    rotation_degrees: float
    scale: float
    translate_x: float
    translate_y: float


def run_3d_stack_reconstruction(
    cfg: ThreeDStackReconstructionConfig,
) -> ThreeDStackReconstructionResult:
    """Run 32-slice style Xenium contour alignment and 3D reconstruction."""

    _validate_stack_config(cfg)
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.contour_manifest is not None:
        structures: list[MultiStructureSpec | Mapping[str, Any]] = []
        slices = _read_precomputed_contour_manifest(cfg.contour_manifest)
    else:
        structures = list(cfg.structures) if cfg.structures is not None else _read_strategy_specs(cfg)
        slices = discover_xenium_slices(
            cfg.xenium_root,
            slice_dirs=cfg.slice_dirs,
            sample_glob=cfg.sample_glob,
        )
    if not slices:
        raise ValueError(f"No Xenium output folders were found under {cfg.xenium_root!s}.")
    if cfg.reference_slice_index != 0:
        raise ValueError("Only reference_slice_index=0 is supported in v1 to keep order fixed.")

    slice_manifest_path = out_dir / "xenium_slice_manifest.csv"
    pd.DataFrame(
        [
            {
                "order": item.order,
                "sample_id": item.sample_id,
                "sample_dir": str(item.sample_dir),
                "xenium_dir": str(item.xenium_dir),
                "z_um": _slice_z_um(item, cfg),
                "precomputed_geojson": (
                    str(item.precomputed_geojson) if item.precomputed_geojson is not None else None
                ),
            }
            for item in slices
        ]
    ).to_csv(slice_manifest_path, index=False)
    merged_cluster_tables = {} if cfg.contour_manifest is not None else _load_merged_cluster_tables(cfg, slices)

    contour_dir = out_dir / "slice_contours"
    aligned_dir = out_dir / "aligned_contours"
    alignment_dir = out_dir / "pairwise_alignments"
    mesh_dir = out_dir / "meshes"
    for path in (contour_dir, aligned_dir, alignment_dir, mesh_dir):
        path.mkdir(parents=True, exist_ok=True)

    aligned_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    aligned_paths: list[Path] = []

    for slice_input in slices:
        raw_geojson = _prepare_slice_contours(
            slice_input,
            structures,
            contour_dir,
            cfg,
            merged_clusters=merged_cluster_tables.get(slice_input.sample_id),
        )
        aligned_path = aligned_dir / f"{slice_input.order:03d}_{slice_input.sample_id}_aligned.geojson"

        if slice_input.order == 1:
            if cfg.overwrite or not aligned_path.exists():
                shutil.copy2(raw_geojson, aligned_path)
            aligned_paths.append(aligned_path)
            aligned_rows.append(
                {
                    "order": slice_input.order,
                    "sample_id": slice_input.sample_id,
                    "z_um": _slice_z_um(slice_input, cfg),
                    "raw_geojson": str(raw_geojson),
                    "aligned_geojson": str(aligned_path),
                    "alignment_role": "reference",
                }
            )
            continue

        fixed_path = aligned_paths[-1]
        pair_dir = alignment_dir / f"{slice_input.order - 1:03d}_to_{slice_input.order:03d}_{slice_input.sample_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        hard_path = pair_dir / "moving_hard_aligned.geojson"
        hard_summary_path = pair_dir / "hard_similarity_alignment.json"

        hard_summary = _run_hard_alignment_backend(
            cfg,
            fixed_geojson=fixed_path,
            moving_geojson=raw_geojson,
            output_geojson=hard_path,
            summary_json=hard_summary_path,
        )

        semantic_soft_allowed, semantic_soft_skipped_reason = _semantic_soft_alignment_policy(
            hard_summary
        )
        active_soft_mode, soft_alignment_skipped_reason = _resolve_soft_alignment_mode(
            cfg,
            hard_summary,
            semantic_soft_allowed=semantic_soft_allowed,
            semantic_soft_skipped_reason=semantic_soft_skipped_reason,
        )
        hard_summary = {
            **hard_summary,
            "semantic_soft_allowed": semantic_soft_allowed,
            "semantic_soft_skipped_reason": (
                semantic_soft_skipped_reason
                if cfg.run_soft_alignment and not semantic_soft_allowed
                else None
            ),
            "soft_alignment_mode_requested": (
                "none" if not cfg.run_soft_alignment else cfg.soft_alignment_mode
            ),
            "active_soft_alignment_mode": active_soft_mode,
            "soft_alignment_skipped_reason": soft_alignment_skipped_reason,
        }

        if active_soft_mode == "semantic":
            semantic_start = time.perf_counter()
            soft_result = run_3d_contour_reconstruction(
                ThreeDContourReconstructionConfig(
                    fixed_geojson=fixed_path,
                    moving_hard_aligned_geojson=hard_path,
                    out_dir=pair_dir / "soft_tps",
                    group_property=cfg.group_property,
                    sampling_distance_um=cfg.sampling_distance_um,
                    max_landmark_distance_um=cfg.max_landmark_distance_um,
                    landmarks_per_structure=cfg.landmarks_per_structure,
                    diagnostic_structure_landmarks=cfg.diagnostic_structure_landmarks,
                    landmark_candidate_count=cfg.landmark_candidate_count,
                    landmark_candidate_spacing_um=cfg.landmark_candidate_spacing_um,
                    landmark_normal_weight_um=cfg.landmark_normal_weight_um,
                    landmark_normal_step_um=cfg.landmark_normal_step_um,
                    rbf_neighbors=cfg.rbf_neighbors,
                    rbf_smoothing=cfg.rbf_smoothing,
                    rbf_smoothing_candidates=cfg.rbf_smoothing_candidates,
                    topology_grid_size=cfg.topology_grid_size,
                    topology_min_area_ratio=cfg.topology_min_area_ratio,
                    topology_max_area_ratio=cfg.topology_max_area_ratio,
                    diagnostic_structure=cfg.diagnostic_structure,
                    dpi=cfg.dpi,
                    save_preview_png=cfg.save_alignment_preview_png,
                    curvature_landmark_weight=cfg.curvature_landmark_weight,
                    mutual_nn_check=cfg.mutual_nn_check,
                    icp_iterations=cfg.icp_iterations,
                    zero_anchor_count=cfg.zero_anchor_count,
                    landmark_outlier_mad_threshold=cfg.landmark_outlier_mad_threshold,
                    per_structure_soft_acceptance=cfg.per_structure_soft_acceptance,
                )
            )
            soft_summary = json.loads(soft_result.summary_json.read_text(encoding="utf-8"))
            hard_summary = {
                **hard_summary,
                "soft_alignment_runtime_seconds": time.perf_counter() - semantic_start,
            }
            topology_valid = bool(soft_summary["qc"].get("topology_check", {}).get("valid", True))
            geometry_valid = int(soft_summary["qc"].get("geometry_status_counts", {}).get("invalid", 0)) == 0
            if cfg.overwrite or not aligned_path.exists():
                if topology_valid and geometry_valid and cfg.per_structure_soft_acceptance:
                    # Build a per-structure mixed GeoJSON: use soft geometry for structures
                    # where soft IoU ≥ hard IoU, hard geometry for regressed structures.
                    hard_payload = _read_geojson(hard_path)
                    soft_payload = _read_geojson(soft_result.soft_aligned_geojson)
                    mixed_payload = _build_per_structure_soft_geojson(
                        hard_payload, soft_payload, cfg.group_property, soft_summary
                    )
                    aligned_path.write_text(
                        json.dumps(mixed_payload, ensure_ascii=False), encoding="utf-8"
                    )
                    soft_accepted = True
                else:
                    soft_improved = (
                        soft_summary["qc"]["union_iou_soft_after"]
                        >= soft_summary["qc"]["union_iou_hard_before_soft"]
                    )
                    soft_accepted = soft_improved and topology_valid and geometry_valid
                    shutil.copy2(
                        soft_result.soft_aligned_geojson if soft_accepted else hard_path,
                        aligned_path,
                    )
            else:
                soft_improved = (
                    soft_summary["qc"]["union_iou_soft_after"]
                    >= soft_summary["qc"]["union_iou_hard_before_soft"]
                )
                soft_accepted = soft_improved and topology_valid and geometry_valid
            pairwise_rows.append(
                _pairwise_row(
                    slice_input,
                    hard_summary,
                    soft_summary,
                    soft_result,
                    soft_accepted=soft_accepted,
                )
            )
        elif active_soft_mode == "anchor-only":
            anchor_start = time.perf_counter()
            anchor_csv = _resolve_label_free_anchor_landmarks_csv(hard_summary)
            if anchor_csv is None:
                hard_summary = {
                    **hard_summary,
                    "soft_alignment_skipped_reason": "missing_label_free_anchor_landmarks_csv",
                    "soft_alignment_runtime_seconds": 0.0,
                }
                if cfg.overwrite or not aligned_path.exists():
                    shutil.copy2(hard_path, aligned_path)
                pairwise_rows.append(_pairwise_row(slice_input, hard_summary, None, None))
            else:
                from .label_free_alignment import (  # Local import avoids a module cycle.
                    AnchorOnlyResidualTPSConfig,
                    run_anchor_only_residual_tps,
                )

                try:
                    soft_result = run_anchor_only_residual_tps(
                        AnchorOnlyResidualTPSConfig(
                            fixed_geojson=fixed_path,
                            moving_hard_aligned_geojson=hard_path,
                            anchor_landmarks_csv=anchor_csv,
                            out_dir=pair_dir / "anchor_only_soft_tps",
                            group_property=cfg.group_property,
                            min_anchor_count=cfg.label_free_min_anchor_count,
                            residual_limit_um=cfg.label_free_group_residual_limit_um,
                            bbox_padding_fraction=cfg.anchor_only_bbox_padding_fraction,
                            identity_padding_count=cfg.anchor_only_identity_padding_count,
                            rbf_neighbors=cfg.rbf_neighbors,
                            rbf_smoothing=cfg.anchor_only_rbf_smoothing,
                            jacobian_grid_size=cfg.anchor_only_jacobian_grid_size,
                            max_negative_jacobian_ratio=(
                                cfg.anchor_only_max_negative_jacobian_ratio
                            ),
                            save_preview_png=cfg.save_alignment_preview_png,
                            overwrite=cfg.overwrite,
                            dpi=cfg.dpi,
                        )
                    )
                    soft_summary = json.loads(
                        soft_result.summary_json.read_text(encoding="utf-8")
                    )
                    soft_accepted = bool(soft_summary.get("accepted", False))
                    if soft_accepted and cfg.iterative_contour_refinement:
                        (
                            soft_result,
                            soft_summary,
                            soft_accepted,
                        ) = _run_iterative_contour_refinement_if_enabled(
                            cfg=cfg,
                            fixed_geojson=fixed_path,
                            current_soft_result=soft_result,
                            current_soft_summary=soft_summary,
                            pair_dir=pair_dir,
                            round_count=cfg.iterative_contour_refinement_rounds,
                        )
                    hard_summary = {
                        **hard_summary,
                        "soft_alignment_runtime_seconds": (
                            time.perf_counter() - anchor_start
                        ),
                    }
                    if cfg.overwrite or not aligned_path.exists():
                        shutil.copy2(
                            soft_result.soft_aligned_geojson if soft_accepted else hard_path,
                            aligned_path,
                        )
                    pairwise_rows.append(
                        _pairwise_row(
                            slice_input,
                            hard_summary,
                            soft_summary,
                            soft_result,
                            soft_accepted=soft_accepted,
                        )
                    )
                except Exception as exc:
                    hard_summary = {
                        **hard_summary,
                        "soft_alignment_skipped_reason": (
                            f"anchor_only_residual_tps_failed:{type(exc).__name__}"
                        ),
                        "soft_alignment_runtime_seconds": (
                            time.perf_counter() - anchor_start
                        ),
                    }
                    if cfg.overwrite or not aligned_path.exists():
                        shutil.copy2(hard_path, aligned_path)
                    pairwise_rows.append(_pairwise_row(slice_input, hard_summary, None, None))
        else:
            if cfg.overwrite or not aligned_path.exists():
                shutil.copy2(hard_path, aligned_path)
            pairwise_rows.append(_pairwise_row(slice_input, hard_summary, None, None))

        aligned_paths.append(aligned_path)
        aligned_rows.append(
            {
                "order": slice_input.order,
                "sample_id": slice_input.sample_id,
                "z_um": _slice_z_um(slice_input, cfg),
                "raw_geojson": str(raw_geojson),
                "aligned_geojson": str(aligned_path),
                "alignment_role": "moving",
            }
        )

    aligned_manifest_path = out_dir / "aligned_slice_manifest.csv"
    pd.DataFrame(aligned_rows).to_csv(aligned_manifest_path, index=False)

    pairwise_metrics_path = out_dir / "pairwise_alignment_metrics.csv"
    pd.DataFrame(pairwise_rows).to_csv(pairwise_metrics_path, index=False)
    guarded_audit_csv, guarded_audit_html = _write_guarded_alignment_audit(
        pairwise_rows,
        out_dir,
    )

    # Global drift correction: remove linear centroid drift accumulated along the chain.
    if cfg.global_drift_correction and len(aligned_paths) > 2:
        _apply_global_drift_correction(
            aligned_paths,
            aligned_rows,
            group_property=cfg.group_property,
        )

    contour_points_path = out_dir / "aligned_contour_3d_points.csv"
    contour_points = write_3d_contour_points(
        aligned_rows,
        contour_points_path,
        group_property=cfg.group_property,
        point_sample_distance_um=cfg.point_sample_distance_um,
        xenium_pixel_size_um=cfg.xenium_pixel_size_um,
    )

    mesh_payloads = reconstruct_3d_contour_meshes(
        aligned_rows,
        mesh_dir,
        group_property=cfg.group_property,
        voxel_size_um=cfg.voxel_size_um,
        z_spacing_um=cfg.z_spacing_um,
        xenium_pixel_size_um=cfg.xenium_pixel_size_um,
        mesh_method=cfg.mesh_method,
        mesh_smoothing_sigma_um=cfg.mesh_smoothing_sigma_um,
        mesh_level=cfg.mesh_level,
        mesh_export_formats=cfg.mesh_export_formats,
        mesh_cleanup=cfg.mesh_cleanup,
        min_mesh_component_volume_um3=cfg.min_mesh_component_volume_um3,
        max_faces_for_html=cfg.mesh_max_faces_for_html,
    )

    visualization_path = out_dir / "histoseg_3d_contour_stack.html"
    write_3d_visualization_html(
        contour_points,
        mesh_payloads,
        visualization_path,
        title="HistoSeg 3D contour reconstruction",
    )

    local_z_result = None
    if cfg.local_z_orientation == "auto":
        from .local_z_orientation import (
            LocalZOrientationConfig,
            run_local_z_orientation_correction,
        )

        local_z_result = run_local_z_orientation_correction(
            LocalZOrientationConfig(
                xenium_root=cfg.xenium_root,
                stack_root=out_dir,
                out_dir=out_dir,
                sample_glob=cfg.sample_glob,
                transcript_relpath=cfg.transcript_relpath,
                pixel_size_um=cfg.xenium_pixel_size_um,
                orientation_spatial_unit=cfg.orientation_spatial_unit,
                contour_group_property=cfg.contour_group_property,
                contour_min_transcripts=cfg.contour_min_transcripts,
                contour_match_min_iou=cfg.contour_match_min_iou,
                contour_match_max_distance_um=cfg.contour_match_max_distance_um,
                orientation_bootstrap_iterations=cfg.orientation_bootstrap_iterations,
                orientation_bootstrap_seed=cfg.orientation_bootstrap_seed,
                vertical_qc_backend=cfg.vertical_qc_backend,
                apply_local_z_flip=cfg.apply_local_z_flip,
                ovrlpy_n_components=cfg.ovrlpy_n_components,
                ovrlpy_n_workers=cfg.ovrlpy_n_workers,
                ovrlpy_fit_umap=cfg.ovrlpy_fit_umap,
                ovrlpy_min_transcripts=cfg.ovrlpy_min_transcripts,
            )
        )

    summary_path = out_dir / "3d_stack_reconstruction_summary.json"
    local_z_outputs = (
        {
            "local_z_orientation_manifest_csv": str(local_z_result.manifest_csv),
            "aligned_transcripts_parquet": str(local_z_result.aligned_transcripts_parquet),
            "biological_z_report_html": str(local_z_result.biological_report_html),
            "vertical_qc_dir": str(local_z_result.vertical_qc_dir),
        }
        if local_z_result is not None
        else {}
    )
    summary = {
        "config": _jsonable_config(cfg),
        "slice_count": len(slices),
        "structure_count": _count_structures_from_aligned_rows(
            aligned_rows,
            group_property=cfg.group_property,
        ),
        "outputs": {
            "slice_manifest_csv": str(slice_manifest_path),
            "aligned_manifest_csv": str(aligned_manifest_path),
            "pairwise_metrics_csv": str(pairwise_metrics_path),
            "guarded_alignment_audit_csv": str(guarded_audit_csv),
            "guarded_alignment_audit_html": str(guarded_audit_html),
            "contour_points_csv": str(contour_points_path),
            "visualization_html": str(visualization_path),
            "mesh_dir": str(mesh_dir),
            "mesh_qc_summary_json": str(mesh_dir / "mesh_qc_summary.json"),
            **local_z_outputs,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return ThreeDStackReconstructionResult(
        out_dir=out_dir,
        slice_manifest_csv=slice_manifest_path,
        pairwise_metrics_csv=pairwise_metrics_path,
        aligned_manifest_csv=aligned_manifest_path,
        contour_points_csv=contour_points_path,
        summary_json=summary_path,
        visualization_html=visualization_path,
        mesh_dir=mesh_dir,
        guarded_alignment_audit_csv=guarded_audit_csv,
        guarded_alignment_audit_html=guarded_audit_html,
        local_z_orientation_manifest_csv=(
            local_z_result.manifest_csv if local_z_result is not None else None
        ),
        aligned_transcripts_parquet=(
            local_z_result.aligned_transcripts_parquet if local_z_result is not None else None
        ),
        biological_z_report_html=(
            local_z_result.biological_report_html if local_z_result is not None else None
        ),
    )


def discover_xenium_slices(
    xenium_root: PathLike,
    *,
    slice_dirs: Sequence[PathLike] | None = None,
    sample_glob: str = "*",
) -> list[_SliceInput]:
    """Discover Xenium outs directories and return them in numeric slice order."""

    root = Path(xenium_root).expanduser().resolve()
    if slice_dirs is not None:
        candidates = [Path(path).expanduser().resolve() for path in slice_dirs]
    elif _looks_like_xenium_dir(root):
        candidates = [root]
    else:
        candidates = [
            child
            for child in root.glob(sample_glob)
            if child.is_dir() and _find_xenium_output_dir(child) is not None
        ]

    slices: list[_SliceInput] = []
    for candidate in candidates:
        xenium_dir = _find_xenium_output_dir(candidate)
        if xenium_dir is None:
            continue
        slices.append(
            _SliceInput(
                order=0,
                sample_id=_sample_id_from_slice_path(candidate),
                sample_dir=candidate,
                xenium_dir=xenium_dir,
            )
        )

    ordered = sorted(slices, key=lambda item: _sample_sort_key(item.sample_id))
    return [
        _SliceInput(
            order=index,
            sample_id=item.sample_id,
            sample_dir=item.sample_dir,
            xenium_dir=item.xenium_dir,
        )
        for index, item in enumerate(ordered, start=1)
    ]


def _read_precomputed_contour_manifest(manifest: PathLike) -> list[_SliceInput]:
    """Read precomputed per-slice contour GeoJSONs for manifest-driven stack runs."""

    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    table = pd.read_csv(manifest_path)
    required = {"order", "sample_id", "z_um", "geojson"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            "contour_manifest is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if table.empty:
        raise ValueError("contour_manifest must contain at least one slice row.")

    rows: list[_SliceInput] = []
    seen_orders: set[int] = set()
    seen_sample_ids: set[str] = set()
    for _, row in table.sort_values("order").iterrows():
        try:
            order = int(row["order"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid contour_manifest order: {row['order']!r}.") from exc
        sample_id = str(row["sample_id"]).strip()
        if not sample_id:
            raise ValueError("contour_manifest sample_id values must be non-empty.")
        if order in seen_orders:
            raise ValueError(f"Duplicate contour_manifest order: {order}.")
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate contour_manifest sample_id: {sample_id!r}.")
        seen_orders.add(order)
        seen_sample_ids.add(sample_id)

        try:
            z_um = float(row["z_um"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid contour_manifest z_um for sample {sample_id!r}: {row['z_um']!r}."
            ) from exc
        if not math.isfinite(z_um):
            raise ValueError(f"z_um must be finite for sample {sample_id!r}.")

        geojson_path = _resolve_manifest_path(row["geojson"], manifest_path.parent)
        if not geojson_path.exists():
            raise FileNotFoundError(geojson_path)

        xenium_value = _first_present_manifest_value(
            row,
            ("xenium_dir", "source_xenium_dir", "source_dir"),
        )
        sample_value = _first_present_manifest_value(row, ("sample_dir",))
        xenium_dir = (
            _resolve_manifest_path(xenium_value, manifest_path.parent)
            if xenium_value is not None
            else geojson_path.parent
        )
        sample_dir = (
            _resolve_manifest_path(sample_value, manifest_path.parent)
            if sample_value is not None
            else xenium_dir.parent
        )
        rows.append(
            _SliceInput(
                order=order,
                sample_id=sample_id,
                sample_dir=sample_dir,
                xenium_dir=xenium_dir,
                z_um=z_um,
                precomputed_geojson=geojson_path,
            )
        )
    expected_orders = list(range(1, len(rows) + 1))
    observed_orders = [item.order for item in rows]
    if observed_orders != expected_orders:
        raise ValueError(
            "contour_manifest order values must be consecutive and start at 1; "
            f"observed {observed_orders!r}."
        )
    return rows


def _first_present_manifest_value(row: pd.Series, columns: Sequence[str]) -> str | None:
    for column in columns:
        if column not in row.index:
            continue
        value = row[column]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _resolve_manifest_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _slice_z_um(slice_input: _SliceInput, cfg: ThreeDStackReconstructionConfig) -> float:
    if slice_input.z_um is not None:
        return float(slice_input.z_um)
    return float(slice_input.order - 1) * float(cfg.z_spacing_um)


def _run_hard_alignment_backend(
    cfg: ThreeDStackReconstructionConfig,
    *,
    fixed_geojson: PathLike,
    moving_geojson: PathLike,
    output_geojson: PathLike,
    summary_json: PathLike,
) -> dict[str, Any]:
    if cfg.registration_backend == "auto":
        return _run_auto_hard_alignment(
            cfg,
            fixed_geojson=fixed_geojson,
            moving_geojson=moving_geojson,
            output_geojson=output_geojson,
            summary_json=summary_json,
        )
    if cfg.registration_backend == "contour-tps":
        return hard_align_geojson(
            fixed_geojson=fixed_geojson,
            moving_geojson=moving_geojson,
            output_geojson=output_geojson,
            summary_json=summary_json,
            group_property=cfg.group_property,
            maxiter=cfg.hard_alignment_maxiter,
            overwrite=cfg.overwrite,
            multistart=cfg.hard_alignment_multistart,
            affine_fallback_iou_threshold=cfg.affine_fallback_iou_threshold,
            registration_backend="contour-tps",
        )
    if cfg.registration_backend == "label-free-group":
        return _run_label_free_group_hard_alignment(
            cfg,
            fixed_geojson=fixed_geojson,
            moving_geojson=moving_geojson,
            output_geojson=output_geojson,
            summary_json=summary_json,
            registration_backend="label-free-group",
            selected_hard_seed_backend="label-free-group",
        )
    if cfg.registration_backend == "coda-image":
        return _run_tournament_hard_alignment(
            fixed_geojson=fixed_geojson,
            moving_geojson=moving_geojson,
            output_geojson=output_geojson,
            summary_json=summary_json,
            group_property=cfg.group_property,
            maxiter=cfg.hard_alignment_maxiter,
            overwrite=cfg.overwrite,
            multistart=cfg.hard_alignment_multistart,
            affine_fallback_iou_threshold=cfg.affine_fallback_iou_threshold,
            raster_size=cfg.coda_raster_size,
            angle_step=cfg.coda_angle_step,
            phase_upsample_factor=cfg.coda_phase_upsample_factor,
            mask_padding_fraction=cfg.coda_mask_padding_fraction,
        )
    raise ValueError(f"Unsupported registration_backend: {cfg.registration_backend!r}.")


def _run_auto_hard_alignment(
    cfg: ThreeDStackReconstructionConfig,
    *,
    fixed_geojson: PathLike,
    moving_geojson: PathLike,
    output_geojson: PathLike,
    summary_json: PathLike | None = None,
) -> dict[str, Any]:
    """Run contour and label-free group hard seeds, then select the reliable seed."""

    output_path = Path(output_geojson)
    summary_path = Path(summary_json) if summary_json is not None else None
    if (
        output_path.exists()
        and not cfg.overwrite
        and summary_path is not None
        and summary_path.exists()
    ):
        return json.loads(summary_path.read_text(encoding="utf-8"))

    candidate_dir = output_path.parent / "hard_alignment_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    contour_summary_path = candidate_dir / "contour_tps_summary.json"
    contour_output_path = candidate_dir / "contour_tps_moving_hard_aligned.geojson"
    label_free_summary_path = candidate_dir / "label_free_group_summary.json"
    label_free_output_path = candidate_dir / "label_free_group_moving_hard_aligned.geojson"
    translation_summary_path = candidate_dir / "translation_only_summary.json"
    translation_output_path = candidate_dir / "translation_only_moving_hard_aligned.geojson"

    contour_summary = hard_align_geojson(
        fixed_geojson=fixed_geojson,
        moving_geojson=moving_geojson,
        output_geojson=contour_output_path,
        summary_json=contour_summary_path,
        group_property=cfg.group_property,
        maxiter=cfg.hard_alignment_maxiter,
        overwrite=True,
        multistart=cfg.hard_alignment_multistart,
        affine_fallback_iou_threshold=cfg.affine_fallback_iou_threshold,
        registration_backend="contour-tps",
    )
    label_free_summary = _run_label_free_group_hard_alignment(
        cfg,
        fixed_geojson=fixed_geojson,
        moving_geojson=moving_geojson,
        output_geojson=label_free_output_path,
        summary_json=label_free_summary_path,
        artifact_dir=candidate_dir / "label_free_group_artifacts",
        registration_backend="label-free-group",
        selected_hard_seed_backend="label-free-group",
    )

    label_free_ok, label_free_guard = _label_free_group_candidate_guard(
        label_free_summary,
        cfg,
    )
    label_free_summary = {
        **label_free_summary,
        "label_free_guarded_candidate_accepted": bool(label_free_ok),
        "label_free_guarded_rejection_reason": label_free_guard["reason"],
        "label_free_guarded_rotation_abs_degrees": label_free_guard[
            "rotation_abs_degrees"
        ],
        "label_free_guarded_details": label_free_guard,
    }
    if label_free_ok:
        selected_backend = "label-free-group"
        selected_summary = label_free_summary
        selected_output_path = label_free_output_path
    elif _contour_tps_candidate_ok(contour_summary, cfg):
        selected_backend = "contour-tps"
        selected_summary = contour_summary
        selected_output_path = contour_output_path
    else:
        translation_summary = _run_translation_only_hard_alignment(
            fixed_geojson=fixed_geojson,
            moving_geojson=moving_geojson,
            output_geojson=translation_output_path,
            summary_json=translation_summary_path,
            group_property=cfg.group_property,
            registration_backend="translation-only",
        )
        selected_backend = "translation-only"
        selected_summary = translation_summary
        selected_output_path = translation_output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_output_path, output_path)

    summary = copy.deepcopy(selected_summary)
    if selected_backend != "label-free-group":
        summary.update(
            {
                "label_free_fixed_group": None,
                "label_free_moving_group": None,
                "label_free_used_anchor_pair_count": None,
                "label_free_residual_median": None,
                "label_free_guarded_candidate_accepted": bool(label_free_ok),
                "label_free_guarded_rejection_reason": label_free_guard["reason"],
            }
        )
    contour_rotation_abs = _absolute_rotation_degrees(
        (contour_summary.get("transform") or {}).get("rotation_degrees")
    )
    summary.update(
        {
            "fixed_geojson": str(Path(fixed_geojson)),
            "moving_geojson": str(Path(moving_geojson)),
            "output_geojson": str(output_path),
            "registration_backend": "auto",
            "selected_hard_seed_backend": selected_backend,
            "label_free_guarded_candidate_accepted": bool(label_free_ok),
            "label_free_guarded_rejection_reason": label_free_guard["reason"],
            "label_free_guarded_rotation_abs_degrees": label_free_guard[
                "rotation_abs_degrees"
            ],
            "method_note": (
                "Auto hard alignment compares the same-label contour seed with "
                "label-free group correspondence. The label-free seed is promoted only "
                "when its group-RANSAC anchors, residual, rotation, and spatial-coverage "
                "guards pass configured thresholds."
            ),
            "hard_alignment_tournament": {
                "strategy": "prefer_valid_label_free_group_else_contour_tps",
                "selected_backend": selected_backend,
                "label_free_candidate_accepted": bool(label_free_ok),
                "label_free_guard": label_free_guard,
                "contour_tps_candidate_accepted": _contour_tps_candidate_ok(
                    contour_summary,
                    cfg,
                ),
                "contour_tps_rotation_abs_degrees": contour_rotation_abs,
                "contour_tps_rejection_reason": (
                    None
                    if _contour_tps_candidate_ok(contour_summary, cfg)
                    else "contour_tps_rotation_exceeds_serial_slice_prior"
                ),
                "label_free_min_anchor_count": int(cfg.label_free_min_anchor_count),
                "label_free_group_residual_limit_um": float(
                    cfg.label_free_group_residual_limit_um
                ),
                "label_free_guarded_alignment": bool(cfg.label_free_guarded_alignment),
            },
            "hard_alignment_candidates": [
                _hard_alignment_candidate_payload(
                    contour_summary,
                    summary_json=contour_summary_path,
                ),
                _hard_alignment_candidate_payload(
                    label_free_summary,
                    summary_json=label_free_summary_path,
                ),
                *(
                    [
                        _hard_alignment_candidate_payload(
                            selected_summary,
                            summary_json=translation_summary_path,
                        )
                    ]
                    if selected_backend == "translation-only"
                    else []
                ),
            ],
        }
    )
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary


def _run_label_free_group_hard_alignment(
    cfg: ThreeDStackReconstructionConfig,
    *,
    fixed_geojson: PathLike,
    moving_geojson: PathLike,
    output_geojson: PathLike,
    summary_json: PathLike | None = None,
    artifact_dir: PathLike | None = None,
    registration_backend: str = "label-free-group",
    selected_hard_seed_backend: str = "label-free-group",
) -> dict[str, Any]:
    """Run label-free group correspondence and adapt it to the hard-summary schema."""

    from .label_free_alignment import (  # Local import avoids a multislice import cycle.
        LabelFreeContourAlignmentConfig,
        align_contours_label_free,
    )

    output_path = Path(output_geojson)
    summary_path = Path(summary_json) if summary_json is not None else None
    if (
        output_path.exists()
        and not cfg.overwrite
        and summary_path is not None
        and summary_path.exists()
    ):
        return json.loads(summary_path.read_text(encoding="utf-8"))

    label_free_dir = (
        Path(artifact_dir)
        if artifact_dir is not None
        else output_path.parent / "hard_alignment_candidates" / "label_free_group_artifacts"
    )
    result = align_contours_label_free(
        LabelFreeContourAlignmentConfig(
            fixed_geojson=fixed_geojson,
            moving_geojson=moving_geojson,
            out_dir=label_free_dir,
            maxiter=cfg.hard_alignment_maxiter,
            multistart=cfg.hard_alignment_multistart,
            affine_fallback_iou_threshold=cfg.affine_fallback_iou_threshold,
            run_soft_tps=False,
            partial_correspondence=True,
            diagnostic_only=False,
            search_window=cfg.label_free_search_window,
            knn_neighbors=cfg.label_free_knn_neighbors,
            min_anchor_count=cfg.label_free_min_anchor_count,
            group_correspondence=True,
            group_candidate_count=cfg.label_free_group_candidate_count,
            group_ransac_trials=cfg.label_free_group_ransac_trials,
            group_min_descriptor_score=cfg.label_free_group_min_descriptor_score,
            group_residual_limit_um=cfg.label_free_group_residual_limit_um,
            group_min_component_area_um2=cfg.label_free_group_min_component_area_um2,
            save_preview_png=cfg.save_alignment_preview_png,
            overwrite=True,
            dpi=cfg.dpi,
        )
    )
    if result.aligned_geojson is None:
        raise RuntimeError("label-free group alignment did not produce an aligned GeoJSON.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result.aligned_geojson, output_path)

    label_free_summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    anchor_transform = label_free_summary.get("anchor_transform") or {}
    partial_summary = label_free_summary.get("partial_correspondence_summary") or {}
    entropy_summary = partial_summary.get("anchor_spatial_entropy") or {}
    global_scores = label_free_summary.get("global_context_scores_not_used_for_fitting") or {}
    transform = _label_free_transform_for_hard_schema(label_free_summary.get("transform") or {})
    summary = {
        "fixed_geojson": str(Path(fixed_geojson)),
        "moving_geojson": str(Path(moving_geojson)),
        "output_geojson": str(output_path),
        "registration_backend": registration_backend,
        "selected_hard_seed_backend": selected_hard_seed_backend,
        "transform": transform,
        "affine_params": None,
        "optimization": {
            "method": "label_free_group_correspondence_ransac",
            "success": bool(anchor_transform.get("accepted", False)),
            "accepted": bool(anchor_transform.get("accepted", False)),
            "accepted_reason": anchor_transform.get("reason"),
            "union_iou_before": global_scores.get("union_iou_before"),
            "union_iou_final": global_scores.get("union_iou_after"),
        },
        "union_iou_before_hard": global_scores.get("union_iou_before"),
        "union_iou_after_hard": global_scores.get("union_iou_after"),
        "hard_alignment_accepted": bool(anchor_transform.get("accepted", False)),
        "per_structure_iou_after_hard": {},
        "label_free_summary_json": str(result.summary_json),
        "label_free_anchor_landmarks_csv": (
            str(result.landmarks_csv) if result.landmarks_csv else None
        ),
        "label_free_partial_matches_csv": (
            str(result.partial_matches_csv) if result.partial_matches_csv else None
        ),
        "label_free_partial_nodes_csv": (
            str(result.partial_nodes_csv) if result.partial_nodes_csv else None
        ),
        "label_free_overlay_html": str(result.overlay_html) if result.overlay_html else None,
        "label_free_group_matrix_csv": (
            str(result.group_matrix_csv) if result.group_matrix_csv else None
        ),
        "label_free_group_matrix_html": (
            str(result.group_matrix_html) if result.group_matrix_html else None
        ),
        "label_free_fixed_group": anchor_transform.get("fixed_group"),
        "label_free_moving_group": anchor_transform.get("moving_group"),
        "label_free_used_anchor_pair_count": anchor_transform.get("used_anchor_pair_count"),
        "label_free_residual_median": anchor_transform.get("residual_median"),
        "label_free_residual_p90": anchor_transform.get("residual_p90"),
        "label_free_partial_anchor_count": partial_summary.get("anchor_count"),
        "label_free_anchor_coverage_ratio": partial_summary.get(
            "anchor_convex_hull_coverage_ratio"
        ),
        "label_free_anchor_occupied_quadrants": entropy_summary.get(
            "occupied_quadrants"
        ),
        "label_free_partial_warnings": partial_summary.get("warnings"),
        "label_free_anchor_transform": anchor_transform,
        "method_note": (
            "Label-free group correspondence hard seed. Contour labels are preserved; "
            "fixed and moving groups may have different names and are used only to find "
            "a geometric overlap constellation."
        ),
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary


def _label_free_transform_for_hard_schema(transform: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": transform.get("kind", "similarity"),
        "origin_x": transform.get("origin_x", 0.0),
        "origin_y": transform.get("origin_y", 0.0),
        "rotation_degrees": transform.get("rotation_degrees", 0.0),
        "scale": transform.get("scale", 1.0),
        "translate_x": transform.get("translate_x", 0.0),
        "translate_y": transform.get("translate_y", 0.0),
    }


def _run_translation_only_hard_alignment(
    *,
    fixed_geojson: PathLike,
    moving_geojson: PathLike,
    output_geojson: PathLike,
    summary_json: PathLike | None = None,
    group_property: str = "structure",
    registration_backend: str = "translation-only",
) -> dict[str, Any]:
    """Translate moving contours by union-centroid offset without rotation or scaling."""

    output_path = Path(output_geojson)
    summary_path = Path(summary_json) if summary_json is not None else None
    fixed_payload = _read_geojson(Path(fixed_geojson))
    moving_payload = _read_geojson(Path(moving_geojson))
    fixed_records = _records_from_geojson(fixed_payload, group_property, "fixed")
    moving_records = _records_from_geojson(moving_payload, group_property, "moving")
    fixed_union = unary_union([record.geometry for record in fixed_records])
    moving_union = unary_union([record.geometry for record in moving_records])
    transform = _SimilarityTransform(
        origin_x=float(moving_union.centroid.x),
        origin_y=float(moving_union.centroid.y),
        rotation_degrees=0.0,
        scale=1.0,
        translate_x=float(fixed_union.centroid.x - moving_union.centroid.x),
        translate_y=float(fixed_union.centroid.y - moving_union.centroid.y),
    )
    before_iou = _iou(fixed_union, moving_union)
    aligned_union = _apply_similarity_to_geometry(moving_union, transform)
    after_iou = _iou(fixed_union, aligned_union)
    aligned_payload = _apply_similarity_to_geojson(moving_payload, transform)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aligned_payload, ensure_ascii=False), encoding="utf-8")
    per_structure = _per_structure_iou(
        fixed_records,
        _records_from_geojson(aligned_payload, group_property, "aligned"),
    )
    summary = {
        "fixed_geojson": str(Path(fixed_geojson)),
        "moving_geojson": str(Path(moving_geojson)),
        "output_geojson": str(output_path),
        "registration_backend": registration_backend,
        "selected_hard_seed_backend": registration_backend,
        "transform": asdict(transform),
        "affine_params": None,
        "optimization": {
            "method": "union_centroid_translation_only",
            "success": True,
            "accepted": True,
            "accepted_reason": "safe_fallback_after_rotation_guard_rejected_candidates",
            "union_iou_before": before_iou,
            "union_iou_final": after_iou,
        },
        "union_iou_before_hard": before_iou,
        "union_iou_after_hard": after_iou,
        "hard_alignment_accepted": True,
        "per_structure_iou_after_hard": per_structure,
        "method_note": (
            "Translation-only hard seed. Used when label-free group evidence is "
            "insufficient and contour-TPS violates the serial-slice rotation prior."
        ),
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary


def _run_tournament_hard_alignment(
    *,
    fixed_geojson: PathLike,
    moving_geojson: PathLike,
    output_geojson: PathLike,
    summary_json: PathLike | None = None,
    group_property: str = "structure",
    maxiter: int = 80,
    overwrite: bool = False,
    multistart: bool = True,
    affine_fallback_iou_threshold: float = 0.0,
    raster_size: int = 512,
    angle_step: float = 1.0,
    phase_upsample_factor: int = 1,
    mask_padding_fraction: float = 0.05,
) -> dict[str, Any]:
    """Run contour and CODA-inspired hard seeds, then promote the higher-IoU winner."""

    output_path = Path(output_geojson)
    summary_path = Path(summary_json) if summary_json is not None else None
    if output_path.exists() and not overwrite and summary_path is not None and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    candidate_dir = output_path.parent / "hard_alignment_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    contour_summary_path = candidate_dir / "contour_tps_summary.json"
    contour_output_path = candidate_dir / "contour_tps_moving_hard_aligned.geojson"
    coda_summary_path = candidate_dir / "coda_image_summary.json"
    coda_output_path = candidate_dir / "coda_image_moving_hard_aligned.geojson"

    contour_summary = hard_align_geojson(
        fixed_geojson=fixed_geojson,
        moving_geojson=moving_geojson,
        output_geojson=contour_output_path,
        summary_json=contour_summary_path,
        group_property=group_property,
        maxiter=maxiter,
        overwrite=True,
        multistart=multistart,
        affine_fallback_iou_threshold=affine_fallback_iou_threshold,
        registration_backend="contour-tps",
    )
    coda_summary = _coda_image_align_geojson(
        fixed_geojson=fixed_geojson,
        moving_geojson=moving_geojson,
        output_geojson=coda_output_path,
        summary_json=coda_summary_path,
        group_property=group_property,
        overwrite=True,
        raster_size=raster_size,
        angle_step=angle_step,
        phase_upsample_factor=phase_upsample_factor,
        mask_padding_fraction=mask_padding_fraction,
    )

    contour_iou = _hard_alignment_iou(contour_summary)
    coda_iou = _hard_alignment_iou(coda_summary)
    if contour_iou >= coda_iou:
        selected_backend = "contour-tps"
        selected_summary = contour_summary
        selected_output_path = contour_output_path
    else:
        selected_backend = "coda-image"
        selected_summary = coda_summary
        selected_output_path = coda_output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_output_path, output_path)

    summary = copy.deepcopy(selected_summary)
    summary.update(
        {
            "fixed_geojson": str(Path(fixed_geojson)),
            "moving_geojson": str(Path(moving_geojson)),
            "output_geojson": str(output_path),
            "registration_backend": "coda-image",
            "selected_hard_seed_backend": selected_backend,
            "method_credit": CODA_METHOD_CREDIT,
            "method_reference_doi": CODA_METHOD_REFERENCE_DOI,
            "method_reference_url": CODA_METHOD_REFERENCE_URL,
            "methodology_url": CODA_METHODOLOGY_URL,
            "method_note": (
                "CODA-inspired tournament hard alignment: contour similarity and "
                "CODA-inspired Radon/phase seeds are both evaluated, then the "
                "higher-IoU hard seed is promoted before topology-safe contour TPS. "
                "This is not a full CODA reimplementation."
            ),
            "hard_alignment_tournament": {
                "strategy": "max_union_iou_after_hard",
                "selected_backend": selected_backend,
                "rotation_difference_degrees": _rotation_difference_degrees(
                    contour_summary.get("transform", {}).get("rotation_degrees", 0.0),
                    coda_summary.get("transform", {}).get("rotation_degrees", 0.0),
                ),
            },
            "hard_alignment_candidates": [
                _hard_alignment_candidate_payload(
                    contour_summary,
                    summary_json=contour_summary_path,
                ),
                _hard_alignment_candidate_payload(
                    coda_summary,
                    summary_json=coda_summary_path,
                ),
            ],
            "coda_image": coda_summary.get("coda_image"),
        }
    )
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary


def _coda_image_align_geojson(
    *,
    fixed_geojson: PathLike,
    moving_geojson: PathLike,
    output_geojson: PathLike,
    summary_json: PathLike | None = None,
    group_property: str = "structure",
    overwrite: bool = False,
    raster_size: int = 512,
    angle_step: float = 1.0,
    phase_upsample_factor: int = 1,
    mask_padding_fraction: float = 0.05,
) -> dict[str, Any]:
    """Hard-align contours with a CODA-inspired image-derived similarity seed."""

    output_path = Path(output_geojson)
    summary_path = Path(summary_json) if summary_json is not None else None
    if output_path.exists() and not overwrite and summary_path is not None and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    fixed_payload = _read_geojson(Path(fixed_geojson))
    moving_payload = _read_geojson(Path(moving_geojson))
    fixed_records = _records_from_geojson(fixed_payload, group_property, "fixed")
    moving_records = _records_from_geojson(moving_payload, group_property, "moving")
    fixed_union = unary_union([record.geometry for record in fixed_records])
    moving_union = unary_union([record.geometry for record in moving_records])

    fixed_mask, moving_mask, raster_metadata = _rasterize_pair_for_coda(
        fixed_union,
        moving_union,
        raster_size=int(raster_size),
        padding_fraction=float(mask_padding_fraction),
    )
    image_result = estimate_coda_image_registration(
        fixed_mask,
        moving_mask,
        CodaImageRegistrationConfig(
            angle_step=float(angle_step),
            phase_upsample_factor=int(phase_upsample_factor),
        ),
    )
    native_units_per_pixel = float(raster_metadata["native_units_per_pixel"])
    bounds = raster_metadata["square_bounds"]
    origin_x = (float(bounds[0]) + float(bounds[2])) / 2.0
    origin_y = (float(bounds[1]) + float(bounds[3])) / 2.0
    transform = _SimilarityTransform(
        origin_x=origin_x,
        origin_y=origin_y,
        rotation_degrees=float(image_result.rotation.rotation_degrees),
        scale=1.0,
        translate_x=float(image_result.translation.shift_x) * native_units_per_pixel,
        translate_y=-float(image_result.translation.shift_y) * native_units_per_pixel,
    )

    before_iou = _iou(fixed_union, moving_union)
    aligned_union = _apply_similarity_to_geometry(moving_union, transform)
    after_iou = _iou(fixed_union, aligned_union)
    accepted = after_iou >= before_iou
    optimization = {
        "method": "coda_image_radon_phase_correlation",
        "success": True,
        "accepted": bool(accepted),
        "accepted_reason": (
            "image_similarity_seed_improved_or_preserved_iou"
            if accepted
            else "identity_kept_because_image_seed_reduced_iou"
        ),
        "union_iou_before": before_iou,
        "union_iou_final": after_iou if accepted else before_iou,
    }
    if not accepted:
        transform = _identity_transform_for_geometry(moving_union)
        aligned_union = moving_union
        after_iou = before_iou

    aligned_payload = _apply_similarity_to_geojson(moving_payload, transform)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aligned_payload, ensure_ascii=False), encoding="utf-8")

    per_structure = _per_structure_iou(
        fixed_records,
        _records_from_geojson(aligned_payload, group_property, "aligned"),
    )
    summary = {
        "fixed_geojson": str(Path(fixed_geojson)),
        "moving_geojson": str(Path(moving_geojson)),
        "output_geojson": str(output_path),
        "registration_backend": "coda-image",
        "method_credit": CODA_METHOD_CREDIT,
        "method_reference_doi": CODA_METHOD_REFERENCE_DOI,
        "method_reference_url": CODA_METHOD_REFERENCE_URL,
        "methodology_url": CODA_METHODOLOGY_URL,
        "method_note": (
            "CODA-inspired tissue-mask Radon rotation plus phase-correlation "
            "translation. This is not a full CODA reimplementation."
        ),
        "transform": asdict(transform),
        "affine_params": None,
        "optimization": optimization,
        "coda_image": _coda_image_summary_payload(
            image_result=image_result,
            raster_metadata=raster_metadata,
            angle_step=angle_step,
            phase_upsample_factor=phase_upsample_factor,
            mask_padding_fraction=mask_padding_fraction,
        ),
        "union_iou_before_hard": before_iou,
        "union_iou_after_hard": after_iou,
        "hard_alignment_accepted": accepted,
        "per_structure_iou_after_hard": per_structure,
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary


def hard_align_geojson(
    *,
    fixed_geojson: PathLike,
    moving_geojson: PathLike,
    output_geojson: PathLike,
    summary_json: PathLike | None = None,
    group_property: str = "structure",
    maxiter: int = 80,
    overwrite: bool = False,
    multistart: bool = True,
    affine_fallback_iou_threshold: float = 0.0,
    registration_backend: str = "contour-tps",
) -> dict[str, Any]:
    """Hard-align one contour GeoJSON to another with a similarity transform."""

    output_path = Path(output_geojson)
    summary_path = Path(summary_json) if summary_json is not None else None
    if output_path.exists() and not overwrite and summary_path is not None and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    fixed_payload = _read_geojson(Path(fixed_geojson))
    moving_payload = _read_geojson(Path(moving_geojson))
    fixed_records = _records_from_geojson(fixed_payload, group_property, "fixed")
    moving_records = _records_from_geojson(moving_payload, group_property, "moving")
    fixed_union = unary_union([record.geometry for record in fixed_records])
    moving_union = unary_union([record.geometry for record in moving_records])

    transform, optimization = _estimate_similarity_transform(
        fixed_union,
        moving_union,
        maxiter=maxiter,
        multistart=multistart,
        fixed_records=fixed_records,
        moving_records=moving_records,
    )
    aligned_union = _apply_similarity_to_geometry(moving_union, transform)
    before_iou = _iou(fixed_union, moving_union)
    after_iou = _iou(fixed_union, aligned_union)

    # Affine fallback: if similarity alignment is insufficient, try a 6-DOF affine transform.
    affine_params: np.ndarray | None = None
    if affine_fallback_iou_threshold > 0 and after_iou < affine_fallback_iou_threshold:
        affine_params, after_iou_affine = _try_affine_alignment(
            fixed_union, moving_union, after_iou, maxiter=maxiter
        )
        if affine_params is not None:
            after_iou = after_iou_affine
            aligned_union = affinity.affine_transform(moving_union, affine_params.tolist())

    accepted = (affine_params is not None and after_iou >= before_iou) or (
        affine_params is None and after_iou >= before_iou
    )
    if not accepted:
        transform = _SimilarityTransform(
            origin_x=float(moving_union.centroid.x),
            origin_y=float(moving_union.centroid.y),
            rotation_degrees=0.0,
            scale=1.0,
            translate_x=0.0,
            translate_y=0.0,
        )
        affine_params = None
        optimization = {
            **optimization,
            "accepted": False,
            "accepted_reason": "identity_kept_because_similarity_alignment_reduced_iou",
        }
        aligned_union = moving_union
        after_iou = before_iou
    else:
        optimization = {**optimization, "accepted": True}

    if affine_params is not None:
        aligned_payload = _apply_affine_to_geojson(moving_payload, affine_params)
    else:
        aligned_payload = _apply_similarity_to_geojson(moving_payload, transform)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aligned_payload, ensure_ascii=False), encoding="utf-8")

    per_structure = _per_structure_iou(
        fixed_records,
        _records_from_geojson(aligned_payload, group_property, "aligned"),
    )
    summary = {
        "fixed_geojson": str(Path(fixed_geojson)),
        "moving_geojson": str(Path(moving_geojson)),
        "output_geojson": str(output_path),
        "registration_backend": registration_backend,
        "transform": asdict(transform),
        "affine_params": affine_params.tolist() if affine_params is not None else None,
        "optimization": optimization,
        "union_iou_before_hard": before_iou,
        "union_iou_after_hard": after_iou,
        "hard_alignment_accepted": accepted,
        "per_structure_iou_after_hard": per_structure,
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary


def write_3d_contour_points(
    aligned_rows: Sequence[Mapping[str, Any]],
    output_csv: PathLike,
    *,
    group_property: str,
    point_sample_distance_um: float,
    xenium_pixel_size_um: float,
) -> pd.DataFrame:
    """Sample aligned slice boundaries into a 3D contour point table."""

    rows: list[dict[str, Any]] = []
    native_distance = max(point_sample_distance_um / xenium_pixel_size_um, 1e-6)
    for row in aligned_rows:
        payload = _read_geojson(Path(row["aligned_geojson"]))
        z_um = float(row["z_um"])
        for feature_index, feature in enumerate(payload["features"]):
            group = _feature_group(feature.get("properties") or {}, group_property)
            geom = _valid_polygonal_geometry(shape(feature["geometry"]))
            if geom is None:
                continue
            for poly_index, polygon in enumerate(_iter_polygons(geom)):
                boundary = polygon.exterior
                if boundary.length <= 0:
                    continue
                distances = np.arange(0.0, boundary.length, native_distance)
                if len(distances) == 0:
                    distances = np.array([0.0])
                polyline_id = (
                    f"{int(row['order']):03d}_{feature_index:05d}_{poly_index:03d}"
                )
                for point_index, distance in enumerate(distances):
                    point = boundary.interpolate(float(distance))
                    rows.append(
                        {
                            "slice_order": int(row["order"]),
                            "sample_id": row["sample_id"],
                            "z_um": z_um,
                            "structure": str(group),
                            "polyline_id": polyline_id,
                            "point_index": point_index,
                            "x_um": float(point.x) * xenium_pixel_size_um,
                            "y_um": float(point.y) * xenium_pixel_size_um,
                        }
                    )

    df = pd.DataFrame(
        rows,
        columns=[
            "slice_order",
            "sample_id",
            "z_um",
            "structure",
            "polyline_id",
            "point_index",
            "x_um",
            "y_um",
        ],
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def reconstruct_3d_contour_meshes(
    aligned_rows: Sequence[Mapping[str, Any]],
    mesh_dir: PathLike,
    *,
    group_property: str,
    voxel_size_um: float,
    z_spacing_um: float,
    xenium_pixel_size_um: float,
    mesh_method: str = "marching_cubes",
    mesh_smoothing_sigma_um: float | None = 40.0,
    mesh_level: float = 0.5,
    mesh_export_formats: Sequence[str] = ("ply", "obj"),
    mesh_cleanup: bool = True,
    min_mesh_component_volume_um3: float | None = None,
    max_faces_for_html: int = 25000,
) -> list[dict[str, Any]]:
    """Voxelize aligned structure masks and export continuous surface meshes."""

    if mesh_method != "marching_cubes":
        raise ValueError("mesh_method currently supports only 'marching_cubes'.")
    if mesh_smoothing_sigma_um is not None and mesh_smoothing_sigma_um < 0:
        raise ValueError("mesh_smoothing_sigma_um must be non-negative or None.")
    if not (0.0 < float(mesh_level) < 1.0):
        raise ValueError("mesh_level must be strictly between 0 and 1.")
    if (
        min_mesh_component_volume_um3 is not None
        and min_mesh_component_volume_um3 < 0
    ):
        raise ValueError("min_mesh_component_volume_um3 must be non-negative.")
    export_formats = _normalize_mesh_export_formats(mesh_export_formats)
    trimesh = _import_trimesh()
    mesh_path = Path(mesh_dir)
    mesh_path.mkdir(parents=True, exist_ok=True)
    slice_records = _load_aligned_slice_geometries(
        aligned_rows,
        group_property=group_property,
        xenium_pixel_size_um=xenium_pixel_size_um,
    )
    structures = sorted({record["structure"] for record in slice_records})
    if not structures:
        return []

    bounds = _combined_scaled_bounds(slice_records)
    x_min, y_min, x_max, y_max = bounds
    x_values = np.arange(x_min, x_max + voxel_size_um, voxel_size_um, dtype=float)
    y_values = np.arange(y_min, y_max + voxel_size_um, voxel_size_um, dtype=float)
    xx, yy = np.meshgrid(x_values, y_values)
    volume_metadata = {
        "x_min_um": float(x_min),
        "y_min_um": float(y_min),
        "x_max_um": float(x_max),
        "y_max_um": float(y_max),
        "voxel_size_um": float(voxel_size_um),
        "z_spacing_um": float(z_spacing_um),
        "mesh_method": mesh_method,
        "mesh_smoothing_sigma_um": (
            None if mesh_smoothing_sigma_um is None else float(mesh_smoothing_sigma_um)
        ),
        "mesh_level": float(mesh_level),
        "mesh_cleanup": bool(mesh_cleanup),
        "min_mesh_component_volume_um3": min_mesh_component_volume_um3,
    }

    mesh_payloads: list[dict[str, Any]] = []
    for structure in structures:
        volume = np.zeros((len(aligned_rows), len(y_values), len(x_values)), dtype=bool)
        for record in slice_records:
            if record["structure"] != structure:
                continue
            slice_index = int(record["slice_order"]) - 1
            mask = _geometry_contains_grid(record["geometry_um"], xx, yy)
            volume[slice_index] |= mask
        if not volume.any():
            continue

        component_stats_before = _volume_component_stats(volume, voxel_size_um, z_spacing_um)
        filtered_volume = _filter_volume_components(
            volume,
            min_mesh_component_volume_um3=min_mesh_component_volume_um3,
            voxel_size_um=voxel_size_um,
            z_spacing_um=z_spacing_um,
        )
        if not filtered_volume.any():
            continue

        component_stats_after = _volume_component_stats(
            filtered_volume,
            voxel_size_um,
            z_spacing_um,
        )
        padded = np.pad(filtered_volume.astype(np.float32), ((1, 1), (1, 1), (1, 1)))
        field = _smooth_volume_field(
            padded,
            mesh_smoothing_sigma_um=mesh_smoothing_sigma_um,
            voxel_size_um=voxel_size_um,
            z_spacing_um=z_spacing_um,
        )
        level = float(mesh_level)
        smoothing_applied = bool(
            mesh_smoothing_sigma_um is not None and mesh_smoothing_sigma_um > 0
        )
        if field.max() <= level and smoothing_applied:
            field = padded
            level = 0.5
            smoothing_applied = False

        try:
            vertices, faces, _, _ = measure.marching_cubes(
                field,
                level=level,
                spacing=(z_spacing_um, voxel_size_um, voxel_size_um),
            )
        except ValueError:
            continue

        xyz = np.column_stack(
            [
                x_min + (vertices[:, 2] - voxel_size_um),
                y_min + (vertices[:, 1] - voxel_size_um),
                vertices[:, 0] - z_spacing_um,
            ]
        )

        mesh = trimesh.Trimesh(vertices=xyz, faces=faces, process=False)
        if mesh_cleanup:
            mesh = _cleanup_trimesh_mesh(mesh)

        base_path = mesh_path / _safe_name(structure)
        export_paths = _export_mesh(mesh, base_path, export_formats)
        component_count = _mesh_component_count(mesh)
        payload = {
            "structure": structure,
            "ply": str(export_paths.get("ply")) if "ply" in export_paths else None,
            "obj": str(export_paths.get("obj")) if "obj" in export_paths else None,
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "surface_area_um2": float(mesh.area),
            "volume_um3": float(abs(mesh.volume)) if len(mesh.faces) else 0.0,
            "is_watertight": bool(mesh.is_watertight),
            "euler_number": int(mesh.euler_number),
            "component_count": int(component_count),
            "volume_component_count_before_filter": int(
                component_stats_before["component_count"]
            ),
            "volume_component_count_after_filter": int(component_stats_after["component_count"]),
            "volume_voxel_count_after_filter": int(component_stats_after["voxel_count"]),
            "smoothing_applied": smoothing_applied,
            "mesh_level_used": float(level),
        }
        if len(mesh.faces) <= max_faces_for_html:
            payload.update(
                {
                    "x": np.asarray(mesh.vertices)[:, 0].round(3).tolist(),
                    "y": np.asarray(mesh.vertices)[:, 1].round(3).tolist(),
                    "z": np.asarray(mesh.vertices)[:, 2].round(3).tolist(),
                    "i": np.asarray(mesh.faces)[:, 0].astype(int).tolist(),
                    "j": np.asarray(mesh.faces)[:, 1].astype(int).tolist(),
                    "k": np.asarray(mesh.faces)[:, 2].astype(int).tolist(),
                }
            )
        mesh_payloads.append(payload)

    manifest_columns = [
        "structure",
        "ply",
        "obj",
        "vertex_count",
        "face_count",
        "surface_area_um2",
        "volume_um3",
        "is_watertight",
        "euler_number",
        "component_count",
        "volume_component_count_before_filter",
        "volume_component_count_after_filter",
        "volume_voxel_count_after_filter",
        "smoothing_applied",
        "mesh_level_used",
    ]
    pd.DataFrame(
        [{column: item.get(column) for column in manifest_columns} for item in mesh_payloads],
        columns=manifest_columns,
    ).to_csv(mesh_path / "mesh_manifest.csv", index=False)
    _write_mesh_qc_summary(mesh_path / "mesh_qc_summary.json", mesh_payloads, volume_metadata)
    return mesh_payloads


def write_3d_visualization_html(
    contour_points: pd.DataFrame,
    mesh_payloads: Sequence[Mapping[str, Any]],
    output_html: PathLike,
    *,
    title: str,
) -> Path:
    """Write a self-contained Plotly HTML view for the 3D contour stack."""

    traces: list[dict[str, Any]] = []
    palette = [
        "#3ddbc7",
        "#5aa9ff",
        "#ffb156",
        "#b891ff",
        "#ff7188",
        "#8be563",
        "#ffd15c",
    ]
    has_mesh_surface = any({"x", "y", "z", "i", "j", "k"}.issubset(mesh.keys()) for mesh in mesh_payloads)
    for color_index, mesh in enumerate(mesh_payloads):
        if not {"x", "y", "z", "i", "j", "k"}.issubset(mesh.keys()):
            continue
        traces.append(
            {
                "type": "mesh3d",
                "name": f"{mesh['structure']} mesh",
                "x": mesh["x"],
                "y": mesh["y"],
                "z": mesh["z"],
                "i": mesh["i"],
                "j": mesh["j"],
                "k": mesh["k"],
                "color": palette[color_index % len(palette)],
                "opacity": 0.36,
                "flatshading": True,
            }
        )
    for color_index, (structure, group) in enumerate(contour_points.groupby("structure")):
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        z_values: list[float | None] = []
        for _, polyline in group.sort_values(
            ["slice_order", "polyline_id", "point_index"]
        ).groupby("polyline_id", sort=False):
            x_values.extend(polyline["x_um"].round(3).tolist())
            y_values.extend(polyline["y_um"].round(3).tolist())
            z_values.extend(polyline["z_um"].round(3).tolist())
            x_values.append(None)
            y_values.append(None)
            z_values.append(None)
        trace = {
            "type": "scatter3d",
            "mode": "lines",
            "name": f"{structure} contours",
            "x": x_values,
            "y": y_values,
            "z": z_values,
            "line": {"width": 3, "color": palette[color_index % len(palette)]},
            "opacity": 0.9,
        }
        if has_mesh_surface:
            trace["visible"] = "legendonly"
        traces.append(trace)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; background: #07111d; color: #d8e2ef; font-family: Arial, sans-serif; }}
    #plot {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const data = {json.dumps(traces)};
    const layout = {{
      title: {{text: {json.dumps(title)}, font: {{color: "#d8e2ef", size: 22}}}},
      paper_bgcolor: "#07111d",
      plot_bgcolor: "#07111d",
      scene: {{
        xaxis: {{title: "X (um)", color: "#b9c7d6", gridcolor: "#27364a"}},
        yaxis: {{title: "Y (um)", color: "#b9c7d6", gridcolor: "#27364a"}},
        zaxis: {{title: "Z (um)", color: "#b9c7d6", gridcolor: "#27364a"}},
        aspectmode: "data"
      }},
      legend: {{font: {{color: "#d8e2ef"}}}},
      margin: {{l: 0, r: 0, t: 50, b: 0}}
    }};
    Plotly.newPlot("plot", data, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _validate_stack_config(cfg: ThreeDStackReconstructionConfig) -> None:
    if cfg.contour_manifest is None and cfg.xenium_root is None:
        raise ValueError("Provide xenium_root or contour_manifest for stack reconstruction.")
    if cfg.contour_manifest is not None and cfg.local_z_orientation == "auto" and cfg.xenium_root is None:
        raise ValueError(
            "local_z_orientation='auto' with contour_manifest requires xenium_root so "
            "transcript tables can be discovered."
        )
    if cfg.registration_backend not in {"auto", "contour-tps", "label-free-group", "coda-image"}:
        raise ValueError(
            "registration_backend must be 'auto', 'contour-tps', 'label-free-group', or 'coda-image'."
        )
    if cfg.soft_alignment_mode not in {"auto", "semantic", "anchor-only", "none"}:
        raise ValueError(
            "soft_alignment_mode must be 'auto', 'semantic', 'anchor-only', or 'none'."
        )
    if cfg.anchor_only_bbox_padding_fraction < 0:
        raise ValueError("anchor_only_bbox_padding_fraction must be non-negative.")
    if cfg.anchor_only_identity_padding_count < 4:
        raise ValueError("anchor_only_identity_padding_count must be at least 4.")
    if cfg.anchor_only_rbf_smoothing < 0:
        raise ValueError("anchor_only_rbf_smoothing must be non-negative.")
    if cfg.anchor_only_jacobian_grid_size < 2:
        raise ValueError("anchor_only_jacobian_grid_size must be at least 2.")
    if not (0.0 <= cfg.anchor_only_max_negative_jacobian_ratio <= 1.0):
        raise ValueError("anchor_only_max_negative_jacobian_ratio must be in [0, 1].")
    if cfg.label_free_search_window <= 0:
        raise ValueError("label_free_search_window must be greater than 0.")
    if cfg.label_free_knn_neighbors < 1:
        raise ValueError("label_free_knn_neighbors must be at least 1.")
    if cfg.label_free_min_anchor_count < 1:
        raise ValueError("label_free_min_anchor_count must be at least 1.")
    if cfg.label_free_group_candidate_count < 1:
        raise ValueError("label_free_group_candidate_count must be at least 1.")
    if cfg.label_free_group_ransac_trials < 1:
        raise ValueError("label_free_group_ransac_trials must be at least 1.")
    if not (0 <= cfg.label_free_group_min_descriptor_score <= 1):
        raise ValueError("label_free_group_min_descriptor_score must be in [0, 1].")
    if cfg.label_free_group_residual_limit_um <= 0:
        raise ValueError("label_free_group_residual_limit_um must be greater than 0.")
    if cfg.label_free_group_min_component_area_um2 < 0:
        raise ValueError("label_free_group_min_component_area_um2 must be non-negative.")
    if cfg.label_free_min_similarity_anchor_count < 1:
        raise ValueError("label_free_min_similarity_anchor_count must be at least 1.")
    if cfg.label_free_max_rotation_degrees < 0:
        raise ValueError("label_free_max_rotation_degrees must be non-negative.")
    if cfg.label_free_high_rotation_min_anchor_count < 1:
        raise ValueError("label_free_high_rotation_min_anchor_count must be at least 1.")
    if cfg.label_free_high_rotation_residual_limit_um <= 0:
        raise ValueError("label_free_high_rotation_residual_limit_um must be greater than 0.")
    if not (0.0 <= cfg.label_free_min_anchor_coverage <= 1.0):
        raise ValueError("label_free_min_anchor_coverage must be in [0, 1].")
    if cfg.label_free_min_anchor_quadrants < 0 or cfg.label_free_min_anchor_quadrants > 4:
        raise ValueError("label_free_min_anchor_quadrants must be between 0 and 4.")
    if not (0.0 <= cfg.label_free_near_180_rotation_degrees <= 180.0):
        raise ValueError("label_free_near_180_rotation_degrees must be in [0, 180].")
    if cfg.iterative_contour_refinement_rounds < 0:
        raise ValueError("iterative_contour_refinement_rounds must be non-negative.")
    if cfg.iterative_contour_min_pair_count < 1:
        raise ValueError("iterative_contour_min_pair_count must be at least 1.")
    if cfg.iterative_contour_min_anchor_count < 1:
        raise ValueError("iterative_contour_min_anchor_count must be at least 1.")
    if not (0.0 <= cfg.iterative_contour_min_pair_score <= 1.0):
        raise ValueError("iterative_contour_min_pair_score must be in [0, 1].")
    if not (0.0 <= cfg.iterative_contour_min_overlap <= 1.0):
        raise ValueError("iterative_contour_min_overlap must be in [0, 1].")
    if cfg.iterative_contour_centroid_search_radius_um <= 0:
        raise ValueError("iterative_contour_centroid_search_radius_um must be positive.")
    if cfg.iterative_contour_accepted_centroid_radius_um <= 0:
        raise ValueError("iterative_contour_accepted_centroid_radius_um must be positive.")
    if cfg.iterative_contour_max_anchor_distance_um <= 0:
        raise ValueError("iterative_contour_max_anchor_distance_um must be positive.")
    if cfg.iterative_contour_residual_limit_um <= 0:
        raise ValueError("iterative_contour_residual_limit_um must be positive.")
    if not (0.0 <= cfg.iterative_contour_max_negative_jacobian_ratio <= 1.0):
        raise ValueError("iterative_contour_max_negative_jacobian_ratio must be in [0, 1].")
    if cfg.local_z_orientation not in {"off", "auto"}:
        raise ValueError("local_z_orientation must be 'off' or 'auto'.")
    if cfg.orientation_spatial_unit not in {"auto", "global", "contour"}:
        raise ValueError("orientation_spatial_unit must be 'auto', 'global', or 'contour'.")
    if cfg.contour_min_transcripts < 1:
        raise ValueError("contour_min_transcripts must be at least 1.")
    if not (0.0 <= cfg.contour_match_min_iou <= 1.0):
        raise ValueError("contour_match_min_iou must be in [0, 1].")
    if cfg.contour_match_max_distance_um < 0:
        raise ValueError("contour_match_max_distance_um must be non-negative.")
    if cfg.orientation_bootstrap_iterations < 0:
        raise ValueError("orientation_bootstrap_iterations must be non-negative.")
    if cfg.vertical_qc_backend not in {"none", "ovrlpy"}:
        raise ValueError("vertical_qc_backend must be 'none' or 'ovrlpy'.")
    if cfg.ovrlpy_n_components < 1:
        raise ValueError("ovrlpy_n_components must be at least 1.")
    if cfg.ovrlpy_n_workers < 1:
        raise ValueError("ovrlpy_n_workers must be at least 1.")
    if cfg.ovrlpy_min_transcripts <= 0:
        raise ValueError("ovrlpy_min_transcripts must be greater than 0.")
    if cfg.z_spacing_um <= 0:
        raise ValueError("z_spacing_um must be greater than 0.")
    if cfg.xenium_pixel_size_um <= 0:
        raise ValueError("xenium_pixel_size_um must be greater than 0.")
    if cfg.voxel_size_um <= 0:
        raise ValueError("voxel_size_um must be greater than 0.")
    if cfg.point_sample_distance_um <= 0:
        raise ValueError("point_sample_distance_um must be greater than 0.")
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
    if cfg.coda_raster_size < 16:
        raise ValueError("coda_raster_size must be at least 16.")
    if cfg.coda_angle_step <= 0:
        raise ValueError("coda_angle_step must be greater than 0.")
    if cfg.coda_phase_upsample_factor < 1:
        raise ValueError("coda_phase_upsample_factor must be at least 1.")
    if cfg.coda_mask_padding_fraction < 0:
        raise ValueError("coda_mask_padding_fraction must be non-negative.")
    if cfg.mesh_method != "marching_cubes":
        raise ValueError("mesh_method currently supports only 'marching_cubes'.")
    if cfg.mesh_smoothing_sigma_um is not None and cfg.mesh_smoothing_sigma_um < 0:
        raise ValueError("mesh_smoothing_sigma_um must be non-negative or None.")
    if not (0.0 < cfg.mesh_level < 1.0):
        raise ValueError("mesh_level must be strictly between 0 and 1.")
    if (
        cfg.min_mesh_component_volume_um3 is not None
        and cfg.min_mesh_component_volume_um3 < 0
    ):
        raise ValueError("min_mesh_component_volume_um3 must be non-negative.")
    _normalize_mesh_export_formats(cfg.mesh_export_formats)
    if isinstance(cfg.rbf_smoothing, str):
        if cfg.rbf_smoothing != "auto":
            raise ValueError("rbf_smoothing must be a non-negative float or 'auto'.")
    elif cfg.rbf_smoothing < 0:
        raise ValueError("rbf_smoothing must be non-negative.")
    if cfg.curvature_landmark_weight < 0 or cfg.curvature_landmark_weight > 1:
        raise ValueError("curvature_landmark_weight must be between 0 and 1.")
    if cfg.icp_iterations < 1:
        raise ValueError("icp_iterations must be at least 1.")
    if cfg.zero_anchor_count < 4:
        raise ValueError("zero_anchor_count must be at least 4.")
    if cfg.landmark_outlier_mad_threshold is not None and cfg.landmark_outlier_mad_threshold <= 0:
        raise ValueError("landmark_outlier_mad_threshold must be positive when provided.")


def _read_strategy_specs(cfg: ThreeDStackReconstructionConfig) -> list[MultiStructureSpec]:
    strategy_path = Path(cfg.segmentation_strategy) if cfg.segmentation_strategy else None
    if strategy_path is None:
        candidate = Path(cfg.xenium_root) / "contour for alignment" / "segmentationstrategy.txt"
        if candidate.exists():
            strategy_path = candidate
    if strategy_path is None or not strategy_path.exists():
        raise ValueError(
            "Provide structures=... or --segmentation-strategy. "
            "No contour for alignment/segmentationstrategy.txt was found."
        )
    specs: list[MultiStructureSpec] = []
    for index, line in enumerate(strategy_path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        clusters = [value.strip() for value in text.split(",") if value.strip()]
        specs.append(
            MultiStructureSpec(
                structure_name=f"Structure {len(specs) + 1}",
                structure_id=len(specs) + 1,
                cluster_ids=clusters,
            )
        )
    if not specs:
        raise ValueError(f"No structures were parsed from {strategy_path}.")
    return specs


def _prepare_slice_contours(
    slice_input: _SliceInput,
    structures: Sequence[MultiStructureSpec | Mapping[str, Any]],
    contour_root: Path,
    cfg: ThreeDStackReconstructionConfig,
    *,
    merged_clusters: pd.DataFrame | None = None,
) -> Path:
    if slice_input.precomputed_geojson is None:
        return _build_slice_contours(
            slice_input,
            structures,
            contour_root,
            cfg,
            merged_clusters=merged_clusters,
        )

    slice_out = contour_root / f"{slice_input.order:03d}_{slice_input.sample_id}"
    contour_geojson = slice_out / "xenium_explorer_annotations.geojson"
    if contour_geojson.exists() and not cfg.overwrite:
        return contour_geojson
    slice_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(slice_input.precomputed_geojson, contour_geojson)
    return contour_geojson


def _build_slice_contours(
    slice_input: _SliceInput,
    structures: Sequence[MultiStructureSpec | Mapping[str, Any]],
    contour_root: Path,
    cfg: ThreeDStackReconstructionConfig,
    *,
    merged_clusters: pd.DataFrame | None = None,
) -> Path:
    slice_out = contour_root / f"{slice_input.order:03d}_{slice_input.sample_id}"
    contour_geojson = slice_out / "xenium_explorer_annotations.geojson"
    if contour_geojson.exists() and not cfg.overwrite:
        return contour_geojson

    slice_out.mkdir(parents=True, exist_ok=True)
    slide = _read_xenium_slide(slice_input.xenium_dir, cfg)
    inputs = _write_slide_contour_inputs(
        slide,
        slice_out,
        cfg,
        merged_clusters=merged_clusters,
    )
    result = run_multi_structure_contours(
        MultiStructureContourConfig(
            clusters_csv=inputs["clusters_csv"],
            cells_parquet=inputs["cells_parquet"],
            out_dir=slice_out,
            structures=structures,
            bins_x=cfg.bins_x,
            bins_y=cfg.bins_y,
            gaussian_sigma=cfg.gaussian_sigma,
            density_scale_quantile=cfg.density_scale_quantile,
            support_quantile=cfg.support_quantile,
            tissue_quantile=cfg.tissue_quantile,
            min_dominance=cfg.min_dominance,
            min_cells=cfg.min_cells,
            min_component_pixels=cfg.min_component_pixels,
            xenium_pixel_size_um=cfg.xenium_pixel_size_um,
            save_preview_png=cfg.save_slice_preview_png,
        )
    )
    return result.geojson


def _read_xenium_slide(xenium_dir: Path, cfg: ThreeDStackReconstructionConfig) -> Any:
    if _looks_like_pyxenium_slide_zarr(xenium_dir):
        return _read_pyxenium_slide_zarr(xenium_dir)

    read_xenium = _import_pyxenium_read_xenium()
    try:
        return read_xenium(
            str(xenium_dir),
            as_="slide",
            prefer="auto",
            include_transcripts=False,
            stream_transcripts=True,
            include_boundaries=False,
            include_images=False,
            clusters_relpath=cfg.clusters_relpath,
            cluster_column_name=cfg.cluster_column_name,
        )
    except TypeError:
        return read_xenium(str(xenium_dir), as_="slide")


def _read_pyxenium_slide_zarr(slide_zarr: Path) -> Any:
    try:
        from pyXenium.io import read_xenium_slide

        return read_xenium_slide(str(slide_zarr))
    except Exception as first_error:
        try:
            from pyXenium.io import read_slide

            return read_slide(str(slide_zarr))
        except Exception as second_error:
            raise ImportError(
                "Reading .pyxenium.slide.zarr inputs requires pyXenium with "
                "read_xenium_slide or read_slide support."
            ) from second_error or first_error


def _import_pyxenium_read_xenium():
    try:
        from pyXenium.io import read_xenium

        return read_xenium
    except Exception as first_error:
        try:
            from pyxenium.io import read_xenium

            return read_xenium
        except Exception as second_error:
            raise ImportError(
                "HistoSeg 3D stack reconstruction requires GitHub pyXenium. "
                "Install it with: pip install "
                "'pyXenium @ git+https://github.com/hutaobo/pyXenium.git'"
            ) from second_error or first_error


def _write_slide_contour_inputs(
    slide: Any,
    out_dir: Path,
    cfg: ThreeDStackReconstructionConfig,
    *,
    merged_clusters: pd.DataFrame | None = None,
) -> dict[str, Path]:
    table = getattr(slide, "table", None)
    if table is None or not hasattr(table, "obs"):
        raise ValueError("pyXenium did not return a XeniumSlide with a table.obs annotation table.")
    obs = table.obs.copy()
    obs["Barcode"] = obs.index.astype(str)

    x_values, y_values = _extract_slide_xy(obs, table)

    cells = pd.DataFrame(
        {
            "Barcode": obs["Barcode"].astype(str).to_numpy(),
            "cell_id": obs["Barcode"].astype(str).to_numpy(),
            "x_centroid": np.asarray(x_values, dtype=float),
            "y_centroid": np.asarray(y_values, dtype=float),
        }
    )
    if merged_clusters is not None:
        clusters = merged_clusters.loc[:, ["Barcode", "Cluster"]].copy()
    else:
        cluster_col = _find_cluster_column(obs, cfg.cluster_column_name)
        if cluster_col is None:
            raise ValueError(
                f"XeniumSlide.table.obs does not contain cluster column "
                f"{cfg.cluster_column_name!r}, and no merged_h5ad clusters were provided."
            )
        clusters = pd.DataFrame(
            {
                "Barcode": obs["Barcode"].astype(str).to_numpy(),
                "Cluster": obs[cluster_col].astype(str).to_numpy(),
            }
        )
    cell_keep = (
        np.isfinite(cells["x_centroid"].to_numpy(dtype=float))
        & np.isfinite(cells["y_centroid"].to_numpy(dtype=float))
    )
    cells = cells.loc[cell_keep].reset_index(drop=True)
    clusters = clusters.loc[clusters["Cluster"].astype(str).str.strip() != ""].reset_index(
        drop=True
    )

    cells_path = out_dir / "pyxenium_cells_for_contours.parquet"
    clusters_path = out_dir / "pyxenium_clusters_for_contours.csv"
    cells.to_parquet(cells_path, index=False)
    clusters.to_csv(clusters_path, index=False)
    return {"cells_parquet": cells_path, "clusters_csv": clusters_path}


def _load_merged_cluster_tables(
    cfg: ThreeDStackReconstructionConfig,
    slices: Sequence[_SliceInput],
) -> dict[str, pd.DataFrame]:
    h5ad_path = _find_merged_h5ad(cfg)
    if h5ad_path is None:
        return {}
    try:
        import scanpy as sc
    except Exception as exc:  # pragma: no cover - scanpy is a package dependency.
        raise ImportError("Reading merged_h5ad cluster labels requires scanpy.") from exc

    adata = sc.read_h5ad(h5ad_path, backed="r")
    try:
        obs = adata.obs
        sample_col = cfg.merged_sample_column
        barcode_col = cfg.merged_barcode_column
        if sample_col not in obs.columns:
            raise ValueError(f"merged_h5ad obs is missing sample column {sample_col!r}.")
        if barcode_col not in obs.columns:
            raise ValueError(f"merged_h5ad obs is missing barcode column {barcode_col!r}.")
        cluster_col = _choose_merged_cluster_column(obs, cfg.merged_cluster_column)
        sample_ids = {item.sample_id for item in slices}
        selected = obs.loc[
            obs[sample_col].astype(str).isin(sample_ids),
            [sample_col, barcode_col, cluster_col],
        ].copy()
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()

    selected.columns = ["sample_id", "Barcode", "Cluster"]
    selected["sample_id"] = selected["sample_id"].astype(str)
    selected["Barcode"] = selected["Barcode"].astype(str)
    selected["Cluster"] = selected["Cluster"].astype(str)
    tables: dict[str, pd.DataFrame] = {}
    for sample_id, group in selected.groupby("sample_id", sort=False):
        tables[str(sample_id)] = group.loc[:, ["Barcode", "Cluster"]].reset_index(drop=True)
    return tables


def _find_merged_h5ad(cfg: ThreeDStackReconstructionConfig) -> Path | None:
    if cfg.merged_h5ad is not None:
        path = Path(cfg.merged_h5ad).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    root = Path(cfg.xenium_root).expanduser().resolve()
    merge_dir = root / "pdc_merge_leiden"
    if not merge_dir.exists():
        return None
    candidates = sorted(merge_dir.glob("*processed_leiden*.h5ad"))
    if candidates:
        return candidates[0]
    candidates = sorted(merge_dir.glob("*.h5ad"))
    return candidates[0] if candidates else None


def _choose_merged_cluster_column(obs: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in obs.columns:
            raise ValueError(f"merged_h5ad obs is missing cluster column {requested!r}.")
        return requested
    if "leiden_1_0" in obs.columns:
        return "leiden_1_0"
    candidates = [column for column in obs.columns if str(column).startswith("leiden")]
    if candidates:
        return sorted(candidates)[0]
    raise ValueError("Could not infer a merged_h5ad Leiden cluster column.")


def _extract_slide_xy(obs: pd.DataFrame, table: Any) -> tuple[np.ndarray, np.ndarray]:
    x_candidates = [
        "x_centroid",
        "x_centroid_um",
        "cell_x_centroid",
        "x",
        "x_location",
        "x_coord",
    ]
    y_candidates = [
        "y_centroid",
        "y_centroid_um",
        "cell_y_centroid",
        "y",
        "y_location",
        "y_coord",
    ]
    for x_col in x_candidates:
        for y_col in y_candidates:
            if x_col in obs.columns and y_col in obs.columns:
                return obs[x_col].to_numpy(dtype=float), obs[y_col].to_numpy(dtype=float)
    if hasattr(table, "obsm") and "spatial" in table.obsm:
        spatial = np.asarray(table.obsm["spatial"], dtype=float)
        if spatial.ndim == 2 and spatial.shape[1] >= 2:
            return spatial[:, 0], spatial[:, 1]
    raise ValueError("XeniumSlide.table does not expose x/y centroid coordinates.")


def _find_cluster_column(obs: pd.DataFrame, preferred: str) -> str | None:
    if preferred in obs.columns:
        return preferred
    for candidate in ("Cluster", "cluster", "graphclust", "leiden", "leiden_cluster"):
        if candidate in obs.columns:
            return candidate
    return None


def _find_xenium_output_dir(path: Path) -> Path | None:
    if _looks_like_xenium_dir(path):
        return path
    for child in sorted(path.iterdir()) if path.exists() else []:
        if child.is_dir() and _looks_like_xenium_dir(child):
            return child
    return None


def _looks_like_xenium_dir(path: Path) -> bool:
    return _looks_like_pyxenium_slide_zarr(path) or (
        (path / "cells.parquet").exists()
        and (
            (path / "cell_feature_matrix.h5").exists()
            or (path / "cell_feature_matrix").exists()
        )
    )


def _looks_like_pyxenium_slide_zarr(path: Path) -> bool:
    return (
        path.is_dir()
        and path.name.endswith(".pyxenium.slide.zarr")
        and (path / "zarr.json").exists()
        and (path / "tables").exists()
    )


def _sample_id_from_slice_path(path: Path) -> str:
    name = path.name
    suffix = ".pyxenium.slide.zarr"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def _sample_sort_key(sample_id: str) -> tuple[int, str]:
    matches = re.findall(r"\d+", sample_id)
    if matches:
        return int(matches[-1]), sample_id
    return 10**9, sample_id


def _estimate_similarity_transform(
    fixed_union: Any,
    moving_union: Any,
    *,
    maxiter: int,
    multistart: bool = True,
    fixed_records: Sequence[Any] | None = None,
    moving_records: Sequence[Any] | None = None,
) -> tuple[_SimilarityTransform, dict[str, Any]]:
    origin = moving_union.centroid
    scale0 = math.sqrt(max(fixed_union.area, 1e-9) / max(moving_union.area, 1e-9))
    rotation0 = _principal_axis_angle(fixed_union) - _principal_axis_angle(moving_union)

    # --- Phase 1: Smooth centroid proxy to get a good translation seed ---
    initial = _SimilarityTransform(
        origin_x=float(origin.x),
        origin_y=float(origin.y),
        rotation_degrees=float(rotation0),
        scale=float(scale0),
        translate_x=0.0,
        translate_y=0.0,
    )
    initial_aligned = _apply_similarity_to_geometry(moving_union, initial)

    # Per-structure area-weighted centroid translation init (falls back to union centroid).
    if fixed_records is not None and moving_records is not None:
        fixed_by_group = _group_union(list(fixed_records))
        moving_by_group = _group_union(list(moving_records))
        common_groups = sorted(set(fixed_by_group) & set(moving_by_group))
        if common_groups:
            total_weight = 0.0
            dx_sum = 0.0
            dy_sum = 0.0
            for grp in common_groups:
                fg = fixed_by_group[grp]
                mg_t = _apply_similarity_to_geometry(moving_by_group[grp], initial)
                weight = float(fg.area)
                dx_sum += weight * (float(fg.centroid.x) - float(mg_t.centroid.x))
                dy_sum += weight * (float(fg.centroid.y) - float(mg_t.centroid.y))
                total_weight += weight
            if total_weight > 0:
                centroid_dx = dx_sum / total_weight
                centroid_dy = dy_sum / total_weight
            else:
                centroid_dx = float(fixed_union.centroid.x) - float(initial_aligned.centroid.x)
                centroid_dy = float(fixed_union.centroid.y) - float(initial_aligned.centroid.y)
        else:
            centroid_dx = float(fixed_union.centroid.x) - float(initial_aligned.centroid.x)
            centroid_dy = float(fixed_union.centroid.y) - float(initial_aligned.centroid.y)
    else:
        centroid_dx = float(fixed_union.centroid.x) - float(initial_aligned.centroid.x)
        centroid_dy = float(fixed_union.centroid.y) - float(initial_aligned.centroid.y)

    initial = _SimilarityTransform(
        origin_x=initial.origin_x,
        origin_y=initial.origin_y,
        rotation_degrees=initial.rotation_degrees,
        scale=initial.scale,
        translate_x=centroid_dx,
        translate_y=centroid_dy,
    )
    before = _iou(fixed_union, moving_union)
    initial_iou = _iou(fixed_union, _apply_similarity_to_geometry(moving_union, initial))

    if maxiter <= 0:
        return initial, {
            "method": "pca_centroid",
            "success": True,
            "union_iou_before": before,
            "union_iou_initial": initial_iou,
            "union_iou_final": initial_iou,
        }

    # Smooth centroid proxy pre-optimisation: quickly drive the translation close to optimum.
    params_after_proxy = _smooth_proxy_optimise(
        fixed_union, moving_union, initial, maxiter=max(maxiter // 4, 8)
    )
    # Initialise Nelder-Mead IoU optimisation from the proxy-refined starting point.
    initial_from_proxy = _SimilarityTransform(
        origin_x=initial.origin_x,
        origin_y=initial.origin_y,
        rotation_degrees=float(params_after_proxy[0]),
        scale=float(math.exp(params_after_proxy[1])),
        translate_x=float(params_after_proxy[2]),
        translate_y=float(params_after_proxy[3]),
    )

    # Build candidate starting rotations for multi-start.
    start_rotations: list[float] = [initial_from_proxy.rotation_degrees]
    if multistart:
        for extra_deg in (0.0, 90.0, 180.0, 270.0):
            r = rotation0 + extra_deg
            if not any(abs(r - s) < 5.0 for s in start_rotations):
                start_rotations.append(r)

    best_transform = initial_from_proxy
    best_iou = _iou(fixed_union, _apply_similarity_to_geometry(moving_union, initial_from_proxy))
    best_result_meta: dict[str, Any] = {}

    for rot_seed in start_rotations:
        # Per-rotation centroid re-initialisation.
        seed_transform = _SimilarityTransform(
            origin_x=initial.origin_x,
            origin_y=initial.origin_y,
            rotation_degrees=float(rot_seed),
            scale=float(scale0),
            translate_x=0.0,
            translate_y=0.0,
        )
        seed_aligned = _apply_similarity_to_geometry(moving_union, seed_transform)
        seed_dx = float(fixed_union.centroid.x) - float(seed_aligned.centroid.x)
        seed_dy = float(fixed_union.centroid.y) - float(seed_aligned.centroid.y)
        seed_transform = _SimilarityTransform(
            origin_x=seed_transform.origin_x,
            origin_y=seed_transform.origin_y,
            rotation_degrees=seed_transform.rotation_degrees,
            scale=seed_transform.scale,
            translate_x=seed_dx,
            translate_y=seed_dy,
        )
        params0_seed = np.array(
            [
                seed_transform.rotation_degrees,
                math.log(max(seed_transform.scale, 1e-9)),
                seed_transform.translate_x,
                seed_transform.translate_y,
            ],
            dtype=float,
        )

        def objective(params: np.ndarray) -> float:
            transform = _SimilarityTransform(
                origin_x=initial.origin_x,
                origin_y=initial.origin_y,
                rotation_degrees=float(params[0]),
                scale=float(math.exp(params[1])),
                translate_x=float(params[2]),
                translate_y=float(params[3]),
            )
            return -_iou(fixed_union, _apply_similarity_to_geometry(moving_union, transform))

        result = minimize(
            objective,
            params0_seed,
            method="Nelder-Mead",
            options={"maxiter": int(maxiter), "xatol": 1e-3, "fatol": 1e-5},
        )
        params = result.x if result.fun <= objective(params0_seed) else params0_seed
        candidate = _SimilarityTransform(
            origin_x=initial.origin_x,
            origin_y=initial.origin_y,
            rotation_degrees=float(params[0]),
            scale=float(math.exp(params[1])),
            translate_x=float(params[2]),
            translate_y=float(params[3]),
        )
        candidate_iou = _iou(fixed_union, _apply_similarity_to_geometry(moving_union, candidate))
        if candidate_iou > best_iou:
            best_iou = candidate_iou
            best_transform = candidate
            best_result_meta = {
                "success": bool(result.success) or candidate_iou >= initial_iou,
                "message": str(result.message),
                "iterations": int(getattr(result, "nit", 0)),
                "rotation_seed_deg": float(rot_seed),
            }

    return best_transform, {
        "method": "pca_centroid_multistart_nelder_mead" if multistart else "pca_centroid_nelder_mead",
        "multistart_seeds": len(start_rotations),
        "union_iou_before": before,
        "union_iou_initial": initial_iou,
        "union_iou_final": best_iou,
        **best_result_meta,
    }


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


def _apply_similarity_to_geojson(
    geojson: dict[str, Any],
    transform: _SimilarityTransform,
) -> dict[str, Any]:
    payload = copy.deepcopy(geojson)
    for feature in payload["features"]:
        geom = _apply_similarity_to_geometry(shape(feature["geometry"]), transform)
        feature["geometry"] = mapping(geom)
    return payload


def _apply_similarity_to_geometry(geom: Any, transform: _SimilarityTransform) -> Any:
    origin = (transform.origin_x, transform.origin_y)
    transformed = affinity.rotate(geom, transform.rotation_degrees, origin=origin)
    transformed = affinity.scale(
        transformed,
        xfact=transform.scale,
        yfact=transform.scale,
        origin=origin,
    )
    transformed = affinity.translate(
        transformed,
        xoff=transform.translate_x,
        yoff=transform.translate_y,
    )
    return transformed


def _per_structure_iou(fixed_records: Sequence[Any], moving_records: Sequence[Any]) -> dict[str, float]:
    fixed_by_group = _group_union(fixed_records)
    moving_by_group = _group_union(moving_records)
    return {
        group: _iou(fixed_by_group[group], moving_by_group[group])
        for group in sorted(set(fixed_by_group) & set(moving_by_group))
    }


def _group_union(records: Sequence[Any]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = {}
    for record in records:
        grouped.setdefault(record.group, []).append(record.geometry)
    return {group: unary_union(geoms) for group, geoms in grouped.items()}


def _identity_transform_for_geometry(geom: Any) -> _SimilarityTransform:
    centroid = geom.centroid
    return _SimilarityTransform(
        origin_x=float(centroid.x),
        origin_y=float(centroid.y),
        rotation_degrees=0.0,
        scale=1.0,
        translate_x=0.0,
        translate_y=0.0,
    )


def _rasterize_pair_for_coda(
    fixed_union: Any,
    moving_union: Any,
    *,
    raster_size: int,
    padding_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    bounds = _square_padded_bounds(
        unary_union([fixed_union, moving_union]),
        padding_fraction=padding_fraction,
    )
    x_min, y_min, x_max, y_max = bounds
    x_values = np.linspace(x_min, x_max, int(raster_size), dtype=float)
    y_values = np.linspace(y_max, y_min, int(raster_size), dtype=float)
    xx, yy = np.meshgrid(x_values, y_values)
    fixed_mask = _geometry_contains_grid(fixed_union, xx, yy)
    moving_mask = _geometry_contains_grid(moving_union, xx, yy)
    native_units_per_pixel = (x_max - x_min) / max(int(raster_size) - 1, 1)
    metadata = {
        "input_source": "contour_union_raster_proxy",
        "raster_size": int(raster_size),
        "square_bounds": [float(value) for value in bounds],
        "native_units_per_pixel": float(native_units_per_pixel),
        "fixed_positive_pixels": int(fixed_mask.sum()),
        "moving_positive_pixels": int(moving_mask.sum()),
    }
    return fixed_mask, moving_mask, metadata


def _square_padded_bounds(geom: Any, *, padding_fraction: float) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = map(float, geom.bounds)
    width = max(x_max - x_min, 1e-6)
    height = max(y_max - y_min, 1e-6)
    side = max(width, height, 1.0)
    side *= 1.0 + 2.0 * max(float(padding_fraction), 0.0)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    half = side / 2.0
    return (center_x - half, center_y - half, center_x + half, center_y + half)


def _coda_image_summary_payload(
    *,
    image_result: Any,
    raster_metadata: Mapping[str, Any],
    angle_step: float,
    phase_upsample_factor: int,
    mask_padding_fraction: float,
) -> dict[str, Any]:
    return {
        "radon_rotation_degrees": float(image_result.rotation.rotation_degrees),
        "radon_score": float(image_result.rotation.score),
        "radon_angle_range": [
            float(image_result.rotation.angle_range[0]),
            float(image_result.rotation.angle_range[1]),
        ],
        "radon_angle_step": float(angle_step),
        "phase_shift_y": float(image_result.translation.shift_y),
        "phase_shift_x": float(image_result.translation.shift_x),
        "phase_error": float(image_result.translation.error),
        "phase_difference": float(image_result.translation.phase_difference),
        "phase_upsample_factor": int(phase_upsample_factor),
        "orientation_disambiguation": getattr(
            image_result,
            "orientation_disambiguation",
            None,
        ),
        "orientation_candidates": [
            {key: float(value) for key, value in candidate.items()}
            for candidate in getattr(image_result, "orientation_candidates", ())
        ],
        "preprocessing": {
            **dict(raster_metadata),
            "mask_padding_fraction": float(mask_padding_fraction),
        },
        "deferred_features": [
            "tile_wise_dense_displacement",
            "dense_field_jacobian_validation",
            "semantic_volume_labeling",
        ],
    }


def _hard_alignment_candidate_payload(
    summary: Mapping[str, Any],
    *,
    summary_json: PathLike | None = None,
) -> dict[str, Any]:
    transform = summary.get("transform") or {}
    payload: dict[str, Any] = {
        "backend": summary.get("registration_backend"),
        "summary_json": str(Path(summary_json)) if summary_json is not None else None,
        "output_geojson": summary.get("output_geojson"),
        "union_iou_before_hard": summary.get("union_iou_before_hard"),
        "union_iou_after_hard": summary.get("union_iou_after_hard"),
        "hard_alignment_accepted": summary.get("hard_alignment_accepted"),
        "rotation_degrees": transform.get("rotation_degrees"),
        "scale": transform.get("scale"),
        "translate_x": transform.get("translate_x"),
        "translate_y": transform.get("translate_y"),
    }
    if summary.get("registration_backend") == "label-free-group":
        payload.update(
            {
                "label_free_fixed_group": summary.get("label_free_fixed_group"),
                "label_free_moving_group": summary.get("label_free_moving_group"),
                "label_free_used_anchor_pair_count": summary.get(
                    "label_free_used_anchor_pair_count"
                ),
                "label_free_residual_median": summary.get("label_free_residual_median"),
                "label_free_residual_p90": summary.get("label_free_residual_p90"),
                "label_free_partial_anchor_count": summary.get(
                    "label_free_partial_anchor_count"
                ),
                "label_free_anchor_coverage_ratio": summary.get(
                    "label_free_anchor_coverage_ratio"
                ),
                "label_free_anchor_occupied_quadrants": summary.get(
                    "label_free_anchor_occupied_quadrants"
                ),
                "label_free_guarded_candidate_accepted": summary.get(
                    "label_free_guarded_candidate_accepted"
                ),
                "label_free_guarded_rejection_reason": summary.get(
                    "label_free_guarded_rejection_reason"
                ),
                "label_free_group_matrix_csv": summary.get("label_free_group_matrix_csv"),
                "label_free_group_matrix_html": summary.get("label_free_group_matrix_html"),
            }
        )

    coda_image = summary.get("coda_image")
    if isinstance(coda_image, Mapping):
        payload["coda_image"] = {
            "radon_rotation_degrees": coda_image.get("radon_rotation_degrees"),
            "phase_shift_y": coda_image.get("phase_shift_y"),
            "phase_shift_x": coda_image.get("phase_shift_x"),
            "orientation_disambiguation": coda_image.get("orientation_disambiguation"),
        }
    return payload


def _hard_alignment_iou(summary: Mapping[str, Any]) -> float:
    try:
        value = float(summary.get("union_iou_after_hard", 0.0))
    except (TypeError, ValueError):
        return float("-inf")
    return value if math.isfinite(value) else float("-inf")


def _finite_float(value: Any, default: float = math.inf) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _absolute_rotation_degrees(value: Any) -> float:
    rotation = _finite_float(value, default=0.0)
    return abs((rotation + 180.0) % 360.0 - 180.0)


def _label_free_group_candidate_guard(
    summary: Mapping[str, Any],
    cfg: ThreeDStackReconstructionConfig,
) -> tuple[bool, dict[str, Any]]:
    transform = summary.get("transform") or {}
    used = _int_value(summary.get("label_free_used_anchor_pair_count"), default=0)
    residual = _finite_float(summary.get("label_free_residual_median"))
    rotation_abs = _absolute_rotation_degrees(transform.get("rotation_degrees"))
    coverage = _finite_float(
        summary.get("label_free_anchor_coverage_ratio"),
        default=float("nan"),
    )
    quadrants = _int_value(summary.get("label_free_anchor_occupied_quadrants"), default=0)
    transform_kind = str(transform.get("kind") or "similarity")
    guard: dict[str, Any] = {
        "accepted": False,
        "reason": "accepted",
        "used_anchor_pair_count": used,
        "residual_median": residual if math.isfinite(residual) else None,
        "rotation_abs_degrees": rotation_abs,
        "anchor_coverage_ratio": coverage if math.isfinite(coverage) else None,
        "anchor_occupied_quadrants": quadrants,
        "transform_kind": transform_kind,
        "guarded_alignment": bool(cfg.label_free_guarded_alignment),
        "min_anchor_count": int(cfg.label_free_min_anchor_count),
        "min_similarity_anchor_count": int(cfg.label_free_min_similarity_anchor_count),
        "max_rotation_degrees": float(cfg.label_free_max_rotation_degrees),
        "high_rotation_min_anchor_count": int(
            cfg.label_free_high_rotation_min_anchor_count
        ),
        "high_rotation_residual_limit_um": float(
            cfg.label_free_high_rotation_residual_limit_um
        ),
        "min_anchor_coverage": float(cfg.label_free_min_anchor_coverage),
        "min_anchor_quadrants": int(cfg.label_free_min_anchor_quadrants),
        "near_180_rotation_degrees": float(cfg.label_free_near_180_rotation_degrees),
    }

    def reject(reason: str) -> tuple[bool, dict[str, Any]]:
        guard["reason"] = reason
        return False, guard

    if summary.get("registration_backend") != "label-free-group":
        return reject("not_label_free_group")
    if not bool(summary.get("hard_alignment_accepted")):
        return reject("label_free_candidate_not_accepted")
    if not math.isfinite(residual):
        return reject("missing_label_free_residual")
    if residual > float(cfg.label_free_group_residual_limit_um):
        return reject("label_free_residual_above_limit")
    if used < int(cfg.label_free_min_anchor_count):
        return reject("too_few_label_free_anchors")
    if not bool(cfg.label_free_guarded_alignment):
        guard["accepted"] = True
        return True, guard

    if used < 4 and transform_kind != "translation":
        return reject("too_few_anchors_for_similarity_transform")
    if used < int(cfg.label_free_min_similarity_anchor_count) and transform_kind != "translation":
        return reject("too_few_anchors_for_unconstrained_similarity")
    if rotation_abs >= float(cfg.label_free_near_180_rotation_degrees):
        return reject("near_180_rotation_rejected")

    if rotation_abs > float(cfg.label_free_max_rotation_degrees):
        if used < int(cfg.label_free_high_rotation_min_anchor_count):
            return reject("rotation_exceeds_prior_without_enough_anchors")
        if residual > float(cfg.label_free_high_rotation_residual_limit_um):
            return reject("rotation_exceeds_prior_without_low_residual")
        if not math.isfinite(coverage) or coverage < float(cfg.label_free_min_anchor_coverage):
            return reject("rotation_exceeds_prior_without_anchor_coverage")
        if quadrants < int(cfg.label_free_min_anchor_quadrants):
            return reject("rotation_exceeds_prior_without_spatial_spread")

    guard["accepted"] = True
    return True, guard


def _label_free_group_candidate_ok(
    summary: Mapping[str, Any],
    cfg: ThreeDStackReconstructionConfig,
) -> bool:
    ok, _ = _label_free_group_candidate_guard(summary, cfg)
    return ok


def _contour_tps_candidate_ok(
    summary: Mapping[str, Any],
    cfg: ThreeDStackReconstructionConfig,
) -> bool:
    if summary.get("registration_backend") != "contour-tps":
        return False
    if not bool(summary.get("hard_alignment_accepted")):
        return False
    if not bool(cfg.label_free_guarded_alignment):
        return True
    rotation_abs = _absolute_rotation_degrees(
        (summary.get("transform") or {}).get("rotation_degrees")
    )
    return rotation_abs <= float(cfg.label_free_max_rotation_degrees)


def _resolve_soft_alignment_mode(
    cfg: ThreeDStackReconstructionConfig,
    summary: Mapping[str, Any],
    *,
    semantic_soft_allowed: bool,
    semantic_soft_skipped_reason: str | None,
) -> tuple[str, str | None]:
    if not cfg.run_soft_alignment or cfg.soft_alignment_mode == "none":
        return "none", "soft_alignment_disabled"
    selected = summary.get("selected_hard_seed_backend") or summary.get("registration_backend")
    requested = str(cfg.soft_alignment_mode)
    if requested == "auto":
        if selected == "label-free-group":
            return "anchor-only", None
        if selected == "translation-only":
            return "none", "translation_only_seed_no_semantic_soft"
        if semantic_soft_allowed:
            return "semantic", None
        return "none", semantic_soft_skipped_reason or "semantic_soft_not_allowed"
    if requested == "anchor-only":
        return "anchor-only", None
    if requested == "semantic":
        if semantic_soft_allowed:
            return "semantic", None
        return "none", semantic_soft_skipped_reason or "semantic_soft_not_allowed"
    return "none", "soft_alignment_disabled"


def _semantic_soft_alignment_policy(summary: Mapping[str, Any]) -> tuple[bool, str | None]:
    selected = summary.get("selected_hard_seed_backend") or summary.get("registration_backend")
    if selected != "label-free-group":
        return True, None
    fixed_group = summary.get("label_free_fixed_group")
    moving_group = summary.get("label_free_moving_group")
    if fixed_group is None or moving_group is None:
        return False, "missing_label_free_group_metadata"
    if str(fixed_group) != str(moving_group):
        return False, "cross_named_label_free_group_match"
    return False, "label_free_group_uses_anchor_only_soft_alignment"


def _resolve_label_free_anchor_landmarks_csv(summary: Mapping[str, Any]) -> Path | None:
    for key in ("label_free_anchor_landmarks_csv", "anchor_landmarks_csv"):
        value = summary.get(key)
        if value:
            path = Path(str(value))
            if path.exists():
                return path
    summary_json = summary.get("label_free_summary_json")
    if not summary_json:
        return None
    try:
        payload = json.loads(Path(str(summary_json)).read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("anchor_landmarks_csv", "label_free_anchor_landmarks_csv"):
        value = payload.get(key)
        if value:
            path = Path(str(value))
            if path.exists():
                return path
    return None


def _rotation_difference_degrees(first: Any, second: Any) -> float | None:
    try:
        delta = (float(first) - float(second) + 180.0) % 360.0 - 180.0
    except (TypeError, ValueError):
        return None
    return abs(delta)


def _hard_candidate_iou(summary: Mapping[str, Any], backend: str) -> Any:
    for candidate in summary.get("hard_alignment_candidates") or ():
        if isinstance(candidate, Mapping) and candidate.get("backend") == backend:
            return candidate.get("union_iou_after_hard")
    return None


def _hard_candidate_label_free_residual(summary: Mapping[str, Any]) -> Any:
    return _hard_candidate_label_free_field(summary, "label_free_residual_median")


def _hard_candidate_label_free_field(summary: Mapping[str, Any], key: str) -> Any:
    for candidate in summary.get("hard_alignment_candidates") or ():
        if isinstance(candidate, Mapping) and candidate.get("backend") == "label-free-group":
            return candidate.get(key)
    return None


def _run_iterative_contour_refinement_if_enabled(
    *,
    cfg: ThreeDStackReconstructionConfig,
    fixed_geojson: Path,
    current_soft_result: Any,
    current_soft_summary: Mapping[str, Any],
    pair_dir: Path,
    round_count: int,
) -> tuple[Any, dict[str, Any], bool]:
    from .label_free_alignment import (  # Local import avoids a module cycle.
        IterativeContourRefinementConfig,
        run_iterative_contour_refinement,
    )

    rounds = max(int(round_count), 0)
    if rounds <= 0:
        return current_soft_result, dict(current_soft_summary), bool(
            current_soft_summary.get("accepted", False)
        )

    active_result = current_soft_result
    active_summary = dict(current_soft_summary)
    chain_paths = [str(active_result.summary_json)]
    refinements: list[dict[str, Any]] = []
    accepted_any = False

    for round_index in range(1, rounds + 1):
        refinement = run_iterative_contour_refinement(
            IterativeContourRefinementConfig(
                fixed_geojson=fixed_geojson,
                moving_aligned_geojson=active_result.soft_aligned_geojson,
                out_dir=pair_dir / f"iterative_contour_refinement_round{round_index}",
                group_property=cfg.group_property,
                centroid_search_radius_um=cfg.iterative_contour_centroid_search_radius_um,
                accepted_centroid_radius_um=(
                    cfg.iterative_contour_accepted_centroid_radius_um
                ),
                min_overlap=cfg.iterative_contour_min_overlap,
                min_pair_score=cfg.iterative_contour_min_pair_score,
                max_anchor_distance_um=cfg.iterative_contour_max_anchor_distance_um,
                min_pair_count=cfg.iterative_contour_min_pair_count,
                min_anchor_count=cfg.iterative_contour_min_anchor_count,
                residual_limit_um=cfg.iterative_contour_residual_limit_um,
                rbf_neighbors=cfg.rbf_neighbors,
                rbf_smoothing=max(float(cfg.anchor_only_rbf_smoothing), 2e-4),
                jacobian_grid_size=max(int(cfg.anchor_only_jacobian_grid_size), 45),
                max_negative_jacobian_ratio=(
                    cfg.iterative_contour_max_negative_jacobian_ratio
                ),
                save_preview_png=cfg.save_alignment_preview_png,
                overwrite=cfg.overwrite,
                dpi=cfg.dpi,
            )
        )
        refinement_summary = json.loads(
            refinement.summary_json.read_text(encoding="utf-8")
        )
        refinements.append(refinement_summary)
        if not bool(refinement_summary.get("accepted", False)):
            break
        accepted_any = True
        active_result = SimpleNamespace(
            soft_aligned_geojson=refinement.refined_geojson,
            summary_json=refinement.summary_json,
        )
        active_summary = _combine_anchor_only_and_iterative_soft_summary(
            base_summary=current_soft_summary,
            refinement_summaries=refinements,
            output_geojson=refinement.refined_geojson,
            chain_paths=[*chain_paths, str(refinement.summary_json)],
            out_dir=pair_dir / "anchor_only_iterative_soft_tps",
        )
        active_result = SimpleNamespace(
            soft_aligned_geojson=refinement.refined_geojson,
            summary_json=Path(active_summary["outputs"]["combined_summary_json"]),
        )
        chain_paths = list(active_summary["outputs"]["tps_chain"])

    if not refinements:
        return current_soft_result, dict(current_soft_summary), bool(
            current_soft_summary.get("accepted", False)
        )
    if not accepted_any:
        active_summary = _combine_anchor_only_and_iterative_soft_summary(
            base_summary=current_soft_summary,
            refinement_summaries=refinements,
            output_geojson=Path(current_soft_summary["outputs"]["soft_geojson"]),
            chain_paths=chain_paths,
            out_dir=pair_dir / "anchor_only_iterative_soft_tps",
        )
        active_result = SimpleNamespace(
            soft_aligned_geojson=Path(current_soft_summary["outputs"]["soft_geojson"]),
            summary_json=Path(active_summary["outputs"]["combined_summary_json"]),
        )
    return active_result, active_summary, bool(active_summary.get("accepted", False))


def _combine_anchor_only_and_iterative_soft_summary(
    *,
    base_summary: Mapping[str, Any],
    refinement_summaries: Sequence[Mapping[str, Any]],
    output_geojson: Path,
    chain_paths: Sequence[str],
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "anchor_only_iterative_tps_summary.json"
    combined = copy.deepcopy(dict(base_summary))
    combined["method"] = {
        "name": "anchor_only_residual_tps_with_iterative_contour_refinement",
        "base_method": base_summary.get("method"),
        "rbf_kernel": (
            base_summary.get("config", {}).get("rbf_kernel")
            if isinstance(base_summary.get("config"), Mapping)
            else "thin_plate_spline"
        ),
        "rbf_neighbors": (
            base_summary.get("config", {}).get("rbf_neighbors")
            if isinstance(base_summary.get("config"), Mapping)
            else 96
        ),
        "rbf_smoothing": (
            base_summary.get("config", {}).get("rbf_smoothing")
            if isinstance(base_summary.get("config"), Mapping)
            else 1e-4
        ),
    }
    accepted_refinements = [
        summary for summary in refinement_summaries if bool(summary.get("accepted", False))
    ]
    latest = accepted_refinements[-1] if accepted_refinements else None
    combined["accepted"] = bool(base_summary.get("accepted", False))
    combined["iterative_contour_refinement"] = {
        "attempted": True,
        "round_count": int(len(refinement_summaries)),
        "accepted_round_count": int(len(accepted_refinements)),
        "accepted": bool(latest is not None),
        "latest_reason": refinement_summaries[-1].get("reason")
        if refinement_summaries
        else None,
        "rounds": list(refinement_summaries),
    }
    if latest is not None:
        combined["outputs"]["soft_geojson"] = str(output_geojson)
        combined["qc"]["union_iou_soft_after"] = latest["qc"][
            "union_iou_after_refinement"
        ]
        combined["qc"]["delta_union_iou_soft"] = (
            float(combined["qc"]["union_iou_soft_after"])
            - float(combined["qc"]["union_iou_hard_before_soft"])
        )
        combined["qc"]["geometry_status_counts"] = latest["qc"].get(
            "geometry_status_counts", combined["qc"].get("geometry_status_counts")
        )
        combined["qc"]["topology_check"] = latest["qc"].get(
            "topology_check", combined["qc"].get("topology_check")
        )
        combined["qc"]["jacobian_check"] = latest["qc"].get(
            "jacobian_check", combined["qc"].get("jacobian_check")
        )
        combined["qc"]["post_warp_residual_um"] = latest["qc"].get(
            "post_warp_residual_um", combined["qc"].get("post_warp_residual_um")
        )
    combined.setdefault("outputs", {})
    combined["outputs"]["combined_summary_json"] = str(combined_path)
    combined["outputs"]["tps_chain"] = list(chain_paths)
    combined["outputs"]["iterative_refinement_summary_json"] = (
        str(Path(refinement_summaries[-1]["outputs"]["out_dir"]) / "iterative_contour_refinement_summary.json")
        if refinement_summaries
        else None
    )
    combined_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return combined


def _pairwise_row(
    slice_input: _SliceInput,
    hard_summary: Mapping[str, Any],
    soft_summary: Mapping[str, Any] | None,
    soft_result: Any,
    soft_accepted: bool | None = None,
) -> dict[str, Any]:
    row = {
        "moving_order": slice_input.order,
        "moving_sample_id": slice_input.sample_id,
        "registration_backend": hard_summary.get("registration_backend", "contour-tps"),
        "selected_hard_seed_backend": hard_summary.get("selected_hard_seed_backend"),
        "method_credit": hard_summary.get("method_credit"),
        "method_reference_doi": hard_summary.get("method_reference_doi"),
        "hard_union_iou_before": hard_summary.get("union_iou_before_hard"),
        "hard_union_iou_after": hard_summary.get("union_iou_after_hard"),
        "hard_candidate_contour_iou_after": _hard_candidate_iou(hard_summary, "contour-tps"),
        "hard_candidate_coda_iou_after": _hard_candidate_iou(hard_summary, "coda-image"),
        "hard_candidate_label_free_group_residual_median": _hard_candidate_label_free_residual(
            hard_summary
        ),
        "hard_candidate_label_free_rotation_degrees": _hard_candidate_label_free_field(
            hard_summary, "rotation_degrees"
        ),
        "hard_candidate_label_free_fixed_group": _hard_candidate_label_free_field(
            hard_summary, "label_free_fixed_group"
        ),
        "hard_candidate_label_free_moving_group": _hard_candidate_label_free_field(
            hard_summary, "label_free_moving_group"
        ),
        "hard_candidate_label_free_used_anchor_pair_count": _hard_candidate_label_free_field(
            hard_summary, "label_free_used_anchor_pair_count"
        ),
        "hard_candidate_label_free_anchor_coverage_ratio": _hard_candidate_label_free_field(
            hard_summary, "label_free_anchor_coverage_ratio"
        ),
        "hard_candidate_label_free_anchor_occupied_quadrants": _hard_candidate_label_free_field(
            hard_summary, "label_free_anchor_occupied_quadrants"
        ),
        "hard_candidate_label_free_guarded_candidate_accepted": _hard_candidate_label_free_field(
            hard_summary, "label_free_guarded_candidate_accepted"
        ),
        "hard_candidate_label_free_guarded_rejection_reason": _hard_candidate_label_free_field(
            hard_summary, "label_free_guarded_rejection_reason"
        ),
        "hard_tournament_rotation_difference_degrees": (
            hard_summary.get("hard_alignment_tournament") or {}
        ).get("rotation_difference_degrees"),
        "hard_transform_rotation_degrees": hard_summary["transform"]["rotation_degrees"],
        "hard_transform_scale": hard_summary["transform"]["scale"],
        "hard_transform_translate_x": hard_summary["transform"]["translate_x"],
        "hard_transform_translate_y": hard_summary["transform"]["translate_y"],
        "hard_accepted": hard_summary.get("hard_alignment_accepted"),
        "label_free_fixed_group": hard_summary.get("label_free_fixed_group"),
        "label_free_moving_group": hard_summary.get("label_free_moving_group"),
        "label_free_used_anchor_pair_count": hard_summary.get(
            "label_free_used_anchor_pair_count"
        ),
        "label_free_residual_median": hard_summary.get("label_free_residual_median"),
        "label_free_partial_anchor_count": hard_summary.get(
            "label_free_partial_anchor_count"
        ),
        "label_free_anchor_coverage_ratio": hard_summary.get(
            "label_free_anchor_coverage_ratio"
        ),
        "label_free_anchor_occupied_quadrants": hard_summary.get(
            "label_free_anchor_occupied_quadrants"
        ),
        "label_free_guarded_candidate_accepted": hard_summary.get(
            "label_free_guarded_candidate_accepted"
        ),
        "label_free_guarded_rejection_reason": hard_summary.get(
            "label_free_guarded_rejection_reason"
        ),
        "label_free_guarded_rotation_abs_degrees": hard_summary.get(
            "label_free_guarded_rotation_abs_degrees"
        ),
        "semantic_soft_allowed": hard_summary.get("semantic_soft_allowed"),
        "semantic_soft_skipped_reason": hard_summary.get("semantic_soft_skipped_reason"),
        "soft_alignment_mode_requested": hard_summary.get("soft_alignment_mode_requested"),
        "active_soft_alignment_mode": hard_summary.get("active_soft_alignment_mode"),
        "soft_alignment_skipped_reason": hard_summary.get("soft_alignment_skipped_reason"),
        "soft_alignment_runtime_seconds": hard_summary.get(
            "soft_alignment_runtime_seconds"
        ),
        "coda_radon_rotation_degrees": (hard_summary.get("coda_image") or {}).get(
            "radon_rotation_degrees"
        ),
        "coda_phase_shift_y": (hard_summary.get("coda_image") or {}).get("phase_shift_y"),
        "coda_phase_shift_x": (hard_summary.get("coda_image") or {}).get("phase_shift_x"),
        "soft_union_iou_before": None,
        "soft_union_iou_after": None,
        "soft_accepted": soft_accepted,
        "soft_geometry_valid": None,
        "soft_topology_valid": None,
        "soft_topology_checked_cells": None,
        "soft_topology_min_area_ratio": None,
        "soft_topology_median_area_ratio": None,
        "soft_topology_max_area_ratio": None,
        "soft_topology_folded_cells": None,
        "soft_topology_compressed_cells": None,
        "soft_topology_expanded_cells": None,
        "soft_boundary_landmarks": None,
        "soft_summary_json": None,
        "anchor_only_accepted": None,
        "anchor_only_anchor_count": None,
        "anchor_only_identity_padding_count": None,
        "anchor_only_input_residual_median": None,
        "anchor_only_input_residual_p90": None,
        "anchor_only_post_residual_median": None,
        "anchor_only_post_residual_p90": None,
        "anchor_only_negative_jacobian_ratio": None,
        "anchor_only_min_jacobian_ratio": None,
        "anchor_only_fallback_reason": None,
        "iterative_refinement_attempted": None,
        "iterative_refinement_accepted": None,
        "iterative_refinement_round_count": None,
        "iterative_refinement_accepted_round_count": None,
        "iterative_refinement_pair_count": None,
        "iterative_refinement_anchor_count": None,
        "iterative_refinement_delta_iou": None,
        "iterative_refinement_negative_jacobian_ratio": None,
        "iterative_refinement_min_jacobian_ratio": None,
        "iterative_refinement_fallback_reason": None,
    }
    if soft_summary is not None and soft_result is not None:
        topology = soft_summary["qc"].get("topology_check", {})
        row.update(
            {
                "soft_union_iou_before": soft_summary["qc"]["union_iou_hard_before_soft"],
                "soft_union_iou_after": soft_summary["qc"]["union_iou_soft_after"],
                "soft_geometry_valid": int(
                    soft_summary["qc"].get("geometry_status_counts", {}).get("invalid", 0)
                )
                == 0,
                "soft_topology_valid": topology.get("valid"),
                "soft_topology_checked_cells": topology.get("checked_cells"),
                "soft_topology_min_area_ratio": topology.get("min_area_ratio"),
                "soft_topology_median_area_ratio": topology.get("median_area_ratio"),
                "soft_topology_max_area_ratio": topology.get("max_area_ratio"),
                "soft_topology_folded_cells": topology.get("folded_cell_count"),
                "soft_topology_compressed_cells": topology.get("compressed_cell_count"),
                "soft_topology_expanded_cells": topology.get("expanded_cell_count"),
                "soft_boundary_landmarks": soft_summary["landmarks"]["boundary_landmark_count"],
                "soft_summary_json": str(soft_result.summary_json),
            }
        )
        method_value = soft_summary.get("method")
        method_name = (
            method_value.get("name")
            if isinstance(method_value, Mapping)
            else method_value
        )
        base_method_name = (
            method_value.get("base_method")
            if isinstance(method_value, Mapping)
            else None
        )
        if (
            method_name == "anchor_only_residual_tps"
            or base_method_name == "anchor_only_residual_tps"
        ):
            input_residual = soft_summary.get("landmarks", {}).get("input_residual_um") or {}
            post_residual = soft_summary.get("qc", {}).get("post_warp_residual_um") or {}
            jacobian = soft_summary.get("qc", {}).get("jacobian_check") or {}
            row.update(
                {
                    "anchor_only_accepted": bool(soft_summary.get("accepted", False)),
                    "anchor_only_anchor_count": soft_summary.get("landmarks", {}).get(
                        "anchor_landmark_count"
                    ),
                    "anchor_only_identity_padding_count": soft_summary.get(
                        "landmarks", {}
                    ).get("identity_padding_count"),
                    "anchor_only_input_residual_median": input_residual.get("median"),
                    "anchor_only_input_residual_p90": input_residual.get("p90"),
                    "anchor_only_post_residual_median": post_residual.get("median"),
                    "anchor_only_post_residual_p90": post_residual.get("p90"),
                    "anchor_only_negative_jacobian_ratio": jacobian.get(
                        "negative_jacobian_ratio"
                    ),
                    "anchor_only_min_jacobian_ratio": jacobian.get("min_jacobian_ratio"),
                    "anchor_only_fallback_reason": (
                        None
                        if bool(soft_summary.get("accepted", False))
                        else soft_summary.get("reason")
                    ),
                }
            )
        iterative = soft_summary.get("iterative_contour_refinement") or {}
        if iterative:
            accepted_rounds = [
                summary
                for summary in iterative.get("rounds", [])
                if isinstance(summary, Mapping) and bool(summary.get("accepted", False))
            ]
            latest = accepted_rounds[-1] if accepted_rounds else None
            latest_any = (
                iterative.get("rounds", [])[-1]
                if iterative.get("rounds")
                and isinstance(iterative.get("rounds", [])[-1], Mapping)
                else {}
            )
            qc = latest.get("qc", {}) if isinstance(latest, Mapping) else {}
            jacobian = qc.get("jacobian_check") or {}
            pairs = latest.get("pairs", {}) if isinstance(latest, Mapping) else {}
            landmarks = latest.get("landmarks", {}) if isinstance(latest, Mapping) else {}
            row.update(
                {
                    "iterative_refinement_attempted": bool(
                        iterative.get("attempted", False)
                    ),
                    "iterative_refinement_accepted": bool(
                        iterative.get("accepted", False)
                    ),
                    "iterative_refinement_round_count": iterative.get("round_count"),
                    "iterative_refinement_accepted_round_count": iterative.get(
                        "accepted_round_count"
                    ),
                    "iterative_refinement_pair_count": pairs.get(
                        "accepted_pair_count"
                    ),
                    "iterative_refinement_anchor_count": landmarks.get(
                        "anchor_landmark_count"
                    ),
                    "iterative_refinement_delta_iou": qc.get(
                        "delta_union_iou_refinement"
                    ),
                    "iterative_refinement_negative_jacobian_ratio": jacobian.get(
                        "negative_jacobian_ratio"
                    ),
                    "iterative_refinement_min_jacobian_ratio": jacobian.get(
                        "min_jacobian_ratio"
                    ),
                    "iterative_refinement_fallback_reason": (
                        None
                        if bool(iterative.get("accepted", False))
                        else latest_any.get("reason") or iterative.get("latest_reason")
                    ),
                }
            )
    return row


def _write_guarded_alignment_audit(
    pairwise_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
) -> tuple[Path, Path]:
    audit_csv = out_dir / "pairwise_guarded_alignment_audit.csv"
    audit_html = out_dir / "pairwise_guarded_alignment_audit.html"
    columns = [
        "moving_order",
        "moving_sample_id",
        "registration_backend",
        "selected_hard_seed_backend",
        "label_free_fixed_group",
        "label_free_moving_group",
        "label_free_used_anchor_pair_count",
        "label_free_partial_anchor_count",
        "label_free_residual_median",
        "label_free_anchor_coverage_ratio",
        "label_free_anchor_occupied_quadrants",
        "hard_transform_rotation_degrees",
        "label_free_guarded_rotation_abs_degrees",
        "hard_transform_translate_x",
        "hard_transform_translate_y",
        "label_free_guarded_candidate_accepted",
        "label_free_guarded_rejection_reason",
        "semantic_soft_allowed",
        "semantic_soft_skipped_reason",
        "active_soft_alignment_mode",
        "soft_accepted",
        "anchor_only_accepted",
    ]
    def audit_value(row: Mapping[str, Any], column: str) -> Any:
        candidate_map = {
            "label_free_fixed_group": "hard_candidate_label_free_fixed_group",
            "label_free_moving_group": "hard_candidate_label_free_moving_group",
            "label_free_used_anchor_pair_count": (
                "hard_candidate_label_free_used_anchor_pair_count"
            ),
            "label_free_residual_median": (
                "hard_candidate_label_free_group_residual_median"
            ),
            "label_free_anchor_coverage_ratio": (
                "hard_candidate_label_free_anchor_coverage_ratio"
            ),
            "label_free_anchor_occupied_quadrants": (
                "hard_candidate_label_free_anchor_occupied_quadrants"
            ),
            "hard_transform_rotation_degrees": (
                "hard_candidate_label_free_rotation_degrees"
            ),
            "label_free_guarded_candidate_accepted": (
                "hard_candidate_label_free_guarded_candidate_accepted"
            ),
            "label_free_guarded_rejection_reason": (
                "hard_candidate_label_free_guarded_rejection_reason"
            ),
        }
        value = row.get(column)
        if value is None and column in candidate_map:
            return row.get(candidate_map[column])
        return value

    rows = [
        {column: audit_value(row, column) for column in columns}
        for row in pairwise_rows
    ]
    pd.DataFrame(rows, columns=columns).to_csv(audit_csv, index=False)

    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    html_rows = []
    for row in rows:
        selected = row.get("selected_hard_seed_backend")
        accepted = row.get("label_free_guarded_candidate_accepted")
        reason = row.get("label_free_guarded_rejection_reason") or ""
        row_class = "fallback"
        if selected == "label-free-group" and accepted is True:
            row_class = "accepted"
        elif "rotation" in str(reason) or "too_few" in str(reason):
            row_class = "rejected"
        cells = "".join(
            f"<td>{html.escape('' if row.get(column) is None else str(row.get(column)))}</td>"
            for column in columns
        )
        html_rows.append(f"<tr class=\"{row_class}\">{cells}</tr>")

    audit_html.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <title>HistoSeg guarded pairwise alignment audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 4px 6px; text-align: left; }}
    th {{ position: sticky; top: 0; background: #f3f4f6; }}
    tr.accepted {{ background: #ecfdf5; }}
    tr.fallback {{ background: #fffbeb; }}
    tr.rejected {{ background: #fef2f2; }}
    .note {{ max-width: 900px; line-height: 1.4; }}
  </style>
</head>
<body>
  <h1>Guarded Pairwise Alignment Audit</h1>
  <p class=\"note\">Label-free group candidates are accepted only when anchor count,
  residual, rotation, and anchor-spread guards pass. Rejected candidates fall back to
  the safer hard seed and preserve slice-local contour labels.</p>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>{''.join(html_rows)}</tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )
    return audit_csv, audit_html


def _import_trimesh():
    try:
        import trimesh

        return trimesh
    except Exception as exc:
        raise ImportError(
            "3D surface mesh export requires trimesh. Install with "
            "`pip install -e '.[threed]'` or `pip install trimesh`."
        ) from exc


def _normalize_mesh_export_formats(mesh_export_formats: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(mesh_export_formats, str):
        raw_formats = [
            value.strip().lower()
            for value in mesh_export_formats.split(",")
            if value.strip()
        ]
    else:
        raw_formats = [str(value).strip().lower() for value in mesh_export_formats]
    formats = tuple(dict.fromkeys(raw_formats))
    if not formats:
        raise ValueError("mesh_export_formats must include at least one format.")
    unsupported = sorted(set(formats) - {"ply", "obj"})
    if unsupported:
        raise ValueError(f"Unsupported mesh export format(s): {', '.join(unsupported)}.")
    return formats


def _smooth_volume_field(
    volume: np.ndarray,
    *,
    mesh_smoothing_sigma_um: float | None,
    voxel_size_um: float,
    z_spacing_um: float,
) -> np.ndarray:
    field = np.asarray(volume, dtype=np.float32)
    if mesh_smoothing_sigma_um is None or mesh_smoothing_sigma_um <= 0:
        return field
    sigma = (
        float(mesh_smoothing_sigma_um) / float(z_spacing_um),
        float(mesh_smoothing_sigma_um) / float(voxel_size_um),
        float(mesh_smoothing_sigma_um) / float(voxel_size_um),
    )
    return gaussian_filter(field, sigma=sigma, mode="constant", cval=0.0)


def _volume_component_stats(
    volume: np.ndarray,
    voxel_size_um: float,
    z_spacing_um: float,
) -> dict[str, float | int]:
    labels, component_count = nd_label(volume, structure=np.ones((3, 3, 3), dtype=bool))
    counts = np.bincount(labels.ravel())
    voxel_count = int(volume.sum())
    voxel_volume = float(voxel_size_um) * float(voxel_size_um) * float(z_spacing_um)
    if component_count == 0:
        largest_component_voxels = 0
    else:
        largest_component_voxels = int(counts[1:].max())
    return {
        "component_count": int(component_count),
        "voxel_count": voxel_count,
        "volume_um3": float(voxel_count * voxel_volume),
        "largest_component_voxels": largest_component_voxels,
    }


def _filter_volume_components(
    volume: np.ndarray,
    *,
    min_mesh_component_volume_um3: float | None,
    voxel_size_um: float,
    z_spacing_um: float,
) -> np.ndarray:
    if min_mesh_component_volume_um3 is None or min_mesh_component_volume_um3 <= 0:
        return np.asarray(volume, dtype=bool)

    labels, component_count = nd_label(volume, structure=np.ones((3, 3, 3), dtype=bool))
    if component_count == 0:
        return np.asarray(volume, dtype=bool)
    voxel_volume = float(voxel_size_um) * float(voxel_size_um) * float(z_spacing_um)
    min_voxels = max(int(math.ceil(float(min_mesh_component_volume_um3) / voxel_volume)), 1)
    counts = np.bincount(labels.ravel())
    keep_labels = np.flatnonzero(counts >= min_voxels)
    keep_labels = keep_labels[keep_labels != 0]
    if len(keep_labels) == 0:
        return np.zeros_like(volume, dtype=bool)
    return np.isin(labels, keep_labels)


def _cleanup_trimesh_mesh(mesh):
    mesh = mesh.copy()
    for method_name in (
        "remove_infinite_values",
        "remove_degenerate_faces",
        "remove_duplicate_faces",
    ):
        method = getattr(mesh, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    try:
        mesh.fill_holes()
    except Exception:
        pass
    try:
        processed = mesh.process(validate=True)
        if processed is not None:
            mesh = processed
    except Exception:
        pass
    return mesh


def _mesh_component_count(mesh) -> int:
    try:
        return len(mesh.split(only_watertight=False))
    except Exception:
        return 1 if len(mesh.faces) else 0


def _export_mesh(mesh, base_path: Path, export_formats: Sequence[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for export_format in export_formats:
        path = base_path.with_suffix(f".{export_format}")
        mesh.export(path)
        paths[export_format] = path
    return paths


def _write_mesh_qc_summary(
    path: Path,
    mesh_payloads: Sequence[Mapping[str, Any]],
    volume_metadata: Mapping[str, Any],
) -> None:
    manifest = [
        {
            "structure": item.get("structure"),
            "vertex_count": item.get("vertex_count"),
            "face_count": item.get("face_count"),
            "surface_area_um2": item.get("surface_area_um2"),
            "volume_um3": item.get("volume_um3"),
            "is_watertight": item.get("is_watertight"),
            "euler_number": item.get("euler_number"),
            "component_count": item.get("component_count"),
            "ply": item.get("ply"),
            "obj": item.get("obj"),
        }
        for item in mesh_payloads
    ]
    summary = {
        "mesh_count": len(mesh_payloads),
        "watertight_count": int(sum(1 for item in mesh_payloads if item.get("is_watertight"))),
        "total_vertex_count": int(sum(int(item.get("vertex_count", 0)) for item in mesh_payloads)),
        "total_face_count": int(sum(int(item.get("face_count", 0)) for item in mesh_payloads)),
        "total_surface_area_um2": float(
            sum(float(item.get("surface_area_um2", 0.0)) for item in mesh_payloads)
        ),
        "total_volume_um3": float(
            sum(float(item.get("volume_um3", 0.0)) for item in mesh_payloads)
        ),
        "volume_metadata": dict(volume_metadata),
        "meshes": manifest,
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


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


def _iter_polygons(geom: Any) -> Iterable[Polygon]:
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        yield from geom.geoms


def _load_aligned_slice_geometries(
    aligned_rows: Sequence[Mapping[str, Any]],
    *,
    group_property: str,
    xenium_pixel_size_um: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in aligned_rows:
        payload = _read_geojson(Path(row["aligned_geojson"]))
        for feature in payload["features"]:
            group = _feature_group(feature.get("properties") or {}, group_property)
            geom = _valid_polygonal_geometry(shape(feature["geometry"]))
            if geom is None:
                continue
            scaled = affinity.scale(
                geom,
                xfact=xenium_pixel_size_um,
                yfact=xenium_pixel_size_um,
                origin=(0.0, 0.0),
            )
            records.append(
                {
                    "slice_order": int(row["order"]),
                    "structure": str(group),
                    "geometry_um": scaled,
                }
            )
    return records


def _combined_scaled_bounds(records: Sequence[Mapping[str, Any]]) -> tuple[float, float, float, float]:
    union = unary_union([record["geometry_um"] for record in records])
    return tuple(map(float, union.bounds))


def _geometry_contains_grid(geom: Any, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    if _contains_xy is not None:
        return np.asarray(_contains_xy(geom, xx, yy), dtype=bool)
    flat = [geom.contains(shape({"type": "Point", "coordinates": [x, y]})) for x, y in zip(xx.ravel(), yy.ravel())]
    return np.asarray(flat, dtype=bool).reshape(xx.shape)


def _write_ascii_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex in vertices:
            handle.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "structure"


def _count_structures_from_aligned_rows(
    aligned_rows: Sequence[Mapping[str, Any]],
    *,
    group_property: str,
) -> int:
    structures: set[str] = set()
    for row in aligned_rows:
        path = row.get("aligned_geojson") or row.get("raw_geojson")
        if not path:
            continue
        try:
            payload = _read_geojson(Path(str(path)))
        except Exception:
            continue
        for feature in payload.get("features", []):
            props = feature.get("properties") or {}
            group = _feature_group(props, group_property)
            if group is not None:
                structures.add(str(group))
    return len(structures)


def _jsonable_config(cfg: ThreeDStackReconstructionConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    for key in ("xenium_root", "out_dir", "contour_manifest", "segmentation_strategy", "merged_h5ad"):
        if payload.get(key) is not None:
            payload[key] = str(payload[key])
    if payload.get("slice_dirs") is not None:
        payload["slice_dirs"] = [str(path) for path in payload["slice_dirs"]]
    return payload


# ---------------------------------------------------------------------------
# Hard-alignment helpers: smooth proxy, affine fallback, drift correction,
# per-structure mixed soft/hard acceptance
# ---------------------------------------------------------------------------


def _smooth_proxy_optimise(
    fixed_union: Any,
    moving_union: Any,
    initial: _SimilarityTransform,
    maxiter: int,
) -> np.ndarray:
    """Pre-optimise using a smooth centroid-distance proxy to warm up the main IoU search.

    Returns the 4-element parameter array [rotation_deg, log_scale, tx, ty].
    """
    params0 = np.array(
        [
            initial.rotation_degrees,
            math.log(max(initial.scale, 1e-9)),
            initial.translate_x,
            initial.translate_y,
        ],
        dtype=float,
    )

    def proxy_objective(params: np.ndarray) -> float:
        transform = _SimilarityTransform(
            origin_x=initial.origin_x,
            origin_y=initial.origin_y,
            rotation_degrees=float(params[0]),
            scale=float(math.exp(params[1])),
            translate_x=float(params[2]),
            translate_y=float(params[3]),
        )
        aligned = _apply_similarity_to_geometry(moving_union, transform)
        dx = float(fixed_union.centroid.x) - float(aligned.centroid.x)
        dy = float(fixed_union.centroid.y) - float(aligned.centroid.y)
        return dx * dx + dy * dy

    result = minimize(
        proxy_objective,
        params0,
        method="Nelder-Mead",
        options={"maxiter": int(maxiter), "xatol": 1e-3, "fatol": 1e-5},
    )
    return result.x if proxy_objective(result.x) <= proxy_objective(params0) else params0


def _group_union(records: Sequence[Any]) -> dict[str, Any]:
    """Return per-group union of record geometries (records have .group and .geometry)."""
    grouped: dict[str, list[Any]] = {}
    for record in records:
        grouped.setdefault(record.group, []).append(record.geometry)
    return {group: unary_union(geoms) for group, geoms in grouped.items()}


def _try_affine_alignment(
    fixed_union: Any,
    moving_union: Any,
    similarity_iou: float,
    maxiter: int = 80,
) -> tuple[np.ndarray | None, float]:
    """Try a 6-DOF affine transform as fallback.

    Returns (affine_params, iou) where affine_params is a 6-element vector
    [a, b, d, e, xoff, yoff] for Shapely's affine_transform, or None if the
    affine transform did not improve over the similarity result.
    """
    tx0 = float(fixed_union.centroid.x) - float(moving_union.centroid.x)
    ty0 = float(fixed_union.centroid.y) - float(moving_union.centroid.y)
    params0 = np.array([1.0, 0.0, 0.0, 1.0, tx0, ty0], dtype=float)

    def objective(params: np.ndarray) -> float:
        try:
            aligned = affinity.affine_transform(moving_union, params.tolist())
            iou = _iou(fixed_union, aligned)
            return -iou if math.isfinite(iou) else 0.0
        except Exception:
            return 0.0

    result = minimize(
        objective,
        params0,
        method="Nelder-Mead",
        options={"maxiter": int(maxiter), "xatol": 1e-3, "fatol": 1e-5},
    )
    best_params = result.x if objective(result.x) <= objective(params0) else params0
    try:
        affine_iou = _iou(fixed_union, affinity.affine_transform(moving_union, best_params.tolist()))
    except Exception:
        return None, similarity_iou

    if math.isfinite(affine_iou) and affine_iou > similarity_iou:
        return best_params, affine_iou
    return None, similarity_iou


def _apply_affine_to_geojson(
    geojson: dict[str, Any],
    params: np.ndarray,
) -> dict[str, Any]:
    """Apply a 6-DOF affine transform to all features in a GeoJSON FeatureCollection."""
    payload = copy.deepcopy(geojson)
    param_list = params.tolist()
    for feature in payload["features"]:
        geom = affinity.affine_transform(shape(feature["geometry"]), param_list)
        feature["geometry"] = mapping(geom)
    return payload


def _build_per_structure_soft_geojson(
    hard_payload: dict[str, Any],
    soft_payload: dict[str, Any],
    group_property: str,
    soft_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build a mixed GeoJSON using soft geometry for structures where IoU improved.

    For each structure in *hard_payload*, if the per-structure IoU after soft
    alignment is ≥ that before soft alignment, the feature geometry is replaced
    with the soft-aligned version; otherwise the hard-aligned geometry is kept.
    """
    per_structure_qc = soft_summary.get("qc", {}).get("per_structure", {})
    mixed = copy.deepcopy(hard_payload)
    soft_features = soft_payload.get("features", [])
    for feature_index, feature in enumerate(mixed.get("features", [])):
        group = str(_feature_group(feature.get("properties") or {}, group_property))
        if feature_index >= len(soft_features):
            continue
        soft_feature = soft_features[feature_index]
        soft_group = str(_feature_group(soft_feature.get("properties") or {}, group_property))
        if soft_group != group:
            continue
        qc = per_structure_qc.get(group, {})
        iou_hard = float(qc.get("iou_hard_before_soft", 0.0))
        iou_soft = float(qc.get("iou_soft_after", 0.0))
        if iou_soft >= iou_hard:
            feature["geometry"] = soft_feature["geometry"]
    return mixed


def _apply_global_drift_correction(
    aligned_paths: Sequence[Path],
    aligned_rows: Sequence[Mapping[str, Any]],
    group_property: str,
) -> None:
    """Remove linear centroid drift accumulated along the pairwise alignment chain.

    Computes the centroid of the union of all structures for each slice, fits a
    linear trend in x and y as a function of slice index, then translates each
    slice by the negative of the residual linear drift (anchored at slice 1, the
    reference).  The aligned GeoJSON files are updated in-place.
    """
    if len(aligned_paths) < 3:
        return

    centroids_x: list[float] = []
    centroids_y: list[float] = []
    for path in aligned_paths:
        try:
            payload = _read_geojson(path)
            geoms = [
                shape(f["geometry"])
                for f in payload.get("features", [])
                if f.get("geometry") is not None
            ]
            if not geoms:
                centroids_x.append(float("nan"))
                centroids_y.append(float("nan"))
                continue
            centroid = unary_union(geoms).centroid
            centroids_x.append(float(centroid.x))
            centroids_y.append(float(centroid.y))
        except Exception:
            centroids_x.append(float("nan"))
            centroids_y.append(float("nan"))

    n = len(aligned_paths)
    indices = np.arange(n, dtype=float)
    cx = np.array(centroids_x, dtype=float)
    cy = np.array(centroids_y, dtype=float)
    valid = np.isfinite(cx) & np.isfinite(cy)
    if valid.sum() < 3:
        return

    # Fit linear trend anchored at index 0 (reference slice).
    x_valid, cx_valid = indices[valid], cx[valid]
    y_valid, cy_valid = indices[valid], cy[valid]
    cx_trend = np.polyfit(x_valid, cx_valid, 1)
    cy_trend = np.polyfit(y_valid, cy_valid, 1)

    for i, path in enumerate(aligned_paths):
        if i == 0:
            continue  # reference slice: no correction.
        if not valid[i]:
            continue
        drift_x = float(np.polyval(cx_trend, i) - np.polyval(cx_trend, 0))
        drift_y = float(np.polyval(cy_trend, i) - np.polyval(cy_trend, 0))
        if abs(drift_x) < 1e-6 and abs(drift_y) < 1e-6:
            continue
        try:
            payload = _read_geojson(path)
            for feature in payload.get("features", []):
                if feature.get("geometry") is None:
                    continue
                geom = affinity.translate(shape(feature["geometry"]), xoff=-drift_x, yoff=-drift_y)
                feature["geometry"] = mapping(geom)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
