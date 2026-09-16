"""Transcript-level local-z orientation correction for HistoSeg 3D stacks."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

try:
    from shapely import contains_xy as _contains_xy
except Exception:  # pragma: no cover - compatibility with Shapely 1.x.
    _contains_xy = None

from .cell_cloud import (
    SliceCellTransform,
    apply_similarity_to_points,
    load_cell_alignment_transforms,
)
from .multislice import discover_xenium_slices


PathLike = Union[str, Path]

LOCAL_Z_ORIENTATION_SCHEMA_VERSION = 2
LOCAL_Z_ORIENTATION_STATES = ("preserve", "reverse")
ALIGNED_TRANSCRIPT_OPTIONAL_COLUMNS = (
    "transcript_id",
    "cell_id",
    "overlaps_nucleus",
    "qv",
    "structure",
    "contour_id",
    "contour_structure",
    "contour_feature_index",
)
ALIGNED_TRANSCRIPT_COLUMNS = (
    "sample_id",
    "slice_order",
    "gene",
    "x_raw_um",
    "y_raw_um",
    "z_raw_um",
    "x_3d_um",
    "y_3d_um",
    "z_3d_um",
    "slice_stack_z_um",
    "local_z_corrected_um",
    "local_z_mid_um",
    "local_z_orientation",
    "local_z_flip_applied",
    *ALIGNED_TRANSCRIPT_OPTIONAL_COLUMNS,
)


@dataclass(frozen=True)
class LocalZOrientationConfig:
    """Configuration for transcript-only local-z orientation inference.

    This workflow does not change HistoSeg contour/cell stack reconstruction.
    It applies the existing aligned-stack XY transforms to transcript coordinates,
    infers whether each slice's internal transcript z-axis should be preserved or
    reversed, and writes a transcript-level 3D table.
    """

    xenium_root: PathLike
    stack_root: PathLike
    out_dir: PathLike | None = None
    sample_glob: str = "*"
    transcript_relpath: str = "transcripts.parquet"
    gene_column: str | None = None
    x_column: str | None = None
    y_column: str | None = None
    z_column: str | None = None
    transcript_id_column: str | None = None
    structure_column: str | None = None
    pixel_size_um: float = 0.2125
    z_band_fraction: float = 0.25
    min_band_transcripts: int = 10
    max_signature_genes: int = 256
    low_confidence_margin: float = 0.02
    orientation_spatial_unit: str = "auto"
    contour_group_property: str = "structure"
    contour_min_transcripts: int = 50
    contour_match_min_iou: float = 0.01
    contour_match_max_distance_um: float = 120.0
    orientation_bootstrap_iterations: int = 100
    orientation_bootstrap_seed: int = 0
    orientation_bootstrap_support_threshold: float = 0.7
    contour_max_global_fallback_fraction: float = 0.33
    vertical_qc_backend: str = "ovrlpy"
    apply_local_z_flip: bool = True
    ovrlpy_kde_bandwidth: float = 2.5
    ovrlpy_n_components: int = 20
    ovrlpy_n_workers: int = 1
    ovrlpy_fit_umap: bool = True
    ovrlpy_min_transcripts: float = 10.0
    doublet_min_signal: float = 4.0
    doublet_integrity_sigma: float = 1.0
    doublet_exclusion_radius_um: float = 20.0
    chunk_size: int = 100000


@dataclass
class LocalZOrientationResult:
    out_dir: Path
    manifest_csv: Path
    aligned_transcripts_parquet: Path
    biological_report_html: Path
    marker_gradients_csv: Path
    vertical_qc_dir: Path
    slice_count: int
    transcript_count: int
    best_score: float
    second_score: float
    confidence_margin: float
    low_confidence: bool
    contour_pair_continuity_csv: Path | None = None
    contour_bootstrap_support_csv: Path | None = None


@dataclass(frozen=True)
class _TranscriptSource:
    sample_id: str
    sample_dir: Path
    xenium_dir: Path


@dataclass
class _OvrlpyQC:
    status: str
    doublets_csv: Path | None = None
    doublet_count: int = 0
    signal_integrity_npy: Path | None = None
    signal_map_npy: Path | None = None
    signal_integrity_mean: float = math.nan
    signal_integrity_p05: float = math.nan
    signal_integrity_p95: float = math.nan
    signal_mean: float = math.nan
    signal_p05: float = math.nan
    signal_p95: float = math.nan
    error: str | None = None


@dataclass(frozen=True)
class _ContourFeature:
    sample_id: str
    order: int
    contour_id: str
    structure: str | None
    feature_index: int
    polygon_index: int
    geometry: BaseGeometry
    area_native: float
    area_um2: float
    centroid_x_native: float
    centroid_y_native: float


@dataclass(frozen=True)
class _ContourSet:
    sample_id: str
    order: int
    aligned_geojson: Path | None
    status: str
    features: tuple[_ContourFeature, ...] = ()
    error: str | None = None


@dataclass
class _ContourLayerProfile:
    sample_id: str
    order: int
    contour_id: str
    contour_structure: str | None
    contour_feature_index: int
    contour_polygon_index: int
    area_um2: float
    centroid_x_um: float
    centroid_y_um: float
    transcript_count: int
    signature_transcript_count: int
    excluded_doublet_transcripts: int
    doublet_burden: float
    low_band_count: int
    high_band_count: int
    raw_low_signature: dict[str, float]
    raw_high_signature: dict[str, float]
    usable: bool
    qc_flag: str
    geometry: BaseGeometry


@dataclass(frozen=True)
class _ContourMatch:
    prev: _ContourLayerProfile
    next: _ContourLayerProfile
    match_method: str
    iou: float
    centroid_distance_um: float
    weight: float


@dataclass(frozen=True)
class _EdgeScoreDetail:
    score: float
    backend: str
    used_global_fallback: bool
    contour_pair_count: int
    reason: str


@dataclass(frozen=True)
class _EdgeMatchContext:
    prev_sample_id: str
    next_sample_id: str
    matches: tuple[_ContourMatch, ...]
    fallback_reason: str | None = None


@dataclass
class _SliceLocalZSummary:
    sample_id: str
    order: int
    stack_z_um: float
    transcript_path: Path
    z_min_um: float
    z_max_um: float
    z_mid_um: float
    transcript_count: int
    signature_transcript_count: int
    excluded_doublet_transcripts: int
    low_band_count: int
    high_band_count: int
    raw_low_signature: dict[str, float]
    raw_high_signature: dict[str, float]
    ovrlpy_qc: _OvrlpyQC
    contour_path: Path | None = None
    contour_status: str = "not_requested"
    contour_error: str | None = None
    contour_profile_count: int = 0
    usable_contour_profile_count: int = 0
    contour_assigned_transcripts: int = 0
    contour_excluded_doublet_transcripts: int = 0
    contour_profiles: tuple[_ContourLayerProfile, ...] = ()


@dataclass(frozen=True)
class _OrientationInference:
    states_by_sample: dict[str, str]
    best_score: float
    second_score: float
    confidence_margin: float
    low_confidence: bool
    edge_scores: dict[tuple[str, str], float]
    scoring_backend: str = "global"
    edge_details: dict[tuple[str, str], _EdgeScoreDetail] | None = None
    contour_pair_rows: tuple[dict[str, Any], ...] = ()
    bootstrap_support_by_sample: dict[str, float] | None = None
    bootstrap_rows: tuple[dict[str, Any], ...] = ()
    fallback_edge_count: int = 0
    fallback_edge_fraction: float = 0.0
    low_confidence_reasons: tuple[str, ...] = ()


def run_local_z_orientation_correction(
    cfg: LocalZOrientationConfig,
) -> LocalZOrientationResult:
    """Infer and apply transcript local-z orientation for an aligned HistoSeg stack."""

    _validate_config(cfg)
    stack_root = Path(cfg.stack_root).expanduser().resolve()
    xenium_root = Path(cfg.xenium_root).expanduser().resolve()
    out_dir = Path(cfg.out_dir).expanduser().resolve() if cfg.out_dir else stack_root
    out_dir.mkdir(parents=True, exist_ok=True)
    vertical_qc_dir = out_dir / "vertical_qc"
    vertical_qc_dir.mkdir(parents=True, exist_ok=True)

    transforms = load_cell_alignment_transforms(
        stack_root,
        pixel_size_um=cfg.pixel_size_um,
        chunk_size=cfg.chunk_size,
    )
    sources = _discover_transcript_sources(xenium_root, sample_glob=cfg.sample_glob)
    contour_sets = _load_aligned_contour_sets(stack_root, transforms, cfg)

    summaries: list[_SliceLocalZSummary] = []
    for transform in sorted(transforms, key=lambda item: item.order):
        transcript_path = _find_transcript_path(transform.sample_id, sources, xenium_root, cfg)
        summary = _summarize_slice_transcripts(
            transform=transform,
            transcript_path=transcript_path,
            vertical_qc_dir=vertical_qc_dir,
            contour_set=contour_sets.get(str(transform.sample_id)),
            cfg=cfg,
        )
        summaries.append(summary)

    inference = _infer_orientation_sequence(summaries, cfg)
    manifest = _build_manifest_frame(summaries, inference, cfg)
    manifest_path = out_dir / "local_z_orientation_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    aligned_path = out_dir / "aligned_transcripts_3d.parquet"
    transcript_count = _write_aligned_transcripts(
        summaries=summaries,
        transforms=transforms,
        inference=inference,
        output_parquet=aligned_path,
        cfg=cfg,
    )

    marker_gradients = _build_marker_gradient_frame(summaries, inference)
    marker_gradients_path = vertical_qc_dir / "top_bottom_marker_gradients.csv"
    marker_gradients.to_csv(marker_gradients_path, index=False)

    contour_pair_path = vertical_qc_dir / "contour_pair_continuity.csv"
    _contour_pair_frame(inference).to_csv(contour_pair_path, index=False)
    bootstrap_support_path = vertical_qc_dir / "contour_bootstrap_support.csv"
    _bootstrap_support_frame(inference, summaries).to_csv(bootstrap_support_path, index=False)

    report_path = vertical_qc_dir / "biological_z_report.html"
    _write_biological_report(
        report_path,
        manifest=manifest,
        marker_gradients=marker_gradients,
        inference=inference,
        cfg=cfg,
    )

    return LocalZOrientationResult(
        out_dir=out_dir,
        manifest_csv=manifest_path,
        aligned_transcripts_parquet=aligned_path,
        biological_report_html=report_path,
        marker_gradients_csv=marker_gradients_path,
        vertical_qc_dir=vertical_qc_dir,
        slice_count=len(summaries),
        transcript_count=transcript_count,
        best_score=inference.best_score,
        second_score=inference.second_score,
        confidence_margin=inference.confidence_margin,
        low_confidence=inference.low_confidence,
        contour_pair_continuity_csv=contour_pair_path,
        contour_bootstrap_support_csv=bootstrap_support_path,
    )


def _validate_config(cfg: LocalZOrientationConfig) -> None:
    if cfg.vertical_qc_backend not in {"none", "ovrlpy"}:
        raise ValueError("vertical_qc_backend must be 'none' or 'ovrlpy'.")
    if cfg.pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be greater than 0.")
    if not (0.0 < cfg.z_band_fraction < 0.5):
        raise ValueError("z_band_fraction must be greater than 0 and less than 0.5.")
    if cfg.min_band_transcripts < 1:
        raise ValueError("min_band_transcripts must be at least 1.")
    if cfg.max_signature_genes < 1:
        raise ValueError("max_signature_genes must be at least 1.")
    if cfg.low_confidence_margin < 0:
        raise ValueError("low_confidence_margin must be non-negative.")
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
    if not (0.0 <= cfg.orientation_bootstrap_support_threshold <= 1.0):
        raise ValueError("orientation_bootstrap_support_threshold must be in [0, 1].")
    if not (0.0 <= cfg.contour_max_global_fallback_fraction <= 1.0):
        raise ValueError("contour_max_global_fallback_fraction must be in [0, 1].")
    if cfg.ovrlpy_kde_bandwidth <= 0:
        raise ValueError("ovrlpy_kde_bandwidth must be greater than 0.")
    if cfg.ovrlpy_n_components < 1:
        raise ValueError("ovrlpy_n_components must be at least 1.")
    if cfg.ovrlpy_n_workers < 1:
        raise ValueError("ovrlpy_n_workers must be at least 1.")
    if cfg.ovrlpy_min_transcripts <= 0:
        raise ValueError("ovrlpy_min_transcripts must be greater than 0.")
    if cfg.doublet_exclusion_radius_um < 0:
        raise ValueError("doublet_exclusion_radius_um must be non-negative.")
    if cfg.chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")


def _discover_transcript_sources(
    xenium_root: Path,
    *,
    sample_glob: str,
) -> dict[str, list[_TranscriptSource]]:
    sources: dict[str, list[_TranscriptSource]] = {}
    try:
        for item in discover_xenium_slices(xenium_root, sample_glob=sample_glob):
            sources.setdefault(item.sample_id, []).append(
                _TranscriptSource(
                    sample_id=item.sample_id,
                    sample_dir=item.sample_dir,
                    xenium_dir=item.xenium_dir,
                )
            )
    except Exception:
        pass

    candidates = [xenium_root]
    if xenium_root.exists() and xenium_root.is_dir():
        candidates.extend(child for child in xenium_root.glob(sample_glob) if child.is_dir())
    for candidate in candidates:
        sample_id = _sample_id_from_transcript_source(candidate)
        sources.setdefault(candidate.name, []).append(
            _TranscriptSource(
                sample_id=sample_id,
                sample_dir=candidate,
                xenium_dir=candidate,
            )
        )
        if sample_id != candidate.name:
            sources.setdefault(sample_id, []).append(
                _TranscriptSource(
                    sample_id=sample_id,
                    sample_dir=candidate,
                    xenium_dir=candidate,
                )
            )
    return sources


def _sample_id_from_transcript_source(path: Path) -> str:
    name = path.name
    if name.endswith(".pyxenium.slide.zarr"):
        return name[: -len(".pyxenium.slide.zarr")]
    return name


def _find_transcript_path(
    sample_id: str,
    sources: Mapping[str, Sequence[_TranscriptSource]],
    xenium_root: Path,
    cfg: LocalZOrientationConfig,
) -> Path:
    candidate_dirs: list[Path] = []
    for source in sources.get(sample_id, ()):
        candidate_dirs.extend([source.xenium_dir, source.sample_dir])
    candidate_dirs.extend([xenium_root / sample_id, xenium_root])

    relpaths = [cfg.transcript_relpath] if cfg.transcript_relpath else []
    relpaths.extend(
        [
            "transcripts.parquet",
            "transcripts.csv.gz",
            "transcripts.csv",
            "transcripts.tsv.gz",
            "transcripts.tsv",
        ]
    )
    seen: set[Path] = set()
    for base in candidate_dirs:
        if base.exists() and base.is_dir() and base.name.endswith(".pyxenium.slide.zarr"):
            return base
        for relpath in relpaths:
            candidate = (base / relpath).expanduser().resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Could not find a transcript table for sample {sample_id!r} below {xenium_root}."
    )


def _load_aligned_contour_sets(
    stack_root: Path,
    transforms: Sequence[SliceCellTransform],
    cfg: LocalZOrientationConfig,
) -> dict[str, _ContourSet]:
    if cfg.orientation_spatial_unit == "global":
        return {}
    manifest_path = stack_root / "aligned_slice_manifest.csv"
    if not manifest_path.exists():
        return {}
    try:
        manifest = pd.read_csv(manifest_path)
    except Exception:
        return {}
    if "sample_id" not in manifest.columns:
        return {}

    rows_by_sample = {str(row["sample_id"]): row for _, row in manifest.iterrows()}
    contour_sets: dict[str, _ContourSet] = {}
    for transform in transforms:
        sample_id = str(transform.sample_id)
        row = rows_by_sample.get(sample_id)
        aligned_path = _resolve_aligned_geojson_path(
            row.get("aligned_geojson") if row is not None else None,
            stack_root=stack_root,
            sample_id=sample_id,
        )
        if aligned_path is None:
            contour_sets[sample_id] = _ContourSet(
                sample_id=sample_id,
                order=int(transform.order),
                aligned_geojson=None,
                status="missing_aligned_geojson",
            )
            continue
        try:
            features = _read_contour_features(
                aligned_path,
                sample_id=sample_id,
                order=int(transform.order),
                cfg=cfg,
            )
        except Exception as exc:
            contour_sets[sample_id] = _ContourSet(
                sample_id=sample_id,
                order=int(transform.order),
                aligned_geojson=aligned_path,
                status="load_failed",
                error=str(exc),
            )
            continue
        contour_sets[sample_id] = _ContourSet(
            sample_id=sample_id,
            order=int(transform.order),
            aligned_geojson=aligned_path,
            status="ok" if features else "empty",
            features=tuple(features),
        )
    return contour_sets


def _resolve_aligned_geojson_path(
    value: Any,
    *,
    stack_root: Path,
    sample_id: str,
) -> Path | None:
    candidates: list[Path] = []
    if _has_value(value):
        raw = Path(str(value))
        candidates.append(raw)
        if not raw.is_absolute():
            candidates.append(stack_root / raw)
        candidates.append(stack_root / raw.name)
        candidates.append(stack_root / "aligned_contours" / raw.name)
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            continue
    matches = sorted((stack_root / "aligned_contours").glob(f"*{sample_id}*aligned.geojson"))
    return matches[0].resolve() if matches else None


def _read_contour_features(
    path: Path,
    *,
    sample_id: str,
    order: int,
    cfg: LocalZOrientationConfig,
) -> list[_ContourFeature]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
        raise ValueError(f"{path} must be a GeoJSON FeatureCollection.")

    features: list[_ContourFeature] = []
    for feature_index, feature in enumerate(payload["features"]):
        if feature.get("geometry") is None:
            continue
        properties = feature.get("properties") or {}
        structure = _feature_group(properties, cfg.contour_group_property)
        try:
            geometry = shape(feature["geometry"])
        except Exception:
            continue
        for polygon_index, polygon in enumerate(_iter_polygonal_geometries(geometry)):
            if polygon.is_empty or float(polygon.area) <= 0:
                continue
            centroid = polygon.centroid
            contour_id = f"{sample_id}:feature_{feature_index:05d}:polygon_{polygon_index:03d}"
            features.append(
                _ContourFeature(
                    sample_id=sample_id,
                    order=order,
                    contour_id=contour_id,
                    structure=str(structure) if structure is not None else None,
                    feature_index=int(feature_index),
                    polygon_index=int(polygon_index),
                    geometry=polygon,
                    area_native=float(polygon.area),
                    area_um2=float(polygon.area) * float(cfg.pixel_size_um) ** 2,
                    centroid_x_native=float(centroid.x),
                    centroid_y_native=float(centroid.y),
                )
            )
    return features


def _iter_polygonal_geometries(geometry: BaseGeometry) -> list[BaseGeometry]:
    if geometry.is_empty:
        return []
    if not geometry.is_valid:
        try:
            geometry = geometry.buffer(0)
        except Exception:
            return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        polygons: list[BaseGeometry] = []
        for part in geometry.geoms:
            polygons.extend(_iter_polygonal_geometries(part))
        return polygons
    return []


def _feature_group(properties: Mapping[str, Any], group_property: str) -> Any | None:
    if group_property in properties:
        return properties[group_property]
    if "." in group_property:
        value: Any = properties
        for part in group_property.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return None
            value = value[part]
        return value
    if group_property == "structure" and "assigned_structure" in properties:
        return properties["assigned_structure"]
    return None


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _summarize_slice_transcripts(
    *,
    transform: SliceCellTransform,
    transcript_path: Path,
    vertical_qc_dir: Path,
    contour_set: _ContourSet | None,
    cfg: LocalZOrientationConfig,
) -> _SliceLocalZSummary:
    raw = _read_transcript_table(transcript_path)
    transcripts = _normalize_transcripts(raw, cfg)
    if transcripts.empty:
        raise ValueError(f"{transcript_path} does not contain valid transcript x/y/z rows.")

    sample_qc_dir = vertical_qc_dir / str(transform.sample_id)
    sample_qc_dir.mkdir(parents=True, exist_ok=True)
    ovrlpy_qc = _run_vertical_qc(transcripts, sample_qc_dir, cfg)
    doublet_mask = _doublet_region_mask(transcripts, ovrlpy_qc, cfg)
    filtered = transcripts.loc[~doublet_mask].reset_index(drop=True)
    excluded_count = int(doublet_mask.sum())
    if filtered.empty:
        filtered = transcripts
        excluded_count = 0
    signatures = _build_z_band_signatures(filtered, cfg)
    contour_profiles, contour_assigned, contour_excluded = _build_contour_layer_profiles(
        transform=transform,
        transcripts=transcripts,
        doublet_mask=doublet_mask,
        contour_set=contour_set,
        sample_qc_dir=sample_qc_dir,
        cfg=cfg,
    )

    return _SliceLocalZSummary(
        sample_id=str(transform.sample_id),
        order=int(transform.order),
        stack_z_um=float(transform.z_um),
        transcript_path=transcript_path,
        z_min_um=float(transcripts["z_raw_um"].min()),
        z_max_um=float(transcripts["z_raw_um"].max()),
        z_mid_um=float((transcripts["z_raw_um"].min() + transcripts["z_raw_um"].max()) / 2.0),
        transcript_count=int(len(transcripts)),
        signature_transcript_count=int(len(filtered)),
        excluded_doublet_transcripts=int(excluded_count),
        low_band_count=int(signatures["low_count"]),
        high_band_count=int(signatures["high_count"]),
        raw_low_signature=signatures["low_signature"],
        raw_high_signature=signatures["high_signature"],
        ovrlpy_qc=ovrlpy_qc,
        contour_path=contour_set.aligned_geojson if contour_set is not None else None,
        contour_status=contour_set.status if contour_set is not None else "not_available",
        contour_error=contour_set.error if contour_set is not None else None,
        contour_profile_count=int(len(contour_profiles)),
        usable_contour_profile_count=int(sum(profile.usable for profile in contour_profiles)),
        contour_assigned_transcripts=int(contour_assigned),
        contour_excluded_doublet_transcripts=int(contour_excluded),
        contour_profiles=tuple(contour_profiles),
    )


def _read_transcript_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if path.is_dir() and path.name.endswith(".pyxenium.slide.zarr"):
        return _read_pyxenium_slide_transcripts(path)
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t")
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported transcript table format: {path}")


def _read_pyxenium_slide_transcripts(path: Path) -> pd.DataFrame:
    try:
        from pyXenium import read_slide
    except Exception as exc:
        raise ImportError(
            "Reading pyXeniumSlide zarr transcript stores requires pyXenium."
        ) from exc

    slide = read_slide(path)
    try:
        transcripts = slide.points["transcripts"]
    except Exception as exc:
        fallback = _read_pyxenium_slide_sidecar_transcripts(path)
        if fallback is not None:
            return fallback
        raise ValueError(f"{path} does not contain a transcripts point source.") from exc

    frame = transcripts.copy()
    if "valid" in frame.columns:
        frame = frame.loc[frame["valid"].astype(bool)].copy()
    rename = {
        "gene_name": "gene",
        "quality_score": "qv",
    }
    present = {src: dst for src, dst in rename.items() if src in frame.columns and dst not in frame.columns}
    if present:
        frame = frame.rename(columns=present)
    return frame


def _read_pyxenium_slide_sidecar_transcripts(path: Path) -> pd.DataFrame | None:
    sample_id = _sample_id_from_transcript_source(path)
    candidate_dirs = [
        path,
        path.parent / sample_id,
    ]
    names = [
        "transcripts.parquet",
        "transcripts.csv.gz",
        "transcripts.csv",
        "transcripts.tsv.gz",
        "transcripts.tsv",
    ]
    for base in candidate_dirs:
        for name in names:
            candidate = base / name
            if candidate.exists() and candidate.is_file():
                return _read_transcript_table(candidate)
    return None


def _normalize_transcripts(frame: pd.DataFrame, cfg: LocalZOrientationConfig) -> pd.DataFrame:
    gene_col = _resolve_column(frame, cfg.gene_column, ["gene", "feature_name", "gene_name"], "gene")
    x_col = _resolve_column(frame, cfg.x_column, ["x", "x_um", "x_location", "x_centroid"], "x")
    y_col = _resolve_column(frame, cfg.y_column, ["y", "y_um", "y_location", "y_centroid"], "y")
    z_col = _resolve_column(frame, cfg.z_column, ["z", "z_um", "z_location"], "z")
    transcript_id_col = _optional_column(
        frame,
        cfg.transcript_id_column,
        ["transcript_id", "id", "molecule_id"],
    )
    structure_col = _optional_column(frame, cfg.structure_column, ["structure"])

    result = pd.DataFrame(
        {
            "gene": frame[gene_col].astype(str),
            "x_raw_um": pd.to_numeric(frame[x_col], errors="coerce"),
            "y_raw_um": pd.to_numeric(frame[y_col], errors="coerce"),
            "z_raw_um": pd.to_numeric(frame[z_col], errors="coerce"),
        }
    )
    if transcript_id_col is not None:
        result["transcript_id"] = frame[transcript_id_col].astype(str)
    if structure_col is not None:
        result["structure"] = frame[structure_col].astype(str)
    optional_columns = {
        "cell_id": _optional_column(frame, None, ["cell_id"]),
        "overlaps_nucleus": _optional_column(frame, None, ["overlaps_nucleus"]),
        "qv": _optional_column(frame, None, ["qv", "quality_score"]),
    }
    for optional, column in optional_columns.items():
        if column is not None:
            result[optional] = frame[column].to_numpy()

    finite = np.isfinite(result[["x_raw_um", "y_raw_um", "z_raw_um"]].to_numpy(dtype=float)).all(axis=1)
    non_empty_gene = result["gene"].astype(str).str.strip() != ""
    return result.loc[finite & non_empty_gene].reset_index(drop=True)


def _resolve_column(
    frame: pd.DataFrame,
    requested: str | None,
    candidates: Sequence[str],
    label: str,
) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"Transcript table is missing requested {label} column {requested!r}.")
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"Transcript table is missing a {label} column. Tried: {', '.join(candidates)}."
    )


def _optional_column(
    frame: pd.DataFrame,
    requested: str | None,
    candidates: Sequence[str],
) -> str | None:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"Transcript table is missing requested column {requested!r}.")
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _run_vertical_qc(
    transcripts: pd.DataFrame,
    sample_qc_dir: Path,
    cfg: LocalZOrientationConfig,
) -> _OvrlpyQC:
    sample_qc_dir.mkdir(parents=True, exist_ok=True)
    if cfg.vertical_qc_backend == "none":
        empty_doublets = sample_qc_dir / "doublets.csv"
        pd.DataFrame(columns=["x", "y"]).to_csv(empty_doublets, index=False)
        return _OvrlpyQC(status="skipped", doublets_csv=empty_doublets)

    cached_qc = _load_cached_vertical_qc(sample_qc_dir)
    if cached_qc is not None:
        return cached_qc

    try:
        import ovrlpy
    except Exception as exc:
        raise ImportError(
            "Local-z orientation was requested with vertical_qc_backend='ovrlpy', "
            "but ovrlpy is not installed. Install the optional vertical QC dependency "
            "or use vertical_qc_backend='none'."
        ) from exc

    coordinate_df = transcripts.loc[:, ["gene", "x_raw_um", "y_raw_um", "z_raw_um"]].rename(
        columns={"x_raw_um": "x", "y_raw_um": "y", "z_raw_um": "z"}
    )
    try:
        dataset = ovrlpy.Ovrlp(
            coordinate_df,
            KDE_bandwidth=float(cfg.ovrlpy_kde_bandwidth),
            n_components=int(cfg.ovrlpy_n_components),
            n_workers=int(cfg.ovrlpy_n_workers),
        )
        dataset.analyse(
            min_transcripts=float(cfg.ovrlpy_min_transcripts),
            fit_umap=bool(cfg.ovrlpy_fit_umap),
        )
    except Exception as exc:
        return _OvrlpyQC(status="analysis_failed", error=str(exc))

    integrity_value = getattr(dataset, "integrity_map", None)
    signal_value = getattr(dataset, "signal_map", None)
    integrity_path = _save_dataset_array(integrity_value, sample_qc_dir / "signal_integrity.npy")
    signal_path = _save_dataset_array(signal_value, sample_qc_dir / "signal_map.npy")
    integrity_stats = _array_stats(integrity_value)
    signal_stats = _array_stats(signal_value)

    try:
        doublets = dataset.detect_doublets(
            min_signal=float(cfg.doublet_min_signal),
            integrity_sigma=float(cfg.doublet_integrity_sigma),
        )
        doublets_df = _as_pandas_frame(doublets)
    except Exception as exc:
        doublets_df = pd.DataFrame([{"error": str(exc)}])
        status = "doublet_detection_failed"
    else:
        status = "ok"

    doublets_path = sample_qc_dir / "doublets.csv"
    doublets_df.to_csv(doublets_path, index=False)
    return _OvrlpyQC(
        status=status,
        doublets_csv=doublets_path,
        doublet_count=int(len(doublets_df)) if {"x", "y"}.issubset(doublets_df.columns) else 0,
        signal_integrity_npy=integrity_path,
        signal_map_npy=signal_path,
        signal_integrity_mean=integrity_stats["mean"],
        signal_integrity_p05=integrity_stats["p05"],
        signal_integrity_p95=integrity_stats["p95"],
        signal_mean=signal_stats["mean"],
        signal_p05=signal_stats["p05"],
        signal_p95=signal_stats["p95"],
    )


def _load_cached_vertical_qc(sample_qc_dir: Path) -> _OvrlpyQC | None:
    integrity_path = sample_qc_dir / "signal_integrity.npy"
    signal_path = sample_qc_dir / "signal_map.npy"
    doublets_path = sample_qc_dir / "doublets.csv"
    if not (integrity_path.exists() and signal_path.exists() and doublets_path.exists()):
        return None
    try:
        integrity = np.load(integrity_path, mmap_mode="r")
        signal = np.load(signal_path, mmap_mode="r")
        doublets = pd.read_csv(doublets_path)
    except Exception:
        return None
    integrity_stats = _array_stats(integrity)
    signal_stats = _array_stats(signal)
    return _OvrlpyQC(
        status="ok",
        doublets_csv=doublets_path,
        doublet_count=int(len(doublets)) if {"x", "y"}.issubset(doublets.columns) else 0,
        signal_integrity_npy=integrity_path,
        signal_map_npy=signal_path,
        signal_integrity_mean=integrity_stats["mean"],
        signal_integrity_p05=integrity_stats["p05"],
        signal_integrity_p95=integrity_stats["p95"],
        signal_mean=signal_stats["mean"],
        signal_p05=signal_stats["p05"],
        signal_p95=signal_stats["p95"],
    )


def _save_dataset_array(value: Any, output_path: Path) -> Path | None:
    if value is None:
        return None
    try:
        np.save(output_path, np.asarray(value))
    except Exception:
        return None
    return output_path


def _array_stats(value: Any) -> dict[str, float]:
    if value is None:
        return {"mean": math.nan, "p05": math.nan, "p95": math.nan}
    try:
        array = np.asarray(value, dtype=float)
    except Exception:
        return {"mean": math.nan, "p05": math.nan, "p95": math.nan}
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"mean": math.nan, "p05": math.nan, "p95": math.nan}
    return {
        "mean": float(np.mean(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def _as_pandas_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def _filter_doublet_regions(
    transcripts: pd.DataFrame,
    ovrlpy_qc: _OvrlpyQC,
    cfg: LocalZOrientationConfig,
) -> tuple[pd.DataFrame, int]:
    near_doublet = _doublet_region_mask(transcripts, ovrlpy_qc, cfg)
    if not near_doublet.any():
        return transcripts, 0
    kept = transcripts.loc[~near_doublet].reset_index(drop=True)
    if kept.empty:
        return transcripts, 0
    return kept, int(near_doublet.sum())


def _doublet_region_mask(
    transcripts: pd.DataFrame,
    ovrlpy_qc: _OvrlpyQC,
    cfg: LocalZOrientationConfig,
) -> np.ndarray:
    if (
        cfg.doublet_exclusion_radius_um <= 0
        or ovrlpy_qc.doublets_csv is None
        or not ovrlpy_qc.doublets_csv.exists()
    ):
        return np.zeros(len(transcripts), dtype=bool)
    doublets = pd.read_csv(ovrlpy_qc.doublets_csv)
    if doublets.empty:
        return np.zeros(len(transcripts), dtype=bool)
    x_col = _optional_column(doublets, None, ["x", "x_um", "x_location"])
    y_col = _optional_column(doublets, None, ["y", "y_um", "y_location"])
    if x_col is None or y_col is None:
        return np.zeros(len(transcripts), dtype=bool)
    xy = transcripts[["x_raw_um", "y_raw_um"]].to_numpy(dtype=float)
    doublet_xy = doublets[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
    if len(doublet_xy) == 0:
        return np.zeros(len(transcripts), dtype=bool)
    tree = cKDTree(doublet_xy)
    return np.asarray(
        tree.query_ball_point(xy, r=float(cfg.doublet_exclusion_radius_um), return_length=True)
    ) > 0


def _build_contour_layer_profiles(
    *,
    transform: SliceCellTransform,
    transcripts: pd.DataFrame,
    doublet_mask: np.ndarray,
    contour_set: _ContourSet | None,
    sample_qc_dir: Path,
    cfg: LocalZOrientationConfig,
) -> tuple[list[_ContourLayerProfile], int, int]:
    if contour_set is None or contour_set.status != "ok" or not contour_set.features:
        if cfg.orientation_spatial_unit in {"auto", "contour"}:
            _empty_contour_profile_frame().to_parquet(
                sample_qc_dir / "contour_layer_profiles.parquet",
                index=False,
            )
        return [], 0, 0

    xy_native = _aligned_xy_native(transcripts, transform, cfg)
    assignments = _assign_contours_to_points(xy_native, contour_set.features)
    assigned = assignments["contour_id"].notna().to_numpy()
    doublet_mask = np.asarray(doublet_mask, dtype=bool)
    rows: list[dict[str, Any]] = []
    profiles: list[_ContourLayerProfile] = []

    for feature in contour_set.features:
        in_contour = assignments["contour_id"].eq(feature.contour_id).fillna(False)
        in_contour_mask = in_contour.to_numpy(dtype=bool)
        contour_count = int(in_contour_mask.sum())
        excluded_count = int((in_contour_mask & doublet_mask).sum())
        kept = transcripts.loc[in_contour_mask & ~doublet_mask].reset_index(drop=True)
        doublet_burden = float(excluded_count / contour_count) if contour_count > 0 else 0.0
        usable = bool(len(kept) >= int(cfg.contour_min_transcripts))
        if len(kept) == 0:
            signatures = {
                "low_count": 0,
                "high_count": 0,
                "low_signature": {},
                "high_signature": {},
            }
        else:
            signatures = _build_z_band_signatures(kept, cfg)
            usable = usable and signatures["low_count"] > 0 and signatures["high_count"] > 0
        qc_flag = "ok" if usable else "too_few_transcripts"
        profile = _ContourLayerProfile(
            sample_id=feature.sample_id,
            order=feature.order,
            contour_id=feature.contour_id,
            contour_structure=feature.structure,
            contour_feature_index=feature.feature_index,
            contour_polygon_index=feature.polygon_index,
            area_um2=feature.area_um2,
            centroid_x_um=feature.centroid_x_native * float(cfg.pixel_size_um),
            centroid_y_um=feature.centroid_y_native * float(cfg.pixel_size_um),
            transcript_count=contour_count,
            signature_transcript_count=int(len(kept)),
            excluded_doublet_transcripts=excluded_count,
            doublet_burden=doublet_burden,
            low_band_count=int(signatures["low_count"]),
            high_band_count=int(signatures["high_count"]),
            raw_low_signature=signatures["low_signature"],
            raw_high_signature=signatures["high_signature"],
            usable=usable,
            qc_flag=qc_flag,
            geometry=feature.geometry,
        )
        profiles.append(profile)
        rows.append(_contour_profile_row(profile))

    frame = pd.DataFrame(rows) if rows else _empty_contour_profile_frame()
    frame.to_parquet(sample_qc_dir / "contour_layer_profiles.parquet", index=False)
    return profiles, int(assigned.sum()), int((assigned & doublet_mask).sum())


def _contour_profile_row(profile: _ContourLayerProfile) -> dict[str, Any]:
    return {
        "sample_id": profile.sample_id,
        "order": profile.order,
        "contour_id": profile.contour_id,
        "contour_structure": profile.contour_structure,
        "contour_feature_index": profile.contour_feature_index,
        "contour_polygon_index": profile.contour_polygon_index,
        "area_um2": profile.area_um2,
        "centroid_x_um": profile.centroid_x_um,
        "centroid_y_um": profile.centroid_y_um,
        "transcript_count": profile.transcript_count,
        "signature_transcript_count": profile.signature_transcript_count,
        "excluded_doublet_transcripts": profile.excluded_doublet_transcripts,
        "doublet_burden": profile.doublet_burden,
        "low_band_count": profile.low_band_count,
        "high_band_count": profile.high_band_count,
        "raw_low_signature_json": json.dumps(profile.raw_low_signature, sort_keys=True),
        "raw_high_signature_json": json.dumps(profile.raw_high_signature, sort_keys=True),
        "usable": profile.usable,
        "qc_flag": profile.qc_flag,
    }


def _empty_contour_profile_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_id",
            "order",
            "contour_id",
            "contour_structure",
            "contour_feature_index",
            "contour_polygon_index",
            "area_um2",
            "centroid_x_um",
            "centroid_y_um",
            "transcript_count",
            "signature_transcript_count",
            "excluded_doublet_transcripts",
            "doublet_burden",
            "low_band_count",
            "high_band_count",
            "raw_low_signature_json",
            "raw_high_signature_json",
            "usable",
            "qc_flag",
        ]
    )


def _aligned_xy_native(
    frame: pd.DataFrame,
    transform: SliceCellTransform,
    cfg: LocalZOrientationConfig,
) -> np.ndarray:
    xy_native = frame[["x_raw_um", "y_raw_um"]].to_numpy(dtype=float) / float(cfg.pixel_size_um)
    if transform.hard is not None:
        xy_native = apply_similarity_to_points(xy_native, transform.hard)
    if transform.tps is not None:
        xy_native = transform.tps.warp(xy_native, chunk_size=cfg.chunk_size)
    return xy_native


def _assign_contours_to_points(
    xy_native: np.ndarray,
    features: Sequence[_ContourFeature],
) -> pd.DataFrame:
    row_count = int(len(xy_native))
    contour_id = pd.array([pd.NA] * row_count, dtype="string")
    contour_structure = pd.array([pd.NA] * row_count, dtype="string")
    contour_feature_index = pd.array([pd.NA] * row_count, dtype="Int64")
    if row_count == 0 or not features:
        return pd.DataFrame(
            {
                "contour_id": contour_id,
                "contour_structure": contour_structure,
                "contour_feature_index": contour_feature_index,
            }
        )

    assigned = np.zeros(row_count, dtype=bool)
    x = xy_native[:, 0]
    y = xy_native[:, 1]
    for feature in sorted(features, key=lambda item: item.area_native):
        minx, miny, maxx, maxy = feature.geometry.bounds
        candidates = np.flatnonzero(
            (~assigned) & (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
        )
        if candidates.size == 0:
            continue
        if _contains_xy is not None:
            inside = np.asarray(
                _contains_xy(feature.geometry, x[candidates], y[candidates]),
                dtype=bool,
            )
        else:  # pragma: no cover - exercised only with older Shapely.
            prepared = prep(feature.geometry)
            inside = np.array(
                [prepared.contains(Point(float(x[index]), float(y[index]))) for index in candidates],
                dtype=bool,
            )
        selected = candidates[inside]
        if selected.size == 0:
            continue
        contour_id[selected] = feature.contour_id
        contour_structure[selected] = feature.structure if feature.structure is not None else pd.NA
        contour_feature_index[selected] = int(feature.feature_index)
        assigned[selected] = True
    return pd.DataFrame(
        {
            "contour_id": contour_id,
            "contour_structure": contour_structure,
            "contour_feature_index": contour_feature_index,
        }
    )


def _assign_profiles_to_points(
    xy_native: np.ndarray,
    profiles: Sequence[_ContourLayerProfile],
) -> pd.DataFrame:
    row_count = int(len(xy_native))
    contour_id = pd.array([pd.NA] * row_count, dtype="string")
    contour_structure = pd.array([pd.NA] * row_count, dtype="string")
    contour_feature_index = pd.array([pd.NA] * row_count, dtype="Int64")
    if row_count == 0 or not profiles:
        return pd.DataFrame(
            {
                "contour_id": contour_id,
                "contour_structure": contour_structure,
                "contour_feature_index": contour_feature_index,
            }
        )

    assigned = np.zeros(row_count, dtype=bool)
    x = xy_native[:, 0]
    y = xy_native[:, 1]
    for profile in sorted(profiles, key=lambda item: item.area_um2):
        minx, miny, maxx, maxy = profile.geometry.bounds
        candidates = np.flatnonzero(
            (~assigned) & (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
        )
        if candidates.size == 0:
            continue
        if _contains_xy is not None:
            inside = np.asarray(
                _contains_xy(profile.geometry, x[candidates], y[candidates]),
                dtype=bool,
            )
        else:  # pragma: no cover - exercised only with older Shapely.
            prepared = prep(profile.geometry)
            inside = np.array(
                [prepared.contains(Point(float(x[index]), float(y[index]))) for index in candidates],
                dtype=bool,
            )
        selected = candidates[inside]
        if selected.size == 0:
            continue
        contour_id[selected] = profile.contour_id
        contour_structure[selected] = (
            profile.contour_structure if profile.contour_structure is not None else pd.NA
        )
        contour_feature_index[selected] = int(profile.contour_feature_index)
        assigned[selected] = True
    return pd.DataFrame(
        {
            "contour_id": contour_id,
            "contour_structure": contour_structure,
            "contour_feature_index": contour_feature_index,
        }
    )


def _build_z_band_signatures(
    transcripts: pd.DataFrame,
    cfg: LocalZOrientationConfig,
) -> dict[str, Any]:
    z = transcripts["z_raw_um"].to_numpy(dtype=float)
    low_cut = float(np.quantile(z, cfg.z_band_fraction))
    high_cut = float(np.quantile(z, 1.0 - cfg.z_band_fraction))
    low = transcripts.loc[transcripts["z_raw_um"] <= low_cut]
    high = transcripts.loc[transcripts["z_raw_um"] >= high_cut]
    if len(low) < cfg.min_band_transcripts:
        low = transcripts.nsmallest(cfg.min_band_transcripts, "z_raw_um")
    if len(high) < cfg.min_band_transcripts:
        high = transcripts.nlargest(cfg.min_band_transcripts, "z_raw_um")
    return {
        "low_count": int(len(low)),
        "high_count": int(len(high)),
        "low_signature": _gene_signature(low["gene"], cfg.max_signature_genes),
        "high_signature": _gene_signature(high["gene"], cfg.max_signature_genes),
    }


def _gene_signature(genes: pd.Series, max_genes: int) -> dict[str, float]:
    counts = genes.astype(str).value_counts().head(int(max_genes))
    total = float(counts.sum())
    if total <= 0:
        return {}
    return {str(gene): float(count / total) for gene, count in counts.items()}


def _infer_orientation_sequence(
    summaries: Sequence[_SliceLocalZSummary],
    cfg: LocalZOrientationConfig,
) -> _OrientationInference:
    if not summaries:
        raise ValueError("At least one slice is required for local-z orientation inference.")
    if len(summaries) == 1:
        scoring_backend = _select_orientation_scoring_backend(summaries, cfg)
        sample_id = summaries[0].sample_id
        return _OrientationInference(
            states_by_sample={sample_id: "preserve"},
            best_score=0.0,
            second_score=0.0,
            confidence_margin=0.0,
            low_confidence=True,
            edge_scores={},
            scoring_backend=scoring_backend,
            bootstrap_support_by_sample={sample_id: math.nan},
            low_confidence_reasons=("single_slice",),
        )

    scoring_backend = _select_orientation_scoring_backend(summaries, cfg)
    edge_match_contexts = (
        _build_edge_match_contexts(summaries, cfg) if scoring_backend == "contour" else {}
    )
    edge_detail_cache: dict[tuple[int, str, str], _EdgeScoreDetail] = {}

    paths: dict[str, list[tuple[float, tuple[str, ...]]]] = {
        state: [(0.0, (state,))] for state in LOCAL_Z_ORIENTATION_STATES
    }
    for index in range(1, len(summaries)):
        next_paths: dict[str, list[tuple[float, tuple[str, ...]]]] = {}
        for state in LOCAL_Z_ORIENTATION_STATES:
            candidates: list[tuple[float, tuple[str, ...]]] = []
            for prev_state in LOCAL_Z_ORIENTATION_STATES:
                edge = _edge_detail(
                    summaries,
                    edge_index=index - 1,
                    prev_state=prev_state,
                    next_state=state,
                    backend=scoring_backend,
                    cfg=cfg,
                    edge_match_contexts=edge_match_contexts,
                    cache=edge_detail_cache,
                ).score
                for prev_score, prev_path in paths[prev_state]:
                    candidates.append((float(prev_score + edge), prev_path + (state,)))
            next_paths[state] = _top_unique_paths(candidates, limit=2)
        paths = next_paths

    all_paths = _top_unique_paths([candidate for group in paths.values() for candidate in group], limit=2)
    best_score, best_path = all_paths[0]
    second_score = all_paths[1][0] if len(all_paths) > 1 else best_score
    edge_count = max(len(summaries) - 1, 1)
    average_best = float(best_score) / edge_count
    average_second = float(second_score) / edge_count
    margin = average_best - average_second

    states_by_sample = {
        summary.sample_id: best_path[index] for index, summary in enumerate(summaries)
    }
    edge_scores: dict[tuple[str, str], float] = {}
    edge_details: dict[tuple[str, str], _EdgeScoreDetail] = {}
    for index in range(1, len(summaries)):
        prev = summaries[index - 1]
        cur = summaries[index]
        detail = _edge_detail(
            summaries,
            edge_index=index - 1,
            prev_state=states_by_sample[prev.sample_id],
            next_state=states_by_sample[cur.sample_id],
            backend=scoring_backend,
            cfg=cfg,
            edge_match_contexts=edge_match_contexts,
            cache=edge_detail_cache,
        )
        edge_scores[(prev.sample_id, cur.sample_id)] = detail.score
        edge_details[(prev.sample_id, cur.sample_id)] = detail

    fallback_edge_count = int(sum(detail.used_global_fallback for detail in edge_details.values()))
    fallback_edge_fraction = float(fallback_edge_count / max(len(summaries) - 1, 1))
    contour_pair_rows = (
        _build_contour_pair_rows(summaries, states_by_sample, edge_match_contexts, cfg)
        if scoring_backend == "contour"
        else ()
    )
    bootstrap_support, bootstrap_rows = _bootstrap_orientation_support(
        summaries,
        states_by_sample,
        scoring_backend=scoring_backend,
        edge_match_contexts=edge_match_contexts,
        cfg=cfg,
    )
    low_confidence_reasons: list[str] = []
    if margin < cfg.low_confidence_margin:
        low_confidence_reasons.append("score_margin_below_threshold")
    finite_support = [
        support
        for support in (bootstrap_support or {}).values()
        if not math.isnan(float(support))
    ]
    if finite_support and min(finite_support) < float(cfg.orientation_bootstrap_support_threshold):
        low_confidence_reasons.append("bootstrap_support_below_threshold")
    if (
        scoring_backend == "contour"
        and fallback_edge_fraction > float(cfg.contour_max_global_fallback_fraction)
    ):
        low_confidence_reasons.append("too_many_global_fallback_edges")

    return _OrientationInference(
        states_by_sample=states_by_sample,
        best_score=average_best,
        second_score=average_second,
        confidence_margin=float(margin),
        low_confidence=bool(low_confidence_reasons),
        edge_scores=edge_scores,
        scoring_backend=scoring_backend,
        edge_details=edge_details,
        contour_pair_rows=tuple(contour_pair_rows),
        bootstrap_support_by_sample=bootstrap_support,
        bootstrap_rows=tuple(bootstrap_rows),
        fallback_edge_count=fallback_edge_count,
        fallback_edge_fraction=fallback_edge_fraction,
        low_confidence_reasons=tuple(low_confidence_reasons),
    )


def _select_orientation_scoring_backend(
    summaries: Sequence[_SliceLocalZSummary],
    cfg: LocalZOrientationConfig,
) -> str:
    if cfg.orientation_spatial_unit == "global":
        return "global"
    loaded_contour_slice_count = sum(
        1
        for summary in summaries
        if summary.contour_status == "ok" and summary.contour_profile_count > 0
    )
    usable_contour_slice_count = sum(
        1
        for summary in summaries
        if summary.contour_status == "ok" and summary.usable_contour_profile_count > 0
    )
    if cfg.orientation_spatial_unit == "contour":
        if loaded_contour_slice_count == 0:
            raise ValueError(
                "orientation_spatial_unit='contour' was requested, but no aligned "
                "contour GeoJSON/profile data could be loaded."
            )
        return "contour"
    if usable_contour_slice_count == 0:
        return "global"
    return "contour"


def _build_edge_match_contexts(
    summaries: Sequence[_SliceLocalZSummary],
    cfg: LocalZOrientationConfig,
) -> dict[tuple[str, str], _EdgeMatchContext]:
    contexts: dict[tuple[str, str], _EdgeMatchContext] = {}
    for index in range(1, len(summaries)):
        prev = summaries[index - 1]
        cur = summaries[index]
        contexts[(prev.sample_id, cur.sample_id)] = _match_adjacent_contours(prev, cur, cfg)
    return contexts


def _match_adjacent_contours(
    prev_summary: _SliceLocalZSummary,
    next_summary: _SliceLocalZSummary,
    cfg: LocalZOrientationConfig,
) -> _EdgeMatchContext:
    prev_profiles = [profile for profile in prev_summary.contour_profiles if profile.usable]
    next_profiles = [profile for profile in next_summary.contour_profiles if profile.usable]
    if not prev_profiles or not next_profiles:
        return _EdgeMatchContext(
            prev_sample_id=prev_summary.sample_id,
            next_sample_id=next_summary.sample_id,
            matches=(),
            fallback_reason="missing_usable_contour_profiles",
        )

    matches: list[_ContourMatch] = []
    seen: set[tuple[str, str, str]] = set()
    for prev in prev_profiles:
        for nxt in next_profiles:
            if not _contour_structures_compatible(prev, nxt):
                continue
            iou = _geometry_iou(prev.geometry, nxt.geometry)
            if math.isnan(iou) or iou < float(cfg.contour_match_min_iou):
                continue
            match = _make_contour_match(prev, nxt, "overlap", iou=iou, cfg=cfg)
            key = (match.prev.contour_id, match.next.contour_id, match.match_method)
            if key not in seen:
                seen.add(key)
                matches.append(match)

    if matches:
        return _EdgeMatchContext(
            prev_sample_id=prev_summary.sample_id,
            next_sample_id=next_summary.sample_id,
            matches=tuple(matches),
        )

    max_distance = float(cfg.contour_match_max_distance_um)
    if max_distance <= 0:
        return _EdgeMatchContext(
            prev_sample_id=prev_summary.sample_id,
            next_sample_id=next_summary.sample_id,
            matches=(),
            fallback_reason="no_overlap_matches",
        )

    next_by_group = _profiles_by_structure_key(next_profiles)
    for prev in prev_profiles:
        candidate_groups = _candidate_structure_keys(prev, next_by_group)
        for group_key in candidate_groups:
            group = next_by_group.get(group_key, [])
            if not group:
                continue
            coords = np.array(
                [[profile.centroid_x_um, profile.centroid_y_um] for profile in group],
                dtype=float,
            )
            tree = cKDTree(coords)
            prev_xy = np.array([prev.centroid_x_um, prev.centroid_y_um], dtype=float)
            for index in tree.query_ball_point(prev_xy, r=max_distance):
                nxt = group[int(index)]
                if not _contour_structures_compatible(prev, nxt):
                    continue
                distance = float(np.linalg.norm(prev_xy - coords[int(index)]))
                match = _make_contour_match(
                    prev,
                    nxt,
                    "centroid",
                    iou=0.0,
                    centroid_distance_um=distance,
                    cfg=cfg,
                )
                key = (match.prev.contour_id, match.next.contour_id, match.match_method)
                if key not in seen:
                    seen.add(key)
                    matches.append(match)

    reason = None if matches else "no_centroid_matches"
    return _EdgeMatchContext(
        prev_sample_id=prev_summary.sample_id,
        next_sample_id=next_summary.sample_id,
        matches=tuple(matches),
        fallback_reason=reason,
    )


def _profiles_by_structure_key(
    profiles: Sequence[_ContourLayerProfile],
) -> dict[str | None, list[_ContourLayerProfile]]:
    grouped: dict[str | None, list[_ContourLayerProfile]] = {}
    for profile in profiles:
        grouped.setdefault(profile.contour_structure, []).append(profile)
    return grouped


def _candidate_structure_keys(
    profile: _ContourLayerProfile,
    grouped: Mapping[str | None, Sequence[_ContourLayerProfile]],
) -> list[str | None]:
    if profile.contour_structure is not None:
        return [profile.contour_structure, None]
    return list(grouped)


def _contour_structures_compatible(
    prev: _ContourLayerProfile,
    nxt: _ContourLayerProfile,
) -> bool:
    if prev.contour_structure is None or nxt.contour_structure is None:
        return True
    return prev.contour_structure == nxt.contour_structure


def _make_contour_match(
    prev: _ContourLayerProfile,
    nxt: _ContourLayerProfile,
    method: str,
    *,
    iou: float,
    cfg: LocalZOrientationConfig,
    centroid_distance_um: float | None = None,
) -> _ContourMatch:
    if centroid_distance_um is None:
        centroid_distance_um = float(
            math.hypot(
                prev.centroid_x_um - nxt.centroid_x_um,
                prev.centroid_y_um - nxt.centroid_y_um,
            )
        )
    transcript_weight = math.sqrt(
        max(float(prev.signature_transcript_count), 1.0)
        * max(float(nxt.signature_transcript_count), 1.0)
    )
    area_weight = math.sqrt(max(min(float(prev.area_um2), float(nxt.area_um2)), 1.0))
    non_doublet_weight = max(
        0.05,
        1.0 - (float(prev.doublet_burden) + float(nxt.doublet_burden)) / 2.0,
    )
    distance_weight = 1.0
    if method == "centroid" and cfg.contour_match_max_distance_um > 0:
        distance_weight = max(
            0.05,
            1.0 - min(float(centroid_distance_um), float(cfg.contour_match_max_distance_um))
            / float(cfg.contour_match_max_distance_um),
        )
    weight = transcript_weight * area_weight * non_doublet_weight * distance_weight
    return _ContourMatch(
        prev=prev,
        next=nxt,
        match_method=method,
        iou=float(iou),
        centroid_distance_um=float(centroid_distance_um),
        weight=float(weight),
    )


def _geometry_iou(first: BaseGeometry, second: BaseGeometry) -> float:
    try:
        union_area = float(first.union(second).area)
        if union_area <= 0:
            return math.nan
        return float(first.intersection(second).area / union_area)
    except Exception:
        return math.nan


def _edge_detail(
    summaries: Sequence[_SliceLocalZSummary],
    *,
    edge_index: int,
    prev_state: str,
    next_state: str,
    backend: str,
    cfg: LocalZOrientationConfig,
    edge_match_contexts: Mapping[tuple[str, str], _EdgeMatchContext],
    cache: dict[tuple[int, str, str], _EdgeScoreDetail],
) -> _EdgeScoreDetail:
    key = (int(edge_index), prev_state, next_state)
    if key in cache:
        return cache[key]
    prev = summaries[edge_index]
    nxt = summaries[edge_index + 1]
    if backend == "global":
        detail = _EdgeScoreDetail(
            score=_edge_score(prev, prev_state, nxt, next_state),
            backend="global",
            used_global_fallback=False,
            contour_pair_count=0,
            reason="global_mode",
        )
    else:
        context = edge_match_contexts.get((prev.sample_id, nxt.sample_id))
        if context is None or not context.matches:
            reason = context.fallback_reason if context is not None else "missing_context"
            detail = _EdgeScoreDetail(
                score=_edge_score(prev, prev_state, nxt, next_state),
                backend="global_fallback",
                used_global_fallback=True,
                contour_pair_count=0,
                reason=str(reason or "no_contour_matches"),
            )
        else:
            detail = _EdgeScoreDetail(
                score=_weighted_contour_edge_score(context.matches, prev_state, next_state),
                backend="contour",
                used_global_fallback=False,
                contour_pair_count=len(context.matches),
                reason="matched_contours",
            )
    cache[key] = detail
    return detail


def _weighted_contour_edge_score(
    matches: Sequence[_ContourMatch],
    prev_state: str,
    next_state: str,
) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for match in matches:
        score = _contour_match_score(match, prev_state, next_state)
        weight = max(float(match.weight), 0.0)
        if weight <= 0:
            continue
        weighted_sum += score * weight
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return float(weighted_sum / total_weight)


def _contour_match_score(
    match: _ContourMatch,
    prev_state: str,
    next_state: str,
) -> float:
    prev_high = _contour_surface_signature(match.prev, prev_state, "high")
    next_low = _contour_surface_signature(match.next, next_state, "low")
    return _cosine_similarity(prev_high, next_low)


def _contour_surface_signature(
    profile: _ContourLayerProfile,
    state: str,
    side: str,
) -> dict[str, float]:
    if state not in LOCAL_Z_ORIENTATION_STATES:
        raise ValueError(f"Unsupported local-z orientation state: {state!r}.")
    if side == "low":
        return profile.raw_low_signature if state == "preserve" else profile.raw_high_signature
    if side == "high":
        return profile.raw_high_signature if state == "preserve" else profile.raw_low_signature
    raise ValueError("side must be 'low' or 'high'.")


def _build_contour_pair_rows(
    summaries: Sequence[_SliceLocalZSummary],
    states_by_sample: Mapping[str, str],
    edge_match_contexts: Mapping[tuple[str, str], _EdgeMatchContext],
    cfg: LocalZOrientationConfig,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for index in range(1, len(summaries)):
        prev = summaries[index - 1]
        nxt = summaries[index]
        context = edge_match_contexts.get((prev.sample_id, nxt.sample_id))
        if context is None:
            continue
        for match in context.matches:
            scores = {
                f"score_{prev_state}_{next_state}": _contour_match_score(
                    match,
                    prev_state,
                    next_state,
                )
                for prev_state in LOCAL_Z_ORIENTATION_STATES
                for next_state in LOCAL_Z_ORIENTATION_STATES
            }
            selected_score = _contour_match_score(
                match,
                states_by_sample[prev.sample_id],
                states_by_sample[nxt.sample_id],
            )
            rows.append(
                {
                    "prev_sample_id": prev.sample_id,
                    "next_sample_id": nxt.sample_id,
                    "prev_order": prev.order,
                    "next_order": nxt.order,
                    "prev_orientation": states_by_sample[prev.sample_id],
                    "next_orientation": states_by_sample[nxt.sample_id],
                    "prev_contour_id": match.prev.contour_id,
                    "next_contour_id": match.next.contour_id,
                    "prev_structure": match.prev.contour_structure,
                    "next_structure": match.next.contour_structure,
                    "match_method": match.match_method,
                    "iou": match.iou,
                    "centroid_distance_um": match.centroid_distance_um,
                    "weight": match.weight,
                    "selected_score": selected_score,
                    **scores,
                }
            )
    return tuple(rows)


def _bootstrap_orientation_support(
    summaries: Sequence[_SliceLocalZSummary],
    states_by_sample: Mapping[str, str],
    *,
    scoring_backend: str,
    edge_match_contexts: Mapping[tuple[str, str], _EdgeMatchContext],
    cfg: LocalZOrientationConfig,
) -> tuple[dict[str, float], tuple[dict[str, Any], ...]]:
    if scoring_backend != "contour" or cfg.orientation_bootstrap_iterations <= 0:
        return {}, ()
    iterations = int(cfg.orientation_bootstrap_iterations)
    rng = np.random.default_rng(int(cfg.orientation_bootstrap_seed))
    support_counts = {summary.sample_id: 0 for summary in summaries}
    rows: list[dict[str, Any]] = []

    for iteration in range(iterations):
        edge_scores = _bootstrap_edge_score_matrices(
            summaries,
            edge_match_contexts=edge_match_contexts,
            rng=rng,
        )
        _, _, path = _best_path_from_edge_scores(summaries, edge_scores)
        for summary, state in zip(summaries, path):
            if state == states_by_sample[summary.sample_id]:
                support_counts[summary.sample_id] += 1

    support = {
        sample_id: float(count / iterations) if iterations > 0 else math.nan
        for sample_id, count in support_counts.items()
    }
    for summary in summaries:
        rows.append(
            {
                "sample_id": summary.sample_id,
                "order": summary.order,
                "selected_orientation": states_by_sample[summary.sample_id],
                "bootstrap_iterations": iterations,
                "bootstrap_support": support[summary.sample_id],
                "support_threshold": float(cfg.orientation_bootstrap_support_threshold),
            }
        )
    return support, tuple(rows)


def _bootstrap_edge_score_matrices(
    summaries: Sequence[_SliceLocalZSummary],
    *,
    edge_match_contexts: Mapping[tuple[str, str], _EdgeMatchContext],
    rng: np.random.Generator,
) -> dict[tuple[int, str, str], float]:
    edge_scores: dict[tuple[int, str, str], float] = {}
    for edge_index in range(len(summaries) - 1):
        prev = summaries[edge_index]
        nxt = summaries[edge_index + 1]
        context = edge_match_contexts.get((prev.sample_id, nxt.sample_id))
        sampled_matches: tuple[_ContourMatch, ...] = ()
        if context is not None and context.matches:
            indices = rng.integers(0, len(context.matches), size=len(context.matches))
            sampled_matches = tuple(context.matches[int(index)] for index in indices)
        for prev_state in LOCAL_Z_ORIENTATION_STATES:
            for next_state in LOCAL_Z_ORIENTATION_STATES:
                if sampled_matches:
                    score = _weighted_contour_edge_score(sampled_matches, prev_state, next_state)
                else:
                    score = _edge_score(prev, prev_state, nxt, next_state)
                edge_scores[(edge_index, prev_state, next_state)] = float(score)
    return edge_scores


def _best_path_from_edge_scores(
    summaries: Sequence[_SliceLocalZSummary],
    edge_scores: Mapping[tuple[int, str, str], float],
) -> tuple[float, float, tuple[str, ...]]:
    paths: dict[str, list[tuple[float, tuple[str, ...]]]] = {
        state: [(0.0, (state,))] for state in LOCAL_Z_ORIENTATION_STATES
    }
    for index in range(1, len(summaries)):
        next_paths: dict[str, list[tuple[float, tuple[str, ...]]]] = {}
        for state in LOCAL_Z_ORIENTATION_STATES:
            candidates: list[tuple[float, tuple[str, ...]]] = []
            for prev_state in LOCAL_Z_ORIENTATION_STATES:
                edge = edge_scores[(index - 1, prev_state, state)]
                for prev_score, prev_path in paths[prev_state]:
                    candidates.append((float(prev_score + edge), prev_path + (state,)))
            next_paths[state] = _top_unique_paths(candidates, limit=2)
        paths = next_paths
    all_paths = _top_unique_paths([candidate for group in paths.values() for candidate in group], limit=2)
    best_score, best_path = all_paths[0]
    second_score = all_paths[1][0] if len(all_paths) > 1 else best_score
    edge_count = max(len(summaries) - 1, 1)
    return float(best_score) / edge_count, float(second_score) / edge_count, best_path


def _top_unique_paths(
    candidates: Sequence[tuple[float, tuple[str, ...]]],
    *,
    limit: int,
) -> list[tuple[float, tuple[str, ...]]]:
    unique: dict[tuple[str, ...], float] = {}
    for score, path in candidates:
        previous = unique.get(path)
        if previous is None or score > previous:
            unique[path] = float(score)
    ordered = sorted(
        ((score, path) for path, score in unique.items()),
        key=lambda item: (-item[0], item[1]),
    )
    return ordered[:limit]


def _edge_score(
    prev_summary: _SliceLocalZSummary,
    prev_state: str,
    next_summary: _SliceLocalZSummary,
    next_state: str,
) -> float:
    prev_high = _surface_signature(prev_summary, prev_state, "high")
    next_low = _surface_signature(next_summary, next_state, "low")
    return _cosine_similarity(prev_high, next_low)


def _surface_signature(
    summary: _SliceLocalZSummary,
    state: str,
    side: str,
) -> dict[str, float]:
    if state not in LOCAL_Z_ORIENTATION_STATES:
        raise ValueError(f"Unsupported local-z orientation state: {state!r}.")
    if side == "low":
        return summary.raw_low_signature if state == "preserve" else summary.raw_high_signature
    if side == "high":
        return summary.raw_high_signature if state == "preserve" else summary.raw_low_signature
    raise ValueError("side must be 'low' or 'high'.")


def _cosine_similarity(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    genes = sorted(set(first) | set(second))
    if not genes:
        return 0.0
    a = np.array([float(first.get(gene, 0.0)) for gene in genes], dtype=float)
    b = np.array([float(second.get(gene, 0.0)) for gene in genes], dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _build_manifest_frame(
    summaries: Sequence[_SliceLocalZSummary],
    inference: _OrientationInference,
    cfg: LocalZOrientationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, summary in enumerate(summaries):
        selected = inference.states_by_sample[summary.sample_id]
        previous_edge = math.nan
        next_edge = math.nan
        if index > 0:
            previous = summaries[index - 1]
            previous_edge = inference.edge_scores.get((previous.sample_id, summary.sample_id), math.nan)
            previous_detail = (inference.edge_details or {}).get((previous.sample_id, summary.sample_id))
            previous_pairs = previous_detail.contour_pair_count if previous_detail is not None else 0
            previous_fallback = previous_detail.used_global_fallback if previous_detail is not None else False
            previous_backend = previous_detail.backend if previous_detail is not None else ""
        if index + 1 < len(summaries):
            nxt = summaries[index + 1]
            next_edge = inference.edge_scores.get((summary.sample_id, nxt.sample_id), math.nan)
            next_detail = (inference.edge_details or {}).get((summary.sample_id, nxt.sample_id))
            next_pairs = next_detail.contour_pair_count if next_detail is not None else 0
            next_fallback = next_detail.used_global_fallback if next_detail is not None else False
            next_backend = next_detail.backend if next_detail is not None else ""
        if index == 0:
            previous_pairs = 0
            previous_fallback = False
            previous_backend = ""
        if index + 1 >= len(summaries):
            next_pairs = 0
            next_fallback = False
            next_backend = ""
        rows.append(
            {
                "schema_version": LOCAL_Z_ORIENTATION_SCHEMA_VERSION,
                "sample_id": summary.sample_id,
                "order": summary.order,
                "stack_z_um": summary.stack_z_um,
                "transcript_path": str(summary.transcript_path),
                "selected_orientation": selected,
                "applied": bool(cfg.apply_local_z_flip),
                "orientation_spatial_unit": cfg.orientation_spatial_unit,
                "scoring_backend": inference.scoring_backend,
                "low_confidence_reasons": ";".join(inference.low_confidence_reasons),
                "z_min_um": summary.z_min_um,
                "z_max_um": summary.z_max_um,
                "z_mid_um": summary.z_mid_um,
                "transcript_count": summary.transcript_count,
                "signature_transcript_count": summary.signature_transcript_count,
                "excluded_doublet_transcripts": summary.excluded_doublet_transcripts,
                "low_band_count": summary.low_band_count,
                "high_band_count": summary.high_band_count,
                "ovrlpy_status": summary.ovrlpy_qc.status,
                "ovrlpy_doublet_count": summary.ovrlpy_qc.doublet_count,
                "signal_integrity_mean": summary.ovrlpy_qc.signal_integrity_mean,
                "signal_integrity_p05": summary.ovrlpy_qc.signal_integrity_p05,
                "signal_integrity_p95": summary.ovrlpy_qc.signal_integrity_p95,
                "signal_mean": summary.ovrlpy_qc.signal_mean,
                "signal_p05": summary.ovrlpy_qc.signal_p05,
                "signal_p95": summary.ovrlpy_qc.signal_p95,
                "best_score": inference.best_score,
                "second_score": inference.second_score,
                "confidence_margin": inference.confidence_margin,
                "bootstrap_support": (inference.bootstrap_support_by_sample or {}).get(
                    summary.sample_id,
                    math.nan,
                ),
                "bootstrap_support_threshold": cfg.orientation_bootstrap_support_threshold,
                "fallback_edge_count": inference.fallback_edge_count,
                "fallback_edge_fraction": inference.fallback_edge_fraction,
                "low_confidence": inference.low_confidence,
                "edge_score_to_previous": previous_edge,
                "edge_score_to_next": next_edge,
                "edge_backend_to_previous": previous_backend,
                "edge_backend_to_next": next_backend,
                "contour_pairs_to_previous": previous_pairs,
                "contour_pairs_to_next": next_pairs,
                "global_fallback_to_previous": previous_fallback,
                "global_fallback_to_next": next_fallback,
                "aligned_geojson": str(summary.contour_path) if summary.contour_path is not None else "",
                "contour_status": summary.contour_status,
                "contour_error": summary.contour_error or "",
                "contour_profile_count": summary.contour_profile_count,
                "usable_contour_profile_count": summary.usable_contour_profile_count,
                "contour_assigned_transcripts": summary.contour_assigned_transcripts,
                "contour_excluded_doublet_transcripts": summary.contour_excluded_doublet_transcripts,
            }
        )
    return pd.DataFrame(rows)


def _contour_pair_frame(inference: _OrientationInference) -> pd.DataFrame:
    columns = [
        "prev_sample_id",
        "next_sample_id",
        "prev_order",
        "next_order",
        "prev_orientation",
        "next_orientation",
        "prev_contour_id",
        "next_contour_id",
        "prev_structure",
        "next_structure",
        "match_method",
        "iou",
        "centroid_distance_um",
        "weight",
        "selected_score",
        "score_preserve_preserve",
        "score_preserve_reverse",
        "score_reverse_preserve",
        "score_reverse_reverse",
    ]
    if not inference.contour_pair_rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(list(inference.contour_pair_rows)).reindex(columns=columns)


def _bootstrap_support_frame(
    inference: _OrientationInference,
    summaries: Sequence[_SliceLocalZSummary],
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "order",
        "selected_orientation",
        "bootstrap_iterations",
        "bootstrap_support",
        "support_threshold",
    ]
    if inference.bootstrap_rows:
        return pd.DataFrame(list(inference.bootstrap_rows)).reindex(columns=columns)
    rows = [
        {
            "sample_id": summary.sample_id,
            "order": summary.order,
            "selected_orientation": inference.states_by_sample.get(summary.sample_id, ""),
            "bootstrap_iterations": 0,
            "bootstrap_support": math.nan,
            "support_threshold": math.nan,
        }
        for summary in summaries
    ]
    return pd.DataFrame(rows, columns=columns)


def _write_aligned_transcripts(
    *,
    summaries: Sequence[_SliceLocalZSummary],
    transforms: Sequence[SliceCellTransform],
    inference: _OrientationInference,
    output_parquet: Path,
    cfg: LocalZOrientationConfig,
) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - pyarrow is a package dependency.
        raise ImportError("Writing aligned transcript Parquet requires pyarrow.") from exc

    transforms_by_sample = {str(transform.sample_id): transform for transform in transforms}
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    writer: Any | None = None
    total = 0
    try:
        for summary in summaries:
            transform = transforms_by_sample[summary.sample_id]
            frame = _normalize_transcripts(_read_transcript_table(summary.transcript_path), cfg)
            aligned = _aligned_transcript_frame(
                frame,
                summary=summary,
                transform=transform,
                orientation=inference.states_by_sample[summary.sample_id],
                cfg=cfg,
            )
            table = pa.Table.from_pandas(aligned, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_parquet, table.schema)
            writer.write_table(table)
            total += int(len(aligned))
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pd.DataFrame().to_parquet(output_parquet, index=False)
    return total


def _aligned_transcript_frame(
    frame: pd.DataFrame,
    *,
    summary: _SliceLocalZSummary,
    transform: SliceCellTransform,
    orientation: str,
    cfg: LocalZOrientationConfig,
) -> pd.DataFrame:
    xy_native = _aligned_xy_native(frame, transform, cfg)
    xy_um = xy_native * float(cfg.pixel_size_um)
    contour_assignments = _assign_profiles_to_points(xy_native, summary.contour_profiles)

    z_raw = frame["z_raw_um"].to_numpy(dtype=float)
    should_reverse = bool(cfg.apply_local_z_flip and orientation == "reverse")
    if should_reverse:
        local_z = summary.z_min_um + summary.z_max_um - z_raw
    else:
        local_z = z_raw.copy()
    z_3d = float(summary.stack_z_um) + (local_z - float(summary.z_mid_um))

    output = pd.DataFrame(
        {
            "sample_id": summary.sample_id,
            "slice_order": int(summary.order),
            "gene": frame["gene"].astype(str).to_numpy(),
            "x_raw_um": frame["x_raw_um"].to_numpy(dtype=float),
            "y_raw_um": frame["y_raw_um"].to_numpy(dtype=float),
            "z_raw_um": z_raw,
            "x_3d_um": xy_um[:, 0],
            "y_3d_um": xy_um[:, 1],
            "z_3d_um": z_3d,
            "slice_stack_z_um": float(summary.stack_z_um),
            "local_z_corrected_um": local_z,
            "local_z_mid_um": float(summary.z_mid_um),
            "local_z_orientation": orientation,
            "local_z_flip_applied": should_reverse,
        }
    )
    row_count = len(output)
    for optional in ALIGNED_TRANSCRIPT_OPTIONAL_COLUMNS:
        if optional in {"transcript_id", "cell_id", "structure"}:
            if optional in frame.columns:
                output[optional] = frame[optional].astype("string").reset_index(drop=True)
            else:
                output[optional] = pd.array([pd.NA] * row_count, dtype="string")
        elif optional in {"contour_id", "contour_structure"}:
            output[optional] = contour_assignments[optional].astype("string").reset_index(drop=True)
        elif optional == "contour_feature_index":
            output[optional] = contour_assignments[optional].astype("Int64").reset_index(drop=True)
        elif optional == "overlaps_nucleus":
            if optional in frame.columns:
                output[optional] = (
                    pd.to_numeric(frame[optional], errors="coerce")
                    .fillna(0)
                    .astype("int64")
                    .to_numpy()
                )
            else:
                output[optional] = np.zeros(row_count, dtype="int64")
        elif optional == "qv":
            if optional in frame.columns:
                output[optional] = pd.to_numeric(frame[optional], errors="coerce").astype("float64").to_numpy()
            else:
                output[optional] = np.full(row_count, np.nan, dtype="float64")
    return output.loc[:, list(ALIGNED_TRANSCRIPT_COLUMNS)]


def _build_marker_gradient_frame(
    summaries: Sequence[_SliceLocalZSummary],
    inference: _OrientationInference,
    *,
    max_markers_per_slice: int = 12,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        state = inference.states_by_sample[summary.sample_id]
        bottom = _surface_signature(summary, state, "low")
        top = _surface_signature(summary, state, "high")
        genes = sorted(set(bottom) | set(top))
        ranked = sorted(
            genes,
            key=lambda gene: abs(float(top.get(gene, 0.0)) - float(bottom.get(gene, 0.0))),
            reverse=True,
        )[:max_markers_per_slice]
        for gene in ranked:
            rows.append(
                {
                    "sample_id": summary.sample_id,
                    "order": summary.order,
                    "gene": gene,
                    "corrected_top_fraction": float(top.get(gene, 0.0)),
                    "corrected_bottom_fraction": float(bottom.get(gene, 0.0)),
                    "top_minus_bottom": float(top.get(gene, 0.0)) - float(bottom.get(gene, 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _write_biological_report(
    output_path: Path,
    *,
    manifest: pd.DataFrame,
    marker_gradients: pd.DataFrame,
    inference: _OrientationInference,
    cfg: LocalZOrientationConfig,
) -> None:
    manifest_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(row.sample_id))}</td>"
        f"<td>{int(row.order)}</td>"
        f"<td>{escape(str(row.selected_orientation))}</td>"
        f"<td>{escape(str(row.scoring_backend))}</td>"
        f"<td>{escape(str(row.ovrlpy_status))}</td>"
        f"<td>{int(row.ovrlpy_doublet_count)}</td>"
        f"<td>{float(row.signal_integrity_mean):.4g}</td>"
        f"<td>{float(row.signal_mean):.4g}</td>"
        f"<td>{float(row.edge_score_to_previous):.3f}</td>"
        f"<td>{float(row.edge_score_to_next):.3f}</td>"
        f"<td>{int(row.usable_contour_profile_count)}</td>"
        f"<td>{int(row.contour_pairs_to_previous)}</td>"
        f"<td>{int(row.contour_pairs_to_next)}</td>"
        f"<td>{float(row.bootstrap_support):.3f}</td>"
        "</tr>"
        for row in manifest.itertuples(index=False)
    )
    marker_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(row.sample_id))}</td>"
        f"<td>{escape(str(row.gene))}</td>"
        f"<td>{float(row.corrected_top_fraction):.3f}</td>"
        f"<td>{float(row.corrected_bottom_fraction):.3f}</td>"
        f"<td>{float(row.top_minus_bottom):.3f}</td>"
        "</tr>"
        for row in marker_gradients.head(60).itertuples(index=False)
    )
    low_confidence_note = (
        "The orientation sequence is marked low confidence; forced correction still used "
        "the best-scoring preserve/reverse path."
        if inference.low_confidence
        else "The best orientation path exceeded the configured confidence margin."
    )
    z3d_min = (
        manifest["stack_z_um"].astype(float)
        + manifest["z_min_um"].astype(float)
        - manifest["z_mid_um"].astype(float)
    ).min()
    z3d_max = (
        manifest["stack_z_um"].astype(float)
        + manifest["z_max_um"].astype(float)
        - manifest["z_mid_um"].astype(float)
    ).max()
    total_doublets = int(pd.to_numeric(manifest["ovrlpy_doublet_count"], errors="coerce").fillna(0).sum())
    integrity_median = pd.to_numeric(
        manifest["signal_integrity_mean"], errors="coerce"
    ).median()
    contour_profile_total = int(
        pd.to_numeric(manifest["usable_contour_profile_count"], errors="coerce").fillna(0).sum()
    )
    backend_counts = manifest["scoring_backend"].astype(str).value_counts().to_dict()
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HistoSeg local-z biological QC</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #17202a; }}
    table {{ border-collapse: collapse; margin: 18px 0; width: 100%; }}
    th, td {{ border: 1px solid #d5dbe3; padding: 6px 8px; text-align: left; }}
    th {{ background: #eef3f8; }}
    .note {{ background: #fff8dc; border: 1px solid #ead27a; padding: 10px; }}
    code {{ background: #f2f4f7; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>HistoSeg local-z biological QC</h1>
  <p>
    This report summarizes transcript-level local-z orientation correction.
    Contour and cell stack reconstruction are unchanged; only transcript
    <code>local_z_corrected_um</code> and <code>z_3d_um</code> use the selected
    preserve/reverse states.
  </p>
  <p class="note">{escape(low_confidence_note)}</p>
  <ul>
    <li>Vertical QC backend: {escape(cfg.vertical_qc_backend)}</li>
    <li>Orientation spatial unit: {escape(cfg.orientation_spatial_unit)}</li>
    <li>Scoring backend counts: {escape(json.dumps(backend_counts, sort_keys=True))}</li>
    <li>Best adjacent-surface continuity score: {inference.best_score:.4f}</li>
    <li>Second-best score: {inference.second_score:.4f}</li>
    <li>Confidence margin: {inference.confidence_margin:.4f}</li>
    <li>Low-confidence reasons: {escape('; '.join(inference.low_confidence_reasons) or 'none')}</li>
    <li>Global fallback edge fraction: {inference.fallback_edge_fraction:.3f}</li>
    <li>Local-z flip applied: {str(bool(cfg.apply_local_z_flip)).lower()}</li>
    <li>Total Ovrlpy doublet calls: {total_doublets}</li>
    <li>Median signal integrity mean: {float(integrity_median):.4g}</li>
    <li>Usable contour layer profiles: {contour_profile_total}</li>
    <li>Transcript z_3d range: {float(z3d_min):.3f} to {float(z3d_max):.3f} um</li>
  </ul>

  <h2>Per-slice orientation and vertical QC</h2>
  <table>
    <thead>
      <tr>
        <th>Sample</th><th>Order</th><th>Orientation</th><th>Scoring</th><th>Ovrlpy status</th>
        <th>Doublets</th><th>Integrity mean</th><th>Signal mean</th>
        <th>Prev continuity</th><th>Next continuity</th>
        <th>Usable contours</th><th>Prev contour pairs</th><th>Next contour pairs</th>
        <th>Bootstrap support</th>
      </tr>
    </thead>
    <tbody>{manifest_rows}</tbody>
  </table>

  <h2>Top-vs-bottom marker gradients</h2>
  <table>
    <thead>
      <tr>
        <th>Sample</th><th>Gene</th><th>Corrected top fraction</th>
        <th>Corrected bottom fraction</th><th>Top minus bottom</th>
      </tr>
    </thead>
    <tbody>{marker_rows}</tbody>
  </table>

  <p>
    Contour-aware scoring compares matched HistoSeg contour upper/lower transcript
    layers across adjacent slices after excluding Ovrlpy doublet neighborhoods.
    Interpret these outputs as QC and hypothesis-generation evidence for vertical
    overlaps, tissue folds, doublets, and layer-like molecular signal. They are
    not standalone proof of biological structure.
  </p>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


__all__ = [
    "LOCAL_Z_ORIENTATION_SCHEMA_VERSION",
    "LOCAL_Z_ORIENTATION_STATES",
    "LocalZOrientationConfig",
    "LocalZOrientationResult",
    "run_local_z_orientation_correction",
]
