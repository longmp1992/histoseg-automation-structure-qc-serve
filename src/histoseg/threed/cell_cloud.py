"""Project AnnData cell coordinates into a HistoSeg 3D reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator


PathLike = Union[str, Path]


ALIGNMENT_MANIFEST_SCHEMA_VERSION = 2
CELL_CLOUD_CACHE_SCHEMA_VERSION = 1
CELL_CLOUD_OBSM_KEY = "X_histoseg_3d_um"
CELL_CLOUD_ALIGNED_XY_OBSM_KEY = "X_histoseg_aligned_xy_um"
CELL_CLOUD_OBS_SLICE_KEY = "histoseg_slice_id"
CELL_CLOUD_UNS_KEY = "histoseg_3d_alignment"
CELL_CLOUD_RENDER_WARNING_THRESHOLD = 500000

LOGGER = logging.getLogger(__name__)

CELL_CLOUD_PALETTE = (
    "#2dd4bf",
    "#60a5fa",
    "#f59e0b",
    "#a78bfa",
    "#fb7185",
    "#84cc16",
    "#facc15",
    "#22c55e",
    "#38bdf8",
    "#f472b6",
    "#c084fc",
    "#fb923c",
    "#14b8a6",
    "#818cf8",
    "#ef4444",
    "#10b981",
    "#eab308",
    "#06b6d4",
    "#8b5cf6",
    "#f97316",
)


@dataclass(frozen=True)
class CellCloudProjectionConfig:
    h5ad: PathLike
    stack_root: PathLike
    out_parquet: PathLike
    sample_column: str = "sample_id"
    barcode_column: str = "barcode"
    x_column: str = "x_centroid"
    y_column: str = "y_centroid"
    label_columns: Sequence[str] = ()
    pixel_size_um: float = 0.2125
    chunk_size: int = 100000
    ignore_cache: bool = False
    fail_on_stale_cache: bool = False
    write_cache: bool = False
    cache_h5ad: PathLike | None = None
    write_scanpy_spatial: bool = False


@dataclass
class CellCloudProjectionResult:
    out_parquet: Path
    cell_count: int
    projected_cell_count: int
    stack_root: Path
    alignment_hash: str
    cache_status: str
    cache_h5ad: Path | None = None


@dataclass(frozen=True)
class CellCloudRenderConfig:
    aligned_cells_parquet: PathLike
    stack_root: PathLike
    out_html: PathLike
    label_column: str = "leiden_1_0"
    max_points: int = 300000
    random_state: int = 0
    z_visual_scale: float = 8.0
    marker_size: float = 1.4
    opacity: float = 0.48
    include_contours: bool = True
    title: str = "HistoSeg 3D cell cloud"
    performance_warning_threshold: int = CELL_CLOUD_RENDER_WARNING_THRESHOLD


@dataclass
class CellCloudRenderResult:
    out_html: Path
    aligned_cells_parquet: Path
    stack_root: Path
    total_cells: int
    rendered_cells: int
    label_column: str
    label_count: int
    contour_trace_count: int
    z_visual_scale: float


@dataclass(frozen=True)
class SimilarityTransform:
    origin_x: float
    origin_y: float
    rotation_degrees: float
    scale: float
    translate_x: float
    translate_y: float


@dataclass(frozen=True)
class TpsModel:
    interpolator: RBFInterpolator
    center_xy: np.ndarray
    scale: float

    def warp(self, xy: np.ndarray, *, chunk_size: int) -> np.ndarray:
        result = np.asarray(xy, dtype=float).copy()
        for start in range(0, len(result), int(chunk_size)):
            stop = min(start + int(chunk_size), len(result))
            normalized = (result[start:stop] - self.center_xy) / self.scale
            displacement = np.asarray(self.interpolator(normalized), dtype=float) * self.scale
            result[start:stop] += displacement
        return result


@dataclass(frozen=True)
class CompositeTpsModel:
    models: tuple[TpsModel, ...]

    def warp(self, xy: np.ndarray, *, chunk_size: int) -> np.ndarray:
        result = np.asarray(xy, dtype=float)
        for model in self.models:
            result = model.warp(result, chunk_size=chunk_size)
        return result


@dataclass(frozen=True)
class SliceCellTransform:
    sample_id: str
    order: int
    z_um: float
    hard: SimilarityTransform | None = None
    tps: TpsModel | CompositeTpsModel | None = None


def run_cell_cloud_projection(cfg: CellCloudProjectionConfig) -> CellCloudProjectionResult:
    """Project an AnnData cell table into a 3D reconstruction and write Parquet."""

    try:
        import anndata as ad
    except Exception as exc:
        raise ImportError("Cell cloud projection requires anndata to read .h5ad files.") from exc

    stack_root = Path(cfg.stack_root).expanduser().resolve()
    out_parquet = Path(cfg.out_parquet).expanduser().resolve()
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    _validate_projection_config(cfg)

    manifest = build_alignment_manifest(stack_root, pixel_size_um=cfg.pixel_size_um)
    alignment_hash = hash_alignment_manifest(manifest)

    h5ad_path = Path(cfg.h5ad).expanduser().resolve()
    backed = None if cfg.write_cache else "r"
    adata = ad.read_h5ad(h5ad_path, backed=backed)
    try:
        cache_status = "ignored" if cfg.ignore_cache else cell_cloud_cache_status(adata, alignment_hash)
        if cache_status == "stale" and cfg.fail_on_stale_cache:
            raise ValueError(
                "AnnData contains stale HistoSeg 3D coordinates for a different "
                "alignment hash. Use --ignore-cache or refresh with --write-cache."
            )

        obs_columns = _projection_obs_columns(cfg)
        obs = _adata_obs_to_frame(adata, obs_columns)
        if cache_status == "valid":
            projected = _cell_cloud_dataframe_from_cache(obs, adata, cfg)
        else:
            transforms = load_cell_alignment_transforms(
                stack_root,
                pixel_size_um=cfg.pixel_size_um,
                chunk_size=cfg.chunk_size,
            )
            coords_um, slice_order = project_cell_coordinates(
                obs,
                transforms,
                sample_column=cfg.sample_column,
                x_column=cfg.x_column,
                y_column=cfg.y_column,
                pixel_size_um=cfg.pixel_size_um,
                chunk_size=cfg.chunk_size,
            )
            projected = cell_cloud_dataframe_from_coordinates(obs, coords_um, slice_order, cfg)
            if cfg.write_cache:
                provenance = make_cell_cloud_cache_provenance(
                    stack_root=stack_root,
                    alignment_hash=alignment_hash,
                    manifest=manifest,
                    pixel_size_um=cfg.pixel_size_um,
                )
                write_cell_cloud_cache(
                    adata,
                    coords_um,
                    slice_order,
                    provenance,
                    write_scanpy_spatial=cfg.write_scanpy_spatial,
                )

        projected.to_parquet(out_parquet, index=False)

        cache_h5ad = None
        if cfg.write_cache:
            cache_h5ad = Path(cfg.cache_h5ad).expanduser().resolve() if cfg.cache_h5ad else h5ad_path
            cache_h5ad.parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(cache_h5ad)

        return CellCloudProjectionResult(
            out_parquet=out_parquet,
            cell_count=int(len(obs)),
            projected_cell_count=int(len(projected)),
            stack_root=stack_root,
            alignment_hash=alignment_hash,
            cache_status=cache_status,
            cache_h5ad=cache_h5ad,
        )
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def render_cell_cloud_html(cfg: CellCloudRenderConfig) -> CellCloudRenderResult:
    """Render an aligned HistoSeg cell cloud Parquet as an interactive Plotly HTML."""

    _validate_render_config(cfg)
    aligned_cells_parquet = Path(cfg.aligned_cells_parquet).expanduser().resolve()
    stack_root = Path(cfg.stack_root).expanduser().resolve()
    out_html = Path(cfg.out_html).expanduser().resolve()
    out_html.parent.mkdir(parents=True, exist_ok=True)

    cells = pd.read_parquet(aligned_cells_parquet)
    _require_columns(cells, [cfg.label_column, "x_3d_um", "y_3d_um", "z_um"])
    cells = _finite_cell_cloud_rows(cells)
    total_cells = int(len(cells))
    render_target = min(total_cells, int(cfg.max_points))
    warning_threshold = int(cfg.performance_warning_threshold)
    if warning_threshold > 0 and (total_cells > warning_threshold or render_target > warning_threshold):
        LOGGER.warning(
            "Cell cloud has %s valid cells and render target is %s points; "
            "Plotly scatter3d can become slow above %s points. Use --max-points to downsample.",
            total_cells,
            render_target,
            warning_threshold,
        )

    sampled = _sample_cell_cloud(
        cells,
        label_column=cfg.label_column,
        max_points=int(cfg.max_points),
        random_state=int(cfg.random_state),
    )
    contour_traces = (
        _load_cell_cloud_contour_traces(stack_root, z_visual_scale=float(cfg.z_visual_scale))
        if cfg.include_contours
        else []
    )
    _write_cell_cloud_plotly_html(
        sampled,
        out_html,
        label_column=cfg.label_column,
        title=cfg.title,
        z_visual_scale=float(cfg.z_visual_scale),
        marker_size=float(cfg.marker_size),
        opacity=float(cfg.opacity),
        extra_traces=contour_traces,
        total_cells=total_cells,
        parquet_path=aligned_cells_parquet,
    )

    return CellCloudRenderResult(
        out_html=out_html,
        aligned_cells_parquet=aligned_cells_parquet,
        stack_root=stack_root,
        total_cells=total_cells,
        rendered_cells=int(len(sampled)),
        label_column=cfg.label_column,
        label_count=int(sampled[cfg.label_column].astype(str).nunique()),
        contour_trace_count=len(contour_traces),
        z_visual_scale=float(cfg.z_visual_scale),
    )


def build_alignment_manifest(
    stack_root: PathLike,
    *,
    pixel_size_um: float,
) -> dict[str, Any]:
    """Build a deterministic manifest for geometry-defining stack state."""

    root = Path(stack_root).expanduser().resolve()
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be greater than 0.")

    manifest_path = root / "aligned_slice_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest_df = pd.read_csv(manifest_path)
    required = {"order", "sample_id", "z_um"}
    missing = sorted(required.difference(manifest_df.columns))
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {missing}")

    pairwise_path = root / "pairwise_alignment_metrics.csv"
    pairwise_df = pd.read_csv(pairwise_path) if pairwise_path.exists() else pd.DataFrame()
    pairwise_by_sample = {
        str(row["moving_sample_id"]): row
        for _, row in pairwise_df.iterrows()
        if "moving_sample_id" in pairwise_df.columns
    }

    slices: list[dict[str, Any]] = []
    for _, row in manifest_df.sort_values("order").iterrows():
        sample_id = str(row["sample_id"])
        order = int(row["order"])
        slice_payload: dict[str, Any] = {
            "order": order,
            "sample_id": sample_id,
            "z_um": float(row["z_um"]),
            "alignment_role": str(row.get("alignment_role", "")),
        }
        pairwise_row = pairwise_by_sample.get(sample_id)
        if order == 1:
            slice_payload["hard_transform"] = None
            slice_payload["hard_registration"] = None
            slice_payload["soft_transform"] = None
            slices.append(slice_payload)
            continue

        hard_summary_path = _find_hard_summary(root, sample_id, pairwise_row)
        hard_summary = _load_json(hard_summary_path)
        slice_payload["hard_transform"] = _canonicalize(hard_summary["transform"])
        slice_payload["hard_registration"] = _hard_registration_manifest_payload(hard_summary)

        soft_payload = None
        if pairwise_row is not None and _truthy(pairwise_row.get("soft_accepted")):
            soft_summary_path = _find_soft_summary(root, sample_id, pairwise_row)
            if soft_summary_path is not None and soft_summary_path.exists():
                soft_summary = _load_json(soft_summary_path)
                landmarks_path = _resolve_existing_path(
                    soft_summary.get("outputs", {}).get("landmarks_csv"),
                    base=soft_summary_path.parent,
                )
                method_raw = soft_summary.get("method", {})
                method = method_raw if isinstance(method_raw, Mapping) else {}
                soft_payload = {
                    "landmarks_sha256": _sha256_path(landmarks_path) if landmarks_path else None,
                    "method": method_raw if isinstance(method_raw, str) else method.get("name"),
                    "rbf_kernel": method.get("rbf_kernel"),
                    "rbf_neighbors": method.get("rbf_neighbors"),
                    "rbf_smoothing": method.get("rbf_smoothing"),
                    "landmark_candidate_count": method.get("landmark_candidate_count"),
                    "landmark_normal_weight_um": method.get("landmark_normal_weight_um"),
                    "topology_grid_size": method.get("topology_grid_size"),
                    "topology_min_area_ratio": method.get("topology_min_area_ratio"),
                    "topology_max_area_ratio": method.get("topology_max_area_ratio"),
                }
        slice_payload["soft_transform"] = soft_payload
        slices.append(slice_payload)

    return {
        "schema_version": ALIGNMENT_MANIFEST_SCHEMA_VERSION,
        "coordinate_contract": "histoseg_3d_cell_cloud",
        "pixel_size_um": float(pixel_size_um),
        "stack_root_name": root.name,
        "slices": slices,
    }


def _hard_registration_manifest_payload(hard_summary: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "registration_backend": hard_summary.get("registration_backend", "contour-tps"),
    }
    if hard_summary.get("selected_hard_seed_backend") is not None:
        payload["selected_hard_seed_backend"] = hard_summary.get("selected_hard_seed_backend")
    if hard_summary.get("method_credit") is not None:
        payload["method_credit"] = hard_summary.get("method_credit")
    if hard_summary.get("method_reference_doi") is not None:
        payload["method_reference_doi"] = hard_summary.get("method_reference_doi")

    tournament = hard_summary.get("hard_alignment_tournament")
    if isinstance(tournament, Mapping):
        payload["hard_alignment_tournament"] = {
            "strategy": tournament.get("strategy"),
            "selected_backend": tournament.get("selected_backend"),
            "rotation_difference_degrees": tournament.get("rotation_difference_degrees"),
        }
        candidates = []
        for candidate in hard_summary.get("hard_alignment_candidates") or ():
            if not isinstance(candidate, Mapping):
                continue
            candidates.append(
                {
                    "backend": candidate.get("backend"),
                    "union_iou_after_hard": candidate.get("union_iou_after_hard"),
                    "hard_alignment_accepted": candidate.get("hard_alignment_accepted"),
                    "rotation_degrees": candidate.get("rotation_degrees"),
                    "scale": candidate.get("scale"),
                    "translate_x": candidate.get("translate_x"),
                    "translate_y": candidate.get("translate_y"),
                }
            )
        if candidates:
            payload["hard_alignment_candidates"] = candidates

    coda_image = hard_summary.get("coda_image")
    if isinstance(coda_image, Mapping):
        preprocessing = coda_image.get("preprocessing") or {}
        payload["coda_image"] = {
            "radon_angle_range": _canonicalize(coda_image.get("radon_angle_range")),
            "radon_angle_step": coda_image.get("radon_angle_step"),
            "radon_rotation_degrees": coda_image.get("radon_rotation_degrees"),
            "phase_shift_y": coda_image.get("phase_shift_y"),
            "phase_shift_x": coda_image.get("phase_shift_x"),
            "phase_upsample_factor": coda_image.get("phase_upsample_factor"),
            "preprocessing": {
                "input_source": preprocessing.get("input_source"),
                "raster_size": preprocessing.get("raster_size"),
                "square_bounds": _canonicalize(preprocessing.get("square_bounds")),
                "native_units_per_pixel": preprocessing.get("native_units_per_pixel"),
                "mask_padding_fraction": preprocessing.get("mask_padding_fraction"),
            },
        }
    return _canonicalize(payload)


def hash_alignment_manifest(manifest: Mapping[str, Any]) -> str:
    """Hash canonical JSON for a curated alignment manifest."""

    payload = canonical_json(manifest).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize nested manifest data with stable key and numeric ordering."""

    return json.dumps(
        _canonicalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def load_cell_alignment_transforms(
    stack_root: PathLike,
    *,
    pixel_size_um: float,
    chunk_size: int = 100000,
) -> list[SliceCellTransform]:
    """Load hard/TPS transforms needed to map per-slice cells into 3D."""

    root = Path(stack_root).expanduser().resolve()
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be greater than 0.")
    _validate_chunk_size(chunk_size)

    manifest_path = root / "aligned_slice_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest_df = pd.read_csv(manifest_path)
    required = {"order", "sample_id", "z_um"}
    missing = sorted(required.difference(manifest_df.columns))
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {missing}")

    pairwise_path = root / "pairwise_alignment_metrics.csv"
    pairwise_df = pd.read_csv(pairwise_path) if pairwise_path.exists() else pd.DataFrame()
    pairwise_by_sample = {
        str(row["moving_sample_id"]): row
        for _, row in pairwise_df.iterrows()
        if "moving_sample_id" in pairwise_df.columns
    }

    transforms: list[SliceCellTransform] = []
    for _, row in manifest_df.sort_values("order").iterrows():
        sample_id = str(row["sample_id"])
        order = int(row["order"])
        z_um = float(row["z_um"])
        if order == 1:
            transforms.append(SliceCellTransform(sample_id=sample_id, order=order, z_um=z_um))
            continue

        pairwise_row = pairwise_by_sample.get(sample_id)
        hard_summary = _load_json(_find_hard_summary(root, sample_id, pairwise_row))
        hard = _similarity_transform_from_payload(hard_summary["transform"])

        tps = None
        if pairwise_row is not None and _truthy(pairwise_row.get("soft_accepted")):
            soft_summary_path = _find_soft_summary(root, sample_id, pairwise_row)
            if soft_summary_path is not None and soft_summary_path.exists():
                tps = _load_tps_model(soft_summary_path)

        transforms.append(
            SliceCellTransform(
                sample_id=sample_id,
                order=order,
                z_um=z_um,
                hard=hard,
                tps=tps,
            )
        )
    return transforms


def project_cell_coordinates(
    obs: pd.DataFrame,
    transforms: Sequence[SliceCellTransform],
    *,
    sample_column: str,
    x_column: str,
    y_column: str,
    pixel_size_um: float,
    chunk_size: int = 100000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned ``xyz`` coordinates in microns and slice order per obs row."""

    _validate_chunk_size(chunk_size)
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be greater than 0.")
    _require_columns(obs, [sample_column, x_column, y_column])

    coords_um = np.full((len(obs), 3), np.nan, dtype=float)
    slice_order = np.full(len(obs), -1, dtype=int)
    sample_values = obs[sample_column].astype(str).to_numpy()
    x_um = pd.to_numeric(obs[x_column], errors="coerce").to_numpy(dtype=float)
    y_um = pd.to_numeric(obs[y_column], errors="coerce").to_numpy(dtype=float)

    for transform in transforms:
        positions = np.flatnonzero(sample_values == str(transform.sample_id))
        if positions.size == 0:
            continue
        finite = np.isfinite(x_um[positions]) & np.isfinite(y_um[positions])
        positions = positions[finite]
        if positions.size == 0:
            continue

        xy_native = np.column_stack([x_um[positions], y_um[positions]]) / float(pixel_size_um)
        if transform.hard is not None:
            xy_native = apply_similarity_to_points(xy_native, transform.hard)
        if transform.tps is not None:
            xy_native = transform.tps.warp(xy_native, chunk_size=chunk_size)
        coords_um[positions, 0] = xy_native[:, 0] * float(pixel_size_um)
        coords_um[positions, 1] = xy_native[:, 1] * float(pixel_size_um)
        coords_um[positions, 2] = float(transform.z_um)
        slice_order[positions] = int(transform.order)

    return coords_um, slice_order


def cell_cloud_dataframe_from_coordinates(
    obs: pd.DataFrame,
    coords_um: np.ndarray,
    slice_order: np.ndarray,
    cfg: CellCloudProjectionConfig,
) -> pd.DataFrame:
    """Create the public aligned cell table from projected arrays."""

    _require_columns(obs, _projection_obs_columns(cfg))
    coords = np.asarray(coords_um, dtype=float)
    slice_order = np.asarray(slice_order, dtype=int)
    if coords.shape != (len(obs), 3):
        raise ValueError(f"coords_um must have shape ({len(obs)}, 3), got {coords.shape}.")
    if slice_order.shape != (len(obs),):
        raise ValueError(f"slice_order must have shape ({len(obs)},), got {slice_order.shape}.")

    valid = np.isfinite(coords).all(axis=1) & (slice_order >= 0)
    base_columns = [cfg.sample_column, cfg.barcode_column, *cfg.label_columns]
    result = obs.loc[valid, base_columns].copy().reset_index(drop=True)
    result["slice_order"] = slice_order[valid].astype(int)
    result["x_3d_um"] = coords[valid, 0]
    result["y_3d_um"] = coords[valid, 1]
    result["z_um"] = coords[valid, 2]
    result["x_raw_um"] = pd.to_numeric(obs.loc[valid, cfg.x_column], errors="coerce").to_numpy(dtype=float)
    result["y_raw_um"] = pd.to_numeric(obs.loc[valid, cfg.y_column], errors="coerce").to_numpy(dtype=float)
    return result


def apply_similarity_to_points(
    xy: np.ndarray,
    transform: SimilarityTransform,
) -> np.ndarray:
    """Apply the same row-vector similarity convention used by 3D contour alignment."""

    result = np.asarray(xy, dtype=float).copy()
    origin = np.array([transform.origin_x, transform.origin_y], dtype=float)
    theta = math.radians(float(transform.rotation_degrees))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotation = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=float)
    result = (result - origin) @ rotation + origin
    result = (result - origin) * float(transform.scale) + origin
    result[:, 0] += float(transform.translate_x)
    result[:, 1] += float(transform.translate_y)
    return result


def cell_cloud_cache_status(adata: Any, alignment_hash: str) -> str:
    """Return ``valid``, ``stale``, or ``missing`` for an AnnData coordinate cache."""

    obsm = getattr(adata, "obsm", {})
    obs = getattr(adata, "obs", pd.DataFrame())
    uns = getattr(adata, "uns", {})
    if CELL_CLOUD_OBSM_KEY not in obsm or CELL_CLOUD_OBS_SLICE_KEY not in obs:
        return "missing"
    provenance = uns.get(CELL_CLOUD_UNS_KEY, {}) if isinstance(uns, Mapping) else {}
    if not isinstance(provenance, Mapping):
        return "missing"
    cached_hash = provenance.get("alignment_hash")
    if not cached_hash:
        return "missing"
    return "valid" if str(cached_hash) == str(alignment_hash) else "stale"


def write_cell_cloud_cache(
    adata: Any,
    coords_um: np.ndarray,
    slice_order: np.ndarray,
    provenance: Mapping[str, Any],
    *,
    write_scanpy_spatial: bool = False,
) -> None:
    """Write aligned 3D coordinates to AnnData as a cache with provenance."""

    coords = np.asarray(coords_um, dtype=float)
    slice_order = np.asarray(slice_order, dtype=int)
    n_obs = int(getattr(adata, "n_obs", len(getattr(adata, "obs"))))
    if coords.shape != (n_obs, 3):
        raise ValueError(f"coords_um must have shape ({n_obs}, 3), got {coords.shape}.")
    if slice_order.shape != (n_obs,):
        raise ValueError(f"slice_order must have shape ({n_obs},), got {slice_order.shape}.")

    adata.obsm[CELL_CLOUD_OBSM_KEY] = coords
    adata.obsm[CELL_CLOUD_ALIGNED_XY_OBSM_KEY] = coords[:, :2]
    obs_index = getattr(getattr(adata, "obs", None), "index", None)
    adata.obs[CELL_CLOUD_OBS_SLICE_KEY] = pd.Series(
        np.where(slice_order >= 0, slice_order, np.nan),
        index=obs_index,
    )
    adata.uns[CELL_CLOUD_UNS_KEY] = dict(provenance)
    if write_scanpy_spatial:
        adata.obsm["spatial"] = coords[:, :2]


def make_cell_cloud_cache_provenance(
    *,
    stack_root: PathLike,
    alignment_hash: str,
    manifest: Mapping[str, Any],
    pixel_size_um: float,
) -> dict[str, Any]:
    """Build AnnData ``.uns`` provenance for cached HistoSeg 3D coordinates."""

    return {
        "schema_version": CELL_CLOUD_CACHE_SCHEMA_VERSION,
        "alignment_manifest_schema_version": ALIGNMENT_MANIFEST_SCHEMA_VERSION,
        "alignment_hash": str(alignment_hash),
        "stack_root": str(Path(stack_root).expanduser().resolve()),
        "coordinate_units": "um",
        "pixel_size_um": float(pixel_size_um),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "obsm_key": CELL_CLOUD_OBSM_KEY,
        "aligned_xy_obsm_key": CELL_CLOUD_ALIGNED_XY_OBSM_KEY,
        "obs_slice_key": CELL_CLOUD_OBS_SLICE_KEY,
        "manifest": _canonicalize(manifest),
    }


def _finite_cell_cloud_rows(cells: pd.DataFrame) -> pd.DataFrame:
    frame = cells.copy()
    for column in ("x_3d_um", "y_3d_um", "z_um"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(frame[["x_3d_um", "y_3d_um", "z_um"]].to_numpy(dtype=float)).all(axis=1)
    return frame.loc[finite].reset_index(drop=True)


def _sample_cell_cloud(
    cells: pd.DataFrame,
    *,
    label_column: str,
    max_points: int,
    random_state: int,
) -> pd.DataFrame:
    if max_points <= 0:
        raise ValueError("max_points must be greater than 0.")
    if len(cells) <= max_points:
        return cells.copy().reset_index(drop=True)

    strata_columns = [label_column]
    if "slice_order" in cells.columns:
        strata_columns.append("slice_order")
    grouped = list(cells.groupby(strata_columns, sort=False, dropna=False))
    counts = pd.Series([len(group) for _, group in grouped], dtype=float)
    raw_targets = counts * float(max_points) / float(max(int(counts.sum()), 1))
    targets = np.floor(raw_targets).astype(int).clip(upper=counts.astype(int))

    remaining = int(max_points - targets.sum())
    if remaining > 0:
        capacity = counts.astype(int) - targets
        remainders = (raw_targets - np.floor(raw_targets)).sort_values(ascending=False)
        while remaining > 0 and int(capacity.sum()) > 0:
            for index in remainders.index:
                if remaining <= 0:
                    break
                if int(capacity.loc[index]) <= 0:
                    continue
                targets.loc[index] += 1
                capacity.loc[index] -= 1
                remaining -= 1

    pieces: list[pd.DataFrame] = []
    for index, (_, group) in enumerate(grouped):
        target = int(targets.loc[index])
        if target <= 0:
            continue
        if len(group) <= target:
            pieces.append(group)
        else:
            pieces.append(group.sample(n=target, random_state=random_state + index))
    if not pieces:
        return cells.sample(n=max_points, random_state=random_state).reset_index(drop=True)
    return pd.concat(pieces, ignore_index=True)


def _load_cell_cloud_contour_traces(stack_root: Path, *, z_visual_scale: float) -> list[dict[str, Any]]:
    contour_path = Path(stack_root) / "aligned_contour_3d_points.csv"
    if not contour_path.exists():
        return []
    contour = pd.read_csv(contour_path)
    required = {"structure", "slice_order", "polyline_id", "point_index", "x_um", "y_um", "z_um"}
    if not required.issubset(contour.columns):
        return []

    traces: list[dict[str, Any]] = []
    for index, (structure, group) in enumerate(contour.groupby("structure", sort=True)):
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        z_values: list[float | None] = []
        for _, polyline in group.sort_values(
            ["slice_order", "polyline_id", "point_index"]
        ).groupby(["slice_order", "polyline_id"], sort=False):
            x_values.extend(polyline["x_um"].round(3).tolist())
            y_values.extend(polyline["y_um"].round(3).tolist())
            z_values.extend((polyline["z_um"] * z_visual_scale).round(3).tolist())
            x_values.append(None)
            y_values.append(None)
            z_values.append(None)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": f"{structure} contour",
                "x": x_values,
                "y": y_values,
                "z": z_values,
                "line": {"width": 3, "color": CELL_CLOUD_PALETTE[index % len(CELL_CLOUD_PALETTE)]},
                "opacity": 0.72,
                "visible": "legendonly",
                "hoverinfo": "skip",
            }
        )
    return traces


def _write_cell_cloud_plotly_html(
    cells: pd.DataFrame,
    output_html: Path,
    *,
    label_column: str,
    title: str,
    z_visual_scale: float,
    marker_size: float,
    opacity: float,
    extra_traces: Sequence[Mapping[str, Any]],
    total_cells: int,
    parquet_path: Path,
) -> None:
    traces: list[dict[str, Any]] = []
    labels = sorted(cells[label_column].astype(str).unique(), key=_cell_cloud_label_sort_key)
    for index, label in enumerate(labels):
        group = cells.loc[cells[label_column].astype(str) == label]
        z_true = group["z_um"].to_numpy(dtype=float)
        trace_name = _cell_cloud_trace_name(label_column, label)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": trace_name,
                "x": group["x_3d_um"].round(3).tolist(),
                "y": group["y_3d_um"].round(3).tolist(),
                "z": (z_true * z_visual_scale).round(3).tolist(),
                "customdata": z_true.round(3).tolist(),
                "marker": {
                    "size": marker_size,
                    "color": CELL_CLOUD_PALETTE[index % len(CELL_CLOUD_PALETTE)],
                    "opacity": opacity,
                    "line": {"width": 0},
                },
                "hovertemplate": (
                    f"{trace_name}<br>"
                    "x=%{x:.1f} um<br>"
                    "y=%{y:.1f} um<br>"
                    "z=%{customdata:.1f} um<extra></extra>"
                ),
            }
        )

    traces.extend(dict(trace) for trace in extra_traces)
    subtitle = f"Showing {len(cells):,} of {total_cells:,} cells; full table: {parquet_path}"
    z_title = "Z (um)"
    if z_visual_scale != 1:
        z_title = f"Z (um, visual x{z_visual_scale:g})"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; background: #08111f; color: #d8e2ef; font-family: Arial, sans-serif; }}
    #plot {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const data = {json.dumps(traces)};
    const layout = {{
      title: {{
        text: {json.dumps(title + "<br><sup>" + subtitle + "</sup>")},
        font: {{color: "#d8e2ef", size: 20}}
      }},
      paper_bgcolor: "#08111f",
      plot_bgcolor: "#08111f",
      scene: {{
        xaxis: {{title: "X (um)", color: "#b9c7d6", gridcolor: "#27364a"}},
        yaxis: {{title: "Y (um)", color: "#b9c7d6", gridcolor: "#27364a"}},
        zaxis: {{title: {json.dumps(z_title)}, color: "#b9c7d6", gridcolor: "#27364a"}},
        aspectmode: "data"
      }},
      legend: {{font: {{color: "#d8e2ef"}}, itemsizing: "constant"}},
      margin: {{l: 0, r: 0, t: 62, b: 0}}
    }};
    Plotly.newPlot("plot", data, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def _cell_cloud_label_sort_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _cell_cloud_trace_name(label_column: str, label: str) -> str:
    if label_column.lower().startswith("leiden"):
        return f"Leiden {label}"
    return f"{label_column} {label}"


def _cell_cloud_dataframe_from_cache(
    obs: pd.DataFrame,
    adata: Any,
    cfg: CellCloudProjectionConfig,
) -> pd.DataFrame:
    coords = np.asarray(adata.obsm[CELL_CLOUD_OBSM_KEY], dtype=float)
    slice_order = pd.to_numeric(adata.obs[CELL_CLOUD_OBS_SLICE_KEY], errors="coerce").fillna(-1).to_numpy(dtype=int)
    return cell_cloud_dataframe_from_coordinates(obs, coords, slice_order, cfg)


def _load_tps_model(summary_path: Path) -> TpsModel | CompositeTpsModel:
    summary = _load_json(summary_path)
    chain = summary.get("outputs", {}).get("tps_chain")
    if isinstance(chain, list) and chain:
        models: list[TpsModel] = []
        for value in chain:
            chain_path = _resolve_existing_path(value, base=summary_path.parent)
            if chain_path is None:
                raise FileNotFoundError(
                    f"Could not resolve TPS chain summary {value!r} from {summary_path}"
                )
            model = _load_tps_model(chain_path)
            if isinstance(model, CompositeTpsModel):
                models.extend(model.models)
            else:
                models.append(model)
        return CompositeTpsModel(tuple(models))
    landmarks_path = _resolve_existing_path(
        summary.get("outputs", {}).get("landmarks_csv"),
        base=summary_path.parent,
    )
    if landmarks_path is None:
        sibling_landmarks = summary_path.parent / "soft_tps_landmarks.csv"
        if sibling_landmarks.exists():
            landmarks_path = sibling_landmarks
    if landmarks_path is None:
        raise FileNotFoundError(f"Could not resolve landmarks_csv from {summary_path}")
    landmarks = pd.read_csv(landmarks_path)
    required = {"src_x", "src_y", "dst_x", "dst_y"}
    missing = sorted(required.difference(landmarks.columns))
    if missing:
        raise ValueError(f"{landmarks_path} is missing columns: {missing}")
    src = landmarks[["src_x", "src_y"]].to_numpy(dtype=float)
    dst = landmarks[["dst_x", "dst_y"]].to_numpy(dtype=float)
    center = src.mean(axis=0)
    scale = float(np.ptp(src, axis=0).max())
    if scale <= 0:
        raise ValueError(f"TPS landmarks in {landmarks_path} do not span a coordinate range.")

    method_raw = summary.get("method", {})
    method = method_raw if isinstance(method_raw, Mapping) else {}
    neighbors_value = method.get("rbf_neighbors", 96)
    neighbors = int(neighbors_value) if neighbors_value is not None else None
    if neighbors is not None and len(src) <= neighbors:
        neighbors = None

    interpolator = RBFInterpolator(
        (src - center) / scale,
        (dst - src) / scale,
        kernel=str(method.get("rbf_kernel", "thin_plate_spline")),
        smoothing=float(method.get("rbf_smoothing", 1e-4)),
        neighbors=neighbors,
    )
    return TpsModel(interpolator=interpolator, center_xy=center, scale=scale)


def _similarity_transform_from_payload(payload: Mapping[str, Any]) -> SimilarityTransform:
    keys = (
        "origin_x",
        "origin_y",
        "rotation_degrees",
        "scale",
        "translate_x",
        "translate_y",
    )
    missing = sorted(key for key in keys if key not in payload)
    if missing:
        raise ValueError(
            "Hard alignment transform is missing required fields: "
            + ", ".join(missing)
        )
    return SimilarityTransform(**{key: float(payload[key]) for key in keys})


def _find_hard_summary(
    stack_root: Path,
    sample_id: str,
    pairwise_row: Mapping[str, Any] | None,
) -> Path:
    if pairwise_row is not None:
        soft_value = pairwise_row.get("soft_summary_json")
        soft_summary = _resolve_existing_path(soft_value, base=stack_root)
        if soft_summary is not None:
            candidate = soft_summary.parent.parent / "hard_similarity_alignment.json"
            if candidate.exists():
                return candidate

    matches = sorted((stack_root / "pairwise_alignments").glob(f"*_{sample_id}/hard_similarity_alignment.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find hard_similarity_alignment.json for {sample_id}")
    raise ValueError(f"Multiple hard summaries matched {sample_id}: {matches}")


def _find_soft_summary(
    stack_root: Path,
    sample_id: str,
    pairwise_row: Mapping[str, Any] | None,
) -> Path | None:
    if pairwise_row is not None:
        candidate = _resolve_existing_path(pairwise_row.get("soft_summary_json"), base=stack_root)
        if candidate is not None:
            return candidate
    matches = sorted(
        (stack_root / "pairwise_alignments").glob(
            f"*_{sample_id}/soft_tps/soft_tps_alignment_summary.json"
        )
    )
    return matches[0] if matches else None


def _resolve_existing_path(value: Any, *, base: Path) -> Path | None:
    if not _has_value(value):
        return None
    candidate = Path(str(value))
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        rooted = base / candidate
        if rooted.exists():
            return rooted
    return candidate if candidate.exists() else None


def _adata_obs_to_frame(adata: Any, columns: Sequence[str]) -> pd.DataFrame:
    obs = getattr(adata, "obs")
    _require_columns(obs, columns)
    frame = obs.loc[:, list(columns)].copy()
    for column in columns:
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].astype(str)
    return frame


def _projection_obs_columns(cfg: CellCloudProjectionConfig) -> list[str]:
    columns = [cfg.sample_column, cfg.barcode_column, cfg.x_column, cfg.y_column, *cfg.label_columns]
    return list(dict.fromkeys(columns))


def _validate_projection_config(cfg: CellCloudProjectionConfig) -> None:
    if cfg.pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be greater than 0.")
    _validate_chunk_size(cfg.chunk_size)
    if cfg.write_scanpy_spatial and not cfg.write_cache:
        raise ValueError("--write-scanpy-spatial requires --write-cache.")


def _validate_render_config(cfg: CellCloudRenderConfig) -> None:
    if int(cfg.max_points) <= 0:
        raise ValueError("max_points must be greater than 0.")
    if float(cfg.z_visual_scale) <= 0:
        raise ValueError("z_visual_scale must be greater than 0.")
    if float(cfg.marker_size) <= 0:
        raise ValueError("marker_size must be greater than 0.")
    if not 0 <= float(cfg.opacity) <= 1:
        raise ValueError("opacity must be between 0 and 1.")
    if not str(cfg.label_column).strip():
        raise ValueError("label_column must not be empty.")


def _validate_chunk_size(chunk_size: int) -> None:
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be greater than 0.")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip() != ""


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _canonicalize(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Manifest contains a non-finite float.")
        return float(f"{value:.12g}")
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "CELL_CLOUD_ALIGNED_XY_OBSM_KEY",
    "CELL_CLOUD_OBS_SLICE_KEY",
    "CELL_CLOUD_OBSM_KEY",
    "CELL_CLOUD_UNS_KEY",
    "CellCloudProjectionConfig",
    "CellCloudProjectionResult",
    "CellCloudRenderConfig",
    "CellCloudRenderResult",
    "SimilarityTransform",
    "SliceCellTransform",
    "TpsModel",
    "apply_similarity_to_points",
    "build_alignment_manifest",
    "canonical_json",
    "cell_cloud_cache_status",
    "cell_cloud_dataframe_from_coordinates",
    "hash_alignment_manifest",
    "load_cell_alignment_transforms",
    "make_cell_cloud_cache_provenance",
    "project_cell_coordinates",
    "render_cell_cloud_html",
    "run_cell_cloud_projection",
    "write_cell_cloud_cache",
]
