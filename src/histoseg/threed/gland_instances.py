"""Lumen-seeded gland instance segmentation for aligned 3D HistoSeg stacks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import re
from typing import Any, Mapping, Sequence, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union
from skimage import draw, measure, segmentation


PathLike = Union[str, Path]


DEFAULT_EPITHELIAL_STRUCTURES = ("Structure 3", "Structure 4")
DEFAULT_EPITHELIAL_MARKERS = ("EPCAM", "MUC2", "LGR5", "OLFM4", "MKI67")


@dataclass(frozen=True)
class GlandInstanceSegmentationConfig:
    """Configuration for per-slice gland/crypt instance segmentation."""

    stack_root: PathLike
    aligned_cells_parquet: PathLike
    out_dir: PathLike
    structures: Sequence[str] = DEFAULT_EPITHELIAL_STRUCTURES
    epithelial_markers: Sequence[str] = DEFAULT_EPITHELIAL_MARKERS
    lumen_min_area_um2: float = 200.0
    lumen_cell_density_threshold: float = 0.05
    watershed_compactness: float = 0.01
    pixel_size_um: float = 0.2125
    raster_pixel_size_um: float = 5.0
    xenium_cell_boundaries_relpath: str = "cell_boundaries.parquet"
    xenium_nucleus_boundaries_relpath: str = "nucleus_boundaries.parquet"
    xenium_transcripts_relpath: str = "transcripts.parquet"
    min_ring_support_score: float = 0.3
    min_gland_area_um2: float = 500.0
    group_property: str = "structure"
    cell_density_sigma_um: float = 12.0
    epithelial_density_threshold: float = 0.10
    support_closing_um: float = 12.0
    max_slices: int | None = None


@dataclass
class GlandInstanceSegmentationResult:
    out_dir: Path
    slice_gland_instances_geojson: Path
    slice_qc_csv: Path
    slice_count: int
    total_gland_instances: int


@dataclass(frozen=True)
class _SliceContext:
    order: int
    sample_id: str
    z_um: float
    aligned_geojson: Path
    xenium_dir: Path | None


@dataclass(frozen=True)
class _GridSpec:
    x_min: float
    y_min: float
    pixel_size_um: float
    shape: tuple[int, int]

    @property
    def x_edges(self) -> np.ndarray:
        return self.x_min + np.arange(self.shape[1] + 1, dtype=float) * self.pixel_size_um

    @property
    def y_edges(self) -> np.ndarray:
        return self.y_min + np.arange(self.shape[0] + 1, dtype=float) * self.pixel_size_um


def segment_gland_instances(
    cfg: GlandInstanceSegmentationConfig,
) -> GlandInstanceSegmentationResult:
    """Segment per-slice gland instances from semantic epithelial contours."""

    _validate_config(cfg)
    stack_root = Path(cfg.stack_root).expanduser().resolve()
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_contexts = _load_slice_contexts(stack_root)
    if cfg.max_slices is not None:
        slice_contexts = slice_contexts[: int(cfg.max_slices)]

    cells = _load_aligned_cells(cfg)
    features: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    requested = tuple(str(item) for item in cfg.structures)
    markers = tuple(str(item) for item in cfg.epithelial_markers)

    for context in slice_contexts:
        slice_cells = cells.loc[
            (cells["sample_id"].astype(str) == context.sample_id)
            | (cells["slice_order"].astype(int) == context.order)
        ].copy()
        if slice_cells.empty:
            continue
        marker_counts = _marker_counts_for_slice(
            context=context,
            slice_cells=slice_cells,
            markers=markers,
            cfg=cfg,
        )
        slice_cells["_epithelial_marker_count"] = (
            slice_cells["barcode"].astype(str).map(marker_counts).fillna(0.0).astype(float)
        )
        by_structure = _load_aligned_structure_geometries(
            context.aligned_geojson,
            requested,
            group_property=cfg.group_property,
            pixel_size_um=cfg.pixel_size_um,
        )
        for structure_name, geometry in by_structure.items():
            structure_features, structure_rows = _segment_structure_on_slice(
                context=context,
                structure_name=structure_name,
                geometry=geometry,
                cells=slice_cells,
                markers=markers,
                cfg=cfg,
            )
            features.extend(structure_features)
            qc_rows.extend(structure_rows)

    geojson_path = out_dir / "slice_gland_instances.geojson"
    geojson_payload = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "source": "histoseg_gland_instance_segmentation",
            "config": asdict(cfg),
        },
    }
    geojson_path.write_text(
        json.dumps(geojson_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    qc_path = out_dir / "slice_gland_instances_qc.csv"
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False)

    return GlandInstanceSegmentationResult(
        out_dir=out_dir,
        slice_gland_instances_geojson=geojson_path,
        slice_qc_csv=qc_path,
        slice_count=len({row["slice_order"] for row in qc_rows}),
        total_gland_instances=len(qc_rows),
    )


def _segment_structure_on_slice(
    *,
    context: _SliceContext,
    structure_name: str,
    geometry: Any,
    cells: pd.DataFrame,
    markers: Sequence[str],
    cfg: GlandInstanceSegmentationConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grid = _grid_for_geometry(geometry, cfg.raster_pixel_size_um)
    semantic_mask = _geometry_to_mask(geometry, grid)
    if not semantic_mask.any():
        return [], []

    local_cells = _cells_in_grid_bounds(cells, grid)
    cell_hist = _histogram_cells(local_cells, grid, weight_col=None)
    marker_hist = _histogram_cells(local_cells, grid, weight_col="_epithelial_marker_count")
    density_sigma_px = max(float(cfg.cell_density_sigma_um) / float(cfg.raster_pixel_size_um), 0.0)
    density = ndi.gaussian_filter(cell_hist.astype(float), sigma=density_sigma_px)
    marker_density = ndi.gaussian_filter(marker_hist.astype(float), sigma=density_sigma_px)

    support_values = density[semantic_mask]
    positive_support = support_values[support_values > 0]
    density_p95 = float(np.percentile(positive_support, 95)) if positive_support.size else 0.0
    if density_p95 > 0:
        normalized_density = density / density_p95
    else:
        normalized_density = np.zeros_like(density, dtype=float)

    marker_fraction = marker_density / (density + 1e-9)
    epithelial_support = semantic_mask & (
        (normalized_density >= float(cfg.epithelial_density_threshold))
        | (marker_fraction > 0)
    )
    if not epithelial_support.any():
        epithelial_support = semantic_mask.copy()

    closing_radius = max(1, int(round(float(cfg.support_closing_um) / float(cfg.raster_pixel_size_um))))
    epithelial_support = ndi.binary_closing(epithelial_support, structure=_disk(closing_radius))
    epithelial_support &= semantic_mask
    filled_domain = ndi.binary_fill_holes(epithelial_support) & semantic_mask
    if not filled_domain.any():
        filled_domain = semantic_mask.copy()

    lumen_mask = _detect_lumens(
        epithelial_support=epithelial_support,
        filled_domain=filled_domain,
        normalized_density=normalized_density,
        cfg=cfg,
    )
    lumen_labels, lumen_count = ndi.label(lumen_mask)
    domain_labels, domain_count = ndi.label(filled_domain)

    if lumen_count > 0:
        distance_to_lumen = ndi.distance_transform_edt(~lumen_mask)
        watershed_labels = segmentation.watershed(
            distance_to_lumen,
            markers=lumen_labels,
            mask=filled_domain,
            compactness=float(cfg.watershed_compactness),
        )
    else:
        watershed_labels = domain_labels

    features: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for local_index, label_value in enumerate(sorted(np.unique(watershed_labels[watershed_labels > 0])), start=1):
        instance_mask = watershed_labels == int(label_value)
        if not instance_mask.any():
            continue
        outer_polygon = _mask_to_largest_polygon(instance_mask, grid)
        if outer_polygon is None or outer_polygon.is_empty or outer_polygon.area <= 0:
            continue
        lumen_component = lumen_labels == int(label_value) if lumen_count > 0 else np.zeros_like(instance_mask)
        lumen_polygon = _mask_to_largest_polygon(lumen_component, grid) if lumen_component.any() else None
        row, properties = _instance_metrics(
            context=context,
            structure_name=structure_name,
            instance_index=local_index,
            instance_mask=instance_mask,
            lumen_component=lumen_component,
            lumen_labels=lumen_labels,
            outer_polygon=outer_polygon,
            lumen_polygon=lumen_polygon,
            grid=grid,
            epithelial_support=epithelial_support,
            local_cells=local_cells,
            markers=markers,
            cfg=cfg,
        )
        feature = {
            "type": "Feature",
            "id": str(uuid4()),
            "geometry": mapping(outer_polygon),
            "properties": properties,
        }
        features.append(feature)
        qc_rows.append(row)

    return features, qc_rows


def _instance_metrics(
    *,
    context: _SliceContext,
    structure_name: str,
    instance_index: int,
    instance_mask: np.ndarray,
    lumen_component: np.ndarray,
    lumen_labels: np.ndarray,
    outer_polygon: Polygon,
    lumen_polygon: Polygon | None,
    grid: _GridSpec,
    epithelial_support: np.ndarray,
    local_cells: pd.DataFrame,
    markers: Sequence[str],
    cfg: GlandInstanceSegmentationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    slice_instance_id = (
        f"{context.order:03d}_{_slug(structure_name)}_{instance_index:04d}"
    )
    centroid = outer_polygon.centroid
    area_um2 = float(outer_polygon.area)
    lumen_area_um2 = float(lumen_polygon.area) if lumen_polygon is not None else 0.0
    lumen_centroid = lumen_polygon.centroid if lumen_polygon is not None else None
    ring_support_score = _ring_support_score(instance_mask, epithelial_support)
    inside_cells = _cells_inside_mask(local_cells, instance_mask, grid)
    cell_count = int(len(inside_cells))
    marker_sum = float(inside_cells["_epithelial_marker_count"].sum()) if cell_count else 0.0
    epithelial_marker_score = float(marker_sum / max(cell_count, 1))
    stromal_contamination_score = (
        float((inside_cells["_epithelial_marker_count"] <= 0).mean()) if cell_count else math.nan
    )
    lumen_label_count = int(len(set(np.unique(lumen_labels[lumen_component])) - {0}))

    flag_no_lumen_seed = lumen_polygon is None or lumen_area_um2 <= 0
    flag_weak_ring = bool(ring_support_score < float(cfg.min_ring_support_score))
    flag_small_fragment = bool(area_um2 < float(cfg.min_gland_area_um2))
    flag_merged_candidate = bool(
        lumen_label_count > 1
        or lumen_area_um2 >= max(float(cfg.lumen_min_area_um2) * 4.0, float(cfg.lumen_min_area_um2) + 1.0)
    )
    flags = [
        name
        for name, enabled in [
            ("no_lumen_seed", flag_no_lumen_seed),
            ("weak_ring", flag_weak_ring),
            ("merged_candidate", flag_merged_candidate),
            ("small_fragment", flag_small_fragment),
        ]
        if enabled
    ]
    marker_profile = {
        str(marker): float(inside_cells.get(str(marker), pd.Series(dtype=float)).sum())
        for marker in markers
        if str(marker) in inside_cells.columns
    }
    if not marker_profile:
        marker_profile = {"epithelial_marker_count": marker_sum}

    row = {
        "slice_instance_id": slice_instance_id,
        "slice_order": int(context.order),
        "sample_id": context.sample_id,
        "z_um": float(context.z_um),
        "semantic_structure": structure_name,
        "area_um2": area_um2,
        "centroid_x_um": float(centroid.x),
        "centroid_y_um": float(centroid.y),
        "lumen_area_um2": lumen_area_um2,
        "lumen_centroid_x_um": float(lumen_centroid.x) if lumen_centroid is not None else math.nan,
        "lumen_centroid_y_um": float(lumen_centroid.y) if lumen_centroid is not None else math.nan,
        "ring_support_score": float(ring_support_score),
        "epithelial_marker_score": epithelial_marker_score,
        "stromal_immune_contamination_score": stromal_contamination_score,
        "cell_count": cell_count,
        "qc_flags": ";".join(flags),
        "flag_no_lumen_seed": flag_no_lumen_seed,
        "flag_weak_ring": flag_weak_ring,
        "flag_merged_candidate": flag_merged_candidate,
        "flag_small_fragment": flag_small_fragment,
        "marker_profile_json": json.dumps(marker_profile, sort_keys=True),
    }
    for marker, value in marker_profile.items():
        row[f"marker_{_slug(marker)}"] = float(value)

    properties = dict(row)
    properties.update(
        {
            "objectType": "annotation",
            "name": slice_instance_id,
            "structure": "gland_instance",
            "assigned_structure": structure_name,
            "source": "histoseg_gland_instance_segmentation",
            "lumen_polygon_coordinates": (
                [list(map(float, point)) for point in np.asarray(lumen_polygon.exterior.coords)]
                if lumen_polygon is not None
                else []
            ),
        }
    )
    return row, properties


def _detect_lumens(
    *,
    epithelial_support: np.ndarray,
    filled_domain: np.ndarray,
    normalized_density: np.ndarray,
    cfg: GlandInstanceSegmentationConfig,
) -> np.ndarray:
    holes = filled_domain & ~epithelial_support
    low_density = normalized_density <= float(cfg.lumen_cell_density_threshold)
    lumen = holes & low_density
    min_pixels = max(1, int(round(float(cfg.lumen_min_area_um2) / float(cfg.raster_pixel_size_um) ** 2)))
    lumen = _remove_small_objects(lumen, min_pixels)
    if not lumen.any():
        candidate = filled_domain & low_density
        candidate = _remove_small_objects(candidate, min_pixels)
        domain_boundary = filled_domain & ~ndi.binary_erosion(filled_domain)
        labels, count = ndi.label(candidate)
        keep = np.zeros_like(candidate, dtype=bool)
        for label_value in range(1, count + 1):
            component = labels == label_value
            if (component & domain_boundary).any():
                continue
            keep |= component
        lumen = keep
    labels, count = ndi.label(lumen)
    if count == 0:
        return lumen.astype(bool)
    keep = np.zeros_like(lumen, dtype=bool)
    for label_value in range(1, count + 1):
        component = labels == label_value
        if _touches_array_border(component):
            continue
        keep |= component
    return keep


def _load_slice_contexts(stack_root: Path) -> list[_SliceContext]:
    aligned_manifest_path = stack_root / "aligned_slice_manifest.csv"
    xenium_manifest_path = stack_root / "xenium_slice_manifest.csv"
    if not aligned_manifest_path.exists():
        raise FileNotFoundError(aligned_manifest_path)
    aligned = pd.read_csv(aligned_manifest_path)
    required = {"order", "sample_id", "z_um", "aligned_geojson"}
    missing = sorted(required.difference(aligned.columns))
    if missing:
        raise ValueError(f"{aligned_manifest_path} is missing columns: {missing}")

    xenium_lookup: dict[str, Path] = {}
    if xenium_manifest_path.exists():
        xenium = pd.read_csv(xenium_manifest_path)
        if {"sample_id", "xenium_dir"}.issubset(xenium.columns):
            xenium_lookup = {
                str(row["sample_id"]): Path(str(row["xenium_dir"]))
                for _, row in xenium.iterrows()
            }

    contexts: list[_SliceContext] = []
    for _, row in aligned.sort_values("order").iterrows():
        sample_id = str(row["sample_id"])
        contexts.append(
            _SliceContext(
                order=int(row["order"]),
                sample_id=sample_id,
                z_um=float(row["z_um"]),
                aligned_geojson=_resolve_path(row["aligned_geojson"], stack_root),
                xenium_dir=xenium_lookup.get(sample_id),
            )
        )
    return contexts


def _load_aligned_cells(cfg: GlandInstanceSegmentationConfig) -> pd.DataFrame:
    path = Path(cfg.aligned_cells_parquet).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    cells = pd.read_parquet(path)
    rename_map = {}
    if "cell_id" in cells.columns and "barcode" not in cells.columns:
        rename_map["cell_id"] = "barcode"
    cells = cells.rename(columns=rename_map)
    required = {"sample_id", "barcode", "slice_order", "x_3d_um", "y_3d_um"}
    missing = sorted(required.difference(cells.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    cells = cells.replace([np.inf, -np.inf], np.nan).dropna(subset=["x_3d_um", "y_3d_um"])
    cells["barcode"] = cells["barcode"].astype(str)
    return cells.reset_index(drop=True)


def _marker_counts_for_slice(
    *,
    context: _SliceContext,
    slice_cells: pd.DataFrame,
    markers: Sequence[str],
    cfg: GlandInstanceSegmentationConfig,
) -> pd.Series:
    marker_cols = [marker for marker in markers if marker in slice_cells.columns]
    if marker_cols:
        return slice_cells.set_index("barcode")[marker_cols].sum(axis=1).astype(float)
    if context.xenium_dir is None:
        return pd.Series(0.0, index=slice_cells["barcode"].astype(str))
    transcript_path = context.xenium_dir / cfg.xenium_transcripts_relpath
    if not transcript_path.exists():
        return pd.Series(0.0, index=slice_cells["barcode"].astype(str))
    transcripts = pd.read_parquet(
        transcript_path,
        columns=["cell_id", "feature_name"],
    )
    marker_set = {str(marker) for marker in markers}
    transcripts = transcripts.loc[
        transcripts["feature_name"].astype(str).isin(marker_set)
        & (transcripts["cell_id"].astype(str) != "UNASSIGNED")
    ]
    if transcripts.empty:
        return pd.Series(0.0, index=slice_cells["barcode"].astype(str))
    counts = transcripts.groupby(transcripts["cell_id"].astype(str)).size().astype(float)
    return counts


def _load_aligned_structure_geometries(
    geojson_path: Path,
    structures: Sequence[str],
    *,
    group_property: str,
    pixel_size_um: float,
) -> dict[str, Any]:
    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    requested = {str(item) for item in structures}
    by_structure: dict[str, list[Any]] = {structure: [] for structure in requested}
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        structure_name = _feature_structure(props, group_property)
        if structure_name not in requested:
            continue
        geometry_payload = feature.get("geometry")
        if geometry_payload is None:
            continue
        geom = shape(geometry_payload)
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        geom_um = affinity.scale(
            geom,
            xfact=float(pixel_size_um),
            yfact=float(pixel_size_um),
            origin=(0.0, 0.0),
        )
        if not geom_um.is_empty and geom_um.area > 0:
            by_structure.setdefault(structure_name, []).append(geom_um)
    result: dict[str, Any] = {}
    for structure_name, geometries in by_structure.items():
        if geometries:
            result[structure_name] = unary_union(geometries)
    return result


def _grid_for_geometry(geometry: Any, pixel_size_um: float) -> _GridSpec:
    min_x, min_y, max_x, max_y = geometry.bounds
    pad = max(float(pixel_size_um) * 4.0, 20.0)
    x_min = math.floor((min_x - pad) / pixel_size_um) * pixel_size_um
    y_min = math.floor((min_y - pad) / pixel_size_um) * pixel_size_um
    x_max = math.ceil((max_x + pad) / pixel_size_um) * pixel_size_um
    y_max = math.ceil((max_y + pad) / pixel_size_um) * pixel_size_um
    cols = max(1, int(round((x_max - x_min) / pixel_size_um)))
    rows = max(1, int(round((y_max - y_min) / pixel_size_um)))
    return _GridSpec(
        x_min=float(x_min),
        y_min=float(y_min),
        pixel_size_um=float(pixel_size_um),
        shape=(rows, cols),
    )


def _geometry_to_mask(geometry: Any, grid: _GridSpec) -> np.ndarray:
    mask = np.zeros(grid.shape, dtype=bool)
    for polygon in _iter_polygons(geometry):
        _burn_ring(mask, polygon.exterior.coords, grid, value=True)
        for interior in polygon.interiors:
            _burn_ring(mask, interior.coords, grid, value=False)
    return mask


def _burn_ring(mask: np.ndarray, coords: Any, grid: _GridSpec, *, value: bool) -> None:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return
    cols = (arr[:, 0] - grid.x_min) / grid.pixel_size_um
    rows = (arr[:, 1] - grid.y_min) / grid.pixel_size_um
    rr, cc = draw.polygon(rows, cols, shape=mask.shape)
    mask[rr, cc] = value


def _mask_to_largest_polygon(mask: np.ndarray, grid: _GridSpec) -> Polygon | None:
    if not mask.any():
        return None
    contours = measure.find_contours(mask.astype(float), 0.5)
    polygons: list[Polygon] = []
    for contour in contours:
        if contour.shape[0] < 4:
            continue
        xs = grid.x_min + contour[:, 1] * grid.pixel_size_um
        ys = grid.y_min + contour[:, 0] * grid.pixel_size_um
        coords = np.column_stack([xs, ys])
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])
        polygon = Polygon(coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if isinstance(polygon, Polygon) and not polygon.is_empty and polygon.area > 0:
            polygons.append(polygon)
    if not polygons:
        return None
    return max(polygons, key=lambda item: item.area)


def _histogram_cells(
    cells: pd.DataFrame,
    grid: _GridSpec,
    *,
    weight_col: str | None,
) -> np.ndarray:
    if cells.empty:
        return np.zeros(grid.shape, dtype=float)
    weights = None if weight_col is None else cells[weight_col].to_numpy(dtype=float)
    hist, _, _ = np.histogram2d(
        cells["y_3d_um"].to_numpy(dtype=float),
        cells["x_3d_um"].to_numpy(dtype=float),
        bins=[grid.y_edges, grid.x_edges],
        weights=weights,
    )
    return hist.astype(float)


def _cells_in_grid_bounds(cells: pd.DataFrame, grid: _GridSpec) -> pd.DataFrame:
    x_max = grid.x_min + grid.shape[1] * grid.pixel_size_um
    y_max = grid.y_min + grid.shape[0] * grid.pixel_size_um
    return cells.loc[
        (cells["x_3d_um"] >= grid.x_min)
        & (cells["x_3d_um"] <= x_max)
        & (cells["y_3d_um"] >= grid.y_min)
        & (cells["y_3d_um"] <= y_max)
    ].copy()


def _cells_inside_mask(cells: pd.DataFrame, mask: np.ndarray, grid: _GridSpec) -> pd.DataFrame:
    if cells.empty:
        return cells.copy()
    cols = np.floor((cells["x_3d_um"].to_numpy(dtype=float) - grid.x_min) / grid.pixel_size_um).astype(int)
    rows = np.floor((cells["y_3d_um"].to_numpy(dtype=float) - grid.y_min) / grid.pixel_size_um).astype(int)
    valid = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
    inside = np.zeros(len(cells), dtype=bool)
    inside[valid] = mask[rows[valid], cols[valid]]
    return cells.loc[inside].copy()


def _ring_support_score(instance_mask: np.ndarray, epithelial_support: np.ndarray) -> float:
    if not instance_mask.any():
        return 0.0
    boundary = instance_mask & ~ndi.binary_erosion(instance_mask)
    if not boundary.any():
        return 0.0
    return float((boundary & epithelial_support).sum() / boundary.sum())


def _remove_small_objects(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    keep_labels = np.where(sizes >= int(min_pixels))[0]
    keep = np.isin(labels, keep_labels)
    keep[labels == 0] = False
    return keep


def _touches_array_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _iter_polygons(geometry: Any) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [polygon for polygon in geometry.geoms if not polygon.is_empty]
    if hasattr(geometry, "geoms"):
        return [
            polygon
            for geom in geometry.geoms
            for polygon in _iter_polygons(geom)
        ]
    return []


def _feature_structure(props: Mapping[str, Any], group_property: str) -> str:
    value = props.get(group_property)
    if value is None:
        value = props.get("assigned_structure")
    if value is None:
        value = props.get("structure")
    if value is None and isinstance(props.get("classification"), Mapping):
        value = props["classification"].get("name")
    return str(value or "").strip()


def _resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_")
    return slug or "item"


def _validate_config(cfg: GlandInstanceSegmentationConfig) -> None:
    if not cfg.structures:
        raise ValueError("`structures` must contain at least one semantic structure label.")
    if cfg.pixel_size_um <= 0:
        raise ValueError("`pixel_size_um` must be positive.")
    if cfg.raster_pixel_size_um <= 0:
        raise ValueError("`raster_pixel_size_um` must be positive.")
    if cfg.lumen_min_area_um2 <= 0:
        raise ValueError("`lumen_min_area_um2` must be positive.")
    if not 0 <= cfg.lumen_cell_density_threshold <= 1:
        raise ValueError("`lumen_cell_density_threshold` must be in [0, 1].")


__all__ = [
    "DEFAULT_EPITHELIAL_MARKERS",
    "DEFAULT_EPITHELIAL_STRUCTURES",
    "GlandInstanceSegmentationConfig",
    "GlandInstanceSegmentationResult",
    "segment_gland_instances",
]
