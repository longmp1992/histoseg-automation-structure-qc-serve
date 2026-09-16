"""H&E image region segmentation and change detection.

The module intentionally separates neutral geometry outputs from pathology
semantics. Region names come from user configuration or generic component IDs;
the code does not infer diagnostic labels such as tumor or necrosis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import color, exposure, filters, io, measure, morphology, segmentation, util
from sklearn.cluster import KMeans


PathLike = Union[str, Path]

DEFAULT_MEDSAM_MODEL_ID = "wanglab/medsam-vit-base"
DEFAULT_PALETTE = [
    "#D65F5F",
    "#4E79A7",
    "#59A14F",
    "#F28E2B",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#9C755F",
    "#FF9DA7",
    "#8CD17D",
]


@dataclass(frozen=True)
class HERegionSpec:
    """Prompt and naming information for one user-defined H&E region."""

    region_name: str = "region"
    boxes: Sequence[Sequence[float]] = field(default_factory=tuple)
    points: Sequence[Sequence[float]] = field(default_factory=tuple)
    point_labels: Sequence[int] = field(default_factory=tuple)
    color: Optional[str] = None
    region_id: Optional[int] = None


@dataclass(frozen=True)
class HESegmentationConfig:
    """Configuration for single-region or all-elements H&E segmentation."""

    image: PathLike
    out_dir: PathLike
    task: str = "single"
    backend: Any = "medsam"
    medsam_model_id: str = DEFAULT_MEDSAM_MODEL_ID
    device: Optional[str] = None
    region_specs: Sequence[HERegionSpec | Mapping[str, Any]] = field(default_factory=tuple)
    n_components: int = 6
    component_names: Sequence[str] = field(default_factory=tuple)
    slic_segments: int = 320
    slic_compactness: float = 10.0
    tissue_od_threshold: Optional[float] = None
    tissue_min_area_px: int = 256
    min_region_area_px: int = 256
    point_radius_px: int = 64
    random_state: int = 0
    save_label_map: bool = True
    save_overlay_png: bool = True
    save_geojson: bool = True
    save_tables: bool = True


@dataclass
class HESegmentationResult:
    out_dir: Path
    task: str
    n_regions: int
    coordinate_space: str
    label_map_png: Optional[Path] = None
    overlay_png: Optional[Path] = None
    geojson: Optional[Path] = None
    regions_csv: Optional[Path] = None
    regions_parquet: Optional[Path] = None
    params_json: Optional[Path] = None
    metrics_json: Optional[Path] = None


@dataclass(frozen=True)
class HEChangeDetectionConfig:
    """Configuration for two-timepoint H&E change detection."""

    before_image: PathLike
    after_image: PathLike
    out_dir: PathLike
    resize_after_to_before: bool = False
    change_quantile: float = 0.92
    min_change_area_px: int = 256
    smooth_sigma: float = 1.5
    tissue_od_threshold: Optional[float] = None
    save_overlay_png: bool = True
    save_geojson: bool = True
    save_tables: bool = True


@dataclass
class HEChangeDetectionResult:
    out_dir: Path
    n_regions: int
    coordinate_space: str
    heatmap_png: Path
    change_mask_png: Path
    overlay_png: Optional[Path] = None
    geojson: Optional[Path] = None
    changes_csv: Optional[Path] = None
    changes_parquet: Optional[Path] = None
    params_json: Optional[Path] = None
    metrics_json: Optional[Path] = None


def run_he_segmentation(cfg: HESegmentationConfig) -> HESegmentationResult:
    """Run single-region or all-elements H&E region segmentation."""

    task = _normalize_segmentation_task(cfg.task)
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(cfg.image).expanduser().resolve()
    image = _read_rgb_image(image_path)
    spatial = _read_spatial_metadata(image_path)
    tissue_mask = _compute_tissue_mask(
        image,
        od_threshold=cfg.tissue_od_threshold,
        min_area_px=cfg.tissue_min_area_px,
    )

    region_specs = _normalize_region_specs(cfg.region_specs)
    if task == "single":
        label_map, regions = _run_single_region_segmentation(
            image=image,
            tissue_mask=tissue_mask,
            cfg=cfg,
            region_specs=region_specs,
        )
    else:
        label_map, regions = _run_all_elements_segmentation(
            image=image,
            tissue_mask=tissue_mask,
            cfg=cfg,
        )

    rows = _region_rows(
        image=image,
        label_map=label_map,
        regions=regions,
        score_image=None,
    )

    label_map_path: Optional[Path] = None
    if cfg.save_label_map:
        label_map_path = out_dir / f"he_{task}_label_map.png"
        _save_label_map_png(label_map, label_map_path)

    overlay_path: Optional[Path] = None
    if cfg.save_overlay_png:
        overlay_path = out_dir / f"he_{task}_overlay.png"
        _save_overlay_png(image=image, label_map=label_map, regions=regions, output_path=overlay_path)

    geojson_path: Optional[Path] = None
    if cfg.save_geojson:
        geojson_path = out_dir / f"he_{task}_regions.geojson"
        _write_geojson(
            label_map=label_map,
            regions=regions,
            output_path=geojson_path,
            spatial=spatial,
            score_image=None,
        )

    csv_path: Optional[Path] = None
    parquet_path: Optional[Path] = None
    if cfg.save_tables:
        table = pd.DataFrame(rows)
        csv_path = out_dir / f"he_{task}_regions.csv"
        table.to_csv(csv_path, index=False)
        try:
            parquet_path = out_dir / f"he_{task}_regions.parquet"
            table.to_parquet(parquet_path, index=False)
        except Exception:
            parquet_path = None

    params_path = out_dir / "params.json"
    params_path.write_text(
        json.dumps(_jsonable_dataclass(cfg), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    metrics_path = out_dir / "metrics.json"
    metrics = {
        "task": task,
        "backend": _backend_name(cfg.backend),
        "image": str(image_path),
        "image_shape": list(image.shape),
        "tissue_area_px": int(tissue_mask.sum()),
        "n_regions": int(len(regions)),
        "coordinate_space": spatial["coordinate_space"],
        "outputs": {
            "label_map_png": str(label_map_path) if label_map_path is not None else None,
            "overlay_png": str(overlay_path) if overlay_path is not None else None,
            "geojson": str(geojson_path) if geojson_path is not None else None,
            "regions_csv": str(csv_path) if csv_path is not None else None,
            "regions_parquet": str(parquet_path) if parquet_path is not None else None,
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return HESegmentationResult(
        out_dir=out_dir,
        task=task,
        n_regions=len(regions),
        coordinate_space=str(spatial["coordinate_space"]),
        label_map_png=label_map_path.resolve() if label_map_path is not None else None,
        overlay_png=overlay_path.resolve() if overlay_path is not None else None,
        geojson=geojson_path.resolve() if geojson_path is not None else None,
        regions_csv=csv_path.resolve() if csv_path is not None else None,
        regions_parquet=parquet_path.resolve() if parquet_path is not None else None,
        params_json=params_path.resolve(),
        metrics_json=metrics_path.resolve(),
    )


def run_he_change_detection(cfg: HEChangeDetectionConfig) -> HEChangeDetectionResult:
    """Detect changed H&E regions between two aligned image snapshots."""

    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    before_path = Path(cfg.before_image).expanduser().resolve()
    after_path = Path(cfg.after_image).expanduser().resolve()
    before = _read_rgb_image(before_path)
    after = _read_rgb_image(after_path)
    if before.shape != after.shape:
        if not cfg.resize_after_to_before:
            raise ValueError(
                "before_image and after_image must have the same shape unless "
                "`resize_after_to_before=True` is set."
            )
        after = _resize_rgb_to_shape(after, before.shape)

    spatial = _read_spatial_metadata(before_path)
    before_tissue = _compute_tissue_mask(
        before,
        od_threshold=cfg.tissue_od_threshold,
        min_area_px=cfg.min_change_area_px,
    )
    after_tissue = _compute_tissue_mask(
        after,
        od_threshold=cfg.tissue_od_threshold,
        min_area_px=cfg.min_change_area_px,
    )
    tissue_mask = before_tissue | after_tissue
    score = _stain_normalized_difference(before, after, tissue_mask)
    if cfg.smooth_sigma > 0:
        score = ndi.gaussian_filter(score, sigma=float(cfg.smooth_sigma))
    change_mask = _threshold_change_score(
        score=score,
        tissue_mask=tissue_mask,
        quantile=cfg.change_quantile,
        min_area_px=cfg.min_change_area_px,
    )
    label_map = measure.label(change_mask, connectivity=1).astype(np.uint16)
    regions = [
        {
            "label_value": int(value),
            "region_id": int(value),
            "region_name": f"change_{int(value)}",
            "color": _palette_color(int(value) - 1),
        }
        for value in np.unique(label_map)
        if int(value) > 0
    ]

    heatmap_path = out_dir / "he_change_heatmap.png"
    _save_heatmap_png(image=before, score=score, output_path=heatmap_path)

    mask_path = out_dir / "he_change_mask.png"
    _save_label_map_png(label_map, mask_path)

    overlay_path: Optional[Path] = None
    if cfg.save_overlay_png:
        overlay_path = out_dir / "he_change_overlay.png"
        _save_overlay_png(image=after, label_map=label_map, regions=regions, output_path=overlay_path)

    geojson_path: Optional[Path] = None
    if cfg.save_geojson:
        geojson_path = out_dir / "he_change_regions.geojson"
        _write_geojson(
            label_map=label_map,
            regions=regions,
            output_path=geojson_path,
            spatial=spatial,
            score_image=score,
        )

    csv_path: Optional[Path] = None
    parquet_path: Optional[Path] = None
    rows = _region_rows(image=after, label_map=label_map, regions=regions, score_image=score)
    if cfg.save_tables:
        table = pd.DataFrame(rows)
        csv_path = out_dir / "he_change_regions.csv"
        table.to_csv(csv_path, index=False)
        try:
            parquet_path = out_dir / "he_change_regions.parquet"
            table.to_parquet(parquet_path, index=False)
        except Exception:
            parquet_path = None

    params_path = out_dir / "params.json"
    params_path.write_text(
        json.dumps(_jsonable_dataclass(cfg), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    metrics_path = out_dir / "metrics.json"
    metrics = {
        "task": "change",
        "before_image": str(before_path),
        "after_image": str(after_path),
        "image_shape": list(before.shape),
        "tissue_area_px": int(tissue_mask.sum()),
        "change_area_px": int(change_mask.sum()),
        "n_regions": int(len(regions)),
        "coordinate_space": spatial["coordinate_space"],
        "outputs": {
            "heatmap_png": str(heatmap_path),
            "change_mask_png": str(mask_path),
            "overlay_png": str(overlay_path) if overlay_path is not None else None,
            "geojson": str(geojson_path) if geojson_path is not None else None,
            "changes_csv": str(csv_path) if csv_path is not None else None,
            "changes_parquet": str(parquet_path) if parquet_path is not None else None,
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return HEChangeDetectionResult(
        out_dir=out_dir,
        n_regions=len(regions),
        coordinate_space=str(spatial["coordinate_space"]),
        heatmap_png=heatmap_path.resolve(),
        change_mask_png=mask_path.resolve(),
        overlay_png=overlay_path.resolve() if overlay_path is not None else None,
        geojson=geojson_path.resolve() if geojson_path is not None else None,
        changes_csv=csv_path.resolve() if csv_path is not None else None,
        changes_parquet=parquet_path.resolve() if parquet_path is not None else None,
        params_json=params_path.resolve(),
        metrics_json=metrics_path.resolve(),
    )


def _normalize_segmentation_task(task: str) -> str:
    normalized = str(task).strip().lower().replace("-", "_")
    aliases = {
        "single": "single",
        "single_region": "single",
        "foreground": "single",
        "tissue": "single",
        "all": "all_elements",
        "all_elements": "all_elements",
        "elements": "all_elements",
    }
    if normalized not in aliases:
        raise ValueError("task must be 'single' or 'all_elements'.")
    return aliases[normalized]


def _read_rgb_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    image = io.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    elif image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Unsupported image shape for {path}: {image.shape!r}")

    if image.dtype != np.uint8:
        image = util.img_as_ubyte(exposure.rescale_intensity(image, out_range=(0, 1)))
    return np.ascontiguousarray(image[:, :, :3])


def _resize_rgb_to_shape(image: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    from skimage.transform import resize

    resized = resize(
        image,
        output_shape=shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    return np.clip(resized, 0, 255).astype(np.uint8)


def _read_spatial_metadata(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return {"coordinate_space": "pixel", "transform": None, "crs": None}
    try:
        import rasterio
    except Exception:
        return {"coordinate_space": "pixel", "transform": None, "crs": None}

    try:
        with rasterio.open(path) as dataset:
            transform = dataset.transform
            crs = dataset.crs.to_string() if dataset.crs is not None else None
            if transform is None or transform.is_identity:
                return {"coordinate_space": "pixel", "transform": None, "crs": crs}
            return {"coordinate_space": "source_crs", "transform": transform, "crs": crs}
    except Exception:
        return {"coordinate_space": "pixel", "transform": None, "crs": None}


def _compute_tissue_mask(
    image: np.ndarray,
    *,
    od_threshold: Optional[float],
    min_area_px: int,
) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    od = -np.log(np.clip(rgb, 1e-4, 1.0))
    od_mean = od.mean(axis=2)
    gray = color.rgb2gray(rgb)

    if od_threshold is None:
        sample = od_mean[np.isfinite(od_mean)]
        if sample.size == 0:
            threshold = 0.08
        else:
            try:
                threshold = max(0.04, float(filters.threshold_otsu(sample)))
            except ValueError:
                threshold = 0.08
    else:
        threshold = float(od_threshold)

    mask = (od_mean > threshold) & (gray < 0.93)
    mask = morphology.closing(mask, morphology.disk(3))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=max(1, int(min_area_px)))
    mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=max(1, int(min_area_px)))
    return mask.astype(bool)


def _normalize_region_specs(
    specs: Sequence[HERegionSpec | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(specs, start=1):
        if isinstance(item, HERegionSpec):
            payload = asdict(item)
        elif isinstance(item, Mapping):
            payload = dict(item)
        else:
            raise TypeError("region_specs entries must be HERegionSpec or mapping values.")

        name = str(payload.get("region_name") or payload.get("name") or f"region_{index}").strip()
        if not name:
            name = f"region_{index}"
        region_id_value = payload.get("region_id", payload.get("id"))
        region_id = int(region_id_value) if region_id_value is not None else index
        boxes = [_validate_box(box) for box in payload.get("boxes", ()) or ()]
        points = [_validate_point(point) for point in payload.get("points", ()) or ()]
        point_labels = [int(value) for value in payload.get("point_labels", ()) or ()]
        if point_labels and len(point_labels) != len(points):
            raise ValueError(f"Region {name!r} has point_labels length that does not match points.")
        if not point_labels and points:
            point_labels = [1] * len(points)
        normalized.append(
            {
                "region_id": region_id,
                "region_name": name,
                "boxes": boxes,
                "points": points,
                "point_labels": point_labels,
                "color": str(payload.get("color") or _palette_color(index - 1)),
                "label_value": index,
            }
        )
    return normalized


def _validate_box(box: Sequence[float]) -> list[float]:
    values = [float(value) for value in box]
    if len(values) != 4:
        raise ValueError("Each box must be [x_min, y_min, x_max, y_max].")
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid box with non-positive size: {box!r}")
    return [x0, y0, x1, y1]


def _validate_point(point: Sequence[float]) -> list[float]:
    values = [float(value) for value in point]
    if len(values) < 2:
        raise ValueError("Each point must contain at least [x, y].")
    return values[:2]


def _run_single_region_segmentation(
    *,
    image: np.ndarray,
    tissue_mask: np.ndarray,
    cfg: HESegmentationConfig,
    region_specs: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not region_specs:
        label_map = tissue_mask.astype(np.uint16)
        regions = [
            {
                "label_value": 1,
                "region_id": 1,
                "region_name": "tissue_foreground",
                "color": _palette_color(0),
            }
        ]
        return label_map, regions

    backend = _resolve_mask_backend(cfg)
    label_map = np.zeros(tissue_mask.shape, dtype=np.uint16)
    kept_regions: list[dict[str, Any]] = []
    for index, spec in enumerate(region_specs, start=1):
        mask = backend.segment(
            image=image,
            boxes=spec["boxes"],
            points=spec["points"],
            point_labels=spec["point_labels"],
            tissue_mask=tissue_mask,
            point_radius_px=int(cfg.point_radius_px),
        )
        mask = np.asarray(mask, dtype=bool) & tissue_mask
        mask = morphology.remove_small_objects(mask, min_size=max(1, int(cfg.min_region_area_px)))
        if int(mask.sum()) < int(cfg.min_region_area_px):
            continue
        label_value = int(spec["label_value"])
        label_map[mask] = label_value
        kept = dict(spec)
        kept["label_value"] = label_value
        kept_regions.append(kept)

    return label_map, kept_regions


def _run_all_elements_segmentation(
    *,
    image: np.ndarray,
    tissue_mask: np.ndarray,
    cfg: HESegmentationConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if int(tissue_mask.sum()) == 0:
        return np.zeros(tissue_mask.shape, dtype=np.uint16), []

    rgb = image.astype(np.float32) / 255.0
    segments = segmentation.slic(
        rgb,
        n_segments=max(2, int(cfg.slic_segments)),
        compactness=float(cfg.slic_compactness),
        start_label=1,
        mask=tissue_mask,
        channel_axis=-1,
    )
    segment_ids = np.array([value for value in np.unique(segments) if value > 0], dtype=int)
    if segment_ids.size == 0:
        return np.zeros(tissue_mask.shape, dtype=np.uint16), []

    features = _superpixel_features(image, tissue_mask, segments, segment_ids)
    n_clusters = max(1, min(int(cfg.n_components), len(segment_ids)))
    if n_clusters == 1:
        cluster_labels = np.zeros(len(segment_ids), dtype=int)
    else:
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=int(cfg.random_state))
        cluster_labels = model.fit_predict(features)

    label_map = np.zeros(tissue_mask.shape, dtype=np.uint16)
    for segment_id, cluster_label in zip(segment_ids, cluster_labels):
        label_map[segments == int(segment_id)] = int(cluster_label) + 1

    label_map = _clean_component_label_map(label_map, min_area_px=max(1, int(cfg.min_region_area_px)))
    present_labels = [int(value) for value in np.unique(label_map) if int(value) > 0]
    component_names = list(cfg.component_names)
    regions: list[dict[str, Any]] = []
    for output_index, label_value in enumerate(present_labels, start=1):
        if output_index - 1 < len(component_names) and str(component_names[output_index - 1]).strip():
            name = str(component_names[output_index - 1]).strip()
        else:
            name = f"component_{output_index}"
        regions.append(
            {
                "label_value": label_value,
                "region_id": output_index,
                "region_name": name,
                "color": _palette_color(output_index - 1),
            }
        )
    return label_map, regions


def _superpixel_features(
    image: np.ndarray,
    tissue_mask: np.ndarray,
    segments: np.ndarray,
    segment_ids: np.ndarray,
) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    hed = color.rgb2hed(rgb)
    gray = color.rgb2gray(rgb)
    edges = filters.sobel(gray)
    od = -np.log(np.clip(rgb, 1e-4, 1.0)).mean(axis=2)

    rows: list[list[float]] = []
    h, w = tissue_mask.shape
    yy, xx = np.indices((h, w))
    for segment_id in segment_ids:
        mask = segments == int(segment_id)
        if not np.any(mask):
            rows.append([0.0] * 10)
            continue
        rows.append(
            [
                float(rgb[:, :, 0][mask].mean()),
                float(rgb[:, :, 1][mask].mean()),
                float(rgb[:, :, 2][mask].mean()),
                float(hed[:, :, 0][mask].mean()),
                float(hed[:, :, 1][mask].mean()),
                float(hed[:, :, 2][mask].mean()),
                float(od[mask].mean()),
                float(edges[mask].mean()),
                float(xx[mask].mean() / max(1, w - 1)),
                float(yy[mask].mean() / max(1, h - 1)),
            ]
        )
    features = np.asarray(rows, dtype=np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-6)


def _clean_component_label_map(label_map: np.ndarray, min_area_px: int) -> np.ndarray:
    cleaned = np.zeros_like(label_map, dtype=np.uint16)
    next_label = 1
    for label_value in [int(value) for value in np.unique(label_map) if int(value) > 0]:
        mask = label_map == label_value
        mask = morphology.remove_small_objects(mask, min_size=min_area_px)
        components = measure.label(mask, connectivity=1)
        for component in [int(value) for value in np.unique(components) if int(value) > 0]:
            component_mask = components == component
            if int(component_mask.sum()) >= min_area_px:
                cleaned[component_mask] = next_label
                next_label += 1
    return cleaned


def _resolve_mask_backend(cfg: HESegmentationConfig) -> Any:
    backend = cfg.backend
    if hasattr(backend, "segment"):
        return backend
    name = str(backend).strip().lower()
    if name in {"heuristic", "local", "prompt_heuristic"}:
        return _HeuristicPromptBackend()
    if name in {"medsam", "hf", "huggingface", "hugging_face"}:
        return _MedSAMPromptBackend(model_id=cfg.medsam_model_id, device=cfg.device)
    raise ValueError("backend must be 'medsam', 'heuristic', or an object with segment().")


class _HeuristicPromptBackend:
    def segment(
        self,
        *,
        image: np.ndarray,
        boxes: Sequence[Sequence[float]],
        points: Sequence[Sequence[float]],
        point_labels: Sequence[int],
        tissue_mask: np.ndarray,
        point_radius_px: int,
    ) -> np.ndarray:
        mask = np.zeros(tissue_mask.shape, dtype=bool)
        h, w = tissue_mask.shape
        for box in boxes:
            x0, y0, x1, y1 = _clip_box(box, width=w, height=h)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = True

        if points:
            tissue_components = measure.label(tissue_mask, connectivity=1)
            for point, label_value in zip(points, point_labels):
                if int(label_value) <= 0:
                    continue
                x = int(round(float(point[0])))
                y = int(round(float(point[1])))
                if 0 <= x < w and 0 <= y < h and tissue_components[y, x] > 0:
                    mask |= tissue_components == tissue_components[y, x]
                else:
                    yy, xx = np.ogrid[:h, :w]
                    mask |= (xx - x) ** 2 + (yy - y) ** 2 <= int(point_radius_px) ** 2

        if not boxes and not points:
            mask = tissue_mask.copy()
        return mask & tissue_mask


class _MedSAMPromptBackend:
    def __init__(self, *, model_id: str, device: Optional[str]) -> None:
        try:
            import torch
            from transformers import SamModel, SamProcessor
        except Exception as exc:
            raise ImportError(
                "MedSAM backend requires optional dependencies. Install with "
                "`pip install histoseg[he]` or use backend='heuristic'."
            ) from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self.device = device
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device)
        self.model.eval()

    def segment(
        self,
        *,
        image: np.ndarray,
        boxes: Sequence[Sequence[float]],
        points: Sequence[Sequence[float]],
        point_labels: Sequence[int],
        tissue_mask: np.ndarray,
        point_radius_px: int,
    ) -> np.ndarray:
        del point_radius_px
        if not boxes and not points:
            return tissue_mask.copy()

        from PIL import Image

        pil_image = Image.fromarray(image)
        mask = np.zeros(tissue_mask.shape, dtype=bool)
        for box in boxes:
            inputs = self.processor(
                pil_image,
                input_boxes=[[[float(v) for v in box]]],
                return_tensors="pt",
            ).to(self.device)
            mask |= self._predict_prompt_mask(inputs)

        positive_points = [
            [float(point[0]), float(point[1])]
            for point, label_value in zip(points, point_labels)
            if int(label_value) > 0
        ]
        if positive_points:
            inputs = self.processor(
                pil_image,
                input_points=[positive_points],
                return_tensors="pt",
            ).to(self.device)
            mask |= self._predict_prompt_mask(inputs)
        return mask & tissue_mask

    def _predict_prompt_mask(self, inputs: Any) -> np.ndarray:
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
        )[0]
        masks_np = masks.detach().cpu().numpy()
        if masks_np.ndim == 4:
            masks_np = masks_np.reshape((-1,) + masks_np.shape[-2:])
        elif masks_np.ndim == 3:
            masks_np = masks_np.reshape((-1,) + masks_np.shape[-2:])
        else:
            raise RuntimeError(f"Unexpected MedSAM mask tensor shape: {masks_np.shape!r}")
        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            scores = outputs.iou_scores.detach().cpu().numpy().reshape(-1)
            best_index = int(np.argmax(scores[: masks_np.shape[0]]))
        else:
            best_index = 0
        return masks_np[best_index] > 0


def _clip_box(box: Sequence[float], *, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(round(float(value))) for value in box]
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    return x0, y0, x1, y1


def _region_rows(
    *,
    image: np.ndarray,
    label_map: np.ndarray,
    regions: Sequence[Mapping[str, Any]],
    score_image: Optional[np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for region in regions:
        label_value = int(region["label_value"])
        mask = label_map == label_value
        if not np.any(mask):
            continue
        ys, xs = np.where(mask)
        pixels = image[mask]
        row: dict[str, Any] = {
            "region_id": int(region.get("region_id", label_value)),
            "region_name": str(region.get("region_name", f"region_{label_value}")),
            "label_value": label_value,
            "pixel_count": int(mask.sum()),
            "bbox_xmin": int(xs.min()),
            "bbox_ymin": int(ys.min()),
            "bbox_xmax": int(xs.max()) + 1,
            "bbox_ymax": int(ys.max()) + 1,
            "centroid_x": float(xs.mean()),
            "centroid_y": float(ys.mean()),
            "mean_r": float(pixels[:, 0].mean()),
            "mean_g": float(pixels[:, 1].mean()),
            "mean_b": float(pixels[:, 2].mean()),
        }
        if score_image is not None:
            values = score_image[mask]
            row["score_mean"] = float(values.mean())
            row["score_max"] = float(values.max())
        rows.append(row)
    return rows


def _save_label_map_png(label_map: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(output_path, label_map.astype(np.uint16), check_contrast=False)


def _save_overlay_png(
    *,
    image: np.ndarray,
    label_map: np.ndarray,
    regions: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    plt = _get_pyplot()
    overlay = image.astype(np.float32).copy()
    for region in regions:
        label_value = int(region["label_value"])
        color_rgb = np.asarray(_hex_to_rgb(str(region.get("color") or _palette_color(label_value - 1))))
        mask = label_map == label_value
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * color_rgb

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    for region in regions:
        label_value = int(region["label_value"])
        contours = measure.find_contours((label_map == label_value).astype(float), 0.5)
        for contour in contours:
            if contour.shape[0] >= 3:
                ax.plot(contour[:, 1], contour[:, 0], color=str(region.get("color")), linewidth=1.1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_heatmap_png(*, image: np.ndarray, score: np.ndarray, output_path: Path) -> None:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(score, cmap="inferno", alpha=0.55)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _write_geojson(
    *,
    label_map: np.ndarray,
    regions: Sequence[Mapping[str, Any]],
    output_path: Path,
    spatial: Mapping[str, Any],
    score_image: Optional[np.ndarray],
) -> None:
    features: list[dict[str, Any]] = []
    for region in regions:
        label_value = int(region["label_value"])
        mask = label_map == label_value
        contours = measure.find_contours(mask.astype(float), 0.5)
        polygon_index = 0
        for contour in contours:
            if contour.shape[0] < 4:
                continue
            coords = [[float(x), float(y)] for y, x in contour]
            coords = [_transform_xy(x, y, spatial) for x, y in coords]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            if len(coords) < 4:
                continue
            component_mask = _contour_component_mask(mask, contour)
            properties: dict[str, Any] = {
                "region_id": int(region.get("region_id", label_value)),
                "region_name": str(region.get("region_name", f"region_{label_value}")),
                "label_value": label_value,
                "polygon_index": polygon_index,
                "coordinate_space": str(spatial["coordinate_space"]),
                "color": str(region.get("color") or _palette_color(label_value - 1)),
                "pixel_count": int(component_mask.sum()) if component_mask is not None else int(mask.sum()),
            }
            if score_image is not None and component_mask is not None and np.any(component_mask):
                values = score_image[component_mask]
                properties["score_mean"] = float(values.mean())
                properties["score_max"] = float(values.max())
            features.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords],
                    },
                }
            )
            polygon_index += 1

    payload: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "coordinate_space": str(spatial["coordinate_space"]),
            "crs": spatial.get("crs"),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _contour_component_mask(mask: np.ndarray, contour: np.ndarray) -> Optional[np.ndarray]:
    if contour.shape[0] < 3:
        return None
    min_y = max(0, int(np.floor(contour[:, 0].min())) - 1)
    max_y = min(mask.shape[0], int(np.ceil(contour[:, 0].max())) + 2)
    min_x = max(0, int(np.floor(contour[:, 1].min())) - 1)
    max_x = min(mask.shape[1], int(np.ceil(contour[:, 1].max())) + 2)
    if min_y >= max_y or min_x >= max_x:
        return None
    submask = np.zeros_like(mask, dtype=bool)
    submask[min_y:max_y, min_x:max_x] = mask[min_y:max_y, min_x:max_x]
    return submask


def _transform_xy(x: float, y: float, spatial: Mapping[str, Any]) -> list[float]:
    transform = spatial.get("transform")
    if transform is None:
        return [float(x), float(y)]
    x_out, y_out = transform * (float(x), float(y))
    return [float(x_out), float(y_out)]


def _stain_normalized_difference(
    before: np.ndarray,
    after: np.ndarray,
    tissue_mask: np.ndarray,
) -> np.ndarray:
    before_features = _stain_feature_stack(before)
    after_features = _stain_feature_stack(after)
    if not np.any(tissue_mask):
        tissue_mask = np.ones(before.shape[:2], dtype=bool)
    before_norm = _robust_normalize_feature_stack(before_features, tissue_mask)
    after_norm = _robust_normalize_feature_stack(after_features, tissue_mask)
    score = np.mean(np.abs(after_norm - before_norm), axis=2)
    score[~tissue_mask] = 0.0
    return score.astype(np.float32)


def _stain_feature_stack(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    hed = color.rgb2hed(rgb)
    od = -np.log(np.clip(rgb, 1e-4, 1.0))
    return np.dstack([hed[:, :, 0], hed[:, :, 1], od.mean(axis=2)]).astype(np.float32)


def _robust_normalize_feature_stack(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(features, dtype=np.float32)
    for channel in range(features.shape[2]):
        values = features[:, :, channel][mask]
        if values.size == 0:
            out[:, :, channel] = features[:, :, channel]
            continue
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        scale = max(float(q75 - q25), 1e-4)
        out[:, :, channel] = (features[:, :, channel] - median) / scale
    return out


def _threshold_change_score(
    *,
    score: np.ndarray,
    tissue_mask: np.ndarray,
    quantile: float,
    min_area_px: int,
) -> np.ndarray:
    values = score[tissue_mask]
    if values.size == 0:
        return np.zeros(score.shape, dtype=bool)
    q = min(max(float(quantile), 0.0), 1.0)
    threshold = float(np.quantile(values, q))
    try:
        otsu = float(filters.threshold_otsu(values))
        threshold = max(threshold, otsu)
    except ValueError:
        pass
    mask = (score >= threshold) & tissue_mask
    mask = morphology.closing(mask, morphology.disk(2))
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=max(1, int(min_area_px)))
    return mask.astype(bool)


def _palette_color(index: int) -> str:
    return DEFAULT_PALETTE[int(index) % len(DEFAULT_PALETTE)]


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) != 6:
        return (214, 95, 95)
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _get_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _jsonable_dataclass(value: Any) -> Any:
    if is_dataclass(value):
        payload = asdict(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        return value
    if "backend" in payload:
        payload["backend"] = _backend_name(payload["backend"])
    return payload


def _backend_name(backend: Any) -> str:
    if isinstance(backend, str):
        return backend
    return backend.__class__.__name__
