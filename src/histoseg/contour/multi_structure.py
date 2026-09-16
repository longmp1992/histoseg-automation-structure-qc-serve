"""Multi-structure contour partitioning and Xenium Explorer exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    label as nd_label,
)
from skimage import measure

from .pattern1_isoline import (
    _normalize_cluster_label,
    align_clusters_with_cells,
    extract_contour_paths,
)

PathLike = Union[str, Path]

GROUP_PALETTE = [
    "#6EF0D4",
    "#78B9FF",
    "#FFB870",
    "#C8A2FF",
    "#FF8DA1",
    "#90F184",
    "#FFD76C",
    "#80E1FF",
    "#F4A6FF",
    "#FFA07A",
    "#B8F0DE",
    "#A7BFFF",
]

__all__ = [
    "MultiStructureContourConfig",
    "MultiStructureContourResult",
    "MultiStructureSpec",
    "run_multi_structure_contours",
]


@dataclass(frozen=True)
class MultiStructureSpec:
    structure_name: str
    cluster_ids: Sequence[Union[int, str]]
    structure_color: str | None = None
    structure_id: int | None = None


@dataclass(frozen=True)
class MultiStructureContourConfig:
    clusters_csv: PathLike
    cells_parquet: PathLike
    out_dir: PathLike
    structures: Sequence[MultiStructureSpec | Mapping[str, Any]]
    barcode_col: str = "Barcode"
    cluster_col: str = "Cluster"
    bins_x: int = 900
    bins_y: int = 700
    gaussian_sigma: float = 2.25
    density_scale_quantile: float = 0.98
    support_quantile: float = 0.18
    tissue_quantile: float = 0.06
    min_dominance: float = 0.34
    closing_iterations: int = 2
    opening_iterations: int = 1
    fill_holes: bool = True
    min_cells: int = 500
    min_component_pixels: int = 180
    xenium_pixel_size_um: float = 0.2125
    save_preview_png: bool = True


@dataclass
class MultiStructureContourResult:
    out_dir: Path
    geojson: Path
    csv: Path
    summary: Path
    preview_png: Path | None
    partition_table: Path
    structure_count_csv: Path
    metrics_json: Path


def run_multi_structure_contours(cfg: MultiStructureContourConfig) -> MultiStructureContourResult:
    """Generate non-overlapping structure contours and Xenium Explorer exports."""

    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    structure_specs = _normalize_structure_specs(cfg.structures)

    merged, id_col_used, x_col, y_col = align_clusters_with_cells(
        cfg.clusters_csv,
        cfg.cells_parquet,
        barcode_col=cfg.barcode_col,
        cluster_col=cfg.cluster_col,
    )
    merged = merged.copy()
    merged["cluster"] = merged["cluster"].map(_normalize_cluster_label)
    merged = merged.loc[merged["cluster"] != ""].copy()
    merged["_selected_structure_id"] = 0

    input_cell_counts: dict[int, int] = {}
    for spec in structure_specs:
        structure_id = int(spec["structure_id"])
        structure_mask = merged["cluster"].isin(set(spec["cluster_ids_normalized"]))
        merged.loc[structure_mask, "_selected_structure_id"] = structure_id
        input_cell_counts[structure_id] = int(structure_mask.sum())

    missing = [
        str(spec["structure_name"])
        for spec in structure_specs
        if input_cell_counts.get(int(spec["structure_id"]), 0) == 0
    ]
    if missing:
        raise ValueError(
            "Some requested structures could not be matched to any cells after cluster "
            f"normalization: {', '.join(missing)}"
        )

    isoline_cfg = _isoline_config_from_dataclass(cfg)
    structure_contours, partition_data, isoline_metrics = _build_structure_isolines(
        cells=merged,
        structure_specs=structure_specs,
        x_col=x_col,
        y_col=y_col,
        isoline_cfg=isoline_cfg,
    )
    assigned_cells, cell_assignment_metrics = _assign_cells_to_partition(
        cells=merged,
        partition_data=partition_data,
        structure_specs=structure_specs,
        x_col=x_col,
        y_col=y_col,
    )
    structure_contours = _attach_partition_contours(structure_contours)

    preview_path: Path | None = None
    if cfg.save_preview_png:
        preview_path = out_dir / "multi_structure_contour_preview.png"
        _render_multi_structure_preview(
            assigned_cells=assigned_cells,
            structure_contours=structure_contours,
            structure_specs=structure_specs,
            x_col=x_col,
            y_col=y_col,
            output_path=preview_path,
        )

    partition_table_path: Path
    try:
        partition_table_path = out_dir / "cells_with_structure_partition.parquet"
        assigned_cells.to_parquet(partition_table_path, index=False)
    except Exception:
        partition_table_path = out_dir / "cells_with_structure_partition.csv"
        assigned_cells.to_csv(partition_table_path, index=False)

    structure_count_path = out_dir / "structure_contour_cell_counts.csv"
    pd.DataFrame(
        _structure_count_rows(
            structure_specs=structure_specs,
            structure_contours=structure_contours,
            assigned_cells=assigned_cells,
            input_cell_counts=input_cell_counts,
        )
    ).to_csv(structure_count_path, index=False)

    explorer_exports = _write_xenium_explorer_annotation_exports(
        output_dir=out_dir,
        partition_data=partition_data,
        structure_specs=structure_specs,
        xenium_pixel_size_um=cfg.xenium_pixel_size_um,
    )

    metrics_path = out_dir / "structure_contour_metrics.json"
    metrics_payload = {
        "id_col_used": id_col_used,
        "x_col": x_col,
        "y_col": y_col,
        "selected_structures": [
            {
                "structure_id": int(spec["structure_id"]),
                "structure_name": str(spec["structure_name"]),
                "structure_color": str(spec["structure_color"]),
                "cluster_ids": list(spec["cluster_ids_raw"]),
            }
            for spec in structure_specs
        ],
        "config": asdict(cfg),
        "input_cell_counts": {str(key): int(value) for key, value in input_cell_counts.items()},
        "isoline_metrics": isoline_metrics,
        "cell_assignment_metrics": cell_assignment_metrics,
        "output_files": {
            "geojson": str(explorer_exports["geojson"]),
            "csv": str(explorer_exports["csv"]),
            "summary": str(explorer_exports["summary"]),
            "preview_png": str(preview_path) if preview_path is not None else None,
            "partition_table": str(partition_table_path),
            "structure_count_csv": str(structure_count_path),
        },
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return MultiStructureContourResult(
        out_dir=out_dir,
        geojson=Path(explorer_exports["geojson"]).resolve(),
        csv=Path(explorer_exports["csv"]).resolve(),
        summary=Path(explorer_exports["summary"]).resolve(),
        preview_png=preview_path.resolve() if preview_path is not None else None,
        partition_table=partition_table_path.resolve(),
        structure_count_csv=structure_count_path.resolve(),
        metrics_json=metrics_path.resolve(),
    )


def _group_color(index: int) -> str:
    if index <= 0:
        raise ValueError(f"Structure color index must be positive, got {index}.")
    return GROUP_PALETTE[(index - 1) % len(GROUP_PALETTE)]


def _normalize_structure_specs(
    structures: Sequence[MultiStructureSpec | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not structures:
        raise ValueError("`structures` must contain at least one structure definition.")

    normalized: list[dict[str, Any]] = []
    raw_ids: list[int | None] = []
    for index, item in enumerate(structures, start=1):
        if isinstance(item, MultiStructureSpec):
            payload = {
                "structure_name": item.structure_name,
                "cluster_ids": item.cluster_ids,
                "structure_color": item.structure_color,
                "structure_id": item.structure_id,
            }
        elif isinstance(item, Mapping):
            payload = dict(item)
        else:
            raise TypeError(
                "`structures` entries must be MultiStructureSpec or mapping values, got "
                f"{type(item)!r}."
            )

        structure_name = str(payload.get("structure_name", "")).strip()
        if not structure_name:
            raise ValueError(f"Structure {index} is missing a non-empty `structure_name`.")

        raw_cluster_ids = payload.get("cluster_ids")
        if raw_cluster_ids is None:
            raw_cluster_ids = payload.get("clusters")
        if raw_cluster_ids is None:
            raise ValueError(f"Structure {index} is missing `cluster_ids`.")

        cluster_ids_raw = [str(value) for value in list(raw_cluster_ids)]
        cluster_ids_normalized = [
            _normalize_cluster_label(value)
            for value in raw_cluster_ids
            if _normalize_cluster_label(value) != ""
        ]
        cluster_ids_normalized = list(dict.fromkeys(cluster_ids_normalized))
        if not cluster_ids_normalized:
            raise ValueError(f"Structure {index} does not contain any valid cluster IDs.")

        structure_id_value = payload.get("structure_id")
        if structure_id_value is None:
            structure_id_value = payload.get("id")
        structure_id = int(structure_id_value) if structure_id_value is not None else None
        structure_color = payload.get("structure_color")
        if structure_color is None:
            structure_color = payload.get("color")
        normalized.append(
            {
                "structure_name": structure_name,
                "structure_id": structure_id,
                "structure_color": str(structure_color).strip() if structure_color else None,
                "cluster_ids_raw": cluster_ids_raw,
                "cluster_ids_normalized": cluster_ids_normalized,
            }
        )
        raw_ids.append(structure_id)

    assigned_ids: list[int] = []
    next_id = 1
    for raw_id in raw_ids:
        if raw_id is None:
            while next_id in assigned_ids:
                next_id += 1
            assigned_ids.append(next_id)
        else:
            assigned_ids.append(int(raw_id))

    if len(set(assigned_ids)) != len(assigned_ids):
        raise ValueError("Each structure must have a unique `structure_id`.")
    if any(value <= 0 for value in assigned_ids):
        raise ValueError("Each `structure_id` must be a positive integer.")

    seen_clusters: set[str] = set()
    for index, spec in enumerate(normalized):
        spec["structure_id"] = assigned_ids[index]
        if spec["structure_color"] is None:
            spec["structure_color"] = _group_color(index + 1)

        overlap = seen_clusters.intersection(spec["cluster_ids_normalized"])
        if overlap:
            raise ValueError(
                "A cluster ID cannot belong to more than one structure in the same run. "
                f"Repeated cluster IDs: {', '.join(sorted(overlap))}"
            )
        seen_clusters.update(spec["cluster_ids_normalized"])
    return normalized


def _isoline_config_from_dataclass(cfg: MultiStructureContourConfig) -> dict[str, Any]:
    return {
        "bins_x": int(cfg.bins_x),
        "bins_y": int(cfg.bins_y),
        "gaussian_sigma": float(cfg.gaussian_sigma),
        "density_scale_quantile": float(cfg.density_scale_quantile),
        "support_quantile": float(cfg.support_quantile),
        "tissue_quantile": float(cfg.tissue_quantile),
        "min_dominance": float(cfg.min_dominance),
        "closing_iterations": int(cfg.closing_iterations),
        "opening_iterations": int(cfg.opening_iterations),
        "fill_holes": bool(cfg.fill_holes),
        "min_cells": int(cfg.min_cells),
        "min_component_pixels": int(cfg.min_component_pixels),
    }


def _remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    labeled, n_components = nd_label(mask)
    if n_components == 0:
        return mask

    component_sizes = np.bincount(labeled.ravel())
    keep_labels = np.where(component_sizes >= int(min_pixels))[0]
    keep_mask = np.isin(labeled, keep_labels)
    keep_mask[labeled == 0] = False
    return keep_mask


def _prepare_binary_mask(mask: np.ndarray, isoline_cfg: dict[str, object]) -> np.ndarray:
    structure = generate_binary_structure(2, 1)
    result = mask.copy()

    if int(isoline_cfg["closing_iterations"]) > 0:
        result = binary_closing(
            result,
            structure=structure,
            iterations=int(isoline_cfg["closing_iterations"]),
        )
    if int(isoline_cfg["opening_iterations"]) > 0:
        result = binary_opening(
            result,
            structure=structure,
            iterations=int(isoline_cfg["opening_iterations"]),
        )
    if bool(isoline_cfg["fill_holes"]):
        result = binary_fill_holes(result)

    return _remove_small_components(result, int(isoline_cfg["min_component_pixels"]))


def _build_structure_isolines(
    *,
    cells: pd.DataFrame,
    structure_specs: list[dict[str, Any]],
    x_col: str,
    y_col: str,
    isoline_cfg: dict[str, object],
) -> tuple[dict[int, dict[str, object]], dict[str, np.ndarray], dict[str, object]]:
    x_min = float(cells[x_col].min())
    x_max = float(cells[x_col].max())
    y_min = float(cells[y_col].min())
    y_max = float(cells[y_col].max())

    x_edges = np.linspace(x_min, x_max, int(isoline_cfg["bins_x"]) + 1)
    y_edges = np.linspace(y_min, y_max, int(isoline_cfg["bins_y"]) + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)

    density_stack: list[np.ndarray] = []
    normalized_stack: list[np.ndarray] = []
    support_thresholds: list[float] = []
    valid_specs: list[dict[str, Any]] = []
    raw_density_by_structure: dict[int, np.ndarray] = {}

    all_hist, _, _ = np.histogram2d(
        cells[x_col].to_numpy(dtype=float),
        cells[y_col].to_numpy(dtype=float),
        bins=[x_edges, y_edges],
    )
    occupied_mask = all_hist.T > 0

    skipped_structures: list[dict[str, object]] = []
    for spec in structure_specs:
        structure_id = int(spec["structure_id"])
        subset = cells.loc[cells["_selected_structure_id"] == structure_id, [x_col, y_col]]
        if len(subset) < int(isoline_cfg["min_cells"]):
            skipped_structures.append(
                {
                    "structure_id": structure_id,
                    "structure_name": str(spec["structure_name"]),
                    "reason": (
                        f"Only {len(subset)} cells, below min_cells={int(isoline_cfg['min_cells'])}"
                    ),
                }
            )
            continue

        hist, _, _ = np.histogram2d(
            subset[x_col].to_numpy(dtype=float),
            subset[y_col].to_numpy(dtype=float),
            bins=[x_edges, y_edges],
        )
        density = gaussian_filter(hist.T, sigma=float(isoline_cfg["gaussian_sigma"]))
        positive = density[density > 0]
        if positive.size == 0:
            skipped_structures.append(
                {
                    "structure_id": structure_id,
                    "structure_name": str(spec["structure_name"]),
                    "reason": "Density map contained no positive pixels after smoothing.",
                }
            )
            continue

        scale_value = float(
            np.quantile(positive, float(isoline_cfg["density_scale_quantile"]))
        )
        support_value = float(np.quantile(positive, float(isoline_cfg["support_quantile"])))
        if scale_value <= 0 or support_value <= 0:
            skipped_structures.append(
                {
                    "structure_id": structure_id,
                    "structure_name": str(spec["structure_name"]),
                    "reason": "Could not derive positive thresholds for this structure.",
                }
            )
            continue

        density_stack.append(density)
        normalized_stack.append(density / scale_value)
        support_thresholds.append(support_value)
        valid_specs.append(spec)
        raw_density_by_structure[structure_id] = density

    if not density_stack:
        raise ValueError(
            "No selected structure had enough support to build a contour partition. "
            "Try lowering the minimum cell threshold or selecting more populated structures."
        )

    density_stack_np = np.stack(density_stack, axis=0)
    normalized_stack_np = np.stack(normalized_stack, axis=0)
    total_density = density_stack_np.sum(axis=0)
    total_positive = total_density[total_density > 0]
    tissue_threshold = float(np.quantile(total_positive, float(isoline_cfg["tissue_quantile"])))
    tissue_mask = total_density >= tissue_threshold
    tissue_mask = tissue_mask | occupied_mask
    tissue_mask = _prepare_binary_mask(
        tissue_mask,
        {
            **isoline_cfg,
            "min_component_pixels": max(500, int(isoline_cfg["min_component_pixels"])),
        },
    )
    tissue_mask = tissue_mask | occupied_mask

    posterior = normalized_stack_np / (normalized_stack_np.sum(axis=0, keepdims=True) + 1e-12)
    assignment_idx = posterior.argmax(axis=0)
    best_posterior = posterior.max(axis=0)

    structure_contours: dict[int, dict[str, object]] = {}
    seed_masks: dict[int, np.ndarray] = {}
    for idx, spec in enumerate(valid_specs):
        structure_id = int(spec["structure_id"])
        own_support = raw_density_by_structure[structure_id] >= support_thresholds[idx]
        dominant_mask = assignment_idx == idx
        confident_mask = best_posterior >= float(isoline_cfg["min_dominance"])
        seed_mask = tissue_mask & own_support & dominant_mask & confident_mask
        seed_mask = _prepare_binary_mask(seed_mask, isoline_cfg)
        if not seed_mask.any():
            fallback_mask = tissue_mask & dominant_mask & (raw_density_by_structure[structure_id] > 0)
            fallback_mask = _remove_small_components(
                fallback_mask,
                max(20, int(isoline_cfg["min_component_pixels"]) // 2),
            )
            if fallback_mask.any():
                seed_mask = fallback_mask
            else:
                peak_y, peak_x = np.unravel_index(
                    np.argmax(raw_density_by_structure[structure_id]),
                    raw_density_by_structure[structure_id].shape,
                )
                seed_mask = np.zeros_like(tissue_mask, dtype=bool)
                seed_mask[peak_y, peak_x] = True

        seed_masks[structure_id] = seed_mask
        structure_contours[structure_id] = {
            "structure_name": str(spec["structure_name"]),
            "structure_color": str(spec["structure_color"]),
            "cluster_ids_normalized": list(spec["cluster_ids_normalized"]),
            "grid_x": grid_x,
            "grid_y": grid_y,
            "seed_mask": seed_mask.astype(float),
            "density": raw_density_by_structure[structure_id],
            "posterior": posterior[idx],
        }

    overlap_after = 0
    if seed_masks:
        sorted_ids = sorted(seed_masks)
        mask_stack = np.stack([seed_masks[sid] for sid in sorted_ids], axis=0)
        overlap_map = mask_stack.sum(axis=0)
        if np.any(overlap_map >= 2):
            posterior_lookup = {
                int(spec["structure_id"]): posterior[idx] for idx, spec in enumerate(valid_specs)
            }
            posterior_stack = np.stack([posterior_lookup[sid] for sid in sorted_ids], axis=0)
            winner_idx = posterior_stack.argmax(axis=0)

            for idx, structure_id in enumerate(sorted_ids):
                exclusive_mask = mask_stack[idx] & ((overlap_map == 1) | (winner_idx == idx))
                exclusive_mask = _remove_small_components(
                    exclusive_mask,
                    int(isoline_cfg["min_component_pixels"]),
                )
                seed_masks[structure_id] = exclusive_mask
                structure_contours[structure_id]["seed_mask"] = exclusive_mask.astype(float)

            mask_stack = np.stack([seed_masks[sid] for sid in sorted_ids], axis=0)
            overlap_map = mask_stack.sum(axis=0)
        overlap_after = int((overlap_map >= 2).sum())

    seed_labels = np.zeros_like(tissue_mask, dtype=np.int32)
    for structure_id in sorted(seed_masks):
        seed_labels[seed_masks[structure_id]] = int(structure_id)

    posterior_defined_mask = tissue_mask & (normalized_stack_np.sum(axis=0) > 0)
    posterior_structure_ids = np.array(
        [int(spec["structure_id"]) for spec in valid_specs],
        dtype=np.int32,
    )[assignment_idx]
    partition_labels = np.zeros_like(seed_labels, dtype=np.int32)
    partition_labels[posterior_defined_mask] = posterior_structure_ids[posterior_defined_mask]
    partition_labels[seed_labels > 0] = seed_labels[seed_labels > 0]

    unassigned_mask = tissue_mask & (partition_labels == 0)
    if unassigned_mask.any() and np.any(seed_labels > 0):
        _, nearest_indices = distance_transform_edt(seed_labels == 0, return_indices=True)
        nearest_seed_labels = seed_labels[nearest_indices[0], nearest_indices[1]]
        partition_labels[unassigned_mask] = nearest_seed_labels[unassigned_mask]

    partition_labels[occupied_mask & (partition_labels == 0)] = posterior_structure_ids[
        occupied_mask & (partition_labels == 0)
    ]

    for spec in valid_specs:
        structure_id = int(spec["structure_id"])
        partition_mask = partition_labels == structure_id
        if not partition_mask.any():
            continue
        structure_contours[structure_id]["partition_mask"] = partition_mask.astype(float)

    metrics = {
        "valid_structures": [int(spec["structure_id"]) for spec in valid_specs],
        "skipped_structures": skipped_structures,
        "grid_shape": [int(grid_x.shape[0]), int(grid_x.shape[1])],
        "tissue_threshold": tissue_threshold,
        "overlap_pixels_after": overlap_after,
        "occupied_pixels": int(occupied_mask.sum()),
        "partition_pixels_total": int((partition_labels > 0).sum()),
        "structure_pixels": {
            str(spec["structure_id"]): int((partition_labels == int(spec["structure_id"])).sum())
            for spec in valid_specs
        },
        "seed_pixels": {
            str(structure_id): int(seed_masks[structure_id].sum())
            for structure_id in sorted(seed_masks)
        },
    }
    partition_data = {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "partition_labels": partition_labels,
    }
    return structure_contours, partition_data, metrics


def _assign_cells_to_partition(
    *,
    cells: pd.DataFrame,
    partition_data: dict[str, np.ndarray],
    structure_specs: list[dict[str, object]],
    x_col: str,
    y_col: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    x_edges = np.asarray(partition_data["x_edges"], dtype=float)
    y_edges = np.asarray(partition_data["y_edges"], dtype=float)
    partition_labels = np.asarray(partition_data["partition_labels"], dtype=np.int32)

    x_idx = np.searchsorted(x_edges, cells[x_col].to_numpy(dtype=float), side="right") - 1
    y_idx = np.searchsorted(y_edges, cells[y_col].to_numpy(dtype=float), side="right") - 1
    x_idx = np.clip(x_idx, 0, partition_labels.shape[1] - 1)
    y_idx = np.clip(y_idx, 0, partition_labels.shape[0] - 1)

    assigned_ids = partition_labels[y_idx, x_idx]
    name_lookup = {
        int(spec["structure_id"]): str(spec["structure_name"]) for spec in structure_specs
    }
    assigned_names = [name_lookup.get(int(value), "unassigned") for value in assigned_ids]

    assigned_cells = cells.copy()
    assigned_cells["isoline_structure_id"] = assigned_ids.astype(int)
    assigned_cells["isoline_structure_name"] = assigned_names

    assignment_mask = assigned_cells["isoline_structure_id"] > 0
    metrics = {
        "cell_count": int(len(assigned_cells)),
        "assigned_cell_count": int(assignment_mask.sum()),
        "assigned_cell_fraction": float(assignment_mask.mean()),
        "structure_cell_counts": {
            str(spec["structure_id"]): int(
                (assigned_cells["isoline_structure_id"] == int(spec["structure_id"])).sum()
            )
            for spec in structure_specs
        },
    }
    return assigned_cells, metrics


def _attach_partition_contours(
    structure_contours: dict[int, dict[str, object]],
) -> dict[int, dict[str, object]]:
    updated: dict[int, dict[str, object]] = {}
    for structure_id, payload in structure_contours.items():
        data = dict(payload)
        partition_mask = np.asarray(data.get("partition_mask"))
        if partition_mask.size == 0:
            data["contours"] = []
            updated[structure_id] = data
            continue
        contours = extract_contour_paths(
            np.asarray(data["grid_x"], dtype=float),
            np.asarray(data["grid_y"], dtype=float),
            partition_mask.astype(float),
            level=0.5,
        )
        data["contours"] = [np.asarray(contour, dtype=float) for contour in contours if len(contour) >= 3]
        updated[structure_id] = data
    return updated


def _render_multi_structure_preview(
    *,
    assigned_cells: pd.DataFrame,
    structure_contours: dict[int, dict[str, object]],
    structure_specs: list[dict[str, object]],
    x_col: str,
    y_col: str,
    output_path: Path,
) -> Path:
    sampled_cells = assigned_cells
    if len(sampled_cells) > 60000:
        sampled_cells = sampled_cells.sample(n=60000, random_state=0).copy()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        sampled_cells[x_col],
        sampled_cells[y_col],
        s=2,
        c="#D3D7DE",
        alpha=0.25,
        linewidths=0,
        label="cells",
    )

    for spec in structure_specs:
        structure_id = int(spec["structure_id"])
        structure_color = str(spec["structure_color"])
        structure_name = str(spec["structure_name"])
        structure_cells = sampled_cells.loc[
            sampled_cells["isoline_structure_id"] == structure_id,
            [x_col, y_col],
        ]
        if not structure_cells.empty:
            ax.scatter(
                structure_cells[x_col],
                structure_cells[y_col],
                s=4,
                color=structure_color,
                alpha=0.45,
                linewidths=0,
                label=structure_name,
            )

        for contour in structure_contours.get(structure_id, {}).get("contours", []):
            contour_array = np.asarray(contour, dtype=float)
            if contour_array.ndim != 2 or contour_array.shape[1] != 2:
                continue
            ax.plot(
                contour_array[:, 0],
                contour_array[:, 1],
                color=structure_color,
                linewidth=2,
            )

    ax.set_aspect("equal")
    ax.set_title(f"Multi-structure contour preview | structures={len(structure_specs)}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _hex_to_rgb_triplet(hex_color: str) -> list[int]:
    value = str(hex_color).strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {hex_color!r}")
    return [int(value[index : index + 2], 16) for index in (0, 2, 4)]


def _selection_base_name(structure_id: int, structure_name: str) -> str:
    return f"S{int(structure_id)} {str(structure_name).replace('[', '').replace(']', '')}".strip()


def _component_mask_to_polygons_xenium(
    component_mask: np.ndarray,
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    max_vertices: int = 100000,
) -> list[np.ndarray]:
    if not component_mask.any():
        return []

    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    padded = np.pad(binary_fill_holes(component_mask).astype(np.uint8), 1, mode="constant")
    contours = measure.find_contours(padded.astype(float), 0.5)
    polygons: list[np.ndarray] = []

    x0 = float(x_edges[0] - dx)
    y0 = float(y_edges[0] - dy)
    for contour in contours:
        if contour.shape[0] < 4:
            continue

        rows = contour[:, 0]
        cols = contour[:, 1]
        x_coords = x0 + (cols + 0.5) * dx
        y_coords = y0 + (rows + 0.5) * dy
        polygon = np.column_stack([x_coords, y_coords]).astype(float)

        if not np.allclose(polygon[0], polygon[-1]):
            polygon = np.vstack([polygon, polygon[0]])

        rounded_polygon = np.round(polygon, 6)
        dedup_points = [rounded_polygon[0]]
        for point in rounded_polygon[1:]:
            if not np.allclose(point, dedup_points[-1]):
                dedup_points.append(point)
        polygon = np.asarray(dedup_points, dtype=float)
        if polygon.shape[0] < 4:
            continue
        if not np.allclose(polygon[0], polygon[-1]):
            polygon = np.vstack([polygon, polygon[0]])

        if polygon.shape[0] > max_vertices:
            step = int(np.ceil(polygon.shape[0] / max_vertices))
            polygon = polygon[::step]
            if not np.allclose(polygon[0], polygon[-1]):
                polygon = np.vstack([polygon, polygon[0]])

        if polygon.shape[0] >= 4:
            polygons.append(polygon)
    return polygons


def _write_xenium_explorer_annotation_exports(
    *,
    output_dir: Path,
    partition_data: dict[str, np.ndarray],
    structure_specs: list[dict[str, object]],
    xenium_pixel_size_um: float,
) -> dict[str, str]:
    features: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    partition_labels = np.asarray(partition_data["partition_labels"], dtype=np.int32)
    x_edges = np.asarray(partition_data["x_edges"], dtype=float) / float(xenium_pixel_size_um)
    y_edges = np.asarray(partition_data["y_edges"], dtype=float) / float(xenium_pixel_size_um)

    for spec in structure_specs:
        structure_id = int(spec["structure_id"])
        structure_name = str(spec["structure_name"])
        structure_color = str(spec["structure_color"])
        structure_mask = partition_labels == structure_id
        if not structure_mask.any():
            continue

        labeled_components, n_components = nd_label(structure_mask)
        base_name = _selection_base_name(structure_id, structure_name)
        rgb_color = _hex_to_rgb_triplet(structure_color)

        for component_index in range(1, int(n_components) + 1):
            component_mask = labeled_components == component_index
            polygons = _component_mask_to_polygons_xenium(
                component_mask,
                x_edges=x_edges,
                y_edges=y_edges,
            )
            if not polygons:
                continue

            for polygon_index, polygon in enumerate(polygons, start=1):
                selection_name = (
                    base_name
                    if int(n_components) == 1 and len(polygons) == 1
                    else f"{base_name} #{component_index}.{polygon_index}"
                )
                feature = {
                    "type": "Feature",
                    "id": str(uuid4()),
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[list(map(float, point)) for point in polygon]],
                    },
                    "properties": {
                        "objectType": "annotation",
                        "name": selection_name,
                        "classification": {
                            "name": selection_name,
                            "color": rgb_color,
                        },
                        "structure_id": structure_id,
                        "assigned_structure": structure_name,
                        "component_index": int(component_index),
                        "polygon_index": int(polygon_index),
                    },
                }
                features.append(feature)

                for x_value, y_value in polygon:
                    csv_rows.append(
                        {
                            "Selection": selection_name,
                            "X": float(x_value),
                            "Y": float(y_value),
                        }
                    )

                summary_rows.append(
                    {
                        "Selection": selection_name,
                        "StructureID": structure_id,
                        "AssignedStructure": structure_name,
                        "ComponentIndex": int(component_index),
                        "PolygonIndex": int(polygon_index),
                        "VertexCount": int(polygon.shape[0]),
                    }
                )

    geojson_payload = {
        "type": "FeatureCollection",
        "features": features,
    }
    geojson_path = output_dir / "xenium_explorer_annotations.geojson"
    geojson_path.write_text(
        json.dumps(geojson_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = output_dir / "xenium_explorer_annotations.csv"
    pd.DataFrame(csv_rows, columns=["Selection", "X", "Y"]).to_csv(csv_path, index=False)

    summary_path = output_dir / "xenium_explorer_annotations_summary.csv"
    pd.DataFrame(
        summary_rows,
        columns=[
            "Selection",
            "StructureID",
            "AssignedStructure",
            "ComponentIndex",
            "PolygonIndex",
            "VertexCount",
        ],
    ).to_csv(summary_path, index=False)

    return {
        "geojson": str(geojson_path),
        "csv": str(csv_path),
        "summary": str(summary_path),
    }


def _structure_count_rows(
    *,
    structure_specs: Sequence[Mapping[str, Any]],
    structure_contours: dict[int, dict[str, object]],
    assigned_cells: pd.DataFrame,
    input_cell_counts: Mapping[int, int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in structure_specs:
        structure_id = int(spec["structure_id"])
        payload = structure_contours.get(structure_id, {})
        contour_count = int(len(payload.get("contours", [])))
        if contour_count == 0 and "partition_mask" in payload:
            _, contour_count = nd_label(np.asarray(payload["partition_mask"], dtype=bool))
        rows.append(
            {
                "structure_id": structure_id,
                "structure_name": str(spec["structure_name"]),
                "selected_cluster_ids": ", ".join(str(item) for item in spec["cluster_ids_raw"]),
                "input_cell_count": int(input_cell_counts.get(structure_id, 0)),
                "assigned_cell_count": int(
                    (assigned_cells["isoline_structure_id"] == structure_id).sum()
                ),
                "contour_count": int(contour_count),
            }
        )
    return rows
