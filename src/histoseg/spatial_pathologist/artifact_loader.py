from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib
import json
import sys

import pandas as pd

from .config import SpatialPathologistConfig


_BASE_PATH_KEYS = {
    "dataset_root",
    "cluster_csv",
    "cells_parquet",
    "cluster_annotation_csv",
    "hierarchical_reference_json",
    "celltype_harmonization_json",
    "reference_compartments_csv",
    "keyword_rules_json",
    "analysis_output_dir",
    "validation_output_dir",
    "he_alignment_csv",
    "he_image_tif",
    "contour_pattern_dir",
}


def _resolve_value(key: str, value: Any, base_dir: Path) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_value(k, v, base_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(key, v, base_dir) for v in value]
    if isinstance(value, str) and key in _BASE_PATH_KEYS:
        return (base_dir / value).resolve()
    return value


def load_base_pipeline_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = {key: _resolve_value(key, value, config_path.parent) for key, value in raw.items()}
    cfg["config_path"] = config_path
    cfg["project_root"] = config_path.parent.parent
    return cfg


def _find_segmentation_repo_root(config_path: Path) -> Path:
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "src" / "tissue_structure_pipeline").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate segmentation_methods repo root from {config_path}"
    )


def ensure_base_pipeline_outputs(cfg: SpatialPathologistConfig) -> dict[str, Any]:
    base_cfg = load_base_pipeline_config(cfg.base_pipeline_config)
    analysis_dir = Path(base_cfg["analysis_output_dir"])
    validation_dir = Path(base_cfg["validation_output_dir"])

    required = [
        analysis_dir / "structure_assignments.csv",
        analysis_dir / "structure_assignment_details.json",
        analysis_dir / "cluster_structure_lookup.csv",
        validation_dir / "structure_isoline_metrics.json",
        validation_dir / "xenium_explorer_annotations_summary.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if not missing and not cfg.force_recompute_pipeline:
        return base_cfg

    repo_root = _find_segmentation_repo_root(Path(base_cfg["config_path"]))
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    run_structure_assignment = importlib.import_module(
        "tissue_structure_pipeline.structure_assignment"
    ).run_structure_assignment
    run_he_overlay = importlib.import_module(
        "tissue_structure_pipeline.he_overlay"
    ).run_he_overlay

    run_structure_assignment(base_cfg)
    run_he_overlay(base_cfg)
    return base_cfg


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_case_bundle(cfg: SpatialPathologistConfig) -> dict[str, Any]:
    base_cfg = ensure_base_pipeline_outputs(cfg)
    analysis_dir = Path(base_cfg["analysis_output_dir"])
    validation_dir = Path(base_cfg["validation_output_dir"])

    structure_assignments = pd.read_csv(analysis_dir / "structure_assignments.csv")
    structure_details = _load_json(analysis_dir / "structure_assignment_details.json")
    detail_map = {int(item["structure_id"]): item for item in structure_details}
    cluster_lookup = pd.read_csv(analysis_dir / "cluster_structure_lookup.csv")
    cells = pd.read_parquet(analysis_dir / "cells_with_structures.parquet")
    partition_path = validation_dir / "cells_with_isoline_partition.parquet"
    if partition_path.exists():
        cells_for_counts = pd.read_parquet(partition_path)
        structure_id_col = "isoline_structure_id"
        structure_name_col = "isoline_structure_name"
    else:
        cells_for_counts = cells
        structure_id_col = "structure_id"
        structure_name_col = "structure_name"
    run_summary = _load_json(analysis_dir / "run_summary.json")
    validation_metrics = _load_json(validation_dir / "structure_isoline_metrics.json")
    annotation_summary = pd.read_csv(
        validation_dir / "xenium_explorer_annotations_summary.csv"
    )

    total_cells = int(len(cells_for_counts))

    cluster_counts = (
        cells_for_counts.groupby(["cluster", "cell_type", structure_id_col, structure_name_col])
        .size()
        .rename("cell_count")
        .reset_index()
    )
    cluster_counts = cluster_counts.rename(
        columns={
            structure_id_col: "structure_id",
            structure_name_col: "structure_name",
        }
    )
    cluster_counts = cluster_counts.sort_values(["structure_id", "cell_count"], ascending=[True, False])

    structure_cell_counts = (
        cells_for_counts.groupby([structure_id_col, structure_name_col])
        .size()
        .rename("cell_count")
        .reset_index()
    )
    structure_cell_counts = structure_cell_counts.rename(
        columns={
            structure_id_col: "structure_id",
            structure_name_col: "structure_name",
        }
    )
    structure_cell_count_map = {
        int(row.structure_id): int(row.cell_count)
        for row in structure_cell_counts.itertuples(index=False)
    }
    cluster_structure_counts = (
        cells_for_counts.groupby(["cluster", structure_id_col, structure_name_col])
        .size()
        .rename("cell_count")
        .reset_index()
    )
    cluster_structure_counts = cluster_structure_counts.rename(
        columns={
            structure_id_col: "structure_id",
            structure_name_col: "structure_name",
        }
    )
    cluster_structure_count_map = {
        (int(row.cluster), int(row.structure_id)): int(row.cell_count)
        for row in cluster_structure_counts.itertuples(index=False)
    }

    polygon_counts = (
        annotation_summary.groupby(["StructureID", "AssignedStructure"])
        .size()
        .rename("polygon_count")
        .reset_index()
    )
    polygon_count_map = {
        int(row.StructureID): int(row.polygon_count)
        for row in polygon_counts.itertuples(index=False)
    }

    structures: list[dict[str, Any]] = []
    for row in structure_assignments.itertuples(index=False):
        structure_id = int(row.structure_id)
        detail = detail_map[structure_id]
        score_total = _safe_float(getattr(row, "score_total", 0.0))
        winner_margin = _safe_float(getattr(row, "winner_margin_to_second_leaf", 0.0))
        top_clusters_df = cluster_counts.loc[cluster_counts["structure_id"] == structure_id].copy()
        top_clusters = []
        for cluster_row in top_clusters_df.head(cfg.top_clusters_per_structure).itertuples(index=False):
            structure_cell_count = structure_cell_count_map[structure_id]
            top_clusters.append(
                {
                    "cluster_id": int(cluster_row.cluster),
                    "label": str(cluster_row.cell_type),
                    "cell_count": int(cluster_row.cell_count),
                    "fraction_of_structure": (
                        float(cluster_row.cell_count) / structure_cell_count
                        if structure_cell_count
                        else 0.0
                    ),
                }
            )

        structures.append(
            {
                "structure_id": structure_id,
                "assigned_label": str(row.assigned_structure),
                "assigned_reference_id": str(row.assigned_reference_id),
                "behavior": str(row.assigned_behavior),
                "top_group": int(row.top_group),
                "n_leaf_clusters": int(row.n_leaf_clusters),
                "score_total": score_total,
                "winner_margin_to_second_leaf": winner_margin,
                "case_cell_count": structure_cell_count_map.get(structure_id, 0),
                "case_cell_fraction": (
                    structure_cell_count_map.get(structure_id, 0) / total_cells if total_cells else 0.0
                ),
                "polygon_count": polygon_count_map.get(structure_id, 0),
                "color_hex": str(row.color_hex),
                "fallback_applied": bool(row.fallback_applied),
                "fallback_reason": None if pd.isna(row.fallback_reason) else str(row.fallback_reason),
                "top_candidates": detail.get("scoring_result", {}).get("top_leaf_candidates", []),
                "harmonized_composition": dict(detail.get("harmonized_composition", {})),
                "raw_empirical_composition": dict(detail.get("raw_empirical_composition", {})),
                "top_clusters": top_clusters,
            }
        )

    cluster_structure_map = {
        int(row.cluster): {
            "structure_id": int(row.structure_id),
            "structure_name": str(row.structure_name),
            "top_group": int(row.top_group),
            "structure_leaf_count": int(row.structure_leaf_count),
            "cell_type": str(row.cell_type),
        }
        for row in cluster_lookup.itertuples(index=False)
    }
    clusters: list[dict[str, Any]] = []
    for cluster_id, info in sorted(cluster_structure_map.items()):
        structure_id = int(info["structure_id"])
        structure_cell_count = structure_cell_count_map.get(structure_id, 0)
        cluster_cell_count = cluster_structure_count_map.get((cluster_id, structure_id), 0)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "label": str(info["cell_type"]),
                "structure_id": structure_id,
                "structure_name": str(info["structure_name"]),
                "top_group": int(info["top_group"]),
                "structure_leaf_count": int(info["structure_leaf_count"]),
                "cluster_cell_count": cluster_cell_count,
                "fraction_of_structure": (
                    cluster_cell_count / structure_cell_count if structure_cell_count else 0.0
                ),
                "fraction_of_case": cluster_cell_count / total_cells if total_cells else 0.0,
            }
        )

    return {
        "case_name": cfg.case_name,
        "study_context": cfg.study_context,
        "total_cells": total_cells,
        "base_pipeline_config": str(base_cfg["config_path"]),
        "analysis_output_dir": str(analysis_dir),
        "validation_output_dir": str(validation_dir),
        "run_summary": run_summary,
        "validation_metrics": validation_metrics,
        "structures": structures,
        "clusters": clusters,
        "images": {
            "he_overlay": str(validation_dir / "he_structure_isoline_overlay.png"),
            "spatial_overlay": str(validation_dir / "spatial_structure_isoline_overlay.png"),
            "clustermap": str(analysis_dir / "structure_clustermap.pdf"),
        },
    }
