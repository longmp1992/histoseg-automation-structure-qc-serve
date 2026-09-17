from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import uuid


def bootstrap_runtime_env() -> None:
    """Point caches to writable paths before importing HistoSeg/matplotlib."""
    os.environ.setdefault("MPLBACKEND", "Agg")

    runtime_root = Path("/tmp") / "histoseg-serve" / str(os.getuid())
    path_defaults = {
        "HOME": "/tmp",
        "XDG_CACHE_HOME": "/tmp/.cache",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "GRADIO_TEMP_DIR": "/tmp/gradio",
    }
    fallback_dirs = {
        "HOME": runtime_root / "home",
        "XDG_CACHE_HOME": runtime_root / "cache",
        "MPLCONFIGDIR": runtime_root / "matplotlib",
        "GRADIO_TEMP_DIR": runtime_root / "gradio",
    }

    for key, default_path in path_defaults.items():
        candidate = Path(os.environ.get(key, default_path))
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError:
            fallback = fallback_dirs[key]
            fallback.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(fallback)


bootstrap_runtime_env()

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage, to_tree
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    label as nd_label,
)
from scipy.spatial.distance import squareform

from histoseg.spatial_pathologist.structure_qc import (
    PARAMS as STRUCTURE_QC_PARAMS,
    run as run_structure_qc,
)
from histoseg.spatial_pathologist.auto_split import EXPLORATION_FILE as AUTOSPLIT_EXPLORATION_FILE
from histoseg.spatial_pathologist.auto_split import Exploration as AutoSplitExploration
from histoseg.spatial_pathologist.auto_split import run_auto_split

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional runtime helper
    pq = None

try:
    from histoseg.contour import (
        Pattern1IsolineConfig,
        Pattern1IsolineResult,
        compute_cophenetic_from_distance_matrix,
        compute_searcher_findee_distance_matrix_from_df,
        plot_cophenetic_heatmap,
    )
    from histoseg.contour.pattern1_isoline import (
        _normalize_cluster_label,
        _validate_label_scheme,
        align_clusters_with_cells,
        extract_contour_paths,
    )

    HISTOSEG_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - startup fallback only
    Pattern1IsolineConfig = None  # type: ignore[assignment]
    Pattern1IsolineResult = None  # type: ignore[assignment]
    _normalize_cluster_label = None  # type: ignore[assignment]
    _validate_label_scheme = None  # type: ignore[assignment]
    align_clusters_with_cells = None  # type: ignore[assignment]
    extract_contour_paths = None  # type: ignore[assignment]
    compute_cophenetic_from_distance_matrix = None  # type: ignore[assignment]
    compute_searcher_findee_distance_matrix_from_df = None  # type: ignore[assignment]
    plot_cophenetic_heatmap = None  # type: ignore[assignment]
    HISTOSEG_IMPORT_ERROR = str(exc)


APP_NAME = "HistoSeg Structure QC"
APP_DESCRIPTION = (
    "An end-to-end Xenium workflow that builds a StructureMap, generates HistoSeg contours, "
    "and then validates every structure with CSR and random-labelling spatial tests."
)
DEFAULT_PATTERN1 = "10,23,19,27,14,20,25,26"
GROUP_SELECTION_EMPTY = (
    "No structures selected yet. Use the checklist below, or type one structure per line manually."
)
SELECTION_NOTES_TEXT = (
    "Choose one or more structures in the checklist, or type cluster IDs manually below. "
    "Nothing is rerun while you are choosing. The app reads your final selection only when you click "
    "'Run multi-structure contour analysis'. If the text box is non-empty, the manual lines take priority."
)


def structure_selection_help_text() -> str:
    return f"{GROUP_SELECTION_EMPTY}\n\n{SELECTION_NOTES_TEXT}"
XENIUM_PIXEL_SIZE_UM = 0.2125
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
DEFAULT_STRUCTURE_ISOLINE_CFG = {
    "bins_x": 900,
    "bins_y": 700,
    "gaussian_sigma": 2.25,
    "density_scale_quantile": 0.98,
    "support_quantile": 0.18,
    "tissue_quantile": 0.06,
    "min_dominance": 0.34,
    "closing_iterations": 2,
    "opening_iterations": 1,
    "fill_holes": True,
    "min_cells": 500,
    "min_component_pixels": 180,
}
PREFERRED_WORK_DIR = Path(os.environ.get("APP_DATA_DIR", "./project-vol")).resolve()
FALLBACK_WORK_DIR = Path("/tmp/project-vol")


@dataclass(frozen=True)
class RuntimeProfile:
    grid_n: int
    bg_max_points: int
    syn_bg_density: float
    syn_bg_min: int
    syn_bg_max: int
    scale_label: str
    notes: tuple[str, ...]


def resolve_work_dir() -> Path:
    for candidate in (PREFERRED_WORK_DIR, FALLBACK_WORK_DIR):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise PermissionError(
        f"Could not find a writable work directory. Tried: {PREFERRED_WORK_DIR} and {FALLBACK_WORK_DIR}"
    )


DEFAULT_WORK_DIR = resolve_work_dir()
RUNS_DIR = DEFAULT_WORK_DIR / "runs"
SELECTIONS_DIR = DEFAULT_WORK_DIR / "structure-selections"


def ensure_workdirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)


def log_event(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def to_internal_label_scheme(label_scheme: str) -> str:
    if label_scheme is None:
        return "p1_is_one"
    return _validate_label_scheme(label_scheme)


def describe_label_scheme(label_scheme: str) -> str:
    internal = to_internal_label_scheme(label_scheme)
    if internal == "p1_is_one":
        return "Selected structures are treated as the signal of interest"
    return "Selected structures are treated as background"


def parse_pattern1_clusters(raw: str) -> list[int | str]:
    values: list[int | str] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if token.lstrip("-").isdigit():
            values.append(int(token))
        else:
            values.append(token)
    if not values:
        raise ValueError("Clusters to outline cannot be empty.")
    return values


def parse_optional_clusters(raw: str) -> list[int | str]:
    if raw is None:
        return []
    if not str(raw).strip():
        return []
    return parse_pattern1_clusters(str(raw))


def stringify_clusters(clusters: list[int | str]) -> str:
    return ",".join(str(item) for item in clusters)


def parse_structure_cluster_groups(raw: str) -> list[list[int | str]]:
    if raw is None or not str(raw).strip():
        raise ValueError("Please select one or more structures, or type cluster IDs with one structure per line.")

    groups: list[list[int | str]] = []
    normalized_text = str(raw).replace(";", "\n")
    for raw_line in normalized_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        parsed = parse_pattern1_clusters(line)
        if parsed:
            groups.append(parsed)

    if not groups:
        raise ValueError("No valid structure groups were found. Use one line per structure, for example '10,23,19'.")
    return groups


def stringify_structure_cluster_groups(cluster_groups: list[list[int | str]]) -> str:
    return "\n".join(stringify_clusters(group) for group in cluster_groups)


def summarize_clusters(clusters: list[str], max_items: int = 8) -> str:
    if len(clusters) <= max_items:
        return ", ".join(clusters)
    head = ", ".join(clusters[:max_items])
    return f"{head}, ... (+{len(clusters) - max_items} more)"


def safe_count_parquet_rows(parquet_path: Path) -> int | None:
    if pq is None:
        return None
    try:
        return int(pq.ParquetFile(parquet_path).metadata.num_rows)
    except Exception:
        return None


def safe_count_csv_rows(csv_path: Path) -> int | None:
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
            count = sum(1 for _ in handle) - 1
        return max(count, 0)
    except Exception:
        return None


def choose_runtime_profile(
    *,
    requested_grid_n: int,
    requested_syn_bg_density: float,
    use_synth_bg: bool,
    estimated_rows: int | None,
) -> RuntimeProfile:
    effective_grid_n = int(requested_grid_n)
    bg_max_points = 60000
    syn_bg_density = float(requested_syn_bg_density)
    syn_bg_min = 20000
    syn_bg_max = 120000
    notes: list[str] = []

    ref_rows = estimated_rows or 0
    if ref_rows >= 80000:
        scale_label = "large"
        effective_grid_n = min(effective_grid_n, 450)
        bg_max_points = 12000
        syn_bg_density = min(syn_bg_density, 0.0015)
        syn_bg_min = 4000
        syn_bg_max = 12000
    elif ref_rows >= 40000:
        scale_label = "medium-large"
        effective_grid_n = min(effective_grid_n, 550)
        bg_max_points = 18000
        syn_bg_density = min(syn_bg_density, 0.0025)
        syn_bg_min = 5000
        syn_bg_max = 18000
    elif ref_rows >= 20000:
        scale_label = "medium"
        effective_grid_n = min(effective_grid_n, 650)
        bg_max_points = 25000
        syn_bg_density = min(syn_bg_density, 0.0035)
        syn_bg_min = 8000
        syn_bg_max = 25000
    elif ref_rows >= 10000:
        scale_label = "small-medium"
        effective_grid_n = min(effective_grid_n, 800)
        bg_max_points = 35000
        syn_bg_density = min(syn_bg_density, 0.005)
        syn_bg_min = 12000
        syn_bg_max = 35000
    else:
        scale_label = "small"

    if effective_grid_n != int(requested_grid_n):
        notes.append(
            f"Auto-reduced grid_n from {requested_grid_n} to {effective_grid_n} for Serve runtime stability."
        )
    if use_synth_bg and syn_bg_density != float(requested_syn_bg_density):
        notes.append(
            f"Auto-reduced synthetic background density from {requested_syn_bg_density:.4f} to {syn_bg_density:.4f}."
        )

    return RuntimeProfile(
        grid_n=effective_grid_n,
        bg_max_points=bg_max_points,
        syn_bg_density=syn_bg_density,
        syn_bg_min=syn_bg_min,
        syn_bg_max=syn_bg_max,
        scale_label=scale_label,
        notes=tuple(notes),
    )


def stage_uploaded_file(uploaded: object | None, target_dir: Path, explicit_name: str | None = None) -> Path | None:
    if uploaded is None:
        return None
    source = Path(str(uploaded))
    if not source.exists():
        raise FileNotFoundError(f"Uploaded file not found: {source}")
    filename = explicit_name or source.name
    destination = target_dir / filename
    shutil.copy2(source, destination)
    return destination


def resolve_inputs(
    *,
    cells_upload: object | None,
    clusters_upload: object | None,
    tissue_upload: object | None,
    target_dir: Path,
) -> tuple[Path, Path, Path | None]:
    cells_path = stage_uploaded_file(cells_upload, target_dir)
    clusters_path = stage_uploaded_file(clusters_upload, target_dir)
    tissue_path = stage_uploaded_file(tissue_upload, target_dir)

    if cells_path is None:
        raise ValueError("Missing cells.parquet. Please upload the cell coordinate file.")
    if clusters_path is None:
        raise ValueError("Missing clusters.csv. Please upload the cluster assignment file.")

    return cells_path, clusters_path, tissue_path


def build_run_dir() -> Path:
    ensure_workdirs()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"run-{stamp}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = RUNS_DIR / f"run-{stamp}-{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_selection_dir() -> Path:
    ensure_workdirs()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    selection_dir = SELECTIONS_DIR / f"selection-{stamp}"
    suffix = 1
    while selection_dir.exists():
        suffix += 1
        selection_dir = SELECTIONS_DIR / f"selection-{stamp}-{suffix}"
    selection_dir.mkdir(parents=True, exist_ok=False)
    return selection_dir


def cleanup_old_runs(max_keep: int = 2) -> list[str]:
    ensure_workdirs()
    runs = sorted(
        [path for path in RUNS_DIR.glob("run-*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    removed: list[str] = []
    for stale in runs[max_keep:]:
        try:
            shutil.rmtree(stale)
            removed.append(stale.name)
        except OSError:
            continue
    return removed


def cleanup_old_selections(max_keep: int = 2) -> list[str]:
    ensure_workdirs()
    selections = sorted(
        [path for path in SELECTIONS_DIR.glob("selection-*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    removed: list[str] = []
    for stale in selections[max_keep:]:
        try:
            shutil.rmtree(stale)
            removed.append(stale.name)
        except OSError:
            continue
    return removed


def directory_size_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def zip_outputs(output_dir: Path, archive_dir: Path | None = None) -> tuple[Path | None, str | None]:
    target_dir = archive_dir if archive_dir is not None else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_base = target_dir / "histoseg_outputs"
    archive_path = Path(f"{archive_base}.zip")
    output_bytes = directory_size_bytes(output_dir)
    free_bytes = shutil.disk_usage(target_dir).free

    # Creating a zip duplicates the output payload temporarily, so we keep a safety margin.
    required_free = max(output_bytes * 2, 256 * 1024 * 1024)
    if free_bytes < required_free:
        return None, (
            "Skipped zip archive because disk space is low on the Serve instance. "
            "The raw output files are still available below."
        )

    try:
        archive_path_str = shutil.make_archive(str(archive_base), "zip", root_dir=output_dir)
        return Path(archive_path_str), None
    except OSError as exc:
        try:
            archive_path.unlink(missing_ok=True)
        except OSError:
            pass
        if getattr(exc, "errno", None) == 28:
            return None, (
                "Skipped zip archive because the Serve instance ran out of disk space. "
                "The raw output files are still available below."
            )
        raise


def prepare_merged_clusters(cells_path: Path, clusters_path: Path) -> tuple[pd.DataFrame, str, str, str]:
    merged, id_col_used, x_col, y_col = align_clusters_with_cells(
        clusters_path,
        cells_path,
        barcode_col="Barcode",
        cluster_col="Cluster",
    )
    merged = merged.copy()
    merged["cluster"] = merged["cluster"].map(_normalize_cluster_label)
    merged = merged.loc[merged["cluster"] != ""].copy()
    return merged, id_col_used, x_col, y_col


def normalize_row_cophenetic(row_coph: pd.DataFrame) -> pd.DataFrame:
    labels = [_normalize_cluster_label(label) for label in row_coph.index]
    normalized = row_coph.copy()
    normalized.index = labels
    normalized.columns = labels
    return normalized


def remap_flat_clusters_by_leaf_order(cluster_ids: list[str], linkage_matrix, flat_labels) -> dict[str, int]:
    if linkage_matrix is None:
        return {cluster_ids[0]: 1}

    raw_map = {str(cluster_id): int(raw_label) for cluster_id, raw_label in zip(cluster_ids, flat_labels)}
    ordered_cluster_ids = [cluster_ids[index] for index in leaves_list(linkage_matrix)]

    raw_order: list[int] = []
    seen_raw: set[int] = set()
    for cluster_id in ordered_cluster_ids:
        raw_label = raw_map[str(cluster_id)]
        if raw_label in seen_raw:
            continue
        seen_raw.add(raw_label)
        raw_order.append(raw_label)

    remap = {raw_label: index + 1 for index, raw_label in enumerate(raw_order)}
    return {str(cluster_id): int(remap[raw_map[str(cluster_id)]]) for cluster_id in cluster_ids}


def group_color(group_id: int) -> str:
    return GROUP_PALETTE[(int(group_id) - 1) % len(GROUP_PALETTE)]


def build_structure_choice_label(group_id: int, clusters: list[str]) -> str:
    return f"Structure {group_id} | {len(clusters)} cluster IDs"


def build_group_table(
    group_state: dict[str, object] | None,
    selected_groups: list[str] | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not group_state:
        return pd.DataFrame(rows, columns=["Structure", "Cluster count", "Cluster IDs"])

    for record in group_state.get("group_records", []):
        rows.append(
            {
                "Structure": str(record["group_name"]),
                "Cluster count": int(record["cluster_count"]),
                "Cluster IDs": ", ".join(str(item) for item in record["clusters"]),
            }
        )
    return pd.DataFrame(rows, columns=["Structure", "Cluster count", "Cluster IDs"])


def build_structure_group_state(
    row_coph: pd.DataFrame,
    *,
    n_groups: int,
    linkage_method: str = "average",
) -> dict[str, object]:
    cluster_ids = [str(value) for value in row_coph.index]
    if not cluster_ids:
        raise ValueError("No clusters were available for dendrogram building.")

    cluster_to_leaf_index = {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}
    if len(cluster_ids) == 1:
        ordered_clusters = cluster_ids
        group_to_clusters = {1: ordered_clusters}
        linkage_matrix = None
        leaf_positions = {0: 5.0}
        node_leaf_map = {0: [0]}
    else:
        condensed = squareform(row_coph.values, checks=False)
        linkage_matrix = linkage(condensed, method=linkage_method)
        n_groups = max(1, min(int(n_groups), len(cluster_ids)))
        flat_labels = fcluster(linkage_matrix, t=n_groups, criterion="maxclust")
        cluster_to_group = remap_flat_clusters_by_leaf_order(cluster_ids, linkage_matrix, flat_labels)
        leaf_order = [int(index) for index in leaves_list(linkage_matrix)]
        ordered_clusters = [cluster_ids[index] for index in leaf_order]
        group_to_clusters: dict[int, list[str]] = {}
        for cluster_id in ordered_clusters:
            group_id = int(cluster_to_group[cluster_id])
            group_to_clusters.setdefault(group_id, []).append(cluster_id)
        leaf_positions = {leaf_id: 5.0 + 10.0 * order_index for order_index, leaf_id in enumerate(leaf_order)}

        root_node, node_list = to_tree(linkage_matrix, rd=True)
        node_leaf_map: dict[int, list[int]] = {}

        def collect_leaf_ids(node) -> list[int]:
            if node.is_leaf():
                leaves = [int(node.id)]
            else:
                leaves = collect_leaf_ids(node.left) + collect_leaf_ids(node.right)
            node_leaf_map[int(node.id)] = leaves
            return leaves

        collect_leaf_ids(root_node)
        _ = node_list  # Keeps the rd=True unpacking explicit for readability.

    leaf_set_to_node: dict[frozenset[int], dict[str, float]] = {}
    if linkage_matrix is None:
        leaf_set_to_node[frozenset({0})] = {"node_id": 0.0, "dist": 0.0}
    else:
        root_node, node_list = to_tree(linkage_matrix, rd=True)
        for node in node_list:
            leaves = node_leaf_map.get(int(node.id), [])
            leaf_set_to_node[frozenset(int(value) for value in leaves)] = {
                "node_id": float(node.id),
                "dist": float(node.dist),
            }

    ordered_cluster_to_position = {cluster_id: index for index, cluster_id in enumerate(ordered_clusters)}

    choices: list[str] = []
    choice_to_clusters: dict[str, list[str]] = {}
    table_rows: list[dict[str, object]] = []
    group_records: list[dict[str, Any]] = []
    for group_id, clusters in sorted(group_to_clusters.items()):
        choice_label = build_structure_choice_label(group_id, clusters)
        leaf_ids = [int(cluster_to_leaf_index[cluster_id]) for cluster_id in clusters]
        x_points = [float(leaf_positions[leaf_id]) for leaf_id in leaf_ids]
        leaf_span = sorted(int(ordered_cluster_to_position[cluster_id]) for cluster_id in clusters)
        node_summary = leaf_set_to_node.get(frozenset(leaf_ids), {"node_id": float(group_id), "dist": 0.0})
        span_left = min(x_points) - 5.0
        span_right = max(x_points) + 5.0
        y_data = float(node_summary["dist"])
        marker_y = y_data
        color = group_color(group_id)

        choices.append(choice_label)
        choice_to_clusters[choice_label] = list(clusters)
        table_rows.append(
            {
                "Selected": "",
                "Structure": f"Structure {group_id}",
                "Cluster count": len(clusters),
                "Cluster IDs": ", ".join(clusters),
            }
        )
        group_records.append(
            {
                "group_id": int(group_id),
                "group_name": f"Structure {group_id}",
                "choice_label": choice_label,
                "clusters": list(clusters),
                "cluster_count": int(len(clusters)),
                "color": color,
                "leaf_start": int(min(leaf_span)),
                "leaf_end": int(max(leaf_span)),
                "span_left": float(span_left),
                "span_right": float(span_right),
                "x_data": float(np.mean(x_points)),
                "y_data": float(y_data),
                "marker_y": float(marker_y),
            }
        )

    return {
        "n_groups": len(group_to_clusters),
        "ordered_clusters": ordered_clusters,
        "choices": choices,
        "choice_to_clusters": choice_to_clusters,
        "table_rows": table_rows,
        "group_records": group_records,
        "row_coph_labels": cluster_ids,
        "row_coph_values": row_coph.to_numpy().tolist(),
        "linkage_matrix": linkage_matrix.tolist() if linkage_matrix is not None else None,
        "selected_groups": [],
    }


def collect_clusters_from_groups(selected_groups: list[str], group_state: dict[str, object] | None) -> list[str]:
    if not group_state:
        return []
    choice_to_clusters = group_state.get("choice_to_clusters", {})
    ordered_clusters = group_state.get("ordered_clusters", [])
    selected_set = set(selected_groups or [])
    cluster_order: list[str] = []
    for cluster_id in ordered_clusters:
        for choice, clusters in choice_to_clusters.items():
            if choice in selected_set and cluster_id in clusters and cluster_id not in cluster_order:
                cluster_order.append(cluster_id)
    return cluster_order


def update_clusters_to_outline_from_groups(
    selected_groups: list[str] | None,
    group_state: dict[str, object] | None,
) -> tuple[str, str]:
    normalized_groups = normalize_selected_groups(selected_groups or [], group_state)
    if not normalized_groups:
        return "", GROUP_SELECTION_EMPTY

    choice_to_clusters = (group_state or {}).get("choice_to_clusters", {})
    grouped_clusters = [list(choice_to_clusters.get(choice, [])) for choice in normalized_groups]
    cluster_text = stringify_structure_cluster_groups(grouped_clusters)
    summary_lines = [f"Selected {len(normalized_groups)} structure(s)."]
    for idx, clusters in enumerate(grouped_clusters, start=1):
        summary_lines.append(f"Structure {idx}: {summarize_clusters([str(item) for item in clusters])}")
    summary = "\n".join(summary_lines)
    return cluster_text, summary


def _normalize_non_empty_lines(raw_text: str | None) -> list[str]:
    if raw_text is None:
        return []
    lines: list[str] = []
    for raw_line in str(raw_text).replace(";", "\n").splitlines():
        line = raw_line.strip()
        if line:
            lines.append(line)
    return lines


def sync_selected_groups_to_text(
    selected_groups: list[str] | None,
    current_text: str | None,
    previous_auto_lines: list[str] | None,
    group_state: dict[str, object] | None,
) -> tuple[str, str, list[str]]:
    normalized_groups = normalize_selected_groups(selected_groups or [], group_state)
    auto_cluster_text, auto_summary = update_clusters_to_outline_from_groups(normalized_groups, group_state)
    next_auto_lines = _normalize_non_empty_lines(auto_cluster_text)
    previous_auto_set = set(_normalize_non_empty_lines("\n".join(previous_auto_lines or [])))

    manual_lines = [line for line in _normalize_non_empty_lines(current_text) if line not in previous_auto_set]
    merged_lines = list(manual_lines)
    for line in next_auto_lines:
        if line not in merged_lines:
            merged_lines.append(line)

    if normalized_groups:
        summary_lines = [auto_summary]
        if manual_lines:
            summary_lines.append(
                f"Manual lines kept in the text box: {len(manual_lines)}. You can still edit any line before Run."
            )
        else:
            summary_lines.append("Selected structures were copied into the text box below. You can edit them before Run.")
        summary = "\n\n".join(summary_lines)
    elif manual_lines:
        summary = (
            "No checklist structures selected right now.\n\n"
            f"Manual lines still present in the text box: {len(manual_lines)}.\n"
            "Those manual lines will be used if you click Run."
        )
    else:
        summary = structure_selection_help_text()

    return "\n".join(merged_lines), summary, next_auto_lines


def normalize_selected_groups(
    selected_groups: list[str] | None,
    group_state: dict[str, object] | None,
) -> list[str]:
    if not group_state:
        return []
    selected_set = set(selected_groups or [])
    return [choice for choice in group_state.get("choices", []) if choice in selected_set]


def render_structure_selector_image(
    group_state: dict[str, object],
    selected_groups: list[str] | None,
) -> tuple[Path, dict[str, object]]:
    output_dir = Path(str(group_state["selector_output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_groups = normalize_selected_groups(selected_groups, group_state)
    selected_set = set(selected_groups)
    group_records = [dict(record) for record in group_state.get("group_records", [])]
    row_coph = pd.DataFrame(
        np.asarray(group_state["row_coph_values"], dtype=float),
        index=list(group_state["row_coph_labels"]),
        columns=list(group_state["row_coph_labels"]),
    )
    ordered_clusters = list(group_state["ordered_clusters"])
    linkage_payload = group_state.get("linkage_matrix")
    linkage_matrix = np.asarray(linkage_payload, dtype=float) if linkage_payload is not None else None

    selector_key = "none"
    if selected_set:
        selected_ids = [str(record["group_id"]) for record in group_records if record["choice_label"] in selected_set]
        selector_key = "_".join(selected_ids)
    selector_path = output_dir / f"interactive_structure_selector_{selector_key}.png"

    fig, ax_dendro = plt.subplots(figsize=(13.5, 5.8), facecolor="#07111D")
    ax_dendro.set_facecolor("#0C1726")

    if linkage_matrix is not None:
        dendrogram(
            linkage_matrix,
            no_labels=True,
            color_threshold=0,
            above_threshold_color="#6B8198",
            link_color_func=lambda _node_id: "#6B8198",
            ax=ax_dendro,
        )
        max_dist = float(np.max(linkage_matrix[:, 2])) if len(linkage_matrix) else 1.0
    else:
        ax_dendro.plot([5.0, 5.0], [0.0, 1.0], color="#6B8198", linewidth=2.5)
        max_dist = 1.0

    marker_offset = max(max_dist * 0.08, 0.24)
    marker_positions: dict[str, dict[str, float]] = {}
    marker_size = 310

    for record in group_records:
        is_selected = record["choice_label"] in selected_set
        color = str(record["color"])
        x_data = float(record["x_data"])
        y_data = float(record["y_data"])
        marker_y = y_data + marker_offset
        record["marker_y"] = marker_y

        ax_dendro.axvspan(
            float(record["span_left"]),
            float(record["span_right"]),
            color=color,
            alpha=0.22 if is_selected else 0.06,
            zorder=0,
        )
        ax_dendro.plot(
            [x_data, x_data],
            [y_data, marker_y - marker_offset * 0.18],
            color=color,
            linewidth=2.1 if is_selected else 1.3,
            alpha=0.95,
            zorder=4,
        )
        ax_dendro.scatter(
            [x_data],
            [marker_y],
            s=marker_size + (95 if is_selected else 0),
            color=color,
            edgecolors="#F7FBFF" if is_selected else "#132236",
            linewidths=2.0,
            zorder=6,
        )
        ax_dendro.text(
            x_data,
            marker_y,
            f"S{record['group_id']}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#07111D",
            zorder=7,
        )
        ax_dendro.text(
            x_data,
            marker_y + marker_offset * 0.72,
            f"{record['cluster_count']} IDs",
            ha="center",
            va="bottom",
            fontsize=8.3,
            color="#D9E8F8" if is_selected else "#9CB0C7",
            zorder=7,
        )

    ax_dendro.text(
        0.015,
        0.96,
        "Click a colored structure badge to add or remove that branch from the final contour run.",
        transform=ax_dendro.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#E9F2FD",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#101C2B", edgecolor="#1C3550", alpha=0.97),
    )
    ax_dendro.set_title("Interactive structure selector", loc="left", fontsize=15, color="#F5F9FF", pad=12)
    ax_dendro.set_ylabel("Cophenetic distance", color="#A8BCD3")
    leaf_positions = [5 + 10 * idx for idx in range(len(ordered_clusters))]
    if len(ordered_clusters) <= 18:
        tick_positions = leaf_positions
        tick_labels = ordered_clusters
    else:
        step = max(1, len(ordered_clusters) // 12)
        keep_indices = list(range(0, len(ordered_clusters), step))
        if keep_indices[-1] != len(ordered_clusters) - 1:
            keep_indices.append(len(ordered_clusters) - 1)
        tick_positions = [leaf_positions[idx] for idx in keep_indices]
        tick_labels = [ordered_clusters[idx] for idx in keep_indices]
    ax_dendro.set_xticks(tick_positions)
    ax_dendro.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8, color="#A9BDD4")
    ax_dendro.tick_params(axis="x", colors="#90A6BF")
    ax_dendro.tick_params(axis="y", colors="#90A6BF")
    ax_dendro.set_xlabel("Cluster IDs ordered by the dendrogram", color="#A8BCD3", labelpad=10)
    for spine in ax_dendro.spines.values():
        spine.set_color("#20354A")
    ax_dendro.set_ylim(-marker_offset * 0.4, max_dist + marker_offset * 2.0)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    for record in group_records:
        x_disp, y_disp = ax_dendro.transData.transform((float(record["x_data"]), float(record["marker_y"])))
        marker_positions[str(record["choice_label"])] = {
            "x": float(x_disp),
            "y": float(height - y_disp),
            "x_norm": float(x_disp / width),
            "y_norm": float((height - y_disp) / height),
            "radius": 32.0,
            "radius_norm": float(32.0 / max(width, height)),
        }

    fig.savefig(selector_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    next_state = dict(group_state)
    next_state["selected_groups"] = list(selected_groups)
    next_state["marker_positions"] = marker_positions
    next_state["selector_path"] = str(selector_path)
    return selector_path, next_state


def resolve_clicked_structure(
    click_index: object,
    group_state: dict[str, object] | None,
) -> str | None:
    if not group_state:
        return None

    marker_positions = group_state.get("marker_positions", {})
    if not marker_positions:
        return None

    candidate_points: list[tuple[float, float]] = []
    if isinstance(click_index, dict):
        if "x" in click_index and "y" in click_index:
            candidate_points.append((float(click_index["x"]), float(click_index["y"])))
    elif isinstance(click_index, (list, tuple)) and len(click_index) >= 2:
        a = float(click_index[0])
        b = float(click_index[1])
        candidate_points.append((a, b))
        if abs(a - b) > 1:
            candidate_points.append((b, a))
    else:
        return None

    best_choice: str | None = None
    best_distance = float("inf")
    best_threshold = float("inf")
    for choice_label, marker in marker_positions.items():
        for x_click, y_click in candidate_points:
            if max(abs(x_click), abs(y_click)) <= 1.5:
                distance = float(np.hypot(x_click - marker["x_norm"], y_click - marker["y_norm"]))
                threshold = float(marker["radius_norm"]) * 1.6
            else:
                distance = float(np.hypot(x_click - marker["x"], y_click - marker["y"]))
                threshold = float(marker["radius"]) * 1.6

            if distance < best_distance:
                best_choice = str(choice_label)
                best_distance = distance
                best_threshold = threshold

    if best_choice is not None and best_distance <= best_threshold:
        return best_choice
    return None


def refresh_structure_selection(
    selected_groups: list[str] | None,
    group_state: dict[str, object] | None,
    note: str | None = None,
) -> tuple[str | None, pd.DataFrame, dict[str, object], str, str, dict[str, object]]:
    if not group_state:
        empty_table = build_group_table({}, [])
        return None, empty_table, gr.update(choices=[], value=[]), "", note or GROUP_SELECTION_EMPTY, {}

    normalized_groups = normalize_selected_groups(selected_groups, group_state)
    selector_path, next_state = render_structure_selector_image(group_state, normalized_groups)
    cluster_text, summary = update_clusters_to_outline_from_groups(normalized_groups, next_state)
    if note:
        summary = f"{summary}\n{note}"

    return (
        str(selector_path),
        build_group_table(next_state, normalized_groups),
        gr.update(choices=next_state["choices"], value=normalized_groups),
        cluster_text,
        summary,
        next_state,
    )


def toggle_structure_group_from_selector(
    group_state: dict[str, object] | None,
    evt: gr.SelectData,
) -> tuple[str | None, pd.DataFrame, dict[str, object], str, str, dict[str, object]]:
    if not group_state:
        empty_table = build_group_table({}, [])
        return None, empty_table, gr.update(choices=[], value=[]), "", GROUP_SELECTION_EMPTY, {}

    current_groups = normalize_selected_groups(group_state.get("selected_groups", []), group_state)
    clicked_choice = resolve_clicked_structure(getattr(evt, "index", None), group_state)
    if clicked_choice is None:
        return refresh_structure_selection(
            current_groups,
            group_state,
            note="Click directly on one of the colored badges labelled S1, S2, S3, ... to toggle a structure.",
        )

    next_groups = list(current_groups)
    if clicked_choice in next_groups:
        next_groups = [choice for choice in next_groups if choice != clicked_choice]
    else:
        next_groups.append(clicked_choice)

    return refresh_structure_selection(next_groups, group_state)


def clear_structure_selection(
    group_state: dict[str, object] | None,
) -> tuple[str | None, pd.DataFrame, dict[str, object], str, str, dict[str, object]]:
    return refresh_structure_selection([], group_state, note="Selection cleared. Choose one or more structures to continue.")


def build_selected_structure_specs(
    raw_groups_text: str,
    selected_groups: list[str] | None,
    group_state: dict[str, object] | None,
) -> list[dict[str, object]]:
    selected_records: list[dict[str, object]] = []
    if group_state:
        record_by_choice = {
            str(record["choice_label"]): dict(record)
            for record in group_state.get("group_records", [])
        }
        for choice in normalize_selected_groups(selected_groups or [], group_state):
            record = record_by_choice.get(choice)
            if record is not None:
                selected_records.append(record)

    raw_groups_text = "" if raw_groups_text is None else str(raw_groups_text)
    parsed_groups = parse_structure_cluster_groups(raw_groups_text) if raw_groups_text.strip() else []

    if not parsed_groups:
        if not selected_records:
            raise ValueError(
                "Please choose one or more structures in the checklist, or type cluster IDs with one structure per line."
            )

        specs: list[dict[str, object]] = []
        seen_clusters: set[str] = set()
        for index, record in enumerate(selected_records, start=1):
            normalized_clusters = []
            for item in record.get("clusters", []):
                normalized = _normalize_cluster_label(item)
                if normalized and normalized not in normalized_clusters:
                    normalized_clusters.append(normalized)
            if not normalized_clusters:
                raise ValueError(f"Structure {index} does not contain any valid cluster IDs.")

            overlap = seen_clusters.intersection(normalized_clusters)
            if overlap:
                raise ValueError(
                    "A cluster ID cannot belong to more than one structure in the same run. "
                    f"Repeated cluster IDs: {', '.join(sorted(overlap))}"
                )
            seen_clusters.update(normalized_clusters)

            specs.append(
                {
                    "structure_id": int(index),
                    "structure_name": str(record["group_name"]),
                    "structure_color": str(record["color"]),
                    "source_label": str(record["choice_label"]),
                    "cluster_ids_raw": [str(item) for item in record.get("clusters", [])],
                    "cluster_ids_normalized": normalized_clusters,
                }
            )
        return specs

    specs: list[dict[str, object]] = []
    seen_clusters: set[str] = set()
    for index, group in enumerate(parsed_groups, start=1):
        normalized_clusters: list[str] = []
        for item in group:
            normalized = _normalize_cluster_label(item)
            if normalized and normalized not in normalized_clusters:
                normalized_clusters.append(normalized)
        if not normalized_clusters:
            raise ValueError(f"Structure {index} does not contain any valid cluster IDs.")

        overlap = seen_clusters.intersection(normalized_clusters)
        if overlap:
            raise ValueError(
                "A cluster ID cannot belong to more than one structure in the same run. "
                f"Repeated cluster IDs: {', '.join(sorted(overlap))}"
            )
        seen_clusters.update(normalized_clusters)

        if index <= len(selected_records):
            record = selected_records[index - 1]
            structure_name = str(record["group_name"])
            structure_color = str(record["color"])
            source_label = str(record["choice_label"])
        else:
            structure_name = f"Structure {index}"
            structure_color = group_color(index)
            source_label = structure_name

        specs.append(
            {
                "structure_id": int(index),
                "structure_name": structure_name,
                "structure_color": structure_color,
                "source_label": source_label,
                "cluster_ids_raw": [str(item) for item in group],
                "cluster_ids_normalized": normalized_clusters,
            }
        )

    return specs


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
        result = binary_closing(result, structure=structure, iterations=int(isoline_cfg["closing_iterations"]))
    if int(isoline_cfg["opening_iterations"]) > 0:
        result = binary_opening(result, structure=structure, iterations=int(isoline_cfg["opening_iterations"]))
    if bool(isoline_cfg["fill_holes"]):
        result = binary_fill_holes(result)

    return _remove_small_components(result, int(isoline_cfg["min_component_pixels"]))


def build_structure_isolines(
    *,
    cells: pd.DataFrame,
    structure_specs: list[dict[str, object]],
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
    valid_specs: list[dict[str, object]] = []
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
                    "structure_name": spec["structure_name"],
                    "reason": f"Only {len(subset)} cells, below min_cells={int(isoline_cfg['min_cells'])}",
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
                    "structure_name": spec["structure_name"],
                    "reason": "Density map contained no positive pixels after smoothing.",
                }
            )
            continue

        scale_value = float(np.quantile(positive, float(isoline_cfg["density_scale_quantile"])))
        support_value = float(np.quantile(positive, float(isoline_cfg["support_quantile"])))
        if scale_value <= 0 or support_value <= 0:
            skipped_structures.append(
                {
                    "structure_id": structure_id,
                    "structure_name": spec["structure_name"],
                    "reason": "Could not derive positive scale/support thresholds for this structure.",
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
            "Try selecting more cells per structure or lowering the minimum-cells threshold."
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
            "structure_name": spec["structure_name"],
            "structure_color": spec["structure_color"],
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
                int(spec["structure_id"]): posterior[idx]
                for idx, spec in enumerate(valid_specs)
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
    posterior_structure_ids = np.array([int(spec["structure_id"]) for spec in valid_specs], dtype=np.int32)[assignment_idx]
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


def assign_cells_to_partition(
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
    name_lookup = {int(spec["structure_id"]): str(spec["structure_name"]) for spec in structure_specs}
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
            str(spec["structure_id"]): int((assigned_cells["isoline_structure_id"] == int(spec["structure_id"])).sum())
            for spec in structure_specs
        },
    }
    return assigned_cells, metrics


def attach_partition_contours(
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


def _hex_to_rgb_triplet(hex_color: str) -> list[int]:
    value = str(hex_color).strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {hex_color!r}")
    return [int(value[index : index + 2], 16) for index in (0, 2, 4)]


def _selection_base_name(structure_id: int, structure_name: str) -> str:
    return f"S{int(structure_id)} {str(structure_name).replace('[', '').replace(']', '')}".strip()


def _resolve_xenium_pixel_size_um(
    partition_data: dict[str, Any],
    *,
    pixel_size_um: float | None = None,
) -> float:
    metadata = partition_data.get("metadata", {})
    candidates = (
        pixel_size_um,
        partition_data.get("pixel_size_um"),
        metadata.get("pixel_size_um") if isinstance(metadata, dict) else None,
        XENIUM_PIXEL_SIZE_UM,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        numeric = float(candidate)
        if numeric <= 0:
            raise ValueError(f"Xenium pixel size must be positive, got {candidate!r}")
        return numeric
    raise ValueError("Could not resolve a Xenium pixel size for polygon export.")


def _component_mask_to_polygons_xenium(
    component_mask: np.ndarray,
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    max_vertices: int = 100000,
) -> list[np.ndarray]:
    """Match the stable segmentation_methods export path for Xenium Explorer polygons."""
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


def write_xenium_explorer_annotation_exports(
    *,
    output_dir: Path,
    partition_data: dict[str, np.ndarray],
    structure_specs: list[dict[str, object]],
    pixel_size_um: float | None = None,
) -> dict[str, str]:
    features: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    partition_labels = np.asarray(partition_data["partition_labels"], dtype=np.int32)
    resolved_pixel_size_um = _resolve_xenium_pixel_size_um(
        partition_data,
        pixel_size_um=pixel_size_um,
    )
    # Internal geometry lives in micron space; Xenium Explorer expects pixel-space polygons.
    x_edges = np.asarray(partition_data["x_edges"], dtype=float) / resolved_pixel_size_um
    y_edges = np.asarray(partition_data["y_edges"], dtype=float) / resolved_pixel_size_um

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
                    "id": str(uuid.uuid4()),
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
                    }
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
    geojson_path.write_text(json.dumps(geojson_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / "xenium_explorer_annotations.csv"
    pd.DataFrame(
        csv_rows,
        columns=["Selection", "X", "Y"],
    ).to_csv(csv_path, index=False)

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


def render_multi_structure_preview(
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
        sampled_cells = assigned_cells.sample(n=60000, random_state=42)

    fig, ax = plt.subplots(figsize=(13.8, 13.2))
    fig.patch.set_facecolor("#09111A")
    ax.set_facecolor("#09111A")

    background_mask = sampled_cells["isoline_structure_id"] <= 0
    if background_mask.any():
        ax.scatter(
            sampled_cells.loc[background_mask, x_col],
            sampled_cells.loc[background_mask, y_col],
            s=1,
            alpha=0.12,
            color="#4A5B70",
            rasterized=True,
        )

    for spec in structure_specs:
        structure_id = int(spec["structure_id"])
        color = str(spec["structure_color"])
        mask = sampled_cells["isoline_structure_id"] == structure_id
        if mask.any():
            ax.scatter(
                sampled_cells.loc[mask, x_col],
                sampled_cells.loc[mask, y_col],
                s=2.2,
                alpha=0.58,
                color=color,
                label=f"S{structure_id}: {spec['structure_name']}",
                rasterized=True,
            )

    for spec in structure_specs:
        structure_id = int(spec["structure_id"])
        contour_data = structure_contours.get(structure_id)
        if not contour_data or "partition_mask" not in contour_data:
            continue
        color = str(spec["structure_color"])
        ax.contourf(
            np.asarray(contour_data["grid_x"], dtype=float),
            np.asarray(contour_data["grid_y"], dtype=float),
            np.asarray(contour_data["partition_mask"], dtype=float),
            levels=[0.5, 1.5],
            colors=[color],
            alpha=0.17,
        )
        ax.contour(
            np.asarray(contour_data["grid_x"], dtype=float),
            np.asarray(contour_data["grid_y"], dtype=float),
            np.asarray(contour_data["partition_mask"], dtype=float),
            levels=[0.5],
            colors=[color],
            linewidths=2.0,
            alpha=0.98,
        )

    ax.set_xlabel("X (um)", color="#B7C7D8", fontsize=10)
    ax.set_ylabel("Y (um)", color="#B7C7D8", fontsize=10)
    ax.tick_params(colors="#70869D", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#26384B")
    ax.set_aspect("equal")
    ax.set_title(
        f"Multi-structure spatial contour map | structures={len(structure_specs)}",
        color="#EAF2FA",
        fontsize=14,
    )
    ax.legend(
        loc="upper right",
        fontsize=8,
        facecolor="#0F1B29",
        edgecolor="#29435D",
        labelcolor="white",
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def build_structure_groups(
    cells_parquet: object | None,
    clusters_csv: object | None,
    n_structure_groups: int,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    if HISTOSEG_IMPORT_ERROR is not None:
        raise gr.Error(
            "HistoSeg could not be imported inside the app container. "
            f"Import error: {HISTOSEG_IMPORT_ERROR}"
        )

    removed_previews = cleanup_old_selections(max_keep=2)
    selection_dir = build_selection_dir()
    input_dir = selection_dir / "inputs"
    output_dir = selection_dir / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress(0.1, desc="Staging files for dendrogram")
    cells_path, clusters_path, _unused_tissue_path = resolve_inputs(
        cells_upload=cells_parquet,
        clusters_upload=clusters_csv,
        tissue_upload=None,
        target_dir=input_dir,
    )

    progress(0.35, desc="Aligning cells and clusters")
    merged, id_col_used, x_col, y_col = prepare_merged_clusters(cells_path, clusters_path)
    if merged["cluster"].nunique() < 2:
        raise gr.Error("Need at least two clusters to build a dendrogram.")

    progress(0.6, desc="Computing cophenetic dendrogram")
    distance_matrix = compute_searcher_findee_distance_matrix_from_df(
        merged,
        x_col=x_col,
        y_col=y_col,
        z_col=None,
        celltype_col="cluster",
    )
    row_coph, _col_coph = compute_cophenetic_from_distance_matrix(
        distance_matrix,
        method="average",
        show_corr=False,
    )
    row_coph = normalize_row_cophenetic(row_coph)

    heatmap_image = plot_cophenetic_heatmap(
        row_coph,
        matrix_name="row_coph",
        sample="all clusters",
        return_image=True,
        dpi=300,
    )
    heatmap_path = output_dir / "cophenetic_heatmap_row_coph.png"
    heatmap_image.save(heatmap_path)
    row_coph.to_csv(output_dir / "cophenetic_heatmap_row_coph.csv", index_label="cluster")

    progress(0.85, desc="Cutting dendrogram into structure groups")
    group_state = build_structure_group_state(row_coph, n_groups=int(n_structure_groups), linkage_method="average")
    group_state["selector_output_dir"] = str(output_dir)
    selector_path, group_state = render_structure_selector_image(group_state, [])
    group_df = build_group_table(group_state, [])
    status_lines = [
        f"Built a cophenetic dendrogram for {merged['cluster'].nunique()} clusters.",
        f"Cut the dendrogram into {group_state['n_groups']} candidate structure(s).",
        "Choose one or more structures in the checklist below, or type one structure per line manually.",
        "The final contour analysis starts only after you click the Run button.",
    ]
    if removed_previews:
        status_lines.append(f"Cleaned old dendrogram sessions: {', '.join(removed_previews)}")
    status_lines.append(f"Merged cells available for grouping: {len(merged)}")
    status_lines.append(f"Coordinate columns: {x_col}, {y_col}")
    status_lines.append(f"Cell identifier column: {id_col_used}")

    progress(1.0, desc="Structure groups ready")
    return (
        "\n".join(status_lines),
        str(selector_path),
        str(heatmap_path),
        group_df,
        gr.update(choices=group_state["choices"], value=[]),
        "",
        structure_selection_help_text(),
        group_state,
        [],
    )


def compute_cophenetic_outputs(
    *,
    merged_df: object,
    pattern1_clusters: list[int | str],
    x_col: str,
    y_col: str,
    output_dir: Path,
    linkage_method: str,
    show_corr: bool,
) -> tuple[float, dict[str, object], Path]:
    distance_matrix = compute_searcher_findee_distance_matrix_from_df(
        merged_df,
        x_col=x_col,
        y_col=y_col,
        z_col=None,
        celltype_col="cluster",
    )
    if getattr(distance_matrix, "shape", (0, 0))[0] < 2 or getattr(distance_matrix, "shape", (0, 0))[1] < 2:
        raise ValueError("Need at least two cluster groups to build the cophenetic heatmap.")

    row_coph, _col_coph = compute_cophenetic_from_distance_matrix(
        distance_matrix,
        method=linkage_method,
        show_corr=show_corr,
    )

    labels = [_normalize_cluster_label(label) for label in row_coph.index]
    row_coph = row_coph.copy()
    row_coph.index = labels
    row_coph.columns = labels

    selected_clusters: list[str] = []
    for item in pattern1_clusters:
        normalized = _normalize_cluster_label(item)
        if normalized and normalized in row_coph.index and normalized not in selected_clusters:
            selected_clusters.append(normalized)

    if not selected_clusters:
        raise ValueError("None of the selected clusters are present in the cophenetic matrix.")

    other_clusters = [label for label in row_coph.index if label not in set(selected_clusters)]
    if not other_clusters:
        raise ValueError("Need both selected and non-selected clusters to compute the cophenetic score.")

    blue_band_matrix = row_coph.loc[selected_clusters, other_clusters]
    band = blue_band_matrix.to_numpy().ravel()
    band = band[~np.isnan(band)]
    if band.size == 0:
        raise ValueError("The cophenetic comparison block has no finite values.")

    stats: dict[str, object] = {
        "n_pairs": int(band.size),
        "min": float(np.min(band)),
        "p05": float(np.quantile(band, 0.05)),
        "median": float(np.median(band)),
        "mean": float(np.mean(band)),
        "p95": float(np.quantile(band, 0.95)),
        "max": float(np.max(band)),
    }

    heatmap_image = plot_cophenetic_heatmap(
        row_coph,
        matrix_name="row_coph",
        sample="selected clusters",
        return_image=True,
        dpi=300,
    )
    heatmap_path = output_dir / "cophenetic_heatmap_row_coph.png"
    heatmap_image.save(heatmap_path)

    return float(stats["mean"]), stats, heatmap_path


def save_cophenetic_heatmap_only(
    *,
    merged_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_dir: Path,
    sample_label: str,
) -> tuple[Path, dict[str, object]]:
    distance_matrix = compute_searcher_findee_distance_matrix_from_df(
        merged_df,
        x_col=x_col,
        y_col=y_col,
        z_col=None,
        celltype_col="cluster",
    )
    row_coph, _col_coph = compute_cophenetic_from_distance_matrix(
        distance_matrix,
        method="average",
        show_corr=False,
    )
    row_coph = normalize_row_cophenetic(row_coph)
    heatmap_image = plot_cophenetic_heatmap(
        row_coph,
        matrix_name="row_coph",
        sample=sample_label,
        return_image=True,
        dpi=300,
    )
    heatmap_path = output_dir / "cophenetic_heatmap_row_coph.png"
    heatmap_image.save(heatmap_path)
    row_coph.to_csv(output_dir / "cophenetic_heatmap_row_coph.csv", index_label="cluster")
    stats = {
        "n_clusters": int(row_coph.shape[0]),
        "matrix_min": float(np.min(row_coph.to_numpy())),
        "matrix_max": float(np.max(row_coph.to_numpy())),
    }
    return heatmap_path, stats


def format_summary(result: object, *, used_tissue_boundary: bool, work_dir: Path) -> dict[str, object]:
    payload = {
        "work_dir": str(work_dir),
        "out_dir": str(result.out_dir),
        "id_col_used": result.id_col_used,
        "x_col": result.x_col,
        "y_col": result.y_col,
        "n_target_cells": result.n_target_cells,
        "n_bg0_points": result.n_bg0_points,
        "n_contours": len(result.contours),
        "label_scheme": describe_label_scheme(result.label_scheme),
        "label_scheme_internal": to_internal_label_scheme(result.label_scheme),
        "used_tissue_boundary": used_tissue_boundary,
    }
    if result.segmentation_confidence_score is not None:
        payload["segmentation_confidence_score"] = result.segmentation_confidence_score
    if result.segmentation_confidence_stats is not None:
        payload["segmentation_confidence_stats"] = result.segmentation_confidence_stats
    return payload


def emit_status(
    *,
    phase: str,
    run_dir: Path,
    lines: list[str],
    summary: dict[str, object],
    preview_path: str | None = None,
    cophenetic_heatmap_path: str | None = None,
    archive_path: str | None = None,
    output_files: list[str] | None = None,
    qc_overview_path: str | None = None,
    qc_structures: pd.DataFrame | None = None,
    qc_clusters: pd.DataFrame | None = None,
) -> tuple[
    str,
    str | None,
    str | None,
    dict[str, object],
    str | None,
    list[str],
    str | None,
    pd.DataFrame,
    pd.DataFrame,
]:
    status_lines = [f"Phase: {phase}", f"Run directory: {run_dir}"]
    status_lines.extend(lines)
    return (
        "\n".join(status_lines),
        preview_path,
        cophenetic_heatmap_path,
        summary,
        archive_path,
        output_files or [],
        qc_overview_path,
        qc_structures if qc_structures is not None else pd.DataFrame(),
        qc_clusters if qc_clusters is not None else pd.DataFrame(),
    )


def run_analysis(
    cells_parquet: object | None,
    clusters_csv: object | None,
    tissue_boundary_csv: object | None,
    pattern1_clusters: str,
    selected_groups: list[str] | None,
    group_state: dict[str, object] | None,
    grid_n: int,
    knn_k: int,
    smooth_sigma: float,
    min_cells_inside: int,
    compute_confidence_score: bool,
    bbox_expand_um: float,
    syn_bg_density: float,
    use_synth_bg: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    if HISTOSEG_IMPORT_ERROR is not None:
        raise gr.Error(
            "HistoSeg could not be imported inside the app container. "
            f"Import error: {HISTOSEG_IMPORT_ERROR}"
        )
    removed_runs = cleanup_old_runs(max_keep=2)
    start_time = time.perf_counter()
    run_dir = build_run_dir()
    upload_dir = run_dir / "inputs"
    output_dir = run_dir / "outputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"work_dir": str(run_dir)}

    try:
        progress(0.03, desc="Staging uploaded files")
        log_event("Staging uploaded files")
        yield emit_status(
            phase="staging-inputs",
            run_dir=run_dir,
            lines=["Copying uploaded files into the app workspace."] + (
                [f"Cleaned old run directories: {', '.join(removed_runs)}"] if removed_runs else []
            ),
            summary=summary,
        )

        cells_path, clusters_path, tissue_path = resolve_inputs(
            cells_upload=cells_parquet,
            clusters_upload=clusters_csv,
            tissue_upload=tissue_boundary_csv,
            target_dir=upload_dir,
        )

        structure_specs = build_selected_structure_specs(
            pattern1_clusters,
            selected_groups,
            group_state,
        )
        estimated_cells_rows = safe_count_parquet_rows(cells_path)
        estimated_cluster_rows = safe_count_csv_rows(clusters_path)
        isoline_cfg = {
            **DEFAULT_STRUCTURE_ISOLINE_CFG,
            "bins_x": int(grid_n),
            "bins_y": int(knn_k),
            "gaussian_sigma": float(smooth_sigma),
            "min_cells": int(min_cells_inside),
            "min_dominance": float(bbox_expand_um),
            "support_quantile": float(syn_bg_density),
            "fill_holes": bool(use_synth_bg),
        }
        summary.update(
            {
                "estimated_cells_rows": estimated_cells_rows,
                "estimated_cluster_rows": estimated_cluster_rows,
                "used_tissue_boundary": tissue_path is not None,
                "selected_structure_count": len(structure_specs),
                "selected_structures": [
                    {
                        "structure_id": int(spec["structure_id"]),
                        "structure_name": str(spec["structure_name"]),
                        "cluster_ids": list(spec["cluster_ids_raw"]),
                    }
                    for spec in structure_specs
                ],
                "isoline_parameters": {
                    "bins_x": int(isoline_cfg["bins_x"]),
                    "bins_y": int(isoline_cfg["bins_y"]),
                    "gaussian_sigma": float(isoline_cfg["gaussian_sigma"]),
                    "min_cells": int(isoline_cfg["min_cells"]),
                    "support_quantile": float(isoline_cfg["support_quantile"]),
                    "tissue_quantile": float(isoline_cfg["tissue_quantile"]),
                    "min_dominance": float(isoline_cfg["min_dominance"]),
                    "fill_holes": bool(isoline_cfg["fill_holes"]),
                    "min_component_pixels": int(isoline_cfg["min_component_pixels"]),
                },
            }
        )

        preflight_lines = [
            f"Estimated cells.parquet rows: {estimated_cells_rows if estimated_cells_rows is not None else 'unknown'}",
            f"Estimated clusters.csv rows: {estimated_cluster_rows if estimated_cluster_rows is not None else 'unknown'}",
            f"Structures requested for contouring: {len(structure_specs)}",
            f"Partition grid: {int(isoline_cfg['bins_x'])} x {int(isoline_cfg['bins_y'])}",
            f"Gaussian sigma: {float(isoline_cfg['gaussian_sigma']):.2f}",
            f"Minimum cells per structure: {int(isoline_cfg['min_cells'])}",
        ]
        for spec in structure_specs:
            preflight_lines.append(
                f"{spec['structure_name']} -> {', '.join(str(item) for item in spec['cluster_ids_raw'])}"
            )
        if tissue_path is not None:
            preflight_lines.append(
                "tissue_boundary.csv was uploaded, but the current multi-structure isoline mode does not use it."
            )

        progress(0.12, desc="Inputs ready")
        log_event(
            f"Inputs ready | cells_rows={estimated_cells_rows} | cluster_rows={estimated_cluster_rows} | "
            f"structures={len(structure_specs)} | bins=({int(isoline_cfg['bins_x'])},{int(isoline_cfg['bins_y'])})"
        )
        yield emit_status(
            phase="preflight",
            run_dir=run_dir,
            lines=preflight_lines,
            summary=summary,
        )

        progress(0.2, desc="Aligning clusters and cell coordinates")
        log_event("Aligning clusters with cells.parquet")
        yield emit_status(
            phase="aligning-cells",
            run_dir=run_dir,
            lines=["Matching GraphClust barcodes with cell coordinates and preparing selected structures."],
            summary=summary,
        )

        merged, id_col_used, x_col, y_col = align_clusters_with_cells(
            clusters_path,
            cells_path,
            barcode_col="Barcode",
            cluster_col="Cluster",
        )
        merged = merged.copy()
        merged["cluster"] = merged["cluster"].map(_normalize_cluster_label)
        merged = merged.loc[merged["cluster"] != ""].copy()
        merged["_selected_structure_id"] = 0
        cell_counts_by_structure: dict[int, int] = {}
        for spec in structure_specs:
            structure_id = int(spec["structure_id"])
            structure_mask = merged["cluster"].isin(set(spec["cluster_ids_normalized"]))
            merged.loc[structure_mask, "_selected_structure_id"] = structure_id
            cell_counts_by_structure[structure_id] = int(structure_mask.sum())

        missing_structures = [
            spec["structure_name"]
            for spec in structure_specs
            if cell_counts_by_structure.get(int(spec["structure_id"]), 0) == 0
        ]
        if missing_structures:
            raise ValueError(
                "Some selected structures could not be matched to any cells after cluster normalization: "
                + ", ".join(str(name) for name in missing_structures)
            )

        selected_cell_count = int((merged["_selected_structure_id"] > 0).sum())
        summary.update(
            {
                "id_col_used": id_col_used,
                "x_col": x_col,
                "y_col": y_col,
                "merged_rows": int(len(merged)),
                "selected_cell_count": selected_cell_count,
                "selected_structure_cell_counts": {
                    str(spec["structure_id"]): int(cell_counts_by_structure[int(spec["structure_id"])])
                    for spec in structure_specs
                },
            }
        )

        progress(0.42, desc="Building non-overlapping structure partitions")
        log_event(
            f"Building structure partitions | selected_cells={selected_cell_count} | "
            f"structures={len(structure_specs)}"
        )
        yield emit_status(
            phase="building-partitions",
            run_dir=run_dir,
            lines=[
                f"Selected cells entering the partition step: {selected_cell_count}",
                "Each selected structure is modeled separately, then the masks are forced to be non-overlapping.",
            ],
            summary=summary,
        )

        structure_contours, partition_data, isoline_metrics = build_structure_isolines(
            cells=merged,
            structure_specs=structure_specs,
            x_col=x_col,
            y_col=y_col,
            isoline_cfg=isoline_cfg,
        )

        progress(0.62, desc="Assigning cells to structure partitions")
        log_event("Assigning cells to partition masks")
        yield emit_status(
            phase="assigning-cells",
            run_dir=run_dir,
            lines=[
                "Projecting every cell onto the non-overlapping partition grid.",
                "This produces one contour-ready region per selected structure.",
            ],
            summary=summary,
        )

        assigned_cells, cell_assignment_metrics = assign_cells_to_partition(
            cells=merged,
            partition_data=partition_data,
            structure_specs=structure_specs,
            x_col=x_col,
            y_col=y_col,
        )
        structure_contours = attach_partition_contours(structure_contours)
        total_contours = int(
            sum(len(payload.get("contours", [])) for payload in structure_contours.values())
        )

        progress(0.78, desc="Rendering multi-structure contour preview")
        log_event(f"Rendering preview | total_contours={total_contours}")
        preview_path = output_dir / "multi_structure_contour_preview.png"
        render_multi_structure_preview(
            assigned_cells=assigned_cells,
            structure_contours=structure_contours,
            structure_specs=structure_specs,
            x_col=x_col,
            y_col=y_col,
            output_path=preview_path,
        )

        cophenetic_heatmap_path: Path | None = None
        cophenetic_matrix_path: Path | None = None
        cophenetic_stats: dict[str, object] | None = None
        if compute_confidence_score:
            progress(0.86, desc="Saving cophenetic heatmap for the final output set")
            log_event("Saving cophenetic heatmap")
            cophenetic_heatmap_path, cophenetic_stats = save_cophenetic_heatmap_only(
                merged_df=merged,
                x_col=x_col,
                y_col=y_col,
                output_dir=output_dir,
                sample_label="selected structures",
            )
            cophenetic_matrix_path = output_dir / "cophenetic_heatmap_row_coph.csv"

        progress(0.93, desc="Writing structure contour outputs")
        log_event("Writing output files")
        yield emit_status(
            phase="saving-outputs",
            run_dir=run_dir,
            lines=[
                f"Valid structures with partition masks: {len(structure_contours)}",
                f"Total contour paths written: {total_contours}",
                "Saving the preview, partition assignments, metrics, and per-structure contour files.",
            ],
            summary=summary,
            preview_path=str(preview_path),
            cophenetic_heatmap_path=str(cophenetic_heatmap_path) if cophenetic_heatmap_path is not None else None,
        )

        metrics_payload = {
            "selected_structures": summary["selected_structures"],
            "isoline_parameters": summary["isoline_parameters"],
            "isoline_metrics": isoline_metrics,
            "cell_assignment_metrics": cell_assignment_metrics,
            "cophenetic_stats": cophenetic_stats,
        }
        params_path = output_dir / "structure_contour_metrics.json"
        params_path.write_text(
            json.dumps(metrics_payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        partitioned_cells_path: Path | None = None
        try:
            partitioned_cells_path = output_dir / "cells_with_structure_partition.parquet"
            assigned_cells.to_parquet(partitioned_cells_path, index=False)
        except Exception:
            partitioned_cells_path = output_dir / "cells_with_structure_partition.csv"
            assigned_cells.to_csv(partitioned_cells_path, index=False)

        structure_count_rows = []
        output_files: list[str] = [str(preview_path), str(params_path), str(partitioned_cells_path)]
        for spec in structure_specs:
            structure_id = int(spec["structure_id"])
            payload = structure_contours.get(structure_id)
            structure_count_rows.append(
                {
                    "structure_id": structure_id,
                    "structure_name": spec["structure_name"],
                    "selected_cluster_ids": ", ".join(str(item) for item in spec["cluster_ids_raw"]),
                    "input_cell_count": int(cell_counts_by_structure.get(structure_id, 0)),
                    "assigned_cell_count": int((assigned_cells["isoline_structure_id"] == structure_id).sum()),
                    "contour_count": int(len(payload.get("contours", [])) if payload else 0),
                }
            )
            if not payload:
                continue
            for contour_index, contour_vertices in enumerate(payload.get("contours", [])):
                contour_path = output_dir / f"structure_{structure_id}_contour_{contour_index}.npy"
                np.save(contour_path, np.asarray(contour_vertices, dtype=float))
                output_files.append(str(contour_path))

        structure_counts_path = output_dir / "structure_contour_cell_counts.csv"
        pd.DataFrame(structure_count_rows).to_csv(structure_counts_path, index=False)
        output_files.append(str(structure_counts_path))

        xenium_annotation_exports = write_xenium_explorer_annotation_exports(
            output_dir=output_dir,
            partition_data=partition_data,
            structure_specs=structure_specs,
        )
        output_files.extend(
            [
                str(xenium_annotation_exports["geojson"]),
                str(xenium_annotation_exports["csv"]),
                str(xenium_annotation_exports["summary"]),
            ]
        )

        if cophenetic_heatmap_path is not None:
            output_files.append(str(cophenetic_heatmap_path))
        if cophenetic_matrix_path is not None and cophenetic_matrix_path.exists():
            output_files.append(str(cophenetic_matrix_path))

        progress(0.96, desc="Running CSR structure quality control")
        log_event("Running CSR structure QC and split recommendation")
        yield emit_status(
            phase="csr-structure-qc",
            run_dir=run_dir,
            lines=[
                "HistoSeg contours are complete.",
                "Running fixed-window CSR and random-labelling tests for every structure and eligible cluster.",
            ],
            summary=summary,
            preview_path=str(preview_path),
            cophenetic_heatmap_path=(
                str(cophenetic_heatmap_path) if cophenetic_heatmap_path is not None else None
            ),
            output_files=output_files,
        )

        qc_dir = output_dir / "csr_structure_qc"
        qc_params = dict(STRUCTURE_QC_PARAMS)
        qc_workers = max(1, min(2, os.cpu_count() or 1))
        qc_structures, qc_clusters, qc_suggestions = run_structure_qc(
            output_dir,
            qc_dir,
            qc_params,
            workers=qc_workers,
            structuremap=cophenetic_matrix_path,
        )
        qc_overview_path = qc_dir / "qc_overview.png"
        qc_report_path = qc_dir / "qc_report.md"
        output_files.extend(str(path) for path in sorted(qc_dir.iterdir()) if path.is_file())

        verdict_counts = {
            str(key): int(value)
            for key, value in qc_structures["verdict"].value_counts().items()
        }
        split_counts = {
            str(key): int(value)
            for key, value in qc_structures["split"].value_counts().items()
        }
        hotspot_count = int((qc_clusters["status"] == "HOTSPOT").sum())
        summary["csr_structure_qc"] = {
            "report_path": str(qc_report_path),
            "overview_path": str(qc_overview_path),
            "workers": qc_workers,
            "params": qc_params,
            "verdict_counts": verdict_counts,
            "split_counts": split_counts,
            "hotspot_count": hotspot_count,
            "suggestion_count": len(qc_suggestions),
        }

        archive_path, archive_note = zip_outputs(output_dir, archive_dir=run_dir)

        summary.update(
            {
                "effective_runtime_seconds": round(time.perf_counter() - start_time, 2),
                "cophenetic_heatmap_generated": cophenetic_heatmap_path is not None,
                "cophenetic_heatmap_path": str(cophenetic_heatmap_path) if cophenetic_heatmap_path is not None else None,
                "cophenetic_matrix_path": str(cophenetic_matrix_path) if cophenetic_matrix_path is not None else None,
                "archive_path": str(archive_path) if archive_path is not None else None,
                "partition_metrics_path": str(params_path),
                "partitioned_cells_path": str(partitioned_cells_path),
                "structure_count_table_path": str(structure_counts_path),
                "xenium_explorer_geojson_path": str(xenium_annotation_exports["geojson"]),
                "xenium_explorer_csv_path": str(xenium_annotation_exports["csv"]),
                "xenium_explorer_summary_path": str(xenium_annotation_exports["summary"]),
                "total_contours": total_contours,
                "valid_structure_ids": sorted(int(key) for key in structure_contours.keys()),
            }
        )

        log_event(
            f"Run finished successfully | structures={len(structure_contours)} | "
            f"total_contours={total_contours} | elapsed_s={summary['effective_runtime_seconds']}"
        )
        progress(1.0, desc="Finished")
        yield emit_status(
            phase="finished",
            run_dir=run_dir,
            lines=[
                f"{APP_NAME} finished successfully.",
                f"Structure partitions generated: {len(structure_contours)}",
                f"Contour paths generated: {total_contours}",
                f"CSR QC verdicts: {verdict_counts}",
                f"CSR split recommendations: {split_counts}",
                f"Structure-level HOTSPOT clusters: {hotspot_count}",
                "Xenium Explorer annotations were exported as GeoJSON and CSV.",
                f"Elapsed time: {summary['effective_runtime_seconds']} seconds",
            ]
            + (["A complete ZIP archive of all outputs is available below."] if archive_path is not None else [])
            + (
                [
                    f"{spec['structure_name']} -> assigned cells: "
                    f"{int((assigned_cells['isoline_structure_id'] == int(spec['structure_id'])).sum())}"
                    for spec in structure_specs
                ]
            )
            + ([archive_note] if archive_note else []),
            summary=summary,
            preview_path=str(preview_path),
            cophenetic_heatmap_path=str(cophenetic_heatmap_path) if cophenetic_heatmap_path is not None else None,
            archive_path=str(archive_path) if archive_path is not None else None,
            output_files=output_files,
            qc_overview_path=str(qc_overview_path),
            qc_structures=qc_structures,
            qc_clusters=qc_clusters,
        )
    except Exception as exc:
        log_event(f"Run failed: {exc}")
        print(traceback.format_exc(), flush=True)
        raise gr.Error(str(exc))


AUTOSPLIT_RULE_CHOICES = {
    "Keep the parent, extract only the top-ΔDI child branch": "extract_top_branch",
    "Split the parent as well": "split_parent",
    "Keep the parent, ignore branch points below it": "stop",
}
AUTOSPLIT_MODE_CHOICES = {
    "ΔDI threshold": "threshold",
    "Number of structures (split the top-ranked ΔDI branch points)": "n_structures",
}
AUTOSPLIT_NODE_COLUMNS = ["Branch point", "ΔDI rank", "Height", "Children", "Σ DI before", "Σ DI after", "ΔDI",
                          "Decision", "Reason"]
AUTOSPLIT_CLUSTER_COLUMNS = ["Cluster", "Final structure", "Cells", "DI before (whole tissue)", "DI after (final)",
                             "ΔDI", "Fraction in own contour"]
AUTOSPLIT_STRUCTURE_COLUMNS = ["Structure", "Cluster IDs", "Source", "Kind", "Cells", "Σ DI before", "Σ DI after",
                               "Max DI after"]


def _autosplit_workers() -> int:
    env = os.environ.get("HISTOSEG_QC_WORKERS", "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(2, os.cpu_count() or 1))


def _row_coph_from_group_state(group_state: dict[str, object] | None, clusters: set[str]) -> pd.DataFrame | None:
    if not group_state:
        return None
    labels = [_normalize_cluster_label(item) for item in group_state.get("row_coph_labels", [])]
    values = group_state.get("row_coph_values")
    if not labels or values is None or set(labels) != clusters:
        return None
    return pd.DataFrame(np.asarray(values, dtype=float), index=labels, columns=labels)


def _round_or_blank(value: object, digits: int = 2) -> object:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if np.isnan(number) else round(number, digits)


def _autosplit_partition_setup(cells_path: Path, clusters_path: Path, isoline_cfg: dict[str, object]):
    merged, _id_col, x_col, y_col = prepare_merged_clusters(cells_path, clusters_path)
    base_cells = merged[[x_col, y_col, "cluster"]].copy()
    cluster_values = base_cells["cluster"].to_numpy()

    def partition_fn(groups: list[list[str]]):
        specs, selected = [], np.zeros(len(base_cells), dtype=int)
        for gid, group in enumerate(groups, start=1):
            specs.append(
                {
                    "structure_id": gid,
                    "structure_name": f"Structure {gid}",
                    "structure_color": group_color(gid),
                    "cluster_ids_raw": list(group),
                    "cluster_ids_normalized": list(group),
                }
            )
            selected[np.isin(cluster_values, list(group))] = gid
        frame = base_cells.copy()
        frame["_selected_structure_id"] = selected
        _contours, partition_data, _metrics = build_structure_isolines(
            cells=frame, structure_specs=specs, x_col=x_col, y_col=y_col, isoline_cfg=isoline_cfg
        )
        assigned, _assign_metrics = assign_cells_to_partition(
            cells=frame, partition_data=partition_data, structure_specs=specs, x_col=x_col, y_col=y_col
        )
        return (
            partition_data["partition_labels"],
            partition_data["x_edges"],
            partition_data["y_edges"],
            assigned["isoline_structure_id"].to_numpy(),
        )

    return merged, base_cells, x_col, y_col, partition_fn


def _autosplit_split_kwargs(mode_label: str, n_structures: float, min_ddi: float, rule_label: str) -> dict[str, object]:
    return {
        "mode": AUTOSPLIT_MODE_CHOICES.get(str(mode_label), "threshold"),
        "n_structures": int(n_structures),
        "min_ddi": float(min_ddi),
        "descendant_rule": AUTOSPLIT_RULE_CHOICES.get(str(rule_label), "extract_top_branch"),
    }


def _autosplit_outputs(result, run_dir: Path, header_lines: list[str], context: dict[str, object]):
    archive_path = Path(shutil.make_archive(str(run_dir / "autosplit_outputs"), "zip", root_dir=result.out_dir))
    nodes_table = pd.DataFrame(
        [
            [row.node, int(row.dDI_rank), round(row.height, 3), row.children, _round_or_blank(row.sum_DI_before),
             _round_or_blank(row.sum_DI_after), _round_or_blank(row.dDI), row.decision, row.reason]
            for row in result.nodes.sort_values("dDI_rank").itertuples()
        ],
        columns=AUTOSPLIT_NODE_COLUMNS,
    )
    clusters_table = pd.DataFrame(
        [
            [f"C{row.cluster}", row.final_structure, int(row.n_cells), _round_or_blank(row.DI_before_tissue),
             _round_or_blank(row.DI_after_final), _round_or_blank(row.dDI), f"{row.frac_in_own_contour:.0%}"]
            for row in result.clusters.sort_values("DI_before_tissue", ascending=False).itertuples()
        ],
        columns=AUTOSPLIT_CLUSTER_COLUMNS,
    )
    structures_table = pd.DataFrame(
        [
            [row.structure_name, row.cluster_ids, row.source, row.kind, int(row.n_cells),
             _round_or_blank(row.sum_DI_before_tissue), _round_or_blank(row.sum_DI_after_final),
             _round_or_blank(row.max_DI_after_final)]
            for row in result.structures.itertuples()
        ],
        columns=AUTOSPLIT_STRUCTURE_COLUMNS,
    )
    status_lines = header_lines + [
        f"Structures: {len(result.structures)}; Σ DI {result.sum_di_baseline:.2f} (whole tissue) -> "
        f"{result.sum_di_final:.2f} (final structures).",
        "Change the split rule / threshold / number of structures and click 'Re-split' to reuse this ΔDI exploration.",
        "Click 'Use these structures in step 2' to copy them into the cluster-ID box and draw their contours.",
    ]
    return (
        "\n".join(status_lines),
        str(result.dendrogram_png),
        str(result.partition_png),
        str(result.before_after_png),
        nodes_table,
        clusters_table,
        structures_table,
        result.structure_lines,
        result.report_md.read_text(encoding="utf-8"),
        str(archive_path),
        [str(path) for path in result.files],
        context,
    )


def run_auto_structure_split(
    cells_parquet: object | None,
    clusters_csv: object | None,
    group_state: dict[str, object] | None,
    mode_label: str,
    n_structures: float,
    min_ddi: float,
    descendant_rule_label: str,
    n_sim: int,
    grid_n: int,
    knn_k: int,
    smooth_sigma: float,
    min_cells_inside: int,
    bbox_expand_um: float,
    syn_bg_density: float,
    use_synth_bg: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Step 3: explore every StructureMap branch point with HistoSeg + CSR DI, then split the tree."""
    if HISTOSEG_IMPORT_ERROR is not None:
        raise gr.Error(f"HistoSeg could not be imported inside the app container. Import error: {HISTOSEG_IMPORT_ERROR}")
    try:
        started = time.perf_counter()
        removed_runs = cleanup_old_runs(max_keep=2)
        run_dir = build_run_dir()
        upload_dir = run_dir / "inputs"
        out_dir = run_dir / "autosplit"
        upload_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        progress(0.01, desc="Staging uploaded files")
        cells_path, clusters_path, _unused = resolve_inputs(
            cells_upload=cells_parquet, clusters_upload=clusters_csv, tissue_upload=None, target_dir=upload_dir
        )
        isoline_cfg = {
            **DEFAULT_STRUCTURE_ISOLINE_CFG,
            "bins_x": int(grid_n),
            "bins_y": int(knn_k),
            "gaussian_sigma": float(smooth_sigma),
            "min_cells": int(min_cells_inside),
            "min_dominance": float(bbox_expand_um),
            "support_quantile": float(syn_bg_density),
            "fill_holes": bool(use_synth_bg),
        }
        merged, base_cells, x_col, y_col, partition_fn = _autosplit_partition_setup(cells_path, clusters_path, isoline_cfg)
        clusters_present = set(merged["cluster"].unique())
        if len(clusters_present) < 2:
            raise ValueError("Need at least two clusters to split.")

        row_coph = _row_coph_from_group_state(group_state, clusters_present)
        structuremap_source = "StructureMap from step 1"
        if row_coph is None:
            progress(0.02, desc="Computing the StructureMap (step 1 was not run for these files)")
            distance_matrix = compute_searcher_findee_distance_matrix_from_df(
                merged, x_col=x_col, y_col=y_col, z_col=None, celltype_col="cluster"
            )
            row_coph, _col_coph = compute_cophenetic_from_distance_matrix(distance_matrix, method="average", show_corr=False)
            row_coph = normalize_row_cophenetic(row_coph)
            structuremap_source = "StructureMap computed for this run"
        row_coph.to_csv(out_dir / "cophenetic_heatmap_row_coph.csv", index_label="cluster")

        split_kwargs = _autosplit_split_kwargs(mode_label, n_structures, min_ddi, descendant_rule_label)
        workers = _autosplit_workers()
        log_event(
            f"Auto split | clusters={len(clusters_present)} | split={split_kwargs} | n_sim={int(n_sim)} | workers={workers}"
        )

        def report_progress(frac: float, message: str) -> None:
            progress(0.03 + 0.94 * float(frac), desc=message)
            log_event(f"Auto split | {message}")

        result = run_auto_split(
            base_cells,
            row_coph,
            partition_fn,
            out_dir,
            params={**STRUCTURE_QC_PARAMS, "n_sim": int(n_sim)},
            workers=workers,
            x_col=x_col,
            y_col=y_col,
            progress=report_progress,
            **split_kwargs,
        )
        progress(0.98, desc="Packaging outputs")
        context = {
            "run_dir": str(run_dir),
            "out_dir": str(out_dir),
            "cells_path": str(cells_path),
            "clusters_path": str(clusters_path),
            "isoline_cfg": isoline_cfg,
            "n_clusters": len(clusters_present),
        }
        header = [
            f"Automatic ΔDI split finished in {time.perf_counter() - started:.0f} s.",
            f"Run directory: {run_dir}",
            f"{structuremap_source}; {len(clusters_present)} clusters, {len(result.nodes)} branch points explored "
            f"({int(n_sim)} Monte Carlo simulations per DI test, {workers} worker(s)).",
            f"Split rule: {mode_label}"
            + (f" = {int(n_structures)}" if split_kwargs["mode"] == "n_structures"
               else f" {float(min_ddi):g}; {descendant_rule_label}"),
        ]
        if removed_runs:
            header.append(f"Cleaned old run directories: {', '.join(removed_runs)}")
        progress(1.0, desc="Automatic split finished")
        return _autosplit_outputs(result, run_dir, header, context)
    except Exception as exc:
        log_event(f"Auto split failed: {exc}")
        print(traceback.format_exc(), flush=True)
        raise gr.Error(str(exc))


def resplit_auto_structure_split(
    context: dict[str, object] | None,
    mode_label: str,
    n_structures: float,
    min_ddi: float,
    descendant_rule_label: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Re-cut the tree with new settings, reusing the ΔDI exploration of the last step-3 run."""
    try:
        if not context:
            raise ValueError("Run '3. Split structures automatically by ΔDI' first; re-splitting reuses its ΔDI exploration.")
        out_dir = Path(str(context["out_dir"]))
        exploration_path = out_dir / AUTOSPLIT_EXPLORATION_FILE
        if not exploration_path.exists():
            raise ValueError("The ΔDI exploration of the last run is no longer available (old runs are cleaned up). "
                             "Run step 3 again.")
        started = time.perf_counter()
        progress(0.02, desc="Loading the ΔDI exploration")
        exploration = AutoSplitExploration.load(exploration_path)
        _merged, base_cells, x_col, y_col, partition_fn = _autosplit_partition_setup(
            Path(str(context["cells_path"])), Path(str(context["clusters_path"])), dict(context["isoline_cfg"])
        )
        split_kwargs = _autosplit_split_kwargs(mode_label, n_structures, min_ddi, descendant_rule_label)
        log_event(f"Auto split re-cut | split={split_kwargs}")

        def report_progress(frac: float, message: str) -> None:
            progress(0.05 + 0.9 * float(frac), desc=message)

        result = run_auto_split(
            base_cells,
            exploration.M,
            partition_fn,
            out_dir,
            workers=_autosplit_workers(),
            x_col=x_col,
            y_col=y_col,
            progress=report_progress,
            exploration=exploration,
            **split_kwargs,
        )
        header = [
            f"Re-split finished in {time.perf_counter() - started:.0f} s (ΔDI exploration reused, "
            f"{exploration.params['n_sim']} Monte Carlo simulations per DI test).",
            f"Run directory: {context['run_dir']}",
            f"Split rule: {mode_label}"
            + (f" = {int(n_structures)}" if split_kwargs["mode"] == "n_structures"
               else f" {float(min_ddi):g}; {descendant_rule_label}"),
        ]
        progress(1.0, desc="Re-split finished")
        return _autosplit_outputs(result, Path(str(context["run_dir"])), header, context)
    except Exception as exc:
        log_event(f"Auto split re-cut failed: {exc}")
        print(traceback.format_exc(), flush=True)
        raise gr.Error(str(exc))


def use_autosplit_structures(structure_lines: str):
    if not str(structure_lines or "").strip():
        raise gr.Error("Run the automatic ΔDI split first.")
    return str(structure_lines).strip(), [], (
        "The automatically split structures were copied into the cluster-ID box (one structure per line). "
        "Click '2. Run HistoSeg contours + CSR QC' to draw their contours."
    )


CUSTOM_CSS = """
:root {
  --app-bg: #07111d;
  --app-bg-soft: #0b1523;
  --panel-bg: rgba(12, 23, 38, 0.96);
  --panel-bg-strong: rgba(15, 29, 46, 0.98);
  --panel-border: #1f3850;
  --panel-border-strong: #2a5270;
  --text-main: #f4f8ff;
  --text-muted: #9db1c9;
  --accent: #6ef0d4;
  --accent-warm: #ffbd73;
  --accent-cool: #7ab8ff;
  --shadow-strong: 0 24px 70px rgba(0, 0, 0, 0.38);
}

html,
body {
  min-height: 100%;
  background:
    radial-gradient(circle at top left, rgba(110, 240, 212, 0.10), transparent 28%),
    radial-gradient(circle at top right, rgba(122, 184, 255, 0.12), transparent 32%),
    linear-gradient(180deg, #050d16 0%, #08111d 100%) !important;
}

body,
gradio-app {
  color: var(--text-main) !important;
}

.gradio-container {
  width: min(100%, 1480px) !important;
  max-width: 1480px !important;
  min-height: 100vh !important;
  margin: 0 auto !important;
  color: var(--text-main);
  background:
    radial-gradient(circle at top left, rgba(110, 240, 212, 0.10), transparent 28%),
    radial-gradient(circle at top right, rgba(122, 184, 255, 0.12), transparent 32%),
    linear-gradient(180deg, #050d16 0%, #08111d 100%) !important;
  font-family: "Aptos", "Bahnschrift", "Segoe UI Variable", "Segoe UI", sans-serif;
}

.gradio-container .gr-box,
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .gr-accordion,
.gradio-container .gr-dataframe,
.gradio-container .gradio-file {
  background: var(--panel-bg) !important;
  border: 1px solid var(--panel-border) !important;
  box-shadow: none !important;
}

.gradio-container .label-wrap,
.gradio-container .label-wrap span,
.gradio-container label,
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .gr-markdown,
.gradio-container .gr-markdown p,
.gradio-container .gr-markdown li,
.gradio-container .gr-html,
.gradio-container .gr-html p,
.gradio-container .gr-html li,
.gradio-container .gr-json,
.gradio-container .gr-dataframe,
.gradio-container .gr-textbox,
.gradio-container .gr-form label {
  color: var(--text-main) !important;
}

.gradio-container .gr-markdown strong,
.gradio-container .gr-html strong,
.gradio-container .gr-markdown code,
.gradio-container .gr-html code {
  color: #f9fcff !important;
}

.gradio-container code {
  background: rgba(122, 184, 255, 0.12) !important;
  border: 1px solid rgba(122, 184, 255, 0.14) !important;
  border-radius: 5px !important;
  color: #eaf3ff !important;
  overflow-wrap: anywhere !important;
  padding: 1px 5px !important;
  white-space: normal !important;
  word-break: break-word !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: #091321 !important;
  color: var(--text-main) !important;
  border: 1px solid #284059 !important;
}

.gradio-container input[type="checkbox"] {
  accent-color: var(--accent) !important;
  width: 18px !important;
  height: 18px !important;
}

.gradio-container input[type="checkbox"]:checked {
  box-shadow: 0 0 0 1px rgba(110, 240, 212, 0.45);
}

.gradio-container .label-wrap,
.gradio-container .label-wrap > span,
.gradio-container [data-testid="block-label"],
.gradio-container .block-label {
  background: #0f1b29 !important;
  border: 1px solid #29435d !important;
  border-radius: 8px !important;
  color: #eaf3ff !important;
}

.gradio-container .label-wrap svg,
.gradio-container [data-testid="block-label"] svg,
.gradio-container .block-label svg {
  color: #90a6bf !important;
  fill: none !important;
  stroke: currentColor !important;
}

.gradio-container .file-preview,
.gradio-container .file-preview * ,
.gradio-container .upload-container,
.gradio-container .upload-container * ,
.gradio-container [data-testid="file"],
.gradio-container [data-testid="file"] * {
  color: #b7c7d8 !important;
}

.gradio-container .file-preview,
.gradio-container .upload-container,
.gradio-container [data-testid="file"] {
  background: #101b2b !important;
  border-color: #29435d !important;
}

.gradio-container .image-container,
.gradio-container .empty,
.gradio-container .empty * {
  background: #101b2b !important;
  color: #9db1c9 !important;
}

.gradio-container table,
.gradio-container .table-wrap,
.gradio-container .dataframe,
.gradio-container [role="grid"] {
  background: #0c1726 !important;
  color: var(--text-main) !important;
}

.gradio-container th,
.gradio-container [role="columnheader"] {
  background: #101d2d !important;
  color: var(--text-main) !important;
  border-color: #284059 !important;
}

.gradio-container td,
.gradio-container [role="gridcell"] {
  background: #0c1726 !important;
  color: #e6eef8 !important;
  border-color: #22384d !important;
}

.gradio-container .gr-button {
  border-radius: 14px !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em;
}

.gradio-container .gr-button-primary {
  background: linear-gradient(135deg, #6ef0d4 0%, #3ec7ff 100%) !important;
  color: #04111b !important;
  border: none !important;
  box-shadow: 0 14px 30px rgba(62, 199, 255, 0.25);
}

.gradio-container .gr-button-secondary {
  background: #132335 !important;
  color: var(--text-main) !important;
  border: 1px solid #284761 !important;
}

.hero-shell {
  background:
    linear-gradient(135deg, rgba(16, 27, 42, 0.98) 0%, rgba(12, 20, 31, 0.96) 48%, rgba(14, 29, 44, 0.98) 100%);
  border: 1px solid rgba(110, 240, 212, 0.18);
  border-radius: 28px;
  padding: 30px 32px;
  margin-bottom: 18px;
  box-shadow: var(--shadow-strong);
}

.hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(110, 240, 212, 0.10);
  border: 1px solid rgba(110, 240, 212, 0.18);
  color: var(--accent);
  font-size: 0.88rem;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  margin-bottom: 16px;
}

.hero-shell h1 {
  margin: 0;
  font-size: 2.9rem;
  line-height: 1.04;
  letter-spacing: -0.03em;
  color: #f6fbff;
}

.hero-shell p {
  margin: 14px 0 0 0;
  max-width: 980px;
  color: #d5e3f4;
  font-size: 1.08rem;
  line-height: 1.7;
}

.hero-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 20px;
}

.hero-metrics span {
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: #eaf3ff;
  font-size: 0.92rem;
}

.guide-shell {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 18px;
}

.guide-card {
  background: var(--panel-bg-strong);
  border: 1px solid var(--panel-border);
  border-radius: 22px;
  overflow: hidden;
  padding: 18px;
  min-height: 172px;
}

.guide-card .guide-step {
  color: var(--accent-warm);
  font-size: 0.86rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 10px;
}

.guide-card h3 {
  margin: 0 0 10px 0;
  color: #f6fbff;
  font-size: 1.08rem;
}

.guide-card p {
  margin: 0;
  color: #c8d7ea;
  line-height: 1.62;
  font-size: 0.96rem;
}

.app-note {
  margin-bottom: 18px;
  padding: 16px 18px;
  border-radius: 20px;
  background: rgba(13, 23, 38, 0.96);
  border: 1px solid rgba(255, 189, 115, 0.18);
  color: #eff6ff;
  line-height: 1.7;
}

.app-note strong {
  color: var(--accent-warm);
}

.micro-guide {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(14, 24, 38, 0.96);
  border: 1px solid rgba(122, 184, 255, 0.18);
  color: #d9e7f6;
  line-height: 1.62;
  margin-bottom: 14px;
  overflow: hidden;
}

#left-rail,
#right-rail {
  gap: 14px;
}

#structure-selector,
#cophenetic-preview,
#contour-preview,
#autosplit-dendrogram,
#autosplit-partition {
  border-radius: 20px !important;
  overflow: hidden;
}

#selection-summary textarea,
#dendrogram-status textarea,
#run-status textarea,
#autosplit-status textarea {
  min-height: 112px !important;
}

#pattern1-clusters textarea {
  min-height: 210px !important;
}

@media (max-width: 1120px) {
  .guide-shell {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 700px) {
  .hero-shell {
    padding: 24px 20px;
  }
  .hero-shell h1 {
    font-size: 2.2rem;
  }
  .guide-shell {
    grid-template-columns: 1fr;
  }
}
"""


GLOBAL_HEAD = """
<style>
  html,
  body,
  gradio-app {
    min-height: 100%;
    background:
      radial-gradient(circle at top left, rgba(110, 240, 212, 0.10), transparent 28%),
      radial-gradient(circle at top right, rgba(122, 184, 255, 0.12), transparent 32%),
      linear-gradient(180deg, #050d16 0%, #08111d 100%) !important;
  }
</style>
"""


with gr.Blocks(
    title=APP_NAME,
    css=CUSTOM_CSS,
    head=GLOBAL_HEAD,
    fill_width=True,
    theme=gr.themes.Base(primary_hue="cyan", secondary_hue="blue", neutral_hue="slate"),
) as demo:
    ensure_workdirs()

    gr.HTML(
        """
        <div class="hero-shell">
          <div class="hero-kicker">SciLifeLab Serve app | end-to-end Xenium structure analysis</div>
          <h1>HistoSeg Structure QC</h1>
          <p>
            Start from Xenium <code>cells.parquet</code> and <code>clusters.csv</code>. The app builds the
            StructureMap, lets you select candidate branches, generates non-overlapping HistoSeg contours,
            and automatically runs CSR structure QC and split recommendations on the result.
          </p>
          <div class="hero-metrics">
            <span>Dendrogram-guided structure picking</span>
            <span>Multi-structure contour analysis</span>
            <span>CSR QC + automatic split advice</span>
          </div>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="guide-shell">
          <div class="guide-card">
            <div class="guide-step">Step 1</div>
            <h3>Upload the Xenium-derived tables</h3>
            <p>
              Provide <code>cells.parquet</code> and <code>clusters.csv</code>. In a Xenium export, <code>cells.parquet</code> is usually in the
              <code>outs</code> folder root, and one common <code>clusters.csv</code> location is
              <code>outs\\analysis\\clustering\\gene_expression_graphclust\\clusters.csv</code>. Add <code>tissue_boundary.csv</code> if you want synthetic background support during contour generation.
            </p>
          </div>
          <div class="guide-card">
            <div class="guide-step">Step 2</div>
            <h3>Build the structure dendrogram</h3>
            <p>The app groups cluster IDs by spatial similarity and cuts the dendrogram into candidate structures. Each colored badge marks one branch after the current cut.</p>
          </div>
          <div class="guide-card">
            <div class="guide-step">Step 3</div>
            <h3>Select one or more structures</h3>
            <p>Click the dendrogram badges or use the checklist fallback. The selected structures are merged automatically into the cluster-ID textbox used by the final HistoSeg run.</p>
          </div>
          <div class="guide-card">
            <div class="guide-step">Step 4</div>
            <h3>Generate contours and validate them</h3>
            <p>The final run writes the contours, then compares each structure and only its assigned clusters against matched spatial null models. Foreign and unassigned clusters are excluded. You receive PASS/PARTIAL/FAIL calls, assigned-cluster HOTSPOT calls, and tree-consistent split suggestions.</p>
          </div>
          <div class="guide-card">
            <div class="guide-step">Step 5 (optional)</div>
            <h3>Split structures automatically by ΔDI</h3>
            <p>Every dendrogram branch point is split with HistoSeg in turn, and the drop in the CSR deviation index (ΔDI) of its clusters is measured. Choose a ΔDI threshold: branch points that reach it are split, and the resulting structures can be sent straight to step 2.</p>
          </div>
        </div>
        <div class="app-note">
          <strong>What this app is for.</strong> It starts before HistoSeg contour generation; do not upload a
          precomputed HistoSeg ZIP. The dendrogram defines candidate tissue compartments, HistoSeg builds their
          contours, and CSR QC measures whether each contour has actually explained the spatial clustering.
        </div>
        """
    )

    group_state = gr.State(value={})
    auto_structure_lines_state = gr.State(value=[])
    autosplit_context_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1, elem_id="left-rail"):
            gr.HTML(
                """
                <div class="micro-guide">
                  <strong>Where to find the Xenium input files.</strong> In a standard Xenium <code>outs</code> directory,
                  <code>cells.parquet</code> is typically located directly in <code>outs</code>, and one common
                  <code>clusters.csv</code> path is <code>outs\\analysis\\clustering\\gene_expression_graphclust\\clusters.csv</code>.
                </div>
                """
            )
            cells_parquet = gr.File(
                label="Cell coordinates (cells.parquet)",
                file_types=[".parquet"],
                type="filepath",
            )
            clusters_csv = gr.File(
                label="Cluster assignments (clusters.csv)",
                file_types=[".csv"],
                type="filepath",
            )
            tissue_boundary_csv = gr.File(
                label="Tissue boundary (optional: tissue_boundary.csv)",
                file_types=[".csv"],
                type="filepath",
            )
            n_structure_groups = gr.Slider(
                label="Cut the dendrogram into this many candidate structures",
                minimum=2,
                maximum=12,
                step=1,
                value=4,
            )
            build_groups_button = gr.Button("1. Build dendrogram and candidate structures", variant="secondary")
            gr.HTML(
                """
                <div class="micro-guide">
                  Build the dendrogram first. Then choose structures in the checklist below, or type one structure per line manually.
                  Each selected structure will be contoured separately. Nothing reruns while you are choosing; the app reads your final selection only when you click Run.
                </div>
                """
            )
            structure_group_selector = gr.CheckboxGroup(
                label="Structures to include in the final contour run",
                choices=[],
                value=[],
                info="Choose one or more candidate structures here. Checked structures are appended as new lines in the text box below, without rerunning the analysis.",
            )
            pattern1_clusters = gr.Textbox(
                label="Cluster IDs for each structure (one structure per line)",
                value="",
                lines=8,
                placeholder="10,23,19\n27,14,20",
                info="Checked structures are copied here one line at a time. You can still edit, remove, or add lines manually before the final run.",
                elem_id="pattern1-clusters",
            )
            selection_summary = gr.Textbox(
                label="How final structure selection works",
                value=structure_selection_help_text(),
                lines=4,
                elem_id="selection-summary",
                interactive=False,
            )

            with gr.Accordion("Advanced parameters", open=False):
                grid_n = gr.Slider(label="Partition grid bins along X", minimum=300, maximum=1600, step=50, value=900)
                knn_k = gr.Slider(label="Partition grid bins along Y", minimum=250, maximum=1400, step=50, value=700)
                smooth_sigma = gr.Slider(label="Gaussian smoothing sigma", minimum=0.5, maximum=6.0, step=0.25, value=2.25)
                min_cells_inside = gr.Slider(label="Minimum cells required for one structure", minimum=50, maximum=3000, step=50, value=500)
                bbox_expand_um = gr.Slider(label="Minimum dominance needed to claim a pixel", minimum=0.10, maximum=0.80, step=0.01, value=0.34)
                syn_bg_density = gr.Slider(label="Support threshold quantile", minimum=0.05, maximum=0.40, step=0.01, value=0.18)
                use_synth_bg = gr.Checkbox(label="Fill holes inside each structure partition", value=True)
                compute_confidence_score = gr.Checkbox(
                    label="Run CSR structure QC with the original StructureMap matrix",
                    value=True,
                    interactive=False,
                )

            run_button = gr.Button("2. Run HistoSeg contours + CSR QC", variant="primary")

            gr.HTML(
                """
                <div class="micro-guide">
                  <strong>Step 3 (optional) - automatic structure split by ΔDI.</strong>
                  Uses the uploaded tables, the StructureMap from step 1 and the partition parameters above.
                  Every branch point of the dendrogram is split with HistoSeg from the top down, and its ΔDI is the drop
                  in the CSR deviation index of its clusters (DI in the parent contour minus DI in the child contours).
                  Then either split every branch point whose ΔDI reaches a threshold, or choose how many structures you
                  want and split the top-ranked ΔDI branch points (a selected branch point also splits its ancestors).
                  The exploration runs once (one HistoSeg partition per branch point); 'Re-split' reuses it for other settings.
                </div>
                """
            )
            autosplit_mode = gr.Radio(
                label="How to choose the branch points to split",
                choices=list(AUTOSPLIT_MODE_CHOICES),
                value="ΔDI threshold",
            )
            autosplit_n_structures = gr.Slider(
                label="Number of structures (used when splitting by number of structures)",
                minimum=2, maximum=40, step=1, value=5,
            )
            autosplit_threshold = gr.Slider(
                label="Minimum ΔDI to split a branch point", minimum=0.1, maximum=10.0, step=0.1, value=1.0
            )
            autosplit_rule = gr.Dropdown(
                label="When a branch point is below the threshold but a branch point under it passes",
                choices=list(AUTOSPLIT_RULE_CHOICES),
                value="Keep the parent, extract only the top-ΔDI child branch",
            )
            autosplit_n_sim = gr.Slider(
                label="Monte Carlo simulations per DI test", minimum=19, maximum=99, step=10, value=49
            )
            autosplit_button = gr.Button("3. Split structures automatically by ΔDI", variant="primary")
            autosplit_resplit_button = gr.Button(
                "Re-split with the current settings (reuse the ΔDI exploration)", variant="secondary"
            )
            use_autosplit_button = gr.Button("Use these structures in step 2", variant="secondary")

        with gr.Column(scale=1, elem_id="right-rail"):
            structure_status = gr.Textbox(label="Step 1 status", lines=6, elem_id="dendrogram-status")
            structure_selector_image = gr.Image(
                label="Structure dendrogram reference",
                type="filepath",
                interactive=False,
                sources=[],
                elem_id="structure-selector",
            )
            structure_group_table = gr.Dataframe(
                label="Candidate structures and their cluster IDs",
                headers=["Structure", "Cluster count", "Cluster IDs"],
                datatype=["str", "number", "str"],
                interactive=False,
                wrap=True,
            )
            cophenetic_heatmap_image = gr.Image(
                label="Cophenetic heatmap reference",
                type="filepath",
                elem_id="cophenetic-preview",
            )
            status_text = gr.Textbox(label="Step 2 status", lines=8, elem_id="run-status")
            preview_image = gr.Image(label="Multi-structure contour preview", type="filepath", elem_id="contour-preview")
            summary_json = gr.JSON(label="Run summary")
            output_archive = gr.File(label="Download all outputs as ZIP", file_count="single")
            output_files = gr.File(label="Download outputs", file_count="multiple")
            qc_overview_image = gr.Image(
                label="CSR structure QC overview",
                type="filepath",
                interactive=False,
                sources=[],
            )
            qc_structures_table = gr.Dataframe(
                label="Structure-level QC and split recommendation",
                interactive=False,
                wrap=True,
            )
            qc_clusters_table = gr.Dataframe(
                label="Cluster-level CSR status and role",
                interactive=False,
                wrap=True,
            )
            autosplit_status = gr.Textbox(label="Step 3 status (automatic ΔDI split)", lines=7, elem_id="autosplit-status")
            autosplit_dendrogram = gr.Image(
                label="ΔDI of every branch point", type="filepath", interactive=False, sources=[],
                elem_id="autosplit-dendrogram",
            )
            autosplit_partition = gr.Image(
                label="HistoSeg partition of the split structures", type="filepath", interactive=False, sources=[],
                elem_id="autosplit-partition",
            )
            autosplit_bars = gr.Image(
                label="DI of every cluster: original (whole tissue) vs final (split structure)", type="filepath",
                interactive=False, sources=[], elem_id="autosplit-bars",
            )
            autosplit_structures_table = gr.Dataframe(
                label="Automatically split structures", headers=AUTOSPLIT_STRUCTURE_COLUMNS, interactive=False, wrap=True
            )
            autosplit_nodes_table = gr.Dataframe(
                label="ΔDI of every branch point", headers=AUTOSPLIT_NODE_COLUMNS, interactive=False, wrap=True
            )
            autosplit_clusters_table = gr.Dataframe(
                label="DI of every cluster before (whole tissue) and after (final structure)",
                headers=AUTOSPLIT_CLUSTER_COLUMNS, interactive=False, wrap=True,
            )
            autosplit_lines = gr.Textbox(
                label="Split structures (one per line, HistoSeg cluster-ID format)", lines=6, interactive=False
            )
            with gr.Accordion("Automatic split report (includes per-cluster DI at every branch point)", open=False):
                autosplit_report = gr.Markdown()
            autosplit_archive = gr.File(label="Download automatic split outputs as ZIP", file_count="single")
            autosplit_files = gr.File(label="Automatic split files", file_count="multiple")

    build_groups_button.click(
        fn=build_structure_groups,
        inputs=[cells_parquet, clusters_csv, n_structure_groups],
        outputs=[
            structure_status,
            structure_selector_image,
            cophenetic_heatmap_image,
            structure_group_table,
            structure_group_selector,
            pattern1_clusters,
            selection_summary,
            group_state,
            auto_structure_lines_state,
        ],
    )

    structure_group_selector.change(
        fn=sync_selected_groups_to_text,
        inputs=[structure_group_selector, pattern1_clusters, auto_structure_lines_state, group_state],
        outputs=[pattern1_clusters, selection_summary, auto_structure_lines_state],
        show_progress="hidden",
    )

    run_button.click(
        fn=run_analysis,
        inputs=[
            cells_parquet,
            clusters_csv,
            tissue_boundary_csv,
            pattern1_clusters,
            structure_group_selector,
            group_state,
            grid_n,
            knn_k,
            smooth_sigma,
            min_cells_inside,
            compute_confidence_score,
            bbox_expand_um,
            syn_bg_density,
            use_synth_bg,
        ],
        outputs=[
            status_text,
            preview_image,
            cophenetic_heatmap_image,
            summary_json,
            output_archive,
            output_files,
            qc_overview_image,
            qc_structures_table,
            qc_clusters_table,
        ],
    )

    autosplit_button.click(
        fn=run_auto_structure_split,
        inputs=[
            cells_parquet,
            clusters_csv,
            group_state,
            autosplit_mode,
            autosplit_n_structures,
            autosplit_threshold,
            autosplit_rule,
            autosplit_n_sim,
            grid_n,
            knn_k,
            smooth_sigma,
            min_cells_inside,
            bbox_expand_um,
            syn_bg_density,
            use_synth_bg,
        ],
        outputs=[
            autosplit_status,
            autosplit_dendrogram,
            autosplit_partition,
            autosplit_bars,
            autosplit_nodes_table,
            autosplit_clusters_table,
            autosplit_structures_table,
            autosplit_lines,
            autosplit_report,
            autosplit_archive,
            autosplit_files,
            autosplit_context_state,
        ],
    )

    autosplit_resplit_button.click(
        fn=resplit_auto_structure_split,
        inputs=[autosplit_context_state, autosplit_mode, autosplit_n_structures, autosplit_threshold, autosplit_rule],
        outputs=[
            autosplit_status,
            autosplit_dendrogram,
            autosplit_partition,
            autosplit_bars,
            autosplit_nodes_table,
            autosplit_clusters_table,
            autosplit_structures_table,
            autosplit_lines,
            autosplit_report,
            autosplit_archive,
            autosplit_files,
            autosplit_context_state,
        ],
    )

    use_autosplit_button.click(
        fn=use_autosplit_structures,
        inputs=[autosplit_lines],
        outputs=[pattern1_clusters, structure_group_selector, selection_summary],
    )


def main() -> None:
    """Launch the Gradio app used by the SciLifeLab Serve deployment."""
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        show_api=False,
        allowed_paths=[str(DEFAULT_WORK_DIR)],
    )


if __name__ == "__main__":
    main()
