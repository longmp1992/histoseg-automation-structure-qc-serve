from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from histoseg._cellgps import require_cellgps_attrs

from .pattern1_isoline import _normalize_cluster_label, align_clusters_with_cells

PathLike = Union[str, Path]

XENIUM_CLUSTERS_RELPATH = Path("analysis") / "clustering" / "gene_expression_graphclust" / "clusters.csv"
XENIUM_CELLS_RELPATH = Path("cells.parquet")

DEFAULT_STRUCTURE_COLORS = (
    "#6EF0D4",
    "#78B9FF",
    "#FFB870",
    "#C5A3FF",
    "#F87171",
    "#A3E635",
    "#F9A8D4",
    "#67E8F9",
)

IGNORED_STRUCTURE_CLUSTER_LABELS = {"unassigned", "nan", "none", "null"}


@dataclass(frozen=True)
class AutoStructureDiscoveryConfig:
    """Automatically discover coarse structure groups from GraphClust cells.

    The workflow reads Xenium cell centroids and GraphClust cluster assignments,
    computes a spatial cluster-to-cluster relationship matrix, and cuts the
    resulting agglomerative hierarchy into named ``Structure N`` groups. The
    generated ``structures.json`` can be passed directly to
    :class:`histoseg.contour.MultiStructureContourConfig`.
    """

    out_dir: PathLike
    xenium_output: PathLike | None = None
    clusters_csv: PathLike | None = None
    cells_parquet: PathLike | None = None
    barcode_col: str = "Barcode"
    cluster_col: str = "Cluster"
    cluster_count: int | str = "auto"
    min_structure_count: int = 3
    max_structure_count: int = 8
    max_leaf_clusters_per_structure: int = 5
    min_leaf_clusters_per_structure: int = 1
    extended_leaf_clusters_per_structure: int = 10
    extended_leaf_merge_distance_threshold: float = 0.25
    min_structure_cell_fraction: float = 0.01
    linkage_method: str = "average"
    use_cophenetic: bool = True
    dry_run: bool = False
    overwrite: bool = False


@dataclass
class AutoStructureDiscoveryResult:
    out_dir: Path
    structures_json: Path
    cluster_structure_csv: Path
    distance_matrix_csv: Path
    cophenetic_matrix_csv: Path | None
    summary_json: Path
    selected_structures: list[dict[str, Any]]
    selected_structure_count: int
    dry_run: bool = False


def resolve_xenium_output_folder(path: PathLike) -> Path:
    """Resolve either a Xenium ``outs`` folder or a parent containing one."""

    root = Path(path).expanduser().resolve()
    if _is_xenium_outs(root):
        return root

    if root.exists() and root.is_dir():
        direct_children = [child for child in root.iterdir() if child.is_dir() and _is_xenium_outs(child)]
        outs_children = [child for child in direct_children if child.name.endswith("_outs")]
        if len(outs_children) == 1:
            return outs_children[0].resolve()
        if len(direct_children) == 1:
            return direct_children[0].resolve()

    expected = root / XENIUM_CELLS_RELPATH
    expected_clusters = root / XENIUM_CLUSTERS_RELPATH
    raise FileNotFoundError(
        "Could not resolve a Xenium output folder. Expected files such as "
        f"{expected} and {expected_clusters}, or exactly one direct child folder "
        "containing those files."
    )


def discover_auto_structures(cfg: AutoStructureDiscoveryConfig) -> AutoStructureDiscoveryResult:
    """Discover coarse structure groups and write a reusable structures JSON."""

    out_dir = Path(cfg.out_dir).expanduser().resolve()
    structures_json = out_dir / "structures.json"
    cluster_structure_csv = out_dir / "cluster_structure_assignments.csv"
    distance_matrix_csv = out_dir / "cluster_spatial_distance_matrix.csv"
    cophenetic_matrix_csv = out_dir / "cluster_cophenetic_matrix.csv"
    summary_json = out_dir / "auto_structure_summary.json"

    if (
        not cfg.dry_run
        and not cfg.overwrite
        and structures_json.exists()
        and cluster_structure_csv.exists()
        and distance_matrix_csv.exists()
        and summary_json.exists()
    ):
        selected_structures = json.loads(structures_json.read_text(encoding="utf-8"))
        return AutoStructureDiscoveryResult(
            out_dir=out_dir,
            structures_json=structures_json,
            cluster_structure_csv=cluster_structure_csv,
            distance_matrix_csv=distance_matrix_csv,
            cophenetic_matrix_csv=cophenetic_matrix_csv if cophenetic_matrix_csv.exists() else None,
            summary_json=summary_json,
            selected_structures=selected_structures,
            selected_structure_count=len(selected_structures),
            dry_run=False,
        )

    clusters_csv, cells_parquet, xenium_output = _resolve_inputs(cfg)
    merged, id_col_used, x_col, y_col = align_clusters_with_cells(
        clusters_csv,
        cells_parquet,
        barcode_col=cfg.barcode_col,
        cluster_col=cfg.cluster_col,
    )
    merged = merged.loc[merged["cluster"].map(_is_structure_cluster_label).to_numpy()].copy()
    if merged["cluster"].nunique() < 2:
        raise ValueError("Auto-structure discovery requires at least two matched GraphClust clusters.")

    distance_matrix = _compute_spatial_distance_matrix(merged, x_col=x_col, y_col=y_col, symmetrize=False)
    clustering_matrix = _symmetrize_distance_matrix(distance_matrix)
    cophenetic_matrix: pd.DataFrame | None = None
    if cfg.use_cophenetic:
        cophenetic_matrix = _compute_cophenetic_matrix(distance_matrix, method=cfg.linkage_method)
        if cophenetic_matrix is not None:
            clustering_matrix = cophenetic_matrix

    cluster_counts = merged["cluster"].value_counts().reindex(clustering_matrix.index).fillna(0).astype(int)
    cluster_centroids = (
        merged.groupby("cluster")[[x_col, y_col]]
        .mean()
        .reindex(clustering_matrix.index)
        .rename(columns={x_col: "x_centroid", y_col: "y_centroid"})
    )
    selected = _select_structure_count(
        clustering_matrix,
        cluster_counts=cluster_counts,
        cfg=cfg,
    )
    if selected.get("cluster_count_mode") == "leaf_balanced":
        raw_labels = np.asarray(selected["labels"], dtype=int)
    else:
        raw_labels = _cluster_labels(clustering_matrix, selected["structure_count"], linkage_method=cfg.linkage_method)
    structure_specs, assignment_table = _build_structure_specs(
        labels=raw_labels,
        cluster_ids=list(clustering_matrix.index),
        cluster_counts=cluster_counts,
        cluster_centroids=cluster_centroids,
        total_cells=int(len(merged)),
    )

    summary = {
        "xenium_output": str(xenium_output) if xenium_output is not None else None,
        "cells_parquet": str(cells_parquet),
        "clusters_csv": str(clusters_csv),
        "out_dir": str(out_dir),
        "matched_cell_count": int(len(merged)),
        "cluster_count": int(merged["cluster"].nunique()),
        "selected_structure_count": int(selected["structure_count"]),
        "cluster_count_mode": selected.get("cluster_count_mode", cfg.cluster_count),
        "max_leaf_clusters_per_structure": int(cfg.max_leaf_clusters_per_structure),
        "min_leaf_clusters_per_structure": int(cfg.min_leaf_clusters_per_structure),
        "extended_leaf_clusters_per_structure": int(cfg.extended_leaf_clusters_per_structure),
        "extended_leaf_merge_distance_threshold": float(cfg.extended_leaf_merge_distance_threshold),
        "selection_score": selected,
        "candidate_scores": selected.get("candidate_scores", []),
        "id_col_used": id_col_used,
        "x_col": x_col,
        "y_col": y_col,
        "linkage_method": cfg.linkage_method,
        "use_cophenetic": bool(cfg.use_cophenetic and cophenetic_matrix is not None),
        "min_structure_cell_fraction": float(cfg.min_structure_cell_fraction),
        "dry_run": bool(cfg.dry_run),
    }

    if not cfg.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        structures_json.write_text(json.dumps(structure_specs, indent=2, ensure_ascii=False), encoding="utf-8")
        assignment_table.to_csv(cluster_structure_csv, index=False)
        distance_matrix.to_csv(distance_matrix_csv)
        if cophenetic_matrix is not None:
            cophenetic_matrix.to_csv(cophenetic_matrix_csv)
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return AutoStructureDiscoveryResult(
        out_dir=out_dir,
        structures_json=structures_json,
        cluster_structure_csv=cluster_structure_csv,
        distance_matrix_csv=distance_matrix_csv,
        cophenetic_matrix_csv=cophenetic_matrix_csv if cophenetic_matrix is not None else None,
        summary_json=summary_json,
        selected_structures=structure_specs,
        selected_structure_count=len(structure_specs),
        dry_run=bool(cfg.dry_run),
    )


def _is_xenium_outs(path: Path) -> bool:
    return (path / XENIUM_CELLS_RELPATH).exists() and (path / XENIUM_CLUSTERS_RELPATH).exists()


def _resolve_inputs(cfg: AutoStructureDiscoveryConfig) -> tuple[Path, Path, Path | None]:
    if cfg.xenium_output is not None:
        xenium_output = resolve_xenium_output_folder(cfg.xenium_output)
        return xenium_output / XENIUM_CLUSTERS_RELPATH, xenium_output / XENIUM_CELLS_RELPATH, xenium_output

    if cfg.clusters_csv is None or cfg.cells_parquet is None:
        raise ValueError("Provide either xenium_output or both clusters_csv and cells_parquet.")
    return Path(cfg.clusters_csv).expanduser().resolve(), Path(cfg.cells_parquet).expanduser().resolve(), None


def _compute_spatial_distance_matrix(merged: pd.DataFrame, *, x_col: str, y_col: str, symmetrize: bool = True) -> pd.DataFrame:
    try:
        cellgps = require_cellgps_attrs(("compute_searcher_findee_distance_matrix_from_df",))
        raw = cellgps.compute_searcher_findee_distance_matrix_from_df(
            merged[[x_col, y_col, "cluster"]].copy(),
            x_col=x_col,
            y_col=y_col,
            celltype_col="cluster",
        )
        matrix = pd.DataFrame(raw).copy()
    except Exception:
        centroids = merged.groupby("cluster")[[x_col, y_col]].mean()
        values = squareform(pdist(centroids.to_numpy(dtype=float)))
        matrix = pd.DataFrame(values, index=centroids.index, columns=centroids.index)

    matrix.index = [_normalize_cluster_label(value) for value in matrix.index]
    matrix.columns = [_normalize_cluster_label(value) for value in matrix.columns]
    labels = sorted(set(matrix.index).intersection(matrix.columns), key=_cluster_sort_key)
    matrix = matrix.loc[labels, labels].apply(pd.to_numeric, errors="coerce")
    if symmetrize:
        return _symmetrize_distance_matrix(matrix)
    values = matrix.to_numpy(dtype=float)
    np.fill_diagonal(values, 0.0)
    finite = values[np.isfinite(values)]
    fill = float(np.max(finite)) if finite.size else 1.0
    values[~np.isfinite(values)] = fill
    values = np.maximum(values, 0.0)
    return pd.DataFrame(values, index=labels, columns=labels)


def _symmetrize_distance_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    labels = list(matrix.index)
    values = matrix.to_numpy(dtype=float)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 0.0)
    finite = values[np.isfinite(values)]
    fill = float(np.max(finite)) if finite.size else 1.0
    values[~np.isfinite(values)] = fill
    values = np.maximum(values, 0.0)
    return pd.DataFrame(values, index=labels, columns=labels)


def _compute_cophenetic_matrix(distance_matrix: pd.DataFrame, *, method: str) -> pd.DataFrame | None:
    try:
        cellgps = require_cellgps_attrs(("compute_cophenetic_from_distance_matrix",))
        row_coph, _col_coph = cellgps.compute_cophenetic_from_distance_matrix(
            distance_matrix,
            method=method,
            show_corr=False,
        )
        row_coph = pd.DataFrame(row_coph).reindex(index=distance_matrix.index, columns=distance_matrix.columns)
        return row_coph.apply(pd.to_numeric, errors="coerce")
    except Exception:
        try:
            values = distance_matrix.to_numpy(dtype=float)
            tree = linkage(values, method=method)
            _corr, condensed_cophenetic = cophenet(tree, pdist(values))
            coph = squareform(condensed_cophenetic)
            np.fill_diagonal(coph, 0.0)
            finite = coph[np.isfinite(coph)]
            if finite.size:
                cmin = float(np.min(finite))
                cmax = float(np.max(finite))
                if cmax > cmin:
                    coph = (coph - cmin) / (cmax - cmin)
            return pd.DataFrame(coph, index=distance_matrix.index, columns=distance_matrix.columns)
        except Exception:
            return None


def _select_structure_count(
    distance_matrix: pd.DataFrame,
    *,
    cluster_counts: pd.Series,
    cfg: AutoStructureDiscoveryConfig,
) -> dict[str, Any]:
    n_clusters = len(distance_matrix)
    cluster_count_mode = str(cfg.cluster_count).strip().lower()
    if cluster_count_mode in {"leaf", "leaf-balanced", "leaf_balanced", "bottom-up", "bottom_up"}:
        labels = _leaf_balanced_labels(
            distance_matrix,
            linkage_method=cfg.linkage_method,
            max_leaf_count=cfg.max_leaf_clusters_per_structure,
            min_leaf_count=cfg.min_leaf_clusters_per_structure,
            extended_max_leaf_count=cfg.extended_leaf_clusters_per_structure,
            extended_merge_distance_threshold=cfg.extended_leaf_merge_distance_threshold,
        )
        score = _score_labels(distance_matrix, labels, cluster_counts, cfg.min_structure_cell_fraction)
        structure_count = int(len(set(labels.tolist())))
        return {
            "structure_count": structure_count,
            "cluster_count_mode": "leaf_balanced",
            "labels": labels.tolist(),
            **score,
            "candidate_scores": [
                {
                    "structure_count": structure_count,
                    "cluster_count_mode": "leaf_balanced",
                    "max_leaf_clusters_per_structure": int(cfg.max_leaf_clusters_per_structure),
                    "min_leaf_clusters_per_structure": int(cfg.min_leaf_clusters_per_structure),
                    "extended_leaf_clusters_per_structure": int(cfg.extended_leaf_clusters_per_structure),
                    "extended_leaf_merge_distance_threshold": float(cfg.extended_leaf_merge_distance_threshold),
                    **score,
                }
            ],
        }

    if isinstance(cfg.cluster_count, int) or cluster_count_mode != "auto":
        k = int(cfg.cluster_count)
        if k < 2 or k > n_clusters:
            raise ValueError(f"cluster_count must be between 2 and {n_clusters}, got {k}.")
        labels = _cluster_labels(distance_matrix, k, linkage_method=cfg.linkage_method)
        score = _score_labels(distance_matrix, labels, cluster_counts, cfg.min_structure_cell_fraction)
        return {"structure_count": k, **score, "candidate_scores": [score | {"structure_count": k}]}

    min_k = max(2, int(cfg.min_structure_count))
    max_k = min(int(cfg.max_structure_count), n_clusters)
    if min_k > max_k:
        min_k = max_k

    candidates: list[dict[str, Any]] = []
    for k in range(min_k, max_k + 1):
        labels = _cluster_labels(distance_matrix, k, linkage_method=cfg.linkage_method)
        score = _score_labels(distance_matrix, labels, cluster_counts, cfg.min_structure_cell_fraction)
        candidates.append({"structure_count": k, **score})

    best = max(candidates, key=lambda item: float(item["score"]))
    return {**best, "candidate_scores": candidates}


def _leaf_balanced_labels(
    distance_matrix: pd.DataFrame,
    *,
    linkage_method: str,
    max_leaf_count: int,
    min_leaf_count: int,
    extended_max_leaf_count: int | None = None,
    extended_merge_distance_threshold: float | None = None,
) -> np.ndarray:
    """Group nearby leaf clusters without allowing root-level giant structures.

    This mode follows the StructureMap tree from leaves upward.  A merge is kept
    only when the resulting structure would contain no more than
    ``max_leaf_count`` original graph clusters. If a larger local branch remains
    tightly connected in the normalized StructureMap, merging can continue up
    to ``extended_max_leaf_count``. The final structure groups are therefore
    local tree branches instead of a coarse global tree cut.
    """

    n_clusters = len(distance_matrix)
    if n_clusters == 0:
        return np.asarray([], dtype=int)
    if n_clusters == 1:
        return np.zeros(1, dtype=int)

    max_leaf_count = max(1, int(max_leaf_count))
    min_leaf_count = max(1, int(min_leaf_count))
    if extended_max_leaf_count is None:
        extended_max_leaf_count = max_leaf_count
    extended_max_leaf_count = max(max_leaf_count, int(extended_max_leaf_count))
    if extended_merge_distance_threshold is None:
        extended_merge_distance_threshold = -math.inf
    extended_merge_distance_threshold = float(extended_merge_distance_threshold)
    if n_clusters <= max_leaf_count:
        # Keep very small clusterings split enough to preserve contour contrast.
        if n_clusters <= 2:
            return np.arange(n_clusters, dtype=int)
        return np.zeros(n_clusters, dtype=int)

    values = distance_matrix.to_numpy(dtype=float)
    condensed = squareform(values, checks=False)
    method = "average" if linkage_method == "ward" else linkage_method
    tree = linkage(condensed, method=method)

    next_group_id = n_clusters
    active_groups: dict[int, set[int]] = {idx: {idx} for idx in range(n_clusters)}
    node_to_groups: dict[int, list[int]] = {idx: [idx] for idx in range(n_clusters)}

    for row_idx, (left_raw, right_raw, _dist, _count) in enumerate(tree):
        left = int(left_raw)
        right = int(right_raw)
        node_id = n_clusters + row_idx
        child_group_ids = list(node_to_groups.get(left, [])) + list(node_to_groups.get(right, []))
        child_group_ids = [gid for gid in child_group_ids if gid in active_groups]
        if not child_group_ids:
            node_to_groups[node_id] = []
            continue
        combined: set[int] = set()
        for gid in child_group_ids:
            combined.update(active_groups[gid])
        internal_max = _group_internal_max_distance(values, combined)
        can_merge_base = len(combined) <= max_leaf_count
        can_merge_extended = (
            len(combined) <= extended_max_leaf_count
            and internal_max <= extended_merge_distance_threshold
        )
        if can_merge_base or can_merge_extended:
            for gid in child_group_ids:
                active_groups.pop(gid, None)
            active_groups[next_group_id] = combined
            node_to_groups[node_id] = [next_group_id]
            next_group_id += 1
        else:
            node_to_groups[node_id] = child_group_ids

    groups = list(active_groups.values())
    groups = _merge_small_leaf_groups(
        groups,
        distance_matrix=distance_matrix,
        min_leaf_count=min_leaf_count,
        max_leaf_count=max_leaf_count,
    )
    groups = _sort_leaf_groups(groups, distance_matrix.index)
    labels = np.empty(n_clusters, dtype=int)
    for label, group in enumerate(groups):
        for idx in group:
            labels[int(idx)] = label
    return labels


def _merge_small_leaf_groups(
    groups: list[set[int]],
    *,
    distance_matrix: pd.DataFrame,
    min_leaf_count: int,
    max_leaf_count: int,
) -> list[set[int]]:
    groups = [set(group) for group in groups if group]
    values = distance_matrix.to_numpy(dtype=float)
    changed = True
    while changed:
        changed = False
        small_indices = [idx for idx, group in enumerate(groups) if len(group) < min_leaf_count]
        if not small_indices:
            break
        for idx in small_indices:
            if idx >= len(groups) or len(groups[idx]) >= min_leaf_count:
                continue
            best_j = None
            best_dist = math.inf
            for j, other in enumerate(groups):
                if j == idx or len(groups[idx]) + len(other) > max_leaf_count:
                    continue
                sub = values[np.ix_(sorted(groups[idx]), sorted(other))]
                dist = float(np.nanmean(sub)) if sub.size else math.inf
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j is None:
                continue
            groups[best_j].update(groups[idx])
            groups.pop(idx)
            changed = True
            break
    return groups


def _group_internal_max_distance(values: np.ndarray, group: set[int]) -> float:
    if len(group) <= 1:
        return 0.0
    idx = sorted(group)
    sub = values[np.ix_(idx, idx)]
    mask = ~np.eye(len(idx), dtype=bool)
    off_diag = sub[mask]
    finite = off_diag[np.isfinite(off_diag)]
    if not finite.size:
        return math.inf
    return float(np.max(finite))


def _sort_leaf_groups(groups: list[set[int]], cluster_ids: Sequence[str]) -> list[set[int]]:
    def key(group: set[int]) -> tuple[int, Any]:
        first = min((cluster_ids[idx] for idx in group), key=_cluster_sort_key)
        return _cluster_sort_key(first)

    return sorted(groups, key=key)


def _cluster_labels(distance_matrix: pd.DataFrame, n_clusters: int, *, linkage_method: str) -> np.ndarray:
    values = distance_matrix.to_numpy(dtype=float)
    if linkage_method == "ward":
        linkage_method = "average"
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage=linkage_method,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage=linkage_method,
        )
    return np.asarray(model.fit_predict(values), dtype=int)


def _score_labels(
    distance_matrix: pd.DataFrame,
    labels: np.ndarray,
    cluster_counts: pd.Series,
    min_structure_cell_fraction: float,
) -> dict[str, Any]:
    values = distance_matrix.to_numpy(dtype=float)
    counts = cluster_counts.to_numpy(dtype=float)
    total = float(np.sum(counts)) or 1.0
    fractions = []
    within_values: list[float] = []
    between_values: list[float] = []
    for label in sorted(set(labels)):
        idx = np.where(labels == label)[0]
        fractions.append(float(np.sum(counts[idx]) / total))
        if len(idx) > 1:
            sub = values[np.ix_(idx, idx)]
            within_values.extend(sub[np.triu_indices_from(sub, k=1)].tolist())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] != labels[j]:
                between_values.append(float(values[i, j]))
    min_fraction = float(min(fractions)) if fractions else 0.0
    balance_penalty = 0.0 if min_fraction >= min_structure_cell_fraction else -1.0
    silhouette = 0.0
    try:
        if len(set(labels)) < len(labels):
            silhouette = float(silhouette_score(values, labels, metric="precomputed"))
    except Exception:
        silhouette = 0.0
    within = float(np.mean(within_values)) if within_values else 0.0
    between = float(np.mean(between_values)) if between_values else 0.0
    separation = between / (within + 1e-9) if within > 0 else between
    entropy = -sum(frac * math.log(frac + 1e-12) for frac in fractions)
    entropy_norm = entropy / math.log(len(fractions)) if len(fractions) > 1 else 0.0
    score = silhouette + 0.05 * math.log1p(max(separation, 0.0)) + 0.15 * entropy_norm + balance_penalty
    return {
        "score": float(score),
        "silhouette": float(silhouette),
        "within_mean": within,
        "between_mean": between,
        "separation": float(separation),
        "min_structure_cell_fraction_observed": min_fraction,
        "entropy_balance": float(entropy_norm),
        "passes_min_fraction": bool(min_fraction >= min_structure_cell_fraction),
    }


def _build_structure_specs(
    *,
    labels: np.ndarray,
    cluster_ids: Sequence[str],
    cluster_counts: pd.Series,
    cluster_centroids: pd.DataFrame,
    total_cells: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    group_rows: list[dict[str, Any]] = []
    for raw_label in sorted(set(labels)):
        indices = [idx for idx, value in enumerate(labels) if value == raw_label]
        clusters = [cluster_ids[idx] for idx in indices]
        count = int(cluster_counts.reindex(clusters).fillna(0).sum())
        centroids = cluster_centroids.reindex(clusters)
        weighted_x = float(np.average(centroids["x_centroid"], weights=cluster_counts.reindex(clusters).fillna(1)))
        weighted_y = float(np.average(centroids["y_centroid"], weights=cluster_counts.reindex(clusters).fillna(1)))
        group_rows.append(
            {
                "raw_label": int(raw_label),
                "cluster_ids": sorted(clusters, key=_cluster_sort_key),
                "cell_count": count,
                "fraction_of_case": count / total_cells if total_cells else 0.0,
                "x_centroid": weighted_x,
                "y_centroid": weighted_y,
            }
        )
    group_rows = sorted(group_rows, key=lambda item: (item["x_centroid"], item["y_centroid"], item["raw_label"]))

    specs: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    for structure_id, group in enumerate(group_rows, start=1):
        structure_name = f"Structure {structure_id}"
        color = DEFAULT_STRUCTURE_COLORS[(structure_id - 1) % len(DEFAULT_STRUCTURE_COLORS)]
        specs.append(
            {
                "structure_id": structure_id,
                "structure_name": structure_name,
                "cluster_ids": group["cluster_ids"],
                "structure_color": color,
            }
        )
        for cluster_id in group["cluster_ids"]:
            cluster_count = int(cluster_counts.get(cluster_id, 0))
            assignment_rows.append(
                {
                    "cluster": cluster_id,
                    "structure_id": structure_id,
                    "structure_name": structure_name,
                    "cluster_cell_count": cluster_count,
                    "structure_cell_count": int(group["cell_count"]),
                    "fraction_of_structure": (
                        cluster_count / int(group["cell_count"]) if int(group["cell_count"]) else 0.0
                    ),
                    "fraction_of_case": cluster_count / total_cells if total_cells else 0.0,
                }
            )

    assignment_table = pd.DataFrame(assignment_rows).sort_values(["structure_id", "cluster"], key=_sort_series)
    return specs, assignment_table


def _cluster_sort_key(value: Any) -> tuple[int, Any]:
    normalized = _normalize_cluster_label(value)
    try:
        return (0, int(normalized))
    except Exception:
        return (1, normalized)


def _sort_series(series: pd.Series) -> pd.Series:
    def encode(value: Any) -> str:
        kind, payload = _cluster_sort_key(value)
        if kind == 0:
            return f"0:{int(payload):012d}"
        return f"1:{payload}"

    return series.map(encode)


def _is_structure_cluster_label(value: object) -> bool:
    normalized = _normalize_cluster_label(value)
    return normalized != "" and normalized.lower() not in IGNORED_STRUCTURE_CLUSTER_LABELS
