from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import re

import pandas as pd


_CLUSTER_FC_PATTERN = re.compile(r"^Cluster (\d+) Log2 fold change$")


def infer_differential_expression_csv(cluster_csv: str | Path) -> Path:
    cluster_csv = Path(cluster_csv).resolve()
    clustering_name = cluster_csv.parent.name
    analysis_root = cluster_csv.parents[2]
    candidate = analysis_root / "diffexp" / clustering_name / "differential_expression.csv"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not infer differential_expression.csv from {cluster_csv}. Expected {candidate}"
        )
    return candidate


def infer_projection_csv(dataset_root: str | Path) -> Path:
    dataset_root = Path(dataset_root).resolve()
    candidates = sorted((dataset_root / "analysis" / "umap").glob("*/projection.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate projection.csv under {(dataset_root / 'analysis' / 'umap')}"
        )
    raise FileNotFoundError(
        "Multiple projection.csv candidates found. Provide an explicit projection_csv in the full-auto config."
    )


def _pick_column(columns: list[str], candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find {label}. Expected one of: {candidates}")


def _cluster_ids_from_diffexp(columns: list[str]) -> list[int]:
    cluster_ids: list[int] = []
    for column in columns:
        match = _CLUSTER_FC_PATTERN.match(column)
        if match:
            cluster_ids.append(int(match.group(1)))
    return sorted(set(cluster_ids))


def _marker_rows_for_cluster(
    diffexp: pd.DataFrame,
    *,
    cluster_id: int,
    top_positive_markers: int,
    top_negative_markers: int,
    min_log2fc: float,
    max_adjusted_p_value: float,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    gene_col = _pick_column(diffexp.columns.tolist(), ["Feature Name", "feature_name", "gene"], "gene column")
    fc_col = f"Cluster {cluster_id} Log2 fold change"
    p_col = f"Cluster {cluster_id} Adjusted p value"
    mean_col = f"Cluster {cluster_id} Mean Counts"

    if fc_col not in diffexp.columns or p_col not in diffexp.columns:
        raise ValueError(f"Differential expression table is missing columns for cluster {cluster_id}")

    subset = diffexp[[gene_col, fc_col, p_col, mean_col]].copy()
    subset.columns = ["gene", "log2fc", "adjusted_p_value", "mean_counts"]
    subset["gene"] = subset["gene"].astype(str).str.strip()
    subset["log2fc"] = pd.to_numeric(subset["log2fc"], errors="coerce")
    subset["adjusted_p_value"] = pd.to_numeric(subset["adjusted_p_value"], errors="coerce")
    subset["mean_counts"] = pd.to_numeric(subset["mean_counts"], errors="coerce")
    subset = subset.dropna(subset=["gene", "log2fc"]).copy()

    positive = subset.loc[
        (subset["log2fc"] >= float(min_log2fc))
        & (subset["adjusted_p_value"].fillna(1.0) <= float(max_adjusted_p_value))
    ].sort_values(["log2fc", "mean_counts"], ascending=[False, False])
    if positive.empty:
        positive = subset.sort_values(["log2fc", "mean_counts"], ascending=[False, False]).head(top_positive_markers)

    negative = subset.loc[
        (subset["log2fc"] <= -float(min_log2fc))
        & (subset["adjusted_p_value"].fillna(1.0) <= float(max_adjusted_p_value))
    ].sort_values(["log2fc", "mean_counts"], ascending=[True, False])
    if negative.empty:
        negative = subset.sort_values(["log2fc", "mean_counts"], ascending=[True, False]).head(top_negative_markers)

    return (
        positive.head(top_positive_markers).to_dict(orient="records"),
        negative.head(top_negative_markers).to_dict(orient="records"),
    )


def _cluster_umap_summary(
    clusters: pd.DataFrame,
    projection: pd.DataFrame,
) -> tuple[dict[int, dict[str, float]], dict[int, list[dict[str, float | int]]], dict[int, int]]:
    cluster_col = _pick_column(clusters.columns.tolist(), ["Cluster", "cluster"], "cluster column")
    cluster_barcode_col = _pick_column(clusters.columns.tolist(), ["Barcode", "barcode"], "barcode column")
    projection_barcode_col = _pick_column(projection.columns.tolist(), ["Barcode", "barcode"], "projection barcode column")
    umap1_col = _pick_column(projection.columns.tolist(), ["UMAP-1", "umap_1", "UMAP1"], "UMAP-1 column")
    umap2_col = _pick_column(projection.columns.tolist(), ["UMAP-2", "umap_2", "UMAP2"], "UMAP-2 column")

    clusters_df = clusters[[cluster_barcode_col, cluster_col]].copy()
    clusters_df.columns = ["barcode", "cluster_id"]
    clusters_df["cluster_id"] = pd.to_numeric(clusters_df["cluster_id"], errors="coerce").astype("Int64")
    clusters_df = clusters_df.dropna(subset=["cluster_id"]).copy()
    clusters_df["cluster_id"] = clusters_df["cluster_id"].astype(int)
    clusters_df["barcode"] = clusters_df["barcode"].astype(str)

    projection_df = projection[[projection_barcode_col, umap1_col, umap2_col]].copy()
    projection_df.columns = ["barcode", "umap_1", "umap_2"]
    projection_df["barcode"] = projection_df["barcode"].astype(str)
    projection_df["umap_1"] = pd.to_numeric(projection_df["umap_1"], errors="coerce")
    projection_df["umap_2"] = pd.to_numeric(projection_df["umap_2"], errors="coerce")

    merged = clusters_df.merge(projection_df, on="barcode", how="left")
    counts = (
        merged.groupby("cluster_id")
        .size()
        .rename("n_cells")
        .astype(int)
        .to_dict()
    )
    centroids = (
        merged.groupby("cluster_id")[["umap_1", "umap_2"]]
        .mean()
        .rename(columns={"umap_1": "centroid_umap_1", "umap_2": "centroid_umap_2"})
    )
    spreads = (
        merged.groupby("cluster_id")[["umap_1", "umap_2"]]
        .std()
        .fillna(0.0)
        .rename(columns={"umap_1": "spread_umap_1", "umap_2": "spread_umap_2"})
    )
    summary = centroids.join(spreads, how="left").fillna(0.0)

    centroid_map = {
        int(cluster_id): {
            "centroid_umap_1": float(row.centroid_umap_1),
            "centroid_umap_2": float(row.centroid_umap_2),
            "spread_umap_1": float(row.spread_umap_1),
            "spread_umap_2": float(row.spread_umap_2),
        }
        for cluster_id, row in summary.iterrows()
    }

    neighbors: dict[int, list[dict[str, float | int]]] = {}
    for cluster_id, centroid in centroid_map.items():
        items: list[dict[str, float | int]] = []
        for other_id, other_centroid in centroid_map.items():
            if cluster_id == other_id:
                continue
            distance = math.dist(
                [centroid["centroid_umap_1"], centroid["centroid_umap_2"]],
                [other_centroid["centroid_umap_1"], other_centroid["centroid_umap_2"]],
            )
            items.append({"cluster_id": int(other_id), "distance": float(distance)})
        neighbors[cluster_id] = sorted(items, key=lambda item: item["distance"])

    return centroid_map, neighbors, counts


def build_cluster_evidence_pack(
    *,
    cluster_csv: str | Path,
    differential_expression_csv: str | Path,
    projection_csv: str | Path,
    case_name: str,
    study_context: str,
    top_positive_markers: int = 15,
    top_negative_markers: int = 6,
    min_log2fc: float = 0.5,
    max_adjusted_p_value: float = 0.05,
    top_neighbors: int = 5,
) -> dict[str, Any]:
    diffexp = pd.read_csv(differential_expression_csv)
    clusters = pd.read_csv(cluster_csv)
    projection = pd.read_csv(projection_csv)

    centroid_map, neighbors, counts = _cluster_umap_summary(clusters, projection)
    cluster_ids = _cluster_ids_from_diffexp(diffexp.columns.tolist())

    cluster_records: list[dict[str, Any]] = []
    for cluster_id in cluster_ids:
        positive_markers, negative_markers = _marker_rows_for_cluster(
            diffexp,
            cluster_id=cluster_id,
            top_positive_markers=top_positive_markers,
            top_negative_markers=top_negative_markers,
            min_log2fc=min_log2fc,
            max_adjusted_p_value=max_adjusted_p_value,
        )
        cluster_records.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": int(counts.get(cluster_id, 0)),
                "study_context": study_context,
                "umap": centroid_map.get(
                    cluster_id,
                    {
                        "centroid_umap_1": 0.0,
                        "centroid_umap_2": 0.0,
                        "spread_umap_1": 0.0,
                        "spread_umap_2": 0.0,
                    },
                ),
                "nearest_clusters_in_umap": neighbors.get(cluster_id, [])[:top_neighbors],
                "top_positive_markers": positive_markers,
                "top_negative_markers": negative_markers,
            }
        )

    return {
        "case_name": case_name,
        "study_context": study_context,
        "cluster_csv": str(Path(cluster_csv).resolve()),
        "differential_expression_csv": str(Path(differential_expression_csv).resolve()),
        "projection_csv": str(Path(projection_csv).resolve()),
        "cluster_count": len(cluster_records),
        "clusters": cluster_records,
    }
