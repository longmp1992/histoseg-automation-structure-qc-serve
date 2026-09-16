"""Dataset-level contour adjacency aggregation and visualization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Union

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely import wkb, wkt

from .topology import ContourTopologyResult, summarize_contour_topology

PathLike = Union[str, Path]

__all__ = [
    "ContourAdjacencyConfig",
    "ContourAdjacencyResult",
    "build_contour_adjacency_edges",
    "build_contour_adjacency_graph",
    "build_contour_adjacency_matrix",
    "draw_contour_adjacency_heatmap",
    "draw_contour_adjacency_network",
    "load_contours_csv",
    "run_contour_adjacency",
]


@dataclass(frozen=True)
class ContourAdjacencyConfig:
    """Configuration for dataset-level contour adjacency analysis."""

    contours: PathLike | pd.DataFrame
    out_dir: PathLike
    groupby: str = "group_label"
    contour_id_col: str = "contour_id"
    geometry_col: str = "geometry"
    boundary_tolerance: float = 1.0
    min_shared_boundary: float = 0.0
    enclosure_min_fraction: float = 0.95
    dpi: int = 200
    save_preview_png: bool = True


@dataclass
class ContourAdjacencyResult:
    """Output paths produced by :func:`run_contour_adjacency`."""

    out_dir: Path
    edges_csv: Path
    matrix_csv: Path
    metrics_json: Path
    network_png: Path | None
    heatmap_png: Path | None
    combined_png: Path | None


def run_contour_adjacency(cfg: ContourAdjacencyConfig) -> ContourAdjacencyResult:
    """Compute contour-type adjacency CSVs and visualizations for one dataset."""

    _validate_config(cfg)
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    contours = _load_contours_input(
        cfg.contours,
        geometry_col=cfg.geometry_col,
    )
    topology = summarize_contour_topology(
        contours,
        contour_id_col=cfg.contour_id_col,
        geometry_col=cfg.geometry_col,
        groupby=cfg.groupby,
        boundary_tolerance=float(cfg.boundary_tolerance),
        min_shared_boundary=float(cfg.min_shared_boundary),
        enclosure_min_fraction=float(cfg.enclosure_min_fraction),
    )
    groups = _group_labels(contours[cfg.groupby])
    edges = build_contour_adjacency_edges(
        topology,
        groups=groups,
        groupby=cfg.groupby,
        contour_id_col=cfg.contour_id_col,
    )
    matrix = build_contour_adjacency_matrix(edges, groups=groups)
    graph = build_contour_adjacency_graph(edges, groups=groups)

    edges_csv = out_dir / "contour_adjacency_edges.csv"
    matrix_csv = out_dir / "contour_adjacency_matrix.csv"
    metrics_json = out_dir / "contour_adjacency_metrics.json"
    network_png: Path | None = None
    heatmap_png: Path | None = None
    combined_png: Path | None = None

    edges.to_csv(edges_csv, index=False)
    matrix.to_csv(matrix_csv)

    if bool(cfg.save_preview_png):
        network_png = out_dir / "contour_adjacency_network.png"
        ax = draw_contour_adjacency_network(graph)
        ax.figure.savefig(network_png, dpi=int(cfg.dpi), bbox_inches="tight")
        plt.close(ax.figure)

        heatmap_png = out_dir / "contour_adjacency_heatmap.png"
        ax = draw_contour_adjacency_heatmap(matrix)
        ax.figure.savefig(heatmap_png, dpi=int(cfg.dpi), bbox_inches="tight")
        plt.close(ax.figure)

        combined_png = out_dir / "contour_adjacency_overview.png"
        fig, (network_ax, heatmap_ax) = plt.subplots(
            1,
            2,
            figsize=(13, 6),
            gridspec_kw={"width_ratios": [1.05, 1.0]},
        )
        draw_contour_adjacency_network(graph, ax=network_ax)
        draw_contour_adjacency_heatmap(matrix, ax=heatmap_ax)
        fig.tight_layout()
        fig.savefig(combined_png, dpi=int(cfg.dpi), bbox_inches="tight")
        plt.close(fig)

    positive_edges = edges.loc[edges["total_adjacency_length_um"] > 0].copy()
    metrics = {
        "contours": _contours_source_for_metrics(cfg.contours),
        "config": {
            "groupby": cfg.groupby,
            "contour_id_col": cfg.contour_id_col,
            "geometry_col": cfg.geometry_col,
            "boundary_tolerance": float(cfg.boundary_tolerance),
            "min_shared_boundary": float(cfg.min_shared_boundary),
            "enclosure_min_fraction": float(cfg.enclosure_min_fraction),
            "dpi": int(cfg.dpi),
            "save_preview_png": bool(cfg.save_preview_png),
        },
        "n_contours": int(len(contours)),
        "n_groups": int(len(groups)),
        "n_group_pairs": int(len(edges)),
        "n_positive_group_pairs": int(len(positive_edges)),
        "boundary_pair_count": int(edges["boundary_pair_count"].sum()) if not edges.empty else 0,
        "enclosure_pair_count": int(edges["enclosure_pair_count"].sum()) if not edges.empty else 0,
        "total_adjacency_length_um": (
            float(edges["total_adjacency_length_um"].sum()) if not edges.empty else 0.0
        ),
        "output_files": {
            "edges_csv": str(edges_csv),
            "matrix_csv": str(matrix_csv),
            "metrics_json": str(metrics_json),
            "network_png": str(network_png) if network_png is not None else None,
            "heatmap_png": str(heatmap_png) if heatmap_png is not None else None,
            "combined_png": str(combined_png) if combined_png is not None else None,
        },
    }
    metrics_json.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return ContourAdjacencyResult(
        out_dir=out_dir,
        edges_csv=edges_csv.resolve(),
        matrix_csv=matrix_csv.resolve(),
        metrics_json=metrics_json.resolve(),
        network_png=network_png.resolve() if network_png is not None else None,
        heatmap_png=heatmap_png.resolve() if heatmap_png is not None else None,
        combined_png=combined_png.resolve() if combined_png is not None else None,
    )


def load_contours_csv(
    contours_csv: PathLike,
    *,
    geometry_col: str = "geometry",
) -> pd.DataFrame:
    """Load a contour CSV and parse WKT, WKB hex, or GeoJSON geometries."""

    df = pd.read_csv(Path(contours_csv).expanduser().resolve())
    if geometry_col not in df.columns:
        raise ValueError(f"`contours_csv` is missing geometry column `{geometry_col}`.")
    df = df.copy()
    df[geometry_col] = df[geometry_col].map(_parse_geometry)
    return df


def build_contour_adjacency_edges(
    topology: ContourTopologyResult,
    *,
    groups: list[str],
    groupby: str,
    contour_id_col: str = "contour_id",
) -> pd.DataFrame:
    """Aggregate boundary-neighbor and enclosure topology into group-pair rows."""

    pair_stats = {
        pair: {
            "boundary_pair_count": 0,
            "boundary_shared_length_um": 0.0,
            "enclosure_pair_count": 0,
            "enclosure_inner_boundary_length_um": 0.0,
        }
        for pair in _all_group_pairs(groups)
    }

    boundary = topology.boundary_overlap
    group_a_col = f"{groupby}_a"
    group_b_col = f"{groupby}_b"
    if not boundary.empty and group_a_col in boundary.columns and group_b_col in boundary.columns:
        for _, row in boundary.iterrows():
            if not bool(row["is_boundary_neighbor"]):
                continue
            group_a = _clean_group_label(row[group_a_col])
            group_b = _clean_group_label(row[group_b_col])
            if not group_a or not group_b or group_a == group_b:
                continue
            pair = _ordered_group_pair(group_a, group_b)
            stats = pair_stats.setdefault(
                pair,
                {
                    "boundary_pair_count": 0,
                    "boundary_shared_length_um": 0.0,
                    "enclosure_pair_count": 0,
                    "enclosure_inner_boundary_length_um": 0.0,
                },
            )
            stats["boundary_pair_count"] += 1
            stats["boundary_shared_length_um"] += float(row["shared_boundary_length_um"])

    contour_summary = topology.contour_summary.set_index(contour_id_col)
    enclosure = topology.enclosure
    outer_group_col = f"outer_{groupby}"
    inner_group_col = f"inner_{groupby}"
    inner_id_col = f"inner_{contour_id_col}"
    if (
        not enclosure.empty
        and outer_group_col in enclosure.columns
        and inner_group_col in enclosure.columns
        and inner_id_col in enclosure.columns
    ):
        for _, row in enclosure.iterrows():
            outer_group = _clean_group_label(row[outer_group_col])
            inner_group = _clean_group_label(row[inner_group_col])
            if not outer_group or not inner_group or outer_group == inner_group:
                continue
            pair = _ordered_group_pair(outer_group, inner_group)
            stats = pair_stats.setdefault(
                pair,
                {
                    "boundary_pair_count": 0,
                    "boundary_shared_length_um": 0.0,
                    "enclosure_pair_count": 0,
                    "enclosure_inner_boundary_length_um": 0.0,
                },
            )
            inner_id = row[inner_id_col]
            inner_boundary = float(contour_summary.loc[inner_id, "boundary_length_um"])
            stats["enclosure_pair_count"] += 1
            stats["enclosure_inner_boundary_length_um"] += inner_boundary

    rows = []
    for group_a, group_b in _all_group_pairs(groups):
        stats = pair_stats[(group_a, group_b)]
        boundary_pair_count = int(stats["boundary_pair_count"])
        enclosure_pair_count = int(stats["enclosure_pair_count"])
        boundary_shared = float(stats["boundary_shared_length_um"])
        enclosure_inner_boundary = float(stats["enclosure_inner_boundary_length_um"])
        rows.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "boundary_pair_count": boundary_pair_count,
                "boundary_shared_length_um": boundary_shared,
                "enclosure_pair_count": enclosure_pair_count,
                "enclosure_inner_boundary_length_um": enclosure_inner_boundary,
                "total_adjacency_length_um": boundary_shared + enclosure_inner_boundary,
            }
        )

    columns = [
        "group_a",
        "group_b",
        "boundary_pair_count",
        "boundary_shared_length_um",
        "enclosure_pair_count",
        "enclosure_inner_boundary_length_um",
        "total_adjacency_length_um",
    ]
    return pd.DataFrame.from_records(rows, columns=columns).sort_values(
        ["total_adjacency_length_um", "boundary_pair_count", "enclosure_pair_count", "group_a", "group_b"],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def build_contour_adjacency_matrix(edges: pd.DataFrame, *, groups: list[str]) -> pd.DataFrame:
    """Build a symmetric group adjacency matrix with an empty diagonal."""

    matrix = pd.DataFrame(0.0, index=groups, columns=groups)
    for group in groups:
        matrix.loc[group, group] = np.nan
    for _, row in edges.iterrows():
        group_a = str(row["group_a"])
        group_b = str(row["group_b"])
        if group_a == group_b:
            continue
        value = float(row["total_adjacency_length_um"])
        matrix.loc[group_a, group_b] = value
        matrix.loc[group_b, group_a] = value
    return matrix


def build_contour_adjacency_graph(edges: pd.DataFrame, *, groups: list[str]) -> nx.Graph:
    """Build a graph whose positive edges represent group adjacency."""

    graph = nx.Graph()
    for group in groups:
        graph.add_node(
            group,
            group_label=group,
            degree=0,
            total_pair_count=0,
            weighted_degree_adjacency_um=0.0,
        )

    for _, row in edges.iterrows():
        total_length = float(row["total_adjacency_length_um"])
        pair_count = int(row["boundary_pair_count"]) + int(row["enclosure_pair_count"])
        if total_length <= 0 and pair_count <= 0:
            continue
        group_a = str(row["group_a"])
        group_b = str(row["group_b"])
        if group_a == group_b:
            continue
        graph.add_edge(
            group_a,
            group_b,
            boundary_pair_count=int(row["boundary_pair_count"]),
            enclosure_pair_count=int(row["enclosure_pair_count"]),
            total_pair_count=pair_count,
            boundary_shared_length_um=float(row["boundary_shared_length_um"]),
            enclosure_inner_boundary_length_um=float(row["enclosure_inner_boundary_length_um"]),
            total_adjacency_length_um=total_length,
            weight=total_length,
        )

    for node in graph.nodes:
        edge_data = list(graph.edges(node, data=True))
        graph.nodes[node]["degree"] = len(edge_data)
        graph.nodes[node]["total_pair_count"] = int(
            sum(edge[2]["total_pair_count"] for edge in edge_data)
        )
        graph.nodes[node]["weighted_degree_adjacency_um"] = float(
            sum(edge[2]["total_adjacency_length_um"] for edge in edge_data)
        )
    return graph


def draw_contour_adjacency_network(
    graph: nx.Graph,
    ax: Any | None = None,
    *,
    seed: int = 42,
    title: str = "Contour adjacency network",
) -> Any:
    """Draw a contour adjacency network with edge labels as pair counts."""

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No contour groups", ha="center", va="center", fontsize=13)
        ax.set_title(title)
        ax.axis("off")
        return ax

    pos = nx.spring_layout(
        graph,
        weight="total_adjacency_length_um",
        seed=int(seed),
        k=1.25,
        iterations=300,
    )
    node_values = np.asarray(
        [
            graph.nodes[node].get("weighted_degree_adjacency_um", 0.0)
            for node in graph.nodes
        ],
        dtype=float,
    )
    node_sizes = _scale_values(node_values, low=900.0, high=3200.0)
    edge_data = list(graph.edges(data=True))
    edge_weights = np.asarray(
        [edge[2].get("total_adjacency_length_um", 0.0) for edge in edge_data],
        dtype=float,
    )
    edge_widths = _scale_values(edge_weights, low=1.25, high=8.0) if edge_data else []

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color=edge_weights if edge_data else "#A0A7B2",
        edge_cmap=plt.cm.OrRd,
        alpha=0.75,
    )
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_values,
        cmap=plt.cm.YlGnBu,
        edgecolors="#1F2933",
        linewidths=0.9,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9, font_weight="bold")
    edge_labels = {
        (source, target): str(attrs.get("total_pair_count", 0))
        for source, target, attrs in graph.edges(data=True)
    }
    if edge_labels:
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=8,
            label_pos=0.5,
        )

    colorbar = ax.figure.colorbar(nodes, ax=ax, shrink=0.7)
    colorbar.set_label("Weighted adjacency length (um)")
    ax.set_title(title)
    ax.axis("off")
    return ax


def draw_contour_adjacency_heatmap(
    matrix: pd.DataFrame,
    ax: Any | None = None,
    *,
    title: str = "Adjacency heatmap",
) -> Any:
    """Draw a heatmap from a contour adjacency matrix."""

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if matrix.empty:
        ax.text(0.5, 0.5, "No contour groups", ha="center", va="center", fontsize=13)
        ax.set_title(title)
        ax.axis("off")
        return ax

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="mako",
        mask=matrix.isna(),
        square=True,
        linewidths=0.5,
        linecolor="#D8DDE5",
        cbar_kws={"label": "Total adjacency length (um)"},
    )
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    return ax


def _load_contours_input(
    contours: PathLike | pd.DataFrame,
    *,
    geometry_col: str,
) -> pd.DataFrame:
    if isinstance(contours, pd.DataFrame):
        df = contours.copy()
        if geometry_col not in df.columns:
            raise ValueError(f"`contours` is missing geometry column `{geometry_col}`.")
        df[geometry_col] = df[geometry_col].map(
            lambda value: value if isinstance(value, BaseGeometry) else _parse_geometry(value)
        )
        return df
    return load_contours_csv(contours, geometry_col=geometry_col)


def _parse_geometry(value: Any) -> BaseGeometry:
    if isinstance(value, BaseGeometry):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Geometry values must not be empty.")
    text = str(value).strip()
    if not text:
        raise ValueError("Geometry values must not be empty.")
    try:
        return wkt.loads(text)
    except Exception:
        pass
    try:
        return wkb.loads(bytes.fromhex(text))
    except Exception:
        pass
    try:
        payload = json.loads(text)
        return shape(payload)
    except Exception as exc:
        raise ValueError(
            "Geometry values must be Shapely geometries, WKT strings, WKB hex, or GeoJSON."
        ) from exc


def _validate_config(cfg: ContourAdjacencyConfig) -> None:
    if float(cfg.boundary_tolerance) < 0:
        raise ValueError("`boundary_tolerance` must be non-negative.")
    if float(cfg.min_shared_boundary) < 0:
        raise ValueError("`min_shared_boundary` must be non-negative.")
    if not 0 < float(cfg.enclosure_min_fraction) <= 1:
        raise ValueError("`enclosure_min_fraction` must be in the interval (0, 1].")
    if int(cfg.dpi) <= 0:
        raise ValueError("`dpi` must be positive.")


def _contours_source_for_metrics(contours: PathLike | pd.DataFrame) -> str:
    if isinstance(contours, pd.DataFrame):
        return "<dataframe>"
    return str(Path(contours).expanduser().resolve())


def _group_labels(values: pd.Series) -> list[str]:
    groups = [_clean_group_label(value) for value in values]
    groups = [group for group in groups if group]
    return sorted(dict.fromkeys(groups), key=str)


def _all_group_pairs(groups: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for index, group_a in enumerate(groups):
        for group_b in groups[index + 1 :]:
            pairs.append((group_a, group_b))
    return pairs


def _clean_group_label(value: Any) -> str:
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _ordered_group_pair(group_a: str, group_b: str) -> tuple[str, str]:
    return (group_a, group_b) if str(group_a) <= str(group_b) else (group_b, group_a)


def _scale_values(values: np.ndarray, *, low: float, high: float) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=float)
    value_range = float(np.ptp(values))
    if value_range <= 0:
        return np.full(values.shape, (low + high) / 2.0, dtype=float)
    return low + (high - low) * (values - float(values.min())) / value_range
