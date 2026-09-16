"""Group boundary-overlap network analysis and visualization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence, Union

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

PathLike = Union[str, Path]

__all__ = [
    "BoundaryNetworkConfig",
    "BoundaryNetworkResult",
    "build_group_boundary_graph",
    "draw_group_boundary_network",
    "normalize_group_boundary_overlap",
    "run_group_boundary_network",
]


@dataclass(frozen=True)
class BoundaryNetworkConfig:
    """Configuration for group boundary-overlap network rendering."""

    boundary_csv: PathLike
    out_dir: PathLike
    drop_structures: Sequence[Union[str, int]] = ()
    min_shared_boundary: float = 0.0
    min_boundary_pairs: int = 0
    top_n_edges: int | None = None
    dpi: int = 200
    save_preview_png: bool = True


@dataclass
class BoundaryNetworkResult:
    """Output paths produced by :func:`run_group_boundary_network`."""

    out_dir: Path
    filtered_edges_csv: Path
    nodes_csv: Path
    metrics_json: Path
    preview_png: Path | None


def run_group_boundary_network(cfg: BoundaryNetworkConfig) -> BoundaryNetworkResult:
    """Generate a group boundary-overlap network from a CSV edge table."""

    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _validate_config(cfg)
    boundary_csv = Path(cfg.boundary_csv).expanduser().resolve()
    edges = normalize_group_boundary_overlap(pd.read_csv(boundary_csv))
    edges = _filter_edges(
        edges,
        drop_structures=cfg.drop_structures,
        min_shared_boundary=float(cfg.min_shared_boundary),
        min_boundary_pairs=int(cfg.min_boundary_pairs),
        top_n_edges=cfg.top_n_edges,
    )

    graph = build_group_boundary_graph(edges)
    nodes = _node_table_from_graph(graph)

    filtered_edges_csv = out_dir / "group_boundary_overlap_filtered.csv"
    nodes_csv = out_dir / "group_boundary_network_nodes.csv"
    metrics_json = out_dir / "group_boundary_network_metrics.json"
    preview_png: Path | None = None

    edges.to_csv(filtered_edges_csv, index=False)
    nodes.to_csv(nodes_csv, index=False)

    if bool(cfg.save_preview_png):
        preview_png = out_dir / "group_boundary_network.png"
        ax = draw_group_boundary_network(graph)
        ax.figure.savefig(preview_png, dpi=int(cfg.dpi), bbox_inches="tight")
        plt.close(ax.figure)

    metrics = {
        "boundary_csv": str(boundary_csv),
        "config": asdict(cfg),
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "n_boundary_pairs": int(edges["n_boundary_pairs"].sum()) if not edges.empty else 0,
        "total_shared_boundary_length_um": (
            float(edges["shared_boundary_length_um"].sum()) if not edges.empty else 0.0
        ),
        "output_files": {
            "filtered_edges_csv": str(filtered_edges_csv),
            "nodes_csv": str(nodes_csv),
            "metrics_json": str(metrics_json),
            "preview_png": str(preview_png) if preview_png is not None else None,
        },
    }
    metrics_json.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return BoundaryNetworkResult(
        out_dir=out_dir,
        filtered_edges_csv=filtered_edges_csv.resolve(),
        nodes_csv=nodes_csv.resolve(),
        metrics_json=metrics_json.resolve(),
        preview_png=preview_png.resolve() if preview_png is not None else None,
    )


def normalize_group_boundary_overlap(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize group boundary-overlap columns to HistoSeg's internal schema.

    Accepted input styles include legacy CSV exports with
    ``total_shared_boundary``/``mean_shared_boundary`` and current topology
    outputs with ``shared_boundary_length_um``.
    """

    if not isinstance(edges, pd.DataFrame):
        raise TypeError(f"`edges` must be a pandas.DataFrame, got {type(edges)!r}.")

    required = {"group_a", "group_b", "n_boundary_pairs"}
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"`edges` is missing required columns: {sorted(missing)}")

    shared_col = _first_existing_column(
        edges,
        ["shared_boundary_length_um", "total_shared_boundary"],
        "`edges` must contain `shared_boundary_length_um` or `total_shared_boundary`.",
    )
    mean_col = _first_existing_column(
        edges,
        ["mean_shared_boundary_length_um", "mean_shared_boundary"],
        None,
    )

    working = pd.DataFrame(
        {
            "group_a": edges["group_a"].map(_clean_group_label),
            "group_b": edges["group_b"].map(_clean_group_label),
            "n_boundary_pairs": pd.to_numeric(edges["n_boundary_pairs"], errors="coerce"),
            "shared_boundary_length_um": pd.to_numeric(edges[shared_col], errors="coerce"),
        }
    )
    if mean_col is not None:
        working["mean_shared_boundary_length_um"] = pd.to_numeric(
            edges[mean_col],
            errors="coerce",
        )

    working = working.loc[(working["group_a"] != "") & (working["group_b"] != "")].copy()
    working["n_boundary_pairs"] = working["n_boundary_pairs"].fillna(0)
    working["shared_boundary_length_um"] = working["shared_boundary_length_um"].fillna(0.0)

    if (working["n_boundary_pairs"] < 0).any():
        raise ValueError("`n_boundary_pairs` values must be non-negative.")
    if (working["shared_boundary_length_um"] < 0).any():
        raise ValueError("Shared boundary values must be non-negative.")

    ordered_pairs = working.apply(
        lambda row: _ordered_group_pair(row["group_a"], row["group_b"]),
        axis=1,
        result_type="expand",
    )
    if not ordered_pairs.empty:
        working[["group_a", "group_b"]] = ordered_pairs

    grouped = (
        working.groupby(["group_a", "group_b"], dropna=False, as_index=False)
        .agg(
            n_boundary_pairs=("n_boundary_pairs", "sum"),
            shared_boundary_length_um=("shared_boundary_length_um", "sum"),
        )
        .reset_index(drop=True)
    )
    grouped["n_boundary_pairs"] = grouped["n_boundary_pairs"].round().astype(int)
    grouped["mean_shared_boundary_length_um"] = _mean_shared_boundary(
        grouped["shared_boundary_length_um"],
        grouped["n_boundary_pairs"],
    )

    return _sort_edges(grouped)


def build_group_boundary_graph(edges: pd.DataFrame) -> nx.Graph:
    """Build a NetworkX graph from normalized group boundary-overlap edges."""

    normalized = normalize_group_boundary_overlap(edges)
    graph = nx.Graph()

    node_stats: dict[str, dict[str, float | set[str]]] = {}
    for _, row in normalized.iterrows():
        group_a = str(row["group_a"])
        group_b = str(row["group_b"])
        n_pairs = int(row["n_boundary_pairs"])
        shared = float(row["shared_boundary_length_um"])
        mean_shared = float(row["mean_shared_boundary_length_um"])

        endpoint_pairs = [(group_a, group_b)] if group_a == group_b else [
            (group_a, group_b),
            (group_b, group_a),
        ]
        for group, neighbor in endpoint_pairs:
            stats = node_stats.setdefault(
                group,
                {
                    "neighbors": set(),
                    "total_boundary_pairs": 0.0,
                    "weighted_degree_shared_boundary_um": 0.0,
                },
            )
            if neighbor != group:
                stats["neighbors"].add(neighbor)  # type: ignore[union-attr]
            stats["total_boundary_pairs"] = float(stats["total_boundary_pairs"]) + n_pairs
            stats["weighted_degree_shared_boundary_um"] = (
                float(stats["weighted_degree_shared_boundary_um"]) + shared
            )

        graph.add_edge(
            group_a,
            group_b,
            n_boundary_pairs=n_pairs,
            shared_boundary_length_um=shared,
            mean_shared_boundary_length_um=mean_shared,
            weight=shared,
        )

    for group, stats in node_stats.items():
        graph.add_node(
            group,
            group_label=group,
            degree=len(stats["neighbors"]),  # type: ignore[arg-type]
            total_boundary_pairs=int(stats["total_boundary_pairs"]),
            weighted_degree_shared_boundary_um=float(
                stats["weighted_degree_shared_boundary_um"]
            ),
        )

    return graph


def draw_group_boundary_network(
    graph: nx.Graph,
    ax: Any | None = None,
    *,
    seed: int = 42,
    title: str = "Group boundary-overlap network",
) -> Any:
    """Draw a boundary-overlap graph and return the Matplotlib axes."""

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    if graph.number_of_nodes() == 0:
        ax.text(
            0.5,
            0.5,
            "No remaining structures",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_title(title)
        ax.axis("off")
        return ax

    pos = nx.spring_layout(
        graph,
        weight="shared_boundary_length_um",
        seed=int(seed),
        k=1.25,
        iterations=300,
    )

    node_labels = list(graph.nodes())
    node_values = np.asarray(
        [
            graph.nodes[node].get("weighted_degree_shared_boundary_um", 0.0)
            for node in node_labels
        ],
        dtype=float,
    )
    node_sizes = _scale_values(node_values, low=1200.0, high=3600.0)

    edge_data = list(graph.edges(data=True))
    edge_weights = np.asarray(
        [edge[2].get("shared_boundary_length_um", 0.0) for edge in edge_data],
        dtype=float,
    )
    edge_widths = _scale_values(edge_weights, low=1.5, high=9.0) if edge_data else []

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

    colorbar = ax.figure.colorbar(nodes, ax=ax, shrink=0.72)
    colorbar.set_label("Weighted shared boundary (um)")

    ax.set_title(title)
    ax.axis("off")
    return ax


def _validate_config(cfg: BoundaryNetworkConfig) -> None:
    if float(cfg.min_shared_boundary) < 0:
        raise ValueError("`min_shared_boundary` must be non-negative.")
    if int(cfg.min_boundary_pairs) < 0:
        raise ValueError("`min_boundary_pairs` must be non-negative.")
    if cfg.top_n_edges is not None and int(cfg.top_n_edges) < 0:
        raise ValueError("`top_n_edges` must be non-negative when provided.")
    if int(cfg.dpi) <= 0:
        raise ValueError("`dpi` must be positive.")


def _filter_edges(
    edges: pd.DataFrame,
    *,
    drop_structures: Sequence[Union[str, int]],
    min_shared_boundary: float,
    min_boundary_pairs: int,
    top_n_edges: int | None,
) -> pd.DataFrame:
    filtered = edges.copy()
    drop_labels = _drop_structure_labels(drop_structures)
    if drop_labels:
        filtered = filtered.loc[
            ~filtered["group_a"].isin(drop_labels) & ~filtered["group_b"].isin(drop_labels)
        ].copy()
    if min_shared_boundary > 0:
        filtered = filtered.loc[
            filtered["shared_boundary_length_um"] >= float(min_shared_boundary)
        ].copy()
    if min_boundary_pairs > 0:
        filtered = filtered.loc[filtered["n_boundary_pairs"] >= int(min_boundary_pairs)].copy()
    filtered = _sort_edges(filtered)
    if top_n_edges is not None:
        filtered = filtered.head(int(top_n_edges)).copy()
    return filtered.reset_index(drop=True)


def _node_table_from_graph(graph: nx.Graph) -> pd.DataFrame:
    columns = [
        "group_label",
        "degree",
        "total_boundary_pairs",
        "weighted_degree_shared_boundary_um",
    ]
    rows = []
    for node, attrs in graph.nodes(data=True):
        rows.append(
            {
                "group_label": str(attrs.get("group_label", node)),
                "degree": int(attrs.get("degree", 0)),
                "total_boundary_pairs": int(attrs.get("total_boundary_pairs", 0)),
                "weighted_degree_shared_boundary_um": float(
                    attrs.get("weighted_degree_shared_boundary_um", 0.0)
                ),
            }
        )
    return pd.DataFrame.from_records(rows, columns=columns).sort_values(
        ["weighted_degree_shared_boundary_um", "total_boundary_pairs", "group_label"],
        ascending=[False, False, True],
        kind="stable",
    )


def _sort_edges(edges: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "group_a",
        "group_b",
        "n_boundary_pairs",
        "shared_boundary_length_um",
        "mean_shared_boundary_length_um",
    ]
    if edges.empty:
        return pd.DataFrame(columns=columns)
    return edges.loc[:, columns].sort_values(
        ["shared_boundary_length_um", "n_boundary_pairs", "group_a", "group_b"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _mean_shared_boundary(
    shared_boundary: pd.Series,
    n_boundary_pairs: pd.Series,
) -> np.ndarray:
    shared = shared_boundary.to_numpy(dtype=float)
    counts = n_boundary_pairs.to_numpy(dtype=float)
    return np.divide(shared, counts, out=np.zeros_like(shared, dtype=float), where=counts > 0)


def _first_existing_column(
    edges: pd.DataFrame,
    candidates: Sequence[str],
    error_message: str | None,
) -> str | None:
    for candidate in candidates:
        if candidate in edges.columns:
            return candidate
    if error_message is not None:
        raise ValueError(error_message)
    return None


def _drop_structure_labels(values: Sequence[Union[str, int]]) -> set[str]:
    labels: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        labels.add(_clean_group_label(text))
        if text.lower().startswith("structure "):
            labels.add(_canonical_structure_label(text))
        else:
            labels.add(f"Structure {text}")
    return labels


def _clean_group_label(value: Any) -> str:
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    if text.lower().startswith("structure "):
        return _canonical_structure_label(text)
    return text


def _canonical_structure_label(value: str) -> str:
    parts = str(value).strip().split(maxsplit=1)
    if len(parts) == 1:
        return "Structure"
    return f"Structure {parts[1].strip()}"


def _ordered_group_pair(group_a: str, group_b: str) -> tuple[str, str]:
    return (group_a, group_b) if str(group_a) <= str(group_b) else (group_b, group_a)


def _scale_values(values: np.ndarray, *, low: float, high: float) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=float)
    value_range = float(np.ptp(values))
    if value_range <= 0:
        return np.full(values.shape, (low + high) / 2.0, dtype=float)
    return low + (high - low) * (values - float(values.min())) / value_range
