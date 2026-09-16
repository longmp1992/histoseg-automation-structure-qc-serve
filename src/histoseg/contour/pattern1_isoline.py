"""
Pattern1 contour generation via KNN regression + 0.5 isoline.

This module is extracted from the original notebook (contour_generation_pattern1.ipynb)
and turned into a reusable library function.

Core idea:
  - Use GraphClust clusters.csv to define a "target" set of cells (pattern1 clusters).
  - Sample background points from other cells (optionally + synthetic points in tissue bbox).
  - Fit KNeighborsRegressor to predict P(target) in space.
  - Smooth probabilities and take an isoline at level=0.5.
  - Filter out spurious loops by requiring >=min_cells_inside target cells in the loop.

Notes:
  - `hf_hub_download` cache paths should NOT be modified in place.
  - grid_n=1200 can be heavy (1.44M grid points). Reduce grid_n to speed up.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import json
import os
from urllib.parse import quote

import numpy as np
import pandas as pd

from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from sklearn.neighbors import KNeighborsRegressor


PathLike = Union[str, Path]


def _normalize_cluster_label(x: object) -> str:
    """Normalize cluster label to a stable string.

    Goals:
      - Support numeric clusters: 10, 10.0, "10", "10.0" -> "10"
      - Support text clusters: "Luminal A" stays "Luminal A" (trim whitespace)
      - Treat NaN/None as "" (caller should drop or error)
    """
    if x is None:
        return ""
    # pandas NA
    try:
        if pd.isna(x):  # type: ignore[arg-type]
            return ""
    except Exception:
        pass

    s = str(x).strip()
    if s == "":
        return ""
    # "10.0" -> "10" (only if it is exactly an integer-looking float string)
    # This avoids mangling genuine text like "A.0".
    if s.endswith(".0"):
        head = s[:-2]
        if head.isdigit():
            return head
    return s


@dataclass(frozen=True)
class Pattern1IsolineConfig:
    # Required inputs (non-defaults MUST come first for dataclasses)
    clusters_csv: PathLike
    cells_parquet: PathLike
    out_dir: PathLike
    pattern1_clusters: Sequence[Union[int, str]]

    # Optional inputs / schema controls
    tissue_boundary_csv: Optional[PathLike] = None
    barcode_col: str = "Barcode"
    cluster_col: str = "Cluster"

    # Core params (defaults match the notebook)
    grid_n: int = 1200
    knn_k: int = 30
    smooth_sigma: float = 5.0

    margin_um: float = 50.0
    max_dist_threshold: float = 200.0

    bg_d_min: float = 20.0
    bg_d_max: float = 250.0
    bg_max_points: int = 60000
    random_state: int = 0

    min_cells_inside: int = 10

    # Synthetic background
    use_synth_bg: bool = True
    bbox_expand_um: float = 100.0
    syn_bg_density: float = 0.01
    syn_bg_min: int = 20000
    syn_bg_max: int = 120000

    # Contour
    isoline_level: float = 0.5

    # Labeling scheme
    #   - "p1_is_one":   pattern1=1, others=0 (original behavior)
    #   - "p1_is_zero":  pattern1=0, others=1 (requested alternative)
    label_scheme: str = "p1_is_one"

    # Segmentation confidence score (cophenetic blue-band mean)
    compute_confidence_score: bool = False
    confidence_linkage_method: str = "average"
    confidence_show_corr: bool = False

    # Output controls
    save_params_json: bool = True
    save_contours_npy: bool = True
    save_preview_png: bool = True

    # Mesh padding (fraction of bbox size)
    pad_fraction: float = 0.02


@dataclass
class Pattern1IsolineResult:
    out_dir: Path
    id_col_used: str
    x_col: str
    y_col: str
    n_target_cells: int
    n_bg0_points: int
    contours: List[np.ndarray]
    label_scheme: str
    segmentation_confidence_score: Optional[float] = None
    segmentation_confidence_stats: Optional[Mapping[str, Union[int, float]]] = None
    params_json: Optional[Path] = None
    preview_png: Optional[Path] = None


@dataclass
class SegmentationConfidenceResult:
    """Result container for the cophenetic blue-band score."""
    score_mean: float
    stats: Mapping[str, Union[int, float]]
    blue_band_matrix: Optional[pd.DataFrame] = None


def _make_jupyterlab_tree_href(path: Path) -> str:
    """Build an href that opens `path` in JupyterLab's file browser.

    We intentionally return a *relative* href (e.g. "outputs/xxx/") so it works in:
      - plain JupyterLab
      - JupyterHub with base_url prefixes (because the browser resolves it relative to the current notebook URL)

    If reminding: JupyterLab uses the `/lab/tree/<path>` route for file navigation.
    """
    try:
        p = Path(path)
        if p.is_absolute():
            # If it can be made relative to current working directory, do so.
            rel = os.path.relpath(p.as_posix(), start=os.getcwd())
        else:
            rel = p.as_posix()
    except Exception:
        rel = Path(path).as_posix()

    rel = rel.replace("\\", "/")
    if not rel.endswith("/"):
        rel += "/"
    return quote(rel, safe="/")


def make_mesh_from_xy(
    xy: np.ndarray,
    grid_n: int = 800,
    pad_fraction: float = 0.02,
    margin_um: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a square mesh grid covering xy with padding + margin.

    Returns:
        xx, yy: meshgrid arrays (grid_n x grid_n)
        grid: flattened coordinates (grid_n^2 x 2)
    """
    xy = np.asarray(xy, float)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin

    xmin -= dx * pad_fraction
    xmax += dx * pad_fraction
    ymin -= dy * pad_fraction
    ymax += dy * pad_fraction

    xmin -= margin_um
    xmax += margin_um
    ymin -= margin_um
    ymax += margin_um

    xs = np.linspace(xmin, xmax, grid_n)
    ys = np.linspace(ymin, ymax, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def tissue_mask_from_xy(
    all_xy: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    max_dist_threshold: float = 200.0,
) -> np.ndarray:
    """Approximate tissue mask by distance-to-nearest-cell threshold."""
    grid = np.c_[xx.ravel(), yy.ravel()]
    tree = cKDTree(np.asarray(all_xy, float))
    dist, _ = tree.query(grid, k=1)
    return dist.reshape(xx.shape) <= max_dist_threshold


def extract_contour_paths(xx: np.ndarray, yy: np.ndarray, z2d: np.ndarray, level: float = 0.5) -> List[np.ndarray]:
    """Extract contour loop vertices at a given level using matplotlib.contour."""
    fig, ax = plt.subplots()
    cs = ax.contour(xx, yy, z2d, levels=[level])
    plt.close(fig)

    verts_list: List[np.ndarray] = []
    if hasattr(cs, "allsegs") and cs.allsegs and len(cs.allsegs) > 0:
        for seg in cs.allsegs[0]:
            v = np.asarray(seg)
            if v.ndim == 2 and v.shape[0] >= 10 and v.shape[1] == 2:
                verts_list.append(v)
        return verts_list

    # Fallback for older matplotlib
    if hasattr(cs, "collections") and cs.collections:
        for p in cs.collections[0].get_paths():
            v = p.vertices
            if len(v) >= 10:
                verts_list.append(np.asarray(v))
        return verts_list

    return []


def filter_loops_by_cell_count(verts_list: Sequence[np.ndarray], cells_xy: np.ndarray, min_cells_inside: int = 1) -> List[np.ndarray]:
    """Keep only contour loops containing >=min_cells_inside cells."""
    kept: List[np.ndarray] = []
    for v in verts_list:
        path = MplPath(v)
        if int(path.contains_points(cells_xy).sum()) >= min_cells_inside:
            kept.append(v)
    return kept


def load_tissue_boundary_csv(boundary_csv: PathLike) -> np.ndarray:
    """Load tissue boundary xy from tissue_boundary.csv (supports x,y or X,Y)."""
    df = pd.read_csv(boundary_csv)
    if {"x", "y"}.issubset(df.columns):
        return df[["x", "y"]].to_numpy(float)
    if {"X", "Y"}.issubset(df.columns):
        return df[["X", "Y"]].to_numpy(float)
    raise ValueError(f"tissue_boundary.csv 必须包含 x,y 或 X,Y 列，当前列={list(df.columns)}")


def generate_synthetic_bg_in_bbox(
    boundary_xy: np.ndarray,
    expand_um: float = 100.0,
    density: float = 0.01,
    min_n: int = 20000,
    max_n: int = 120000,
    seed: int = 0,
) -> np.ndarray:
    """Generate uniform random synthetic background points in expanded bbox."""
    rng = np.random.default_rng(seed)
    xmin, ymin = boundary_xy.min(axis=0)
    xmax, ymax = boundary_xy.max(axis=0)
    xmin -= expand_um
    xmax += expand_um
    ymin -= expand_um
    ymax += expand_um

    area = (xmax - xmin) * (ymax - ymin)
    n = int(area * density)
    n = max(min_n, min(max_n, n))

    xs = rng.uniform(xmin, xmax, size=n)
    ys = rng.uniform(ymin, ymax, size=n)
    return np.c_[xs, ys].astype(float)


def sample_background_from_other_cells_plus_synth(
    cells_df: pd.DataFrame,
    synthetic_bg_xy: Optional[np.ndarray],
    target_ids: Iterable[str],
    target_xy: np.ndarray,
    cell_id_col: str,
    x_col: str,
    y_col: str,
    d_min: float = 20,
    d_max: float = 250,
    max_points: int = 60000,
    seed: int = 0,
    margin_um: float = 50.0,
) -> np.ndarray:
    """Sample background points from non-target cells, optionally plus synthetic points.

    Filtering:
      - keep only points within bbox(target_xy) expanded by (d_max + margin_um)
      - keep only points with distance to nearest target cell in [d_min, d_max]
      - downsample to max_points
    """
    rng = np.random.default_rng(seed)

    cid_all = cells_df[cell_id_col].astype(str).to_numpy()
    target_ids_arr = np.array(list(target_ids), dtype=str)
    is_bg = ~np.isin(cid_all, target_ids_arr)

    bg_real_df = cells_df.loc[is_bg, [x_col, y_col]].copy()
    bg_real = bg_real_df[[x_col, y_col]].to_numpy(float) if len(bg_real_df) > 0 else np.empty((0, 2), float)

    bg_syn = np.asarray(synthetic_bg_xy, float) if synthetic_bg_xy is not None else np.empty((0, 2), float)

    bg_xy = bg_real
    if len(bg_syn) > 0:
        bg_xy = bg_syn if len(bg_xy) == 0 else np.vstack([bg_xy, bg_syn])
    if len(bg_xy) == 0:
        return np.empty((0, 2), float)

    xmin, ymin = target_xy.min(axis=0)
    xmax, ymax = target_xy.max(axis=0)
    pad = d_max + margin_um
    in_box = (
        (bg_xy[:, 0] >= xmin - pad) & (bg_xy[:, 0] <= xmax + pad) &
        (bg_xy[:, 1] >= ymin - pad) & (bg_xy[:, 1] <= ymax + pad)
    )
    bg_xy = bg_xy[in_box]
    if len(bg_xy) == 0:
        return np.empty((0, 2), float)

    tree = cKDTree(np.asarray(target_xy, float))
    dist, _ = tree.query(bg_xy, k=1)
    keep = (dist >= d_min) & (dist <= d_max)
    bg_xy = bg_xy[keep]
    if len(bg_xy) == 0:
        return np.empty((0, 2), float)

    if len(bg_xy) > max_points:
        idx = rng.choice(len(bg_xy), size=max_points, replace=False)
        bg_xy = bg_xy[idx]

    return bg_xy


def align_clusters_with_cells(
    clusters_csv: PathLike,
    cells_parquet: PathLike,
    barcode_col: str = "Barcode",
    cluster_col: str = "Cluster",
) -> Tuple[pd.DataFrame, str, str, str]:
    """Align clusters.csv(Barcode/Cluster) with cells.parquet.

    Returns:
        merged_df: DataFrame with x/y + cluster label column "cluster"
        id_col_used: which column in cells.parquet was used to join with Barcode
        x_col, y_col: chosen coordinate columns
    """
    cl = pd.read_csv(clusters_csv)
    if barcode_col not in cl.columns or cluster_col not in cl.columns:
        raise ValueError(
            f"clusters.csv 需要包含 {barcode_col}/{cluster_col} 列，"
            f"当前列={list(cl.columns)}"
        )

    cl = cl.copy()
    cl[barcode_col] = cl[barcode_col].astype(str).str.strip()
    # IMPORTANT: keep cluster labels as string to support non-numeric clusters.
    cl[cluster_col] = cl[cluster_col].map(_normalize_cluster_label)

    # Drop rows with empty barcode or empty cluster
    cl = cl.loc[(cl[barcode_col] != "") & (cl[cluster_col] != ""), [barcode_col, cluster_col]].copy()

    cells = pd.read_parquet(cells_parquet)

    # Try to infer coordinate columns
    cand_x = [c for c in cells.columns if c.lower() in ["x", "x_location", "x_centroid", "x_center", "xcoord", "x_coord", "x_centroid_um", "x_centroid_px", "x_centroid_microns"]]
    cand_y = [c for c in cells.columns if c.lower() in ["y", "y_location", "y_centroid", "y_center", "ycoord", "y_coord", "y_centroid_um", "y_centroid_px", "y_centroid_microns"]]
    if not cand_x or not cand_y:
        raise ValueError(f"cells.parquet 找不到 x/y 列。列名示例：{list(cells.columns)[:60]}")

    x_col = cand_x[0]
    y_col = cand_y[0]

    # Try to infer an id/barcode column
    id_candidates: List[str] = []
    for c in cells.columns:
        lc = c.lower()
        if lc in [
            "barcode", "barcodes", "cell_barcode", "cellbarcode", "spot_barcode", "spot_id",
            "cell_id", "cellid", "id"
        ]:
            id_candidates.append(c)

    if not id_candidates:
        id_candidates = [c for c in cells.columns if cells[c].dtype == object][:10]

    def try_merge(cells_id_col: str, strip_suffix: bool = False) -> pd.DataFrame:
        tmp = cells.copy()
        tmp["_join_id"] = tmp[cells_id_col].astype(str)
        cl2 = cl.copy()
        cl2["_join_id"] = cl2[barcode_col].astype(str)

        if strip_suffix:
            tmp["_join_id"] = tmp["_join_id"].str.replace(r"-1$", "", regex=True)
            cl2["_join_id"] = cl2["_join_id"].str.replace(r"-1$", "", regex=True)

        m = tmp.merge(cl2[["_join_id", cluster_col]], on="_join_id", how="inner")
        return m

    best: Optional[pd.DataFrame] = None
    best_info: Optional[Tuple[str, bool]] = None
    for c in id_candidates:
        m1 = try_merge(c, strip_suffix=False)
        if best is None or len(m1) > len(best):
            best, best_info = m1, (c, False)
        m2 = try_merge(c, strip_suffix=True)
        if best is None or len(m2) > len(best):
            best, best_info = m2, (c, True)

    if best is None or len(best) == 0 or best_info is None:
        # Provide debugging hints
        msg = [
            "[FAIL] 无法将 clusters.csv 的 Barcode 对齐到 cells.parquet",
            f"clusters.csv {barcode_col} 示例: {cl[barcode_col].head().tolist()}",
            f"cells.parquet 列名: {list(cells.columns)[:80]}",
        ]
        for c in id_candidates[:6]:
            msg.append(f"cells[{c}] 示例: {cells[c].astype(str).head().tolist()}")
        raise RuntimeError("\n".join(msg))

    id_col_used, _stripped = best_info
    # Rename cluster_col -> cluster (string label)
    out = best.rename(columns={cluster_col: "cluster"})
    out["cluster"] = out["cluster"].map(_normalize_cluster_label)
    return out, id_col_used, x_col, y_col


def _validate_label_scheme(label_scheme: str) -> str:
    s = str(label_scheme).strip().lower()
    if s in {"p1_is_one", "pattern1_is_one", "p1=1", "one"}:
        return "p1_is_one"
    if s in {"p1_is_zero", "pattern1_is_zero", "p1=0", "zero", "invert", "inverted"}:
        return "p1_is_zero"
    raise ValueError(
        f"Unknown label_scheme={label_scheme!r}. Supported: 'p1_is_one' or 'p1_is_zero'."
    )


def compute_segmentation_confidence_score_from_merged(
    merged_df: pd.DataFrame,
    *,
    pattern1_clusters: Sequence[Union[int, str]],
    x_col: str,
    y_col: str,
    celltype_col: str = "cluster",
    z_col: Optional[str] = None,
    linkage_method: str = "average",
    show_corr: bool = False,
    return_blue_band_matrix: bool = False,
) -> SegmentationConfidenceResult:
    """Compute the segmentation confidence score (cophenetic blue-band mean).

    Definition (as requested):
      - Build the searcher→findee distance matrix at the cluster level.
      - Compute the cophenetic (ultrametric) matrix from hierarchical clustering.
      - Take all cophenetic values between PATTERN1 clusters (A) and non-PATTERN1 clusters (B).
      - Return the mean of that cross-block ("blue band") as the confidence score.

    Notes:
      - This function is intentionally *separate* from isoline generation so you can call it independently.
      - It uses Cell-GPS Search-and-Find utilities.

    Returns:
      SegmentationConfidenceResult(score_mean, stats, blue_band_matrix?)
    """
    required = {x_col, y_col, celltype_col}
    missing = required - set(merged_df.columns)
    if missing:
        raise ValueError(f"merged_df missing required columns: {sorted(missing)}")

    from histoseg._cellgps import require_cellgps_attrs

    cellgps = require_cellgps_attrs(
        (
            "compute_searcher_findee_distance_matrix_from_df",
            "compute_cophenetic_from_distance_matrix",
        )
    )

    df = merged_df.copy()
    df[celltype_col] = df[celltype_col].map(_normalize_cluster_label)
    df = df.loc[df[celltype_col] != "", [x_col, y_col, celltype_col] + ([z_col] if z_col else [])].copy()

    if df[celltype_col].nunique() < 2:
        raise ValueError("Need at least 2 clusters to compute cophenetic score.")

    distance_matrix = cellgps.compute_searcher_findee_distance_matrix_from_df(
        df,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        celltype_col=celltype_col,
    )

    if getattr(distance_matrix, "shape", (0, 0))[0] < 2:
        raise ValueError("distance_matrix too small; check cluster sizes.")

    row_coph, col_coph = cellgps.compute_cophenetic_from_distance_matrix(
        distance_matrix,
        method=linkage_method,
        show_corr=show_corr,
    )

    # Use row_coph by default (matches the user's notebook code)
    coph = row_coph

    labels = pd.Index(coph.index).map(_normalize_cluster_label)
    coph = coph.copy()
    coph.index = labels
    coph.columns = labels

    A = pd.Index([_normalize_cluster_label(x) for x in pattern1_clusters if _normalize_cluster_label(x) != ""]).unique()
    A = A.intersection(coph.index)

    if len(A) == 0:
        raise ValueError("No PATTERN1 clusters found in cophenetic matrix index. Check labels.")
    B = pd.Index(coph.index).difference(A)
    if len(B) == 0:
        raise ValueError("No non-PATTERN1 clusters found; cannot compute cross-block mean.")

    blue_band_matrix = coph.loc[A, B]
    band = blue_band_matrix.to_numpy().ravel()
    band = band[~np.isnan(band)]

    if band.size == 0:
        raise ValueError("blue band has no finite values; cannot compute score.")

    stats: Mapping[str, Union[int, float]] = {
        "n_pairs": int(band.size),
        "min": float(np.min(band)),
        "p05": float(np.quantile(band, 0.05)),
        "median": float(np.median(band)),
        "mean": float(np.mean(band)),
        "p95": float(np.quantile(band, 0.95)),
        "max": float(np.max(band)),
    }

    return SegmentationConfidenceResult(
        score_mean=float(stats["mean"]),
        stats=stats,
        blue_band_matrix=blue_band_matrix if return_blue_band_matrix else None,
    )


def compute_segmentation_confidence_score(
    *,
    clusters_csv: PathLike,
    cells_parquet: PathLike,
    pattern1_clusters: Sequence[Union[int, str]],
    barcode_col: str = "Barcode",
    cluster_col: str = "Cluster",
    linkage_method: str = "average",
    show_corr: bool = False,
    return_blue_band_matrix: bool = False,
) -> SegmentationConfidenceResult:
    """Convenience wrapper: load files, merge, and compute confidence score."""
    merged, _id_col_used, x_col, y_col = align_clusters_with_cells(
        clusters_csv=clusters_csv,
        cells_parquet=cells_parquet,
        barcode_col=barcode_col,
        cluster_col=cluster_col,
    )
    return compute_segmentation_confidence_score_from_merged(
        merged,
        pattern1_clusters=pattern1_clusters,
        x_col=x_col,
        y_col=y_col,
        celltype_col="cluster",
        z_col=None,
        linkage_method=linkage_method,
        show_corr=show_corr,
        return_blue_band_matrix=return_blue_band_matrix,
    )


def run_pattern1_isoline(cfg: Pattern1IsolineConfig) -> Pattern1IsolineResult:
    """Run the full pipeline and (optionally) save outputs."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_scheme = _validate_label_scheme(cfg.label_scheme)

    merged, id_col_used, x_col, y_col = align_clusters_with_cells(
        cfg.clusters_csv,
        cfg.cells_parquet,
        barcode_col=cfg.barcode_col,
        cluster_col=cfg.cluster_col,
    )

    merged = merged.copy()
    merged["cluster"] = merged["cluster"].map(_normalize_cluster_label)
    merged = merged.loc[merged["cluster"] != ""].copy()

    # pattern1 cluster labels
    p1 = set(_normalize_cluster_label(x) for x in cfg.pattern1_clusters)
    p1 = {x for x in p1 if x != ""}

    if len(p1) == 0:
        raise ValueError("pattern1_clusters is empty after normalization.")

    merged["_is_p1"] = merged["cluster"].isin(p1)

    p1_df = merged.loc[merged["_is_p1"], [id_col_used, x_col, y_col]].copy()
    if len(p1_df) < 10:
        raise RuntimeError(f"pattern1 cells too few after merge: {len(p1_df)}")

    target_ids = set(p1_df[id_col_used].astype(str))
    target_xy = p1_df[[x_col, y_col]].to_numpy(float)

    syn_bg_xy: Optional[np.ndarray] = None
    if cfg.use_synth_bg:
        if cfg.tissue_boundary_csv is None:
            raise ValueError("use_synth_bg=True 但未提供 tissue_boundary_csv")
        boundary_xy = load_tissue_boundary_csv(cfg.tissue_boundary_csv)
        syn_bg_xy = generate_synthetic_bg_in_bbox(
            boundary_xy,
            expand_um=cfg.bbox_expand_um,
            density=cfg.syn_bg_density,
            min_n=cfg.syn_bg_min,
            max_n=cfg.syn_bg_max,
            seed=cfg.random_state,
        )

    # Build background samples (use merged as the "cells table" because it contains x/y + id_col_used)
    bg0_xy = sample_background_from_other_cells_plus_synth(
        cells_df=merged.rename(columns={id_col_used: "tmp_id"}),
        synthetic_bg_xy=syn_bg_xy,
        target_ids=set([str(x) for x in target_ids]),
        target_xy=target_xy,
        cell_id_col="tmp_id",
        x_col=x_col,
        y_col=y_col,
        d_min=cfg.bg_d_min,
        d_max=cfg.bg_d_max,
        max_points=cfg.bg_max_points,
        seed=cfg.random_state,
        margin_um=cfg.margin_um,
    )
    if len(bg0_xy) == 0:
        raise RuntimeError("No bg0 points sampled. Try relaxing bg_d_min/bg_d_max, or disabling synth bg.")

    # Train KNN regressor
    #  - label_scheme="p1_is_one":  target=1, bg=0  (original)
    #  - label_scheme="p1_is_zero": target=0, bg=1  (inverted)
    X_train = np.vstack([bg0_xy, target_xy])
    if label_scheme == "p1_is_one":
        y_train = np.hstack([np.zeros(len(bg0_xy)), np.ones(len(target_xy))])
    else:
        y_train = np.hstack([np.ones(len(bg0_xy)), np.zeros(len(target_xy))])

    reg = KNeighborsRegressor(n_neighbors=cfg.knn_k, weights="distance")
    reg.fit(X_train, y_train)

    # Predict on mesh + smooth
    xx, yy, grid = make_mesh_from_xy(
        target_xy,
        grid_n=cfg.grid_n,
        pad_fraction=cfg.pad_fraction,
        margin_um=cfg.margin_um,
    )
    prob = reg.predict(grid).reshape(xx.shape)
    prob_smooth = gaussian_filter(prob, sigma=cfg.smooth_sigma)

    # Tissue mask: based on real cell coordinates only (exclude synthetic)
    all_xy = merged[[x_col, y_col]].to_numpy(float)
    tissue_mask = tissue_mask_from_xy(all_xy, xx, yy, max_dist_threshold=cfg.max_dist_threshold)

    prob_smooth_masked = prob_smooth.copy()
    prob_smooth_masked[~tissue_mask] = np.nan

    # 0.5 isoline (or cfg.isoline_level)
    verts_list = extract_contour_paths(xx, yy, prob_smooth_masked, level=cfg.isoline_level)
    verts_list = filter_loops_by_cell_count(verts_list, target_xy, min_cells_inside=cfg.min_cells_inside)

    if len(verts_list) == 0:
        raise RuntimeError(
            "No isoline found.\n"
            "建议：min_cells_inside 降低（如 10->3），smooth_sigma 增大（如 5->8），knn_k 增大（如 30->50），或降低 grid_n。"
        )

    # Optional: compute segmentation confidence score (cophenetic blue-band mean)
    conf_score: Optional[float] = None
    conf_stats: Optional[Mapping[str, Union[int, float]]] = None
    if cfg.compute_confidence_score:
        conf_res = compute_segmentation_confidence_score_from_merged(
            merged,
            pattern1_clusters=cfg.pattern1_clusters,
            x_col=x_col,
            y_col=y_col,
            celltype_col="cluster",
            z_col=None,
            linkage_method=cfg.confidence_linkage_method,
            show_corr=cfg.confidence_show_corr,
            return_blue_band_matrix=False,
        )
        conf_score = conf_res.score_mean
        conf_stats = conf_res.stats

    params_path: Optional[Path] = None
    if cfg.save_params_json:
        params = asdict(cfg)
        params.update(
            dict(
                id_col_used=id_col_used,
                x_col=x_col,
                y_col=y_col,
                n_target_cells=int(len(target_xy)),
                n_bg0=int(len(bg0_xy)),
                n_contours=int(len(verts_list)),
                label_scheme=label_scheme,
                segmentation_confidence_score=conf_score,
                segmentation_confidence_stats=conf_stats,
            )
        )
        params_path = out_dir / "params.json"
        with params_path.open("w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False, default=str)

    if cfg.save_contours_npy:
        for i, v in enumerate(verts_list):
            np.save(out_dir / f"pattern1_isoline_{cfg.isoline_level:g}_{i}.npy", v)

    preview_path: Optional[Path] = None
    if cfg.save_preview_png:
        plt.figure(figsize=(10, 10))
        plt.scatter(bg0_xy[:, 0], bg0_xy[:, 1], s=1, alpha=0.05, label="bg0 (other cells + synth)")
        plt.scatter(target_xy[:, 0], target_xy[:, 1], s=3, alpha=0.85, label="pattern1 cells")
        for v in verts_list:
            plt.plot(v[:, 0], v[:, 1], linewidth=2)
        plt.gca().set_aspect("equal")

        title = f"Pattern1 segmentation | isoline={cfg.isoline_level:g} | contours={len(verts_list)} | label_scheme={label_scheme}"
        if conf_score is not None:
            title += f" | confidence(mean)={conf_score:.4f}"
        plt.title(title)

        plt.legend(frameon=False)
        plt.tight_layout()

        preview_path = out_dir / f"pattern1_isoline_{cfg.isoline_level:g}.png"
        plt.savefig(preview_path, dpi=300)
        plt.close()

    # --- Jupyter-friendly: show clickable output folder link ---
    try:
        from IPython import get_ipython  # type: ignore

        if get_ipython() is not None:
            from IPython.display import HTML, display  # type: ignore

            href = _make_jupyterlab_tree_href(out_dir)
            display(HTML(f'📁 Outputs: <a href="{href}" target="_blank">{out_dir.as_posix()}</a>'))
        else:
            print(f"Outputs: {out_dir}")
    except Exception:
        # Keep it safe in non-notebook usage; still show a plain path.
        print(f"Outputs: {out_dir}")

    return Pattern1IsolineResult(
        out_dir=out_dir,
        id_col_used=id_col_used,
        x_col=x_col,
        y_col=y_col,
        n_target_cells=int(len(target_xy)),
        n_bg0_points=int(len(bg0_xy)),
        contours=list(verts_list),
        label_scheme=label_scheme,
        segmentation_confidence_score=conf_score,
        segmentation_confidence_stats=conf_stats,
        params_json=params_path,
        preview_png=preview_path,
    )



def run_pattern1_isoline_from_hf(
    repo_id: str,
    *,
    revision: str = "main",
    out_dir: PathLike = "outputs/pattern1_isoline0p5_from_graphclust",
    pattern1_clusters: Sequence[Union[int, str]] = (10, 23, 19, 27, 14, 20, 25, 26),
    clusters_relpath: str = "analysis/clustering/gene_expression_graphclust/clusters.csv",
    cache_dir: Optional[PathLike] = None,
    **cfg_overrides,
) -> Pattern1IsolineResult:
    """One-liner: download required files from a HF dataset repo and run the pipeline.

    This requires `huggingface_hub` to be installed.
    """
    # Lazy import to keep core module importable without huggingface_hub.
    from histoseg.io.huggingface import download_xenium_outs

    paths = download_xenium_outs(
        repo_id,
        revision=revision,
        clusters_relpath=clusters_relpath,
        cache_dir=cache_dir,
    )
    cfg = Pattern1IsolineConfig(
        clusters_csv=paths.clusters_csv,
        cells_parquet=paths.cells_parquet,
        tissue_boundary_csv=paths.tissue_boundary_csv,
        out_dir=out_dir,
        pattern1_clusters=pattern1_clusters,
        **cfg_overrides,
    )
    return run_pattern1_isoline(cfg)
