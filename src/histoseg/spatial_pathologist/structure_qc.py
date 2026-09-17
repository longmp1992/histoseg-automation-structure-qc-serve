#!/usr/bin/env python
"""histoseg_csr_qc - CSR-based quality control of HistoSeg multi-structure partitions.

Input : a HistoSeg output zip (or extracted folder) containing
        cells_with_structure_partition.parquet + structure_contour_metrics.json
        + the ORIGINAL StructureMap (row_coph cophenetic matrix, CSV) of the same run,
        found inside the input or given with --structuremap
Output: per-structure verdict (PASS / PARTIAL / FAIL), per-cluster hotspot flags and
        split suggestions cut from the StructureMap dendrogram (new cluster_ids lists).

Algorithm (details in README.md)
  0. Window     exact partition raster, re-built with histoseg's own
                _build_structure_isolines and validated against isoline_structure_id
                (raster fallback if histoseg is missing or the match is < 99.5 %).
  1. Engine     Ripley's L on a fixed r grid + Clark-Evans NND, compared with Monte
                Carlo nulls in the SAME window (edge effects cancel):
                  null "csr" = uniform points in the window
                  null "rl"  = random labelling: n points drawn only from cells whose
                               clusters are assigned to the structure (controls for the
                               assigned structure's own density inhomogeneity)
                DI (deviation index) = mean_{r_min<=r<=r_max} L_obs(r)/L_null(r) - 1
                  DI > 0 clustered, DI ~ 0 random, DI < 0 regular.
  2. Structure  DI_tissue (structure cells, tissue window, csr)
                DI_in     (structure cells, own window,   csr)
                verdict: tissue-level concentration must be significant, then
                DI_in <= t_homog_pass PASS, <= t_homog_partial PARTIAL, else FAIL.
                EF = 1 - max(DI_in, 0) / DI_tissue is reported only (it is biased
                low for structures covering most of the tissue).
  3. Cluster    for every cluster assigned to a structure: DI_csr and DI_rl
                HOTSPOT if DI_rl >= t_hotspot, DI_csr >= t_hotspot, p_rl <= alpha
                Only the structure's assigned clusters are tested; foreign clusters
                spatially falling inside its contour are excluded from all calculations.
  4. Split      CSR decides WHETHER to split, the original StructureMap decides HOW.
                The structure's clusters are taken from the StructureMap as given (no
                distances are recomputed; the ultrametric cophenetic matrix defines the
                tree uniquely). Starting at the structure's subtree root, a branch is
                split into its children while it mixes SUBDOMAIN hotspot clusters with
                non-hotspot clusters; each resulting branch is a proposed sub-structure.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from scipy.spatial import cKDTree

PARAMS = dict(
    n_sim=49,              # Monte Carlo simulations per test (min p = 0.02)
    r_step=10.0,           # µm
    r_min=20.0,            # DI averaged over [r_min, r_max]; r < 20 µm is cell hard-core
    r_max=100.0,
    max_points=20000,      # independent thinning above this (L is thinning-invariant)
    min_cells=100,         # cluster must have >= this many cells inside the structure
    min_share=0.01,        # ... and >= this share of the structure's cells
    alpha=0.05,
    t_homog_pass=0.10,     # DI_in thresholds
    t_homog_partial=0.30,
    t_hotspot=0.30,        # cluster DI thresholds
    t_mild=0.15,
    min_group_share=0.05,  # proposed sub-structures / islands below this share are flagged as small
    t_edge=0.60,           # median distance-to-contour ratio below this = boundary-enriched
    min_match=0.995,       # required cell-label agreement of the reconstructed window
)
ISO_KEYS = ["bins_x", "bins_y", "gaussian_sigma", "density_scale_quantile", "support_quantile",
            "tissue_quantile", "min_dominance", "closing_iterations", "opening_iterations",
            "fill_holes", "min_cells", "min_component_pixels"]
ISO_FALLBACK = dict(bins_x=900, bins_y=700, gaussian_sigma=2.25, density_scale_quantile=0.98,
                    support_quantile=0.18, tissue_quantile=0.06, min_dominance=0.34,
                    closing_iterations=2, opening_iterations=1, fill_holes=True, min_cells=500,
                    min_component_pixels=180)


# --------------------------------------------------------------------------- input
def open_input(path: Path) -> Path:
    if path.is_dir():
        return next(path.rglob("structure_contour_metrics.json")).parent
    tmp = Path(tempfile.mkdtemp(prefix="histoseg_csr_qc_"))
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            base = Path(name).name.lower()
            if base in ("cells_with_structure_partition.parquet", "structure_contour_metrics.json") or \
                    (base.endswith(".csv") and any(k in base for k in STRUCTUREMAP_KEYS)):
                z.extract(name, tmp)
    return next(tmp.rglob("structure_contour_metrics.json")).parent


STRUCTUREMAP_KEYS = ("row_coph", "cophenetic", "structuremap", "structure_map")


def _norm_label(x) -> str:
    t = str(x).strip()
    for prefix in ("cluster_", "cluster", "c"):
        rest = t[len(prefix):].strip()
        if t.lower().startswith(prefix) and rest.replace(".", "", 1).isdigit():
            t = rest
            break
    try:
        f = float(t)
        return str(int(f)) if f.is_integer() else t
    except ValueError:
        return t


def load_structuremap(root: Path, explicit: Path | None):
    if explicit is not None:
        path = explicit
    else:
        cands = [f for f in root.rglob("*.csv") if any(k in f.name.lower() for k in STRUCTUREMAP_KEYS)]
        if not cands:
            return None, None
        if len(cands) > 1:
            raise ValueError(f"several StructureMap candidates found, pass one with --structuremap: {cands}")
        path = cands[0]
    M = pd.read_csv(path, index_col=0)
    M.index = [_norm_label(i) for i in M.index]
    M.columns = [_norm_label(c) for c in M.columns]
    if set(M.index) != set(M.columns):
        raise ValueError(f"StructureMap {path} is not a square cluster x cluster matrix")
    M = M.loc[M.index, M.index].astype(float)
    asym = np.abs(M.to_numpy() - M.to_numpy().T).max()
    if asym > 1e-6 * max(1.0, np.abs(M.to_numpy()).max()):
        raise ValueError(f"StructureMap {path} is not symmetric (max |M - M.T| = {asym:.3g})")
    return M, str(path)


def dendrogram_split(M: pd.DataFrame, clusters: list, hot: set):
    """Cut the StructureMap subtree of `clusters` until no branch mixes hotspot and non-hotspot clusters.

    The cophenetic matrix is ultrametric, so single linkage on it returns exactly the
    tree it encodes (merge heights = cophenetic values); nothing is re-estimated.
    """
    missing = [c for c in clusters if c not in M.index]
    if missing:
        raise ValueError(f"clusters {missing} are not in the StructureMap")
    if len(clusters) == 1:
        return [dict(clusters=clusters, merge_height=0.0)]
    D = M.loc[clusters, clusters].to_numpy()
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    tol = 1e-6 * max(1.0, D.max())
    viol = np.max(D[:, None, :] - np.maximum(D[:, :, None], D[None, :, :]))
    if viol > tol:
        raise ValueError(f"StructureMap is not ultrametric on {clusters} (violation {viol:.3g}); "
                         "expected the cophenetic (row_coph) matrix, not a raw distance matrix")
    root = to_tree(linkage(squareform(D, checks=False), method="single"))

    def leaves(node):
        return [clusters[i] for i in node.pre_order()]

    def children(node):  # collapse tied merges into one multifurcating node
        out = []
        for ch in (node.get_left(), node.get_right()):
            if not ch.is_leaf() and abs(ch.dist - node.dist) <= tol:
                out += children(ch)
            else:
                out.append(ch)
        return out

    def rec(node):
        ls = leaves(node)
        flags = [c in hot for c in ls]
        if node.is_leaf() or all(flags) or not any(flags):
            return [dict(clusters=ls, merge_height=float(node.dist))]
        return [g for ch in children(node) for g in rec(ch)]

    return rec(root)


def load(root: Path):
    meta = json.load(open(root / "structure_contour_metrics.json", encoding="utf-8"))
    cells = pd.read_parquet(root / "cells_with_structure_partition.parquet")
    cells["cluster"] = cells["cluster"].astype(str)
    return meta, cells


# --------------------------------------------------------------------------- step 0: window
def _cell_pixels(cells, xe, ye, shape):
    ix = np.clip(np.searchsorted(xe, cells.x_centroid.to_numpy(), side="right") - 1, 0, shape[1] - 1)
    iy = np.clip(np.searchsorted(ye, cells.y_centroid.to_numpy(), side="right") - 1, 0, shape[0] - 1)
    return iy, ix


def build_window(cells, meta):
    iso = dict(ISO_FALLBACK)
    source = "raster_fallback"
    try:
        from histoseg.contour import multi_structure as ms
        iso.update({f.name: f.default for f in dataclasses.fields(ms.MultiStructureContourConfig)
                    if f.name in ISO_KEYS})
        iso.update(meta.get("isoline_parameters", {}))
        specs = [dict(structure_id=int(s["structure_id"]), structure_name=s["structure_name"],
                      structure_color="#000000", cluster_ids_normalized=list(s["cluster_ids"]))
                 for s in meta["selected_structures"]]
        _, part, _ = ms._build_structure_isolines(cells=cells, structure_specs=specs, x_col="x_centroid",
                                                  y_col="y_centroid", isoline_cfg=iso)
        labels, xe, ye = part["partition_labels"], part["x_edges"], part["y_edges"]
        iy, ix = _cell_pixels(cells, xe, ye, labels.shape)
        match = float((labels[iy, ix] == cells.isoline_structure_id.to_numpy()).mean())
        if match >= PARAMS["min_match"]:
            return labels, xe, ye, "histoseg_rebuild", match
    except Exception as exc:  # histoseg missing / API change
        print(f"[window] histoseg rebuild unavailable ({exc!r}); using raster fallback")

    iso.update(meta.get("isoline_parameters", {}))
    xe = np.linspace(cells.x_centroid.min(), cells.x_centroid.max(), int(iso["bins_x"]) + 1)
    ye = np.linspace(cells.y_centroid.min(), cells.y_centroid.max(), int(iso["bins_y"]) + 1)
    shape = (int(iso["bins_y"]), int(iso["bins_x"]))
    iy, ix = _cell_pixels(cells, xe, ye, shape)
    lab = cells.isoline_structure_id.to_numpy()
    k = lab.max() + 1
    counts = np.bincount((iy * shape[1] + ix) * k + lab, minlength=shape[0] * shape[1] * k)
    counts = counts.reshape(shape[0], shape[1], k)
    labels = counts.argmax(-1).astype(np.int32)
    occupied = counts.sum(-1) > 0
    labels[~occupied] = 0
    tissue = ndi.binary_fill_holes(ndi.binary_closing(occupied, iterations=2)) | occupied
    _, (ny, nx) = ndi.distance_transform_edt(labels == 0, return_indices=True)
    labels = np.where(tissue, labels[ny, nx], 0).astype(np.int32)
    match = float((labels[iy, ix] == lab).mean())
    return labels, xe, ye, source, match


# --------------------------------------------------------------------------- step 1: engine
G: dict = {}


def _init_worker(state):
    G.update(state)


def _seed(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def _L_nnd(xy, area, R):
    n = len(xy)
    tree = cKDTree(xy)
    pairs = tree.count_neighbors(tree, R).astype(float) - n
    L = np.sqrt(area * pairs / (n * (n - 1)) / np.pi)
    return L, tree.query(xy, k=2)[0][:, 1].mean()


def _uniform_in_mask(pix_iy, pix_ix, xe, ye, n, rng):
    k = rng.integers(0, pix_iy.size, n)
    return np.column_stack([xe[pix_ix[k]] + rng.random(n) * (xe[1] - xe[0]),
                            ye[pix_iy[k]] + rng.random(n) * (ye[1] - ye[0])])


def run_test(job):
    """job = (key, structure_id, cluster|None, window 'tissue'|'structure', null 'csr'|'rl')."""
    key, sid, cluster, window, null = job
    P, R, band = G["P"], G["R"], G["band"]
    labels, xe, ye = G["labels"], G["xe"], G["ye"]
    mask = labels > 0 if window == "tissue" else labels == sid
    pix_iy, pix_ix = np.nonzero(mask)
    area = pix_iy.size * (xe[1] - xe[0]) * (ye[1] - ye[0])
    # A structure is represented only by cells from clusters explicitly assigned to it.
    # Spatially overlapping cells from foreign/unassigned clusters are not candidates and
    # are not part of the random-labelling background pool.
    in_s = (G["sid"] == sid) & (G["home_sid"] == sid)
    if window == "tissue" and cluster is not None:
        # original situation: all cells of the cluster, whole tissue, independent of the partition
        sel = G["cluster"] == cluster
    else:
        sel = in_s if cluster is None else in_s & (G["cluster"] == cluster)
    rng = np.random.default_rng(_seed(key))
    xy = G["xy"][sel]
    n_full = len(xy)
    if n_full > P["max_points"]:
        xy = xy[rng.choice(n_full, P["max_points"], replace=False)]
    n = len(xy)
    pool = G["xy"][in_s]

    L_obs, nnd_obs = _L_nnd(xy, area, R)
    L_sim = np.empty((P["n_sim"], R.size))
    nnd_sim = np.empty(P["n_sim"])
    for i in range(P["n_sim"]):
        sim = (_uniform_in_mask(pix_iy, pix_ix, xe, ye, n, rng) if null == "csr"
               else pool[rng.choice(len(pool), n, replace=False)])
        L_sim[i], nnd_sim[i] = _L_nnd(sim, area, R)
    L_null = L_sim.mean(0)
    di_obs = float((L_obs[band] / L_null[band]).mean() - 1)
    loo = (L_sim.sum(0) - L_sim) / (P["n_sim"] - 1)
    di_sim = (L_sim[:, band] / loo[:, band]).mean(1) - 1
    ce = float(nnd_obs / nnd_sim.mean())
    n_lo, n_hi = (nnd_sim <= nnd_obs).sum(), (nnd_sim >= nnd_obs).sum()
    return dict(
        key=key, structure_id=sid, cluster=cluster, window=window, null=null,
        n_cells=n_full, n_used=n, window_area_mm2=area / 1e6,
        DI=di_obs,
        p_clustered=float((1 + (di_sim >= di_obs).sum()) / (P["n_sim"] + 1)),
        p_regular=float((1 + (di_sim <= di_obs).sum()) / (P["n_sim"] + 1)),
        CE_R=ce, p_CE=float(min(1.0, 2 * (1 + min(n_lo, n_hi)) / (P["n_sim"] + 1))),
        L_ratio=(L_obs / L_null).round(4).tolist(),
    )


# --------------------------------------------------------------------------- decisions
def structure_verdict(di_in, di_tis, p_tis, P):
    ef = 1 - max(di_in, 0) / di_tis if di_tis > 0 else np.nan
    if di_tis <= 0 or p_tis > P["alpha"]:
        return "FAIL", ef, "structure cells are not spatially concentrated in the tissue"
    if di_in <= P["t_homog_pass"]:
        return "PASS", ef, "near-random inside the contour"
    if di_in <= P["t_homog_partial"]:
        return "PARTIAL", ef, f"moderate internal clustering (DI_in={di_in:.2f})"
    return "FAIL", ef, f"strong internal clustering (DI_in={di_in:.2f})"


def cluster_status(row, P):
    if row.DI_rl >= P["t_hotspot"] and row.DI_csr >= P["t_hotspot"] and row.p_rl <= P["alpha"]:
        return "HOTSPOT"
    if row.DI_rl >= P["t_mild"] and row.p_rl <= P["alpha"]:
        return "CLUSTERED"
    if row.DI_rl <= -P["t_mild"] and row.p_rl_regular <= P["alpha"]:
        return "DISPERSED"
    return "RANDOM-LIKE"


# --------------------------------------------------------------------------- main
def run(input_path: Path, out_dir: Path, P: dict, workers: int, structuremap: Path | None = None):
    root = open_input(input_path)
    meta, cells = load(root)
    smap, smap_src = load_structuremap(root, structuremap)
    print(f"[structuremap] {smap_src or 'NOT FOUND - split suggestions need the original row_coph matrix (--structuremap)'}")
    if not input_path.is_dir():
        import shutil
        shutil.rmtree(Path(tempfile.gettempdir()) / root.relative_to(tempfile.gettempdir()).parts[0], ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, xe, ye, win_source, match = build_window(cells, meta)
    print(f"[window] {win_source}: cell-label match {match:.4f}")
    np.savez_compressed(out_dir / "partition_labels.npz", labels=labels, x_edges=xe, y_edges=ye)

    R = np.arange(P["r_step"], P["r_max"] + 1e-9, P["r_step"])
    band = (R >= P["r_min"]) & (R <= P["r_max"])
    sid_arr = cells.isoline_structure_id.to_numpy()
    cl_arr = cells.cluster.map(_norm_label).to_numpy(object)
    xy_all = cells[["x_centroid", "y_centroid"]].to_numpy(float)
    pix_area = (xe[1] - xe[0]) * (ye[1] - ye[0])

    specs = {int(s["structure_id"]): s for s in meta["selected_structures"]}
    home = {_norm_label(c): sid for sid, s in specs.items() for c in s["cluster_ids"]}
    home_sid_arr = np.asarray([home.get(c, 0) for c in cl_arr], dtype=int)
    jobs, cluster_rows = [], []
    for sid in specs:
        jobs += [(f"S{sid}|all|tissue|csr", sid, None, "tissue", "csr"),
                 (f"S{sid}|all|structure|csr", sid, None, "structure", "csr")]
        eligible = (sid_arr == sid) & (home_sid_arr == sid)
        n_s = int(eligible.sum())
        if n_s == 0:
            raise ValueError(
                f"Structure S{sid} has no cells from its assigned clusters inside its partition. "
                "Check the StructureMap cluster assignment and HistoSeg partition outputs."
            )
        vc = pd.Series(cl_arr[eligible]).value_counts()
        for c, n in vc.items():
            if n >= P["min_cells"] and n / n_s >= P["min_share"]:
                cluster_rows.append((sid, c, int(n), n / n_s))
                jobs += [(f"S{sid}|C{c}|structure|csr", sid, c, "structure", "csr"),
                         (f"S{sid}|C{c}|structure|rl", sid, c, "structure", "rl")]
    # original (whole tissue) vs final (own structure) DI for every assigned cluster
    queued = {j[0] for j in jobs}
    for c, sid in home.items():
        if not (cl_arr == c).any():
            continue
        jobs.append((f"C{c}|original|tissue|csr", sid, c, "tissue", "csr"))
        if f"S{sid}|C{c}|structure|csr" not in queued and ((sid_arr == sid) & (cl_arr == c)).sum() >= 3:
            jobs.append((f"S{sid}|C{c}|structure|csr", sid, c, "structure", "csr"))
    print(f"[engine] {len(jobs)} Monte Carlo tests x {P['n_sim']} simulations, {workers} workers")
    state = dict(labels=labels, xe=xe, ye=ye, xy=xy_all, sid=sid_arr,
                 cluster=cl_arr, home_sid=home_sid_arr, P=P, R=R, band=band)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(state,)) as ex:
        res = {r["key"]: r for r in ex.map(run_test, jobs)}
    tests = pd.DataFrame(res.values())
    tests.drop(columns="L_ratio").to_csv(out_dir / "qc_tests.csv", index=False)
    pd.DataFrame([dict(key=k, r_um=r, L_ratio=v) for k, t in res.items() for r, v in zip(R, t["L_ratio"])]) \
        .to_csv(out_dir / "qc_L_ratio_curves.csv", index=False)

    # clusters
    crow = []
    for sid, c, n, share in cluster_rows:
        a, b = res[f"S{sid}|C{c}|structure|csr"], res[f"S{sid}|C{c}|structure|rl"]
        crow.append(dict(structure_id=sid, cluster=c, home_structure=home.get(c), is_home=True,
                         n_in_structure=n, share_of_structure=share,
                         frac_of_cluster_here=n / int((cl_arr == c).sum()),
                         DI_csr=a["DI"], p_csr=a["p_clustered"], CE_R_csr=a["CE_R"],
                         DI_rl=b["DI"], p_rl=b["p_clustered"], p_rl_regular=b["p_regular"], CE_R_rl=b["CE_R"]))
    cdf = pd.DataFrame(crow)
    cdf["status"] = [cluster_status(r, P) for r in cdf.itertuples()]
    # distance of every cell to its structure's contour (µm)
    iy, ix = _cell_pixels(cells, xe, ye, labels.shape)
    d_cell = np.zeros(len(cells))
    for sid in specs:
        edt = ndi.distance_transform_edt(labels == sid, sampling=(ye[1] - ye[0], xe[1] - xe[0]))
        m = (sid_arr == sid) & (home_sid_arr == sid)
        d_cell[m] = edt[iy[m], ix[m]]
    med_s = {sid: np.median(d_cell[(sid_arr == sid) & (home_sid_arr == sid)]) for sid in specs}
    cdf["edge_ratio"] = [np.median(d_cell[(sid_arr == r.structure_id) & (cl_arr == r.cluster)]) / med_s[r.structure_id]
                         for r in cdf.itertuples()]
    cdf["role"] = ["SUBDOMAIN" if r.status == "HOTSPOT" else "" for r in cdf.itertuples()]
    before_after = []
    for c, sid in sorted(home.items(), key=lambda kv: (kv[1], _cluster_sort_key(kv[0]))):
        orig = res.get(f"C{c}|original|tissue|csr")
        if orig is None:
            continue
        final = res.get(f"S{sid}|C{c}|structure|csr")
        n_c = int((cl_arr == c).sum())
        n_in = int(((sid_arr == sid) & (cl_arr == c)).sum())
        di_final = final["DI"] if final is not None else np.nan
        before_after.append(dict(cluster=c, structure_id=sid, structure_name=specs[sid]["structure_name"],
                                 n_cells=n_c, n_in_own_structure=n_in, frac_in_own_structure=n_in / max(n_c, 1),
                                 DI_original_tissue=orig["DI"], DI_final_structure=di_final,
                                 dDI=orig["DI"] - di_final))
    before_after = pd.DataFrame(before_after, columns=[
        "cluster", "structure_id", "structure_name", "n_cells", "n_in_own_structure", "frac_in_own_structure",
        "DI_original_tissue", "DI_final_structure", "dDI"])
    before_after.to_csv(out_dir / "qc_cluster_DI_before_after.csv", index=False)

    # structures and split suggestions (dendrogram of the original StructureMap)
    srow, suggestions = [], []
    for sid, spec in specs.items():
        tis, ins = res[f"S{sid}|all|tissue|csr"], res[f"S{sid}|all|structure|csr"]
        verdict, ef, reason = structure_verdict(ins["DI"], tis["DI"], tis["p_clustered"], P)
        sc = cdf[cdf.structure_id == sid]
        n_s = int(((sid_arr == sid) & (home_sid_arr == sid)).sum())
        home_ids = [_norm_label(c) for c in spec["cluster_ids"]]
        single_cluster = len(set(home_ids)) == 1
        if single_cluster:
            verdict = "PASS"
            reason = "single assigned cluster; accepted by definition"
        own = home_sid_arr == sid
        subdom = sc[sc.role == "SUBDOMAIN"]
        islands = sc[sc.role == "EMBEDDED_FOREIGN"]
        big_islands = islands[islands.share_of_structure >= P["min_group_share"]]
        spill = sc[sc.role == "BOUNDARY_SPILLOVER"]
        area = (labels == sid).sum() * pix_area
        eligible = (sid_arr == sid) & own
        share = {_norm_label(k): v / n_s for k, v in pd.Series(cl_arr[eligible]).value_counts().items()}
        di_rl = {_norm_label(k): v for k, v in sc.set_index("cluster").DI_rl.items()}
        hot_ids = {_norm_label(c) for c in subdom.cluster}

        groups, branches = None, []
        if smap is not None and hot_ids:
            branches = dendrogram_split(smap, home_ids, hot_ids)
            # hotspot branches stay as cut from the tree; all non-hotspot clusters form ONE rest group
            groups = [dict(clusters=b["clusters"], merge_height=b["merge_height"], hotspot=True)
                      for b in branches if set(b["clusters"]) & hot_ids]
            rest = [c for b in branches if not set(b["clusters"]) & hot_ids for c in b["clusters"]]
            if rest:
                groups.append(dict(clusters=rest, merge_height=None, hotspot=False))
            for g in groups:
                g["share"] = float(sum(share.get(c, 0.0) for c in g["clusters"]))
                g["max_DI_rl"] = float(np.nanmax([di_rl.get(c, np.nan) for c in g["clusters"]] + [-np.inf]))
                g["small"] = g["share"] < P["min_group_share"]
        can_split = (not single_cluster) and bool(hot_ids) and (groups is None or len(groups) > 1)
        if single_cluster:
            split = "NOT_NEEDED"
        elif verdict != "PASS" and (can_split or len(big_islands)):
            split = "RECOMMENDED"
        elif verdict == "PASS" and can_split and subdom.DI_rl.max() >= 2 * P["t_hotspot"]:
            split = "OPTIONAL"
        elif verdict == "FAIL":
            split = "REVIEW"  # heterogeneous but no splittable hotspot: re-tune isoline / cluster selection
        else:
            split = "NOT_NEEDED"
        kinds = []
        if split in ("RECOMMENDED", "OPTIONAL"):
            if can_split:
                kinds.append("subdomain")
            if len(big_islands):
                kinds.append("carve_island")
        proposed = []
        if "subdomain" in kinds and groups is not None:
            for k, g in enumerate(groups, 1):
                name = f"{spec['structure_name']}.{k}" if g["hotspot"] else f"{spec['structure_name']}.rest"
                proposed.append(dict(structure_name=name, cluster_ids=g["clusters"],
                                     hotspot=g["hotspot"], share=g["share"], merge_height=g["merge_height"],
                                     small=g["small"]))
        suggestions.append(dict(
            structure_id=sid, structure_name=spec["structure_name"], verdict=verdict, split=split,
            split_kind="+".join(kinds), structuremap_available=smap is not None,
            subdomain_hotspots=subdom[["cluster", "share_of_structure", "DI_rl"]].to_dict("records"),
            dendrogram_branches=branches, dendrogram_groups=groups or [], proposed_structures=proposed,
            embedded_islands=islands[["cluster", "home_structure", "share_of_structure", "DI_rl", "edge_ratio"]].to_dict("records"),
            boundary_spillover=spill[["cluster", "share_of_structure", "DI_rl", "edge_ratio"]].to_dict("records")))
        srow.append(dict(structure_id=sid, structure_name=spec["structure_name"], n_cells=n_s,
                         area_mm2=area / 1e6, area_fraction=(labels == sid).sum() / (labels > 0).sum(),
                         purity_selected_clusters=float(own[sid_arr == sid].mean()),
                         recall_selected_clusters=float((sid_arr[own] == sid).mean()),
                         DI_tissue=tis["DI"], p_tissue=tis["p_clustered"], CE_R_tissue=tis["CE_R"],
                         DI_in=ins["DI"], p_in=ins["p_clustered"], CE_R_in=ins["CE_R"],
                         explained_fraction=ef, verdict=verdict, reason=reason,
                         n_subdomain_clusters=int(len(subdom)), n_embedded_foreign=int(len(islands)),
                         boundary_spillover_share=float(spill.share_of_structure.sum()),
                         n_proposed_structures=len(proposed), split=split, split_kind="+".join(kinds)))
    sdf = pd.DataFrame(srow)
    sdf.to_csv(out_dir / "qc_structures.csv", index=False)
    cdf.to_csv(out_dir / "qc_clusters.csv", index=False)
    json.dump(dict(input=str(input_path), window_source=win_source, window_cell_match=match,
                   structuremap=smap_src,
                   params=P, structures=suggestions),
              open(out_dir / "qc_split_suggestions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    write_report(out_dir, input_path, sdf, cdf, suggestions, P, win_source, match, smap_src, before_after)
    plot_overview(out_dir, sdf, cdf, P, before_after)
    return sdf, cdf, suggestions


# --------------------------------------------------------------------------- outputs
def _cluster_sort_key(c):
    s = str(c)
    return (0, int(s), s) if s.isdigit() else (1, 0, s)


def plot_cluster_di_before_after(ax, table, before_col, after_col, group_col, title):
    """Grouped bars: DI of every cluster in the original tissue vs in its final structure."""
    t = table.reset_index(drop=True)
    x, pos, prev, sep = [], 0.0, None, []
    for g in t[group_col]:
        if prev is not None and g != prev:
            sep.append(pos - 0.6)
            pos += 0.6
        x.append(pos)
        pos += 1
        prev = g
    x = np.asarray(x)
    w = 0.38
    before = t[before_col].to_numpy(float)
    after = t[after_col].to_numpy(float)
    ax.bar(x - w / 2, before, width=w, color="#9c9b94", label="original: whole tissue", zorder=3)
    ax.bar(x + w / 2, np.nan_to_num(after), width=w, color="#2a78d6", label="final: own structure", zorder=3)
    for xi, a in zip(x, after):
        if np.isnan(a):
            ax.text(xi + w / 2, 0, "n/a", ha="center", va="bottom", fontsize=6, color="#6b6a64", rotation=90)
    ax.axhline(0, color="#6b6a64", lw=1)
    ax.axhline(0.3, color="#d03b3b", lw=1, ls="--", zorder=2)
    for s in sep:
        ax.axvline(s, color="#d6d5cf", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in t["cluster"]], fontsize=8, rotation=90 if len(t) > 24 else 0)
    top = np.nanmax(np.concatenate([before, after, [0.35]]))
    for g, sub in t.groupby(group_col, sort=False):
        xs = x[sub.index.to_numpy()]
        ax.text(xs.mean(), top * 1.04, str(g), ha="center", va="bottom", fontsize=8, color="#4a4a45")
    ax.set_ylim(min(0, np.nanmin(np.concatenate([before, after]))) * 1.1 - 0.02, top * 1.15)
    ax.set_ylabel("DI vs CSR")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.005, 1.0))
    ax.grid(axis="y", alpha=.25, lw=.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def write_report(out_dir, input_path, sdf, cdf, suggestions, P, win_source, match, smap_src, before_after=None):
    zh = {"PASS": "成功", "PARTIAL": "部分成功", "FAIL": "不成功",
          "RECOMMENDED": "建议进一步分割", "OPTIONAL": "可选分割", "REVIEW": "需人工复核（异质但无主导聚集 cluster）",
          "NOT_NEEDED": "无需分割"}
    def kind_zh(k):
        return " + ".join({"subdomain": "拆分亚区", "carve_island": "切出外来岛"}[x] for x in k.split("+")) if k else ""
    role_zh = {"SUBDOMAIN": "本结构亚区（分割候选）", "EMBEDDED_FOREIGN": "内嵌外来岛（分割候选）",
               "BOUNDARY_SPILLOVER": "边界溢入（轮廓精度问题）"}
    L = [f"# HistoSeg CSR 质控报告", "", f"- 输入: `{input_path}`",
         f"- 窗口: {win_source}（细胞标签一致率 {match:.2%}）",
         f"- StructureMap: `{smap_src}`" if smap_src else
         "- StructureMap: **未提供**（无法给出树状图分割方案，请用 --structuremap 指定原始 row_coph 矩阵）",
         f"- 指标: DI = 平均_{{{P['r_min']:.0f}–{P['r_max']:.0f} µm}} L_obs/L_null − 1；"
         f"{P['n_sim']} 次 Monte Carlo；最多 {P['max_points']} 点（独立稀疏化）", "",
         "## 结构判定", "",
         "| 结构 | 细胞数 | 面积占比 | DI_组织 | DI_结构内 | EF | 判定 | 建议 | 类型 | 原因 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sdf.itertuples():
        L.append(f"| {r.structure_name} | {r.n_cells:,} | {r.area_fraction:.0%} | {r.DI_tissue:.2f} | {r.DI_in:.2f} | "
                 f"{r.explained_fraction:.0%} | **{zh[r.verdict]}** | {zh[r.split]} | {kind_zh(r.split_kind)} | {r.reason} |")
    L += ["", "## 结构内聚集 cluster（HOTSPOT / CLUSTERED）", "",
          "| 结构 | Cluster | 所属结构 | 细胞数 | 占结构 | DI_CSR | DI_随机标记 | CE R | 距轮廓比 | 状态 | 角色 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in cdf[cdf.status.isin(["HOTSPOT", "CLUSTERED"])].sort_values(["structure_id", "DI_rl"], ascending=[True, False]).itertuples():
        L.append(f"| S{r.structure_id} | C{r.cluster} | S{r.home_structure} | {r.n_in_structure:,} | {r.share_of_structure:.1%} | "
                 f"{r.DI_csr:.2f} | {r.DI_rl:.2f} | {r.CE_R_csr:.2f} | {r.edge_ratio:.2f} | {r.status} | {role_zh.get(r.role, '')} |")
    if before_after is not None and len(before_after):
        L += ["", "## 每个 cluster 的 DI：原始（整个组织）vs 最终（所属结构内）", "",
              "| 结构 | Cluster | 细胞数 | 落在所属结构内 | DI 原始 | DI 最终 | ΔDI |", "|---|---|---|---|---|---|---|"]
        for r in before_after.itertuples():
            fin = "—" if pd.isna(r.DI_final_structure) else f"{r.DI_final_structure:.2f}"
            dd = "—" if pd.isna(r.dDI) else f"{r.dDI:.2f}"
            L.append(f"| {r.structure_name} | C{r.cluster} | {r.n_cells:,} | {r.frac_in_own_structure:.0%} | "
                     f"{r.DI_original_tissue:.2f} | {fin} | {dd} |")
    L += ["", "## 分割建议", ""]
    for s in suggestions:
        L.append(f"### {s['structure_name']} — {zh[s['verdict']]} / {zh[s['split']]}")
        if s["subdomain_hotspots"]:
            L.append("- 本结构亚区 HOTSPOT: " + ", ".join(
                f"C{h['cluster']}（{h['share_of_structure']:.1%}, DI_rl {h['DI_rl']:.2f}）" for h in s["subdomain_hotspots"]))
        else:
            L.append("- 无本结构亚区 HOTSPOT")
        if s["proposed_structures"]:
            L.append("- 按原始 StructureMap 树状图切出 HOTSPOT 分支，其余非聚集 cluster 合为一类:")
            for p in s["proposed_structures"]:
                small = "，⚠ 占比小于阈值" if p["small"] else ""
                if p["hotspot"]:
                    L.append(f"  - `{p['structure_name']}`: cluster_ids = {p['cluster_ids']}（HOTSPOT 分支，占结构 {p['share']:.1%}，"
                             f"树上合并高度 {p['merge_height']:.3f}{small}）")
                else:
                    L.append(f"  - `{p['structure_name']}`: cluster_ids = {p['cluster_ids']}（全部非聚集 cluster，占结构 {p['share']:.1%}{small}）")
        elif s["subdomain_hotspots"] and not s["structuremap_available"]:
            L.append("- 需要原始 StructureMap 矩阵才能给出分割方案")
        elif s["subdomain_hotspots"] and len(s["dendrogram_groups"]) == 1:
            L.append("- HOTSPOT 与其他 cluster 在树上无法分开（整棵子树同一状态），不给出亚区分割")
        if s["embedded_islands"]:
            share = sum(b["share_of_structure"] for b in s["embedded_islands"])
            names = ", ".join(f"C{b['cluster']}(属S{b['home_structure']})" for b in s["embedded_islands"])
            L.append(f"- 内嵌外来岛: {names}（合计占 {share:.1%}）——其他结构的细胞在本结构内部成团，轮廓未切出；可降低 min_component_pixels")
        if s["boundary_spillover"]:
            share = sum(b["share_of_structure"] for b in s["boundary_spillover"])
            L.append(f"- 边界溢入 cluster: {', '.join('C' + b['cluster'] for b in s['boundary_spillover'])}（合计占 {share:.1%}，不作为分割依据）")
        L.append("")
    (out_dir / "qc_report.md").write_text("\n".join(L), encoding="utf-8")


def plot_overview(out_dir, sdf, cdf, P, before_after=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vcol = {"PASS": "#1baf7a", "PARTIAL": "#eda100", "FAIL": "#d03b3b"}
    scol = {"HOTSPOT": "#d03b3b", "CLUSTERED": "#eda100", "RANDOM-LIKE": "#b0b0a8", "DISPERSED": "#2a78d6"}
    n_s = len(sdf)
    has_ba = before_after is not None and len(before_after) > 0
    bar_h = 3.6 if has_ba else 0.0
    fig = plt.figure(figsize=(13, 1.2 + 0.55 * max(n_s, 1) + 0.28 * len(cdf) + bar_h))
    gs = fig.add_gridspec(3 if has_ba else 2, 1,
                          height_ratios=[0.55 * n_s + 0.6, 0.28 * len(cdf) + 0.8] + ([bar_h] if has_ba else []))
    ax = fig.add_subplot(gs[0])
    y = -np.arange(n_s)
    for k, r in enumerate(sdf.itertuples()):
        ax.plot([r.DI_in, r.DI_tissue], [y[k]] * 2, color="#c3c2b7", lw=2, zorder=1)
        ax.scatter(r.DI_tissue, y[k], s=70, facecolor="white", edgecolor="#4a4a45", lw=2, zorder=3)
        ax.scatter(r.DI_in, y[k], s=70, color=vcol[r.verdict], edgecolor="white", zorder=3)
    ax.axvline(0, color="#6b6a64", lw=1)
    ax.axvspan(-0.05, P["t_homog_pass"], color="#1baf7a", alpha=.08, lw=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.structure_name} (n={r.n_cells:,})\n{r.verdict} · split {r.split}"
                        f"{' (' + r.split_kind + ')' if r.split_kind else ''}" for r in sdf.itertuples()], fontsize=9)
    ax.set_xlabel("DI vs CSR   ● within own structure   ○ whole tissue   (green band = PASS, amber = PARTIAL for DI_in)")
    ax.set_title("Structure QC", loc="left", fontsize=11)
    ax2 = fig.add_subplot(gs[1])
    c = cdf.sort_values(["structure_id", "DI_rl"], ascending=[True, True]).reset_index(drop=True)
    pos, yy, prev = 0.0, [], None
    for sid in c.structure_id:
        if prev is not None and sid != prev:
            pos += 0.8
        yy.append(pos); pos += 1; prev = sid
    yy = np.array(yy)
    ax2.barh(yy, c.DI_rl, color=[scol[s] for s in c.status], height=0.72)
    ax2.axvline(P["t_hotspot"], color="#d03b3b", lw=1, ls="--")
    ax2.axvline(0, color="#6b6a64", lw=1)
    ax2.set_yticks(yy)
    tag = {"SUBDOMAIN": "  [sub-domain]"}
    ax2.set_yticklabels([f"S{r.structure_id} · C{r.cluster}  ({r.share_of_structure:.0%}){tag.get(r.role, '')}"
                         for r in c.itertuples()], fontsize=8)
    ax2.set_xlabel("DI vs random labelling among clusters assigned to this structure (dashed = hotspot threshold)")
    ax2.set_title("Assigned clusters within each structure", loc="left", fontsize=11)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color=v, label=k) for k, v in scol.items()], frameon=False, fontsize=8, loc="lower right")
    for a in (ax, ax2):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.grid(axis="x", alpha=.25, lw=.6)
    ax.axvspan(P["t_homog_pass"], P["t_homog_partial"], color="#eda100", alpha=.08, lw=0)
    if has_ba:
        ax3 = fig.add_subplot(gs[2])
        plot_cluster_di_before_after(
            ax3, before_after, "DI_original_tissue", "DI_final_structure", "structure_name",
            "DI of every cluster: original (whole tissue) vs final (inside its assigned structure); dashed = 0.3")
    fig.tight_layout()
    fig.savefig(out_dir / "qc_overview.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="histoseg_outputs zip or extracted folder")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--structuremap", type=Path, default=None,
                    help="original StructureMap row_coph matrix CSV (cluster x cluster) of the same HistoSeg run")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    for k, v in PARAMS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()
    P = {k: getattr(a, k) for k in PARAMS}
    out = a.out or a.input.with_name(a.input.stem.replace(" ", "_") + "_csr_qc")
    sdf, cdf, _ = run(a.input, out, P, a.workers, a.structuremap)
    pd.set_option("display.width", 220)
    print(sdf[["structure_name", "n_cells", "area_fraction", "DI_tissue", "DI_in", "explained_fraction", "verdict", "split", "split_kind"]].round(3).to_string(index=False))
    print(cdf[cdf.status == "HOTSPOT"][["structure_id", "cluster", "is_home", "share_of_structure", "DI_csr", "DI_rl", "edge_ratio", "role"]].round(3).to_string(index=False))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
