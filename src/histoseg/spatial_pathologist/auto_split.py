"""Automatic structure splitting along the StructureMap dendrogram, scored by the CSR deviation index (DI).

The DI engine is the one of :mod:`histoseg.spatial_pathologist.structure_qc` (Ripley's L against Monte Carlo
complete spatial randomness in the same window; DI = mean_{r_min<=r<=r_max} L_obs/L_CSR - 1). Partitions are
produced by the caller's HistoSeg partition function, so the split structures are exactly what a HistoSeg run
with those cluster groups would draw.

Algorithm
---------
1. Baseline      DI of every cluster in the whole tissue.
2. Exploration   The frontier starts as one group (all clusters). The highest branch point on the frontier is
                 replaced by its children, HistoSeg re-partitions the tissue with one structure per frontier
                 group, and every cluster under that branch point gets its DI inside its own child contour.
                 dDI(v) = sum_{clusters under v} [DI in v's contour - DI in its child contour], where "DI in v's
                 contour" is measured in the partition right before v is split (HistoSeg partitions are
                 competitive, so v's contour moves as other branches are split). Repeated until every branch
                 point has been evaluated.
3. Decision      with threshold t, from the root:
                 * dDI(v) >= t                                   -> split v, recurse into its children;
                 * dDI(v) <  t and no descendant with dDI >= t   -> keep v as one structure;
                 * dDI(v) <  t but a descendant w has dDI(w) >= t -> depends on ``descendant_rule``:
                     "extract_top_branch" (default): keep v, but extract the child branch of w with the largest
                                           summed per-cluster dDI contribution as its own structure;
                     "split_parent":       split v as well (and recurse);
                     "stop":               keep v, ignore w.
4. Final         HistoSeg partition of the resulting structures; DI of every cluster in its final contour.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from scipy.spatial.distance import squareform

from .structure_qc import PARAMS as QC_PARAMS
from .structure_qc import _L_nnd, _cluster_sort_key, _norm_label, _uniform_in_mask, plot_cluster_di_before_after

DESCENDANT_RULES = ("extract_top_branch", "split_parent", "stop")
PartitionFn = Callable[[list[list[str]]], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


@dataclasses.dataclass
class AutoSplitResult:
    out_dir: Path
    nodes: pd.DataFrame
    node_cluster_di: pd.DataFrame
    clusters: pd.DataFrame
    structures: pd.DataFrame
    structure_lines: str
    report_md: Path
    dendrogram_png: Path
    partition_png: Path
    files: list[Path]
    sum_di_baseline: float
    sum_di_final: float
    before_after_png: Path


# --------------------------------------------------------------------------- DI engine
_W: dict[str, Any] = {}


def _init(state: dict[str, Any]) -> None:
    _W.update(state)


def _di_job(job):
    """job = (key, cell index array, window: 'tissue' or structure id)."""
    key, idx, win = job
    P, R, band = _W["P"], _W["R"], _W["band"]
    labels, xe, ye = _W["labels"], _W["xe"], _W["ye"]
    mask = labels > 0 if win == "tissue" else labels == win
    pix_iy, pix_ix = np.nonzero(mask)
    area = pix_iy.size * (xe[1] - xe[0]) * (ye[1] - ye[0])
    rng = np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
    xy = _W["xy"][idx]
    n_full = len(xy)
    if n_full < 3 or pix_iy.size == 0:
        return dict(key=key, n_cells=n_full, DI=np.nan, p_clustered=np.nan)
    if n_full > P["max_points"]:
        xy = xy[rng.choice(n_full, P["max_points"], replace=False)]
    n = len(xy)
    L_obs, _ = _L_nnd(xy, area, R)
    L_sim = np.empty((P["n_sim"], R.size))
    for i in range(P["n_sim"]):
        L_sim[i], _ = _L_nnd(_uniform_in_mask(pix_iy, pix_ix, xe, ye, n, rng), area, R)
    L_null = L_sim.mean(0)
    di = float((L_obs[band] / L_null[band]).mean() - 1)
    loo = (L_sim.sum(0) - L_sim) / (P["n_sim"] - 1)
    di_sim = (L_sim[:, band] / loo[:, band]).mean(1) - 1
    return dict(key=key, n_cells=n_full, DI=di, p_clustered=float((1 + (di_sim >= di).sum()) / (P["n_sim"] + 1)))


def _run_jobs(jobs, labels, xe, ye, xy, P, workers):
    R = np.arange(P["r_step"], P["r_max"] + 1e-9, P["r_step"])
    state = dict(labels=labels, xe=xe, ye=ye, xy=xy, P=P, R=R, band=(R >= P["r_min"]) & (R <= P["r_max"]))
    if workers <= 1 or len(jobs) <= 1:
        _init(state)
        return {r["key"]: r for r in map(_di_job, jobs)}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(state,)) as ex:
        return {r["key"]: r for r in ex.map(_di_job, jobs, chunksize=1)}


# --------------------------------------------------------------------------- tree
def build_tree(M: pd.DataFrame):
    """Read the exact tree encoded by an ultrametric (cophenetic) StructureMap matrix."""
    D = M.to_numpy(float)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    tol = 1e-6 * max(1.0, float(D.max()))
    viol = float(np.max(D[:, None, :] - np.maximum(D[:, :, None], D[None, :, :])))
    if viol > tol:
        raise ValueError(f"StructureMap matrix is not ultrametric (violation {viol:.3g}); "
                         "the cophenetic row_coph matrix is required")
    Z = linkage(squareform(D, checks=False), method="single")
    root, nodes = to_tree(Z, rd=True)
    labels = list(M.index)
    info: dict[int, dict[str, Any]] = {}
    for nd in nodes:
        info[nd.id] = dict(id=nd.id, height=float(nd.dist), leaves=[labels[i] for i in nd.pre_order()],
                           children=[] if nd.is_leaf() else [nd.get_left().id, nd.get_right().id])
    internal = sorted([k for k, v in info.items() if v["children"]], key=lambda k: (-info[k]["height"], k))
    for rank, k in enumerate(internal, 1):
        info[k]["name"] = f"N{rank}"
    for v in info.values():
        if not v["children"]:
            v["name"] = f"C{v['leaves'][0]}"
    return Z, root.id, info


def _prepare_matrix(row_coph: pd.DataFrame, clusters_present: set[str]) -> pd.DataFrame:
    M = row_coph.copy()
    M.index = [_norm_label(i) for i in M.index]
    M.columns = [_norm_label(c) for c in M.columns]
    if set(M.index) != set(M.columns):
        raise ValueError("StructureMap matrix must be a square cluster x cluster matrix")
    keep = [c for c in M.index if c in clusters_present]
    missing = sorted(clusters_present - set(M.index))
    if missing:
        raise ValueError(f"Clusters without a StructureMap row: {', '.join(missing)}")
    if len(keep) < 2:
        raise ValueError("Need at least two clusters to split")
    return M.loc[keep, keep].astype(float)


# --------------------------------------------------------------------------- main entry
SPLIT_MODES = ("threshold", "n_structures")
EXPLORATION_FILE = "autosplit_exploration.pkl"


@dataclasses.dataclass
class Exploration:
    """Result of the (expensive) top-down exploration; cutting the tree only needs this."""

    M: pd.DataFrame
    baseline: pd.Series
    nodes: pd.DataFrame
    long: pd.DataFrame
    contrib: dict[str, pd.Series]
    params: dict[str, Any]

    def save(self, path: Path) -> Path:
        with open(path, "wb") as fh:
            pickle.dump(dataclasses.asdict(self), fh)
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> "Exploration":
        with open(path, "rb") as fh:
            return cls(**pickle.load(fh))


def explore_tree(
    cells: pd.DataFrame,
    row_coph: pd.DataFrame,
    partition_fn: PartitionFn,
    *,
    params: dict[str, Any] | None = None,
    workers: int = 1,
    x_col: str = "x_centroid",
    y_col: str = "y_centroid",
    progress: Callable[[float, str], None] | None = None,
) -> Exploration:
    """Baseline DI and ΔDI of every StructureMap branch point (HistoSeg re-partition after every split)."""
    P = {**QC_PARAMS, **(params or {})}
    say = progress or (lambda frac, msg: print(f"[auto_split] {msg}", flush=True))
    cl_arr = cells["cluster"].map(_norm_label).to_numpy().astype(str)
    xy = cells[[x_col, y_col]].to_numpy(float)
    M = _prepare_matrix(row_coph, set(cl_arr))
    Z, root, info = build_tree(M)
    n_internal = sum(1 for v in info.values() if v["children"])
    total_steps = n_internal + 1

    say(0.0, "Partitioning the whole tissue (all clusters in one structure)")
    labels_t, xe_t, ye_t, _ = partition_fn([list(M.index)])
    tissue_labels = (np.asarray(labels_t) > 0).astype(np.int32)
    say(0.5 / total_steps, f"Baseline DI of {len(M.index)} clusters in the whole tissue")
    jobs = [(f"tissue|C{c}", np.flatnonzero(cl_arr == c), "tissue") for c in M.index]
    res = _run_jobs(jobs, tissue_labels, xe_t, ye_t, xy, P, workers)
    baseline = pd.Series({c: res[f"tissue|C{c}"]["DI"] for c in M.index}, name="DI_baseline_tissue")
    current = baseline.copy()

    # The partition is competitive: a group's contour moves when other groups are split. "Before" is therefore
    # re-measured in the partition that exists right before v is split (the previous step's partition).
    frontier = [root]
    prev = None
    node_rows, long_rows, contrib = [], [], {}
    step = 0
    while True:
        cands = [n for n in frontier if info[n]["children"]]
        if not cands:
            break
        v = max(cands, key=lambda n: (info[n]["height"], -n))
        step += 1
        frontier = [n for n in frontier if n != v] + info[v]["children"]
        groups = [info[n]["leaves"] for n in frontier]
        say(step / total_steps, f"Branch point {info[v]['name']} ({step}/{n_internal}): HistoSeg partition "
                                f"with {len(groups)} structures")
        labels, xe, ye, assign = partition_fn(groups)
        labels, assign = np.asarray(labels), np.asarray(assign)
        gid_of = {c: gid for gid, g in enumerate(groups, 1) for c in g}
        if prev is not None:
            p_labels, p_xe, p_ye, p_assign, p_gid_of = prev
            before_jobs = [(f"{info[v]['name']}|before|C{c}",
                            np.flatnonzero((cl_arr == c) & (p_assign == p_gid_of[c])), p_gid_of[c])
                           for c in info[v]["leaves"]]
            res_before = _run_jobs(before_jobs, p_labels, p_xe, p_ye, xy, P, workers)
            for c in info[v]["leaves"]:
                current[c] = res_before[f"{info[v]['name']}|before|C{c}"]["DI"]
        jobs = [(f"{info[v]['name']}|C{c}", np.flatnonzero((cl_arr == c) & (assign == gid_of[c])), gid_of[c])
                for c in info[v]["leaves"]]
        res = _run_jobs(jobs, labels, xe, ye, xy, P, workers)
        prev = (labels, xe, ye, assign, gid_of)
        child_of = {c: info[ch]["name"] for ch in info[v]["children"] for c in info[ch]["leaves"]}
        d, befores, afters = {}, [], []
        for c in info[v]["leaves"]:
            r = res[f"{info[v]['name']}|C{c}"]
            before, after = float(current[c]), float(r["DI"])
            d[c] = before - after
            befores.append(before)
            afters.append(after)
            n_c = int((cl_arr == c).sum())
            long_rows.append(dict(node=info[v]["name"], cluster=c, child_branch=child_of[c], DI_before=before,
                                  DI_after=after, dDI=d[c], p_clustered_after=r["p_clustered"],
                                  n_cluster=n_c, n_in_child_contour=int(r["n_cells"]),
                                  frac_in_child_contour=r["n_cells"] / max(n_c, 1)))
            current[c] = after
        contrib[info[v]["name"]] = pd.Series(d)
        node_rows.append(dict(node=info[v]["name"], exploration_step=step, height=info[v]["height"],
                              n_clusters=len(info[v]["leaves"]), clusters=",".join(info[v]["leaves"]),
                              children=" | ".join(",".join(info[ch]["leaves"]) for ch in info[v]["children"]),
                              sum_DI_before=float(np.nansum(befores)), sum_DI_after=float(np.nansum(afters)),
                              dDI=float(np.nansum(list(d.values())))))
    nodes = pd.DataFrame(node_rows)
    nodes["dDI_rank"] = nodes.dDI.rank(ascending=False, method="first").astype(int)
    return Exploration(M=M, baseline=baseline, nodes=nodes, long=pd.DataFrame(long_rows), contrib=contrib, params=P)


def _decide_threshold(info, root, ddi, contrib, min_ddi, descendant_rule):
    def internal_desc(k):
        out = []
        for ch in info[k]["children"]:
            if info[ch]["children"]:
                out += [ch] + internal_desc(ch)
        return out

    decisions: dict[str, dict[str, Any]] = {}
    final_groups: list[dict[str, Any]] = []
    extract_rows: list[dict[str, Any]] = []

    def resolve(k):
        name = info[k]["name"]
        if not info[k]["children"]:
            final_groups.append(dict(source=name, clusters=list(info[k]["leaves"]), kind="single cluster"))
            return
        qualifying = [q for q in internal_desc(k) if ddi[info[q]["name"]] >= min_ddi]
        if ddi[name] >= min_ddi or (descendant_rule == "split_parent" and qualifying):
            decisions[name] = dict(decision="split", reason="own dDI >= threshold" if ddi[name] >= min_ddi
                                   else "descendant " + info[qualifying[0]]["name"] + " dDI >= threshold")
            for ch in info[k]["children"]:
                resolve(ch)
            return
        remaining, extracted = list(info[k]["leaves"]), []
        if descendant_rule == "extract_top_branch":
            for q in sorted(qualifying, key=lambda q: -info[q]["height"]):
                cq = contrib[info[q]["name"]]
                branches = [(float(cq.loc[info[ch]["leaves"]].sum()), ch) for ch in info[q]["children"]
                            if set(info[ch]["leaves"]) <= set(remaining)]
                if not branches:
                    continue
                score, ch = max(branches, key=lambda b: b[0])
                leaves = info[ch]["leaves"]
                if len(leaves) >= len(remaining):
                    continue
                for c in leaves:
                    remaining.remove(c)
                extracted.append(info[ch]["name"])
                extract_rows.append(dict(parent=name, parent_dDI=ddi[name], qualifying_descendant=info[q]["name"],
                                         descendant_dDI=ddi[info[q]["name"]], extracted_branch=info[ch]["name"],
                                         extracted_clusters=",".join(leaves), branch_contribution=score,
                                         branch_contributions="; ".join(
                                             f"{{{','.join(info[c2]['leaves'])}}}: {float(cq.loc[info[c2]['leaves']].sum()):+.3f}"
                                             for c2 in info[q]["children"])))
                final_groups.append(dict(source=info[ch]["name"], clusters=list(leaves),
                                         kind=f"branch extracted from {name} (via {info[q]['name']})"))
        reason = "no branch point below reaches the threshold" if not qualifying else (
            "descendant(s) " + ",".join(info[q]["name"] for q in qualifying) + " reach the threshold"
            + (f"; extracted {','.join(extracted)}" if extracted else "; parent kept (rule: stop)"))
        decisions[name] = dict(decision="keep", reason=reason)
        final_groups.append(dict(source=name + "".join(f"-{e}" for e in extracted), clusters=remaining,
                                 kind="kept branch" + (" minus extracted branches" if extracted else "")))

    resolve(root)
    return decisions, final_groups, extract_rows


def _decide_n_structures(info, root, nodes, n_structures):
    """Select branch points by descending ΔDI; a selected branch point pulls in its unselected ancestors."""
    by_name = {v["name"]: k for k, v in info.items()}
    parent = {ch: k for k, v in info.items() for ch in v["children"]}
    target = int(n_structures) - 1
    selected: dict[int, str] = {}
    for r in nodes.sort_values(["dDI", "height"], ascending=[False, False]).itertuples():
        if len(selected) >= target:
            break
        k = by_name[r.node]
        if k in selected:
            continue
        need, a = [k], parent.get(k)
        while a is not None and a not in selected:
            need.append(a)
            a = parent.get(a)
        if len(selected) + len(need) > target:
            continue
        selected[k] = f"ΔDI rank {r.dDI_rank}"
        for anc in need[1:]:
            selected[anc] = f"ancestor of {r.node} (ΔDI rank {r.dDI_rank})"

    decisions: dict[str, dict[str, Any]] = {}
    final_groups: list[dict[str, Any]] = []

    def resolve(k):
        name = info[k]["name"]
        if not info[k]["children"]:
            final_groups.append(dict(source=name, clusters=list(info[k]["leaves"]), kind="single cluster"))
            return
        if k in selected:
            decisions[name] = dict(decision="split", reason=selected[k])
            for ch in info[k]["children"]:
                resolve(ch)
            return
        decisions[name] = dict(decision="keep", reason="not among the selected top-ΔDI branch points")
        final_groups.append(dict(source=name, clusters=list(info[k]["leaves"]), kind="kept branch"))

    resolve(root)
    return decisions, final_groups, []


def cut_tree(
    exploration: Exploration,
    cells: pd.DataFrame,
    partition_fn: PartitionFn,
    out_dir: Path,
    *,
    mode: str = "threshold",
    min_ddi: float = 1.0,
    descendant_rule: str = "extract_top_branch",
    n_structures: int = 5,
    workers: int = 1,
    x_col: str = "x_centroid",
    y_col: str = "y_centroid",
    progress: Callable[[float, str], None] | None = None,
) -> AutoSplitResult:
    """Cut the explored tree (by ΔDI threshold or by number of structures), re-partition and report."""
    if mode not in SPLIT_MODES:
        raise ValueError(f"mode must be one of {SPLIT_MODES}")
    if descendant_rule not in DESCENDANT_RULES:
        raise ValueError(f"descendant_rule must be one of {DESCENDANT_RULES}")
    say = progress or (lambda frac, msg: print(f"[auto_split] {msg}", flush=True))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ex = exploration
    P, M, baseline = ex.params, ex.M, ex.baseline
    cl_arr = cells["cluster"].map(_norm_label).to_numpy().astype(str)
    xy = cells[[x_col, y_col]].to_numpy(float)
    Z, root, info = build_tree(M)
    nodes = ex.nodes.drop(columns=[c for c in ("decision", "reason", "passes_threshold") if c in ex.nodes]).copy()
    ddi = dict(zip(nodes.node, nodes.dDI))
    n_leaves = len(M.index)

    if mode == "threshold":
        decisions, final_groups, extract_rows = _decide_threshold(info, root, ddi, ex.contrib, min_ddi, descendant_rule)
        mode_text = f"ΔDI threshold {min_ddi:g}; {_RULE_TEXT[descendant_rule]}"
    else:
        n_structures = int(min(max(1, n_structures), n_leaves))
        decisions, final_groups, extract_rows = _decide_n_structures(info, root, nodes, n_structures)
        mode_text = (f"{n_structures} structures: top-ranked ΔDI branch points "
                     "(a selected branch point also splits its ancestors)")
    nodes["decision"] = nodes.node.map(lambda n: decisions.get(n, {}).get("decision", "not reached"))
    nodes["reason"] = nodes.node.map(lambda n: decisions.get(n, {}).get("reason", "inside a kept branch"))
    nodes["passes_threshold"] = nodes.dDI >= min_ddi

    order = sorted(range(len(final_groups)), key=lambda i: (-len(final_groups[i]["clusters"]), final_groups[i]["source"]))
    final_groups = [final_groups[i] for i in order]
    groups = [g["clusters"] for g in final_groups]
    say(0.1, f"Final HistoSeg partition with {len(groups)} structures and DI of every cluster")
    labels_f, xe_f, ye_f, assign_f = partition_fn(groups)
    assign_f = np.asarray(assign_f)
    jobs = [(f"final|C{c}", np.flatnonzero((cl_arr == c) & (assign_f == gid)), gid)
            for gid, g in enumerate(groups, 1) for c in g]
    res = _run_jobs(jobs, np.asarray(labels_f), xe_f, ye_f, xy, P, workers)
    crow = []
    for gid, g in enumerate(final_groups, 1):
        for c in g["clusters"]:
            r = res[f"final|C{c}"]
            n_c = int((cl_arr == c).sum())
            crow.append(dict(cluster=c, final_structure=f"Structure {gid}", n_cells=n_c,
                             DI_before_tissue=float(baseline[c]), DI_after_final=r["DI"],
                             dDI=float(baseline[c]) - r["DI"], p_clustered_after=r["p_clustered"],
                             frac_in_own_contour=r["n_cells"] / max(n_c, 1)))
    clusters = pd.DataFrame(crow)
    srow = []
    for gid, g in enumerate(final_groups, 1):
        sub = clusters[clusters.final_structure == f"Structure {gid}"]
        srow.append(dict(structure_id=gid, structure_name=f"Structure {gid}", source=g["source"], kind=g["kind"],
                         cluster_ids=", ".join(g["clusters"]), n_clusters=len(g["clusters"]),
                         n_cells=int(sub.n_cells.sum()), sum_DI_before_tissue=float(sub.DI_before_tissue.sum()),
                         sum_DI_after_final=float(np.nansum(sub.DI_after_final)),
                         max_DI_after_final=float(np.nanmax(sub.DI_after_final)) if len(sub) else np.nan))
    structures = pd.DataFrame(srow)
    structure_lines = "\n".join(",".join(g["clusters"]) for g in final_groups)

    long = ex.long
    files = []
    for df, name in ((nodes, "autosplit_branch_points.csv"), (long, "autosplit_branch_point_cluster_DI.csv"),
                     (clusters, "autosplit_cluster_DI_before_after.csv"), (structures, "autosplit_structures.csv"),
                     (pd.DataFrame(extract_rows), "autosplit_extracted_branches.csv")):
        df.to_csv(out_dir / name, index=False)
        files.append(out_dir / name)
    (out_dir / "autosplit_histoseg_structures.txt").write_text(structure_lines + "\n", encoding="utf-8")
    np.savez_compressed(out_dir / "autosplit_partition_labels.npz", labels=labels_f, x_edges=xe_f, y_edges=ye_f)
    s0, s1 = float(np.nansum(baseline)), float(np.nansum(clusters.DI_after_final))
    with open(out_dir / "autosplit_result.json", "w", encoding="utf-8") as fh:
        json.dump(dict(mode=mode, mode_text=mode_text, min_dDI=min_ddi, descendant_rule=descendant_rule,
                       n_structures=len(final_groups), params=P, sum_DI_baseline=s0, sum_DI_final=s1,
                       structures=[dict(structure_id=i, cluster_ids=g["clusters"], source=g["source"], kind=g["kind"])
                                   for i, g in enumerate(final_groups, 1)],
                       extracted_branches=extract_rows),
                  fh, indent=2, ensure_ascii=False, default=_json_default)
    files += [out_dir / "autosplit_histoseg_structures.txt", out_dir / "autosplit_partition_labels.npz",
              out_dir / "autosplit_result.json"]
    say(0.8, "Writing report and figures")
    dendro_png = _plot_dendrogram(out_dir, M, Z, info, nodes, final_groups, mode_text)
    part_png = _plot_partition(out_dir, cells, x_col, y_col, assign_f, final_groups)
    bars_png = _plot_before_after(out_dir, clusters)
    report = _write_report(out_dir, nodes, long, clusters, structures, pd.DataFrame(extract_rows), structure_lines,
                           mode_text, mode, min_ddi, P, s0, s1)
    files += [dendro_png, part_png, bars_png, report]
    if (out_dir / EXPLORATION_FILE).exists():
        files.append(out_dir / EXPLORATION_FILE)
    say(1.0, "Automatic split finished")
    return AutoSplitResult(out_dir, nodes, long, clusters, structures, structure_lines, report, dendro_png, part_png,
                           files, s0, s1, bars_png)


def run_auto_split(
    cells: pd.DataFrame,
    row_coph: pd.DataFrame,
    partition_fn: PartitionFn,
    out_dir: Path,
    *,
    mode: str = "threshold",
    min_ddi: float = 1.0,
    descendant_rule: str = "extract_top_branch",
    n_structures: int = 5,
    params: dict[str, Any] | None = None,
    workers: int = 1,
    x_col: str = "x_centroid",
    y_col: str = "y_centroid",
    progress: Callable[[float, str], None] | None = None,
    exploration: Exploration | None = None,
) -> AutoSplitResult:
    """Explore every StructureMap branch point (unless ``exploration`` is given), then cut the tree.

    ``partition_fn(groups)`` must run HistoSeg with one structure per cluster group (structure ids 1..len(groups)
    in the given order) and return ``(partition_labels, x_edges, y_edges, isoline_structure_id per cell)``.
    ``mode="threshold"`` cuts at ``min_ddi`` (with ``descendant_rule``); ``mode="n_structures"`` splits the
    top-ranked ΔDI branch points (plus their ancestors) until ``n_structures`` structures exist.
    """
    say = progress or (lambda frac, msg: print(f"[auto_split] {msg}", flush=True))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if exploration is None:
        exploration = explore_tree(cells, row_coph, partition_fn, params=params, workers=workers, x_col=x_col,
                                   y_col=y_col, progress=lambda f, m: say(0.9 * f, m))
        exploration.save(out_dir / EXPLORATION_FILE)
        cut_progress = lambda f, m: say(0.9 + 0.1 * f, m)  # noqa: E731
    else:
        cut_progress = say
    return cut_tree(exploration, cells, partition_fn, out_dir, mode=mode, min_ddi=min_ddi,
                    descendant_rule=descendant_rule, n_structures=n_structures, workers=workers, x_col=x_col,
                    y_col=y_col, progress=cut_progress)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    return str(o)


# --------------------------------------------------------------------------- report & figures
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cd6", "#8c564b",
            "#17becf", "#bcbd22", "#6b6a64", "#d62728"]
_RULE_TEXT = {
    "extract_top_branch": "failing parent with a qualifying descendant: keep the parent, extract the top-ΔDI child branch",
    "split_parent": "failing parent with a qualifying descendant: split the parent as well",
    "stop": "failing parent: keep it, do not look below",
}


def _f(x, fmt="{:.2f}"):
    try:
        return "—" if x is None or np.isnan(float(x)) else fmt.format(float(x))
    except (TypeError, ValueError):
        return str(x)


def _write_report(out_dir, nodes, long, clusters, structures, extracted, lines, mode_text, mode, t, P, s0, s1) -> Path:
    L = ["# Automatic structure split by ΔDI", "",
         f"- Split rule: **{mode_text}**",
         f"- DI = mean over {int(P['r_min'])}–{int(P['r_max'])} µm of L_obs / L_CSR − 1; "
         f"{P['n_sim']} Monte Carlo simulations; up to {P['max_points']} points per test",
         f"- Σ DI over clusters: {s0:.2f} (whole tissue) → {s1:.2f} (final structures)", "",
         "## Final structures (paste into the cluster-ID box, one structure per line)", "", "```", lines, "```", "",
         "| Structure | Source | Kind | Cluster IDs | Cells | Σ DI before (tissue) | Σ DI after | Max DI after |",
         "|---|---|---|---|---|---|---|---|"]
    for r in structures.itertuples():
        L.append(f"| {r.structure_name} | {r.source} | {r.kind} | {r.cluster_ids} | {r.n_cells:,} | "
                 f"{_f(r.sum_DI_before_tissue)} | {_f(r.sum_DI_after_final)} | {_f(r.max_DI_after_final)} |")
    L += ["", "## ΔDI of every branch point", "",
          "| Branch point | Height | Children | Σ DI before | Σ DI after | ΔDI | ΔDI rank | "
          + ("≥ threshold | " if mode == "threshold" else "") + "Decision | Reason |",
          "|---|---|---|---|---|---|---|" + ("---|" if mode == "threshold" else "") + "---|---|"]
    for r in nodes.itertuples():
        L.append(f"| {r.node} | {r.height:.3f} | {r.children} | {_f(r.sum_DI_before)} | {_f(r.sum_DI_after)} | "
                 f"**{_f(r.dDI)}** | {r.dDI_rank} | "
                 + (f"{'yes' if r.passes_threshold else 'no'} | " if mode == "threshold" else "")
                 + f"{r.decision} | {r.reason} |")
    if len(extracted):
        L += ["", "## Extracted branches", "",
              "| Parent | Parent ΔDI | Qualifying descendant | Descendant ΔDI | Extracted | Branch contributions |",
              "|---|---|---|---|---|---|"]
        for r in extracted.itertuples():
            L.append(f"| {r.parent} | {_f(r.parent_dDI)} | {r.qualifying_descendant} | {_f(r.descendant_dDI)} | "
                     f"{{{r.extracted_clusters}}} | {r.branch_contributions} |")
    L += ["", "## DI of every cluster before (whole tissue) and after (final structure)", "",
          "| Cluster | Final structure | Cells | DI before | DI after | ΔDI | Fraction in own contour |",
          "|---|---|---|---|---|---|---|"]
    for r in clusters.sort_values("DI_before_tissue", ascending=False).itertuples():
        L.append(f"| C{r.cluster} | {r.final_structure} | {r.n_cells:,} | {_f(r.DI_before_tissue)} | "
                 f"{_f(r.DI_after_final)} | {_f(r.dDI)} | {r.frac_in_own_contour:.0%} |")
    L += ["", "## DI of each cluster at every branch-point split", "",
          "| Branch point | Cluster | Child branch | DI before | DI after | ΔDI |", "|---|---|---|---|---|---|"]
    for r in long.itertuples():
        L.append(f"| {r.node} | C{r.cluster} | {r.child_branch} | {_f(r.DI_before)} | {_f(r.DI_after)} | {_f(r.dDI)} |")
    L += ["", "Notes: ΔDI of a branch point is evaluated during a full top-down exploration (HistoSeg is re-run after "
          "every split); DI is computed on the cells of each cluster that fall inside its own contour."]
    path = Path(out_dir) / "autosplit_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def _plot_dendrogram(out_dir, M, Z, info, nodes, final_groups, mode_text) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_of = {c: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(final_groups) for c in g["clusters"]}
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(M) + 4), 6.5))
    dn = dendrogram(Z, labels=[f"C{c}" for c in M.index], ax=ax, color_threshold=0,
                    above_threshold_color="#b0b0a8", leaf_font_size=10)
    for tick in ax.get_xticklabels():
        tick.set_color(color_of[tick.get_text()[1:]])
        tick.set_fontweight("bold")
    xpos = {lab[1:]: 5 + 10 * i for i, lab in enumerate(dn["ivl"])}

    def node_x(n):
        return xpos[n["leaves"][0]] if not n["children"] else (node_x(info[n["children"][0]]) + node_x(info[n["children"][1]])) / 2

    by_name = {v["name"]: v for v in info.values()}
    style = {"split": ("o", "#d03b3b"), "keep": ("X", "#4a4a45"), "not reached": ("o", "#d6d5cf")}
    for r in nodes.itertuples():
        v = by_name[r.node]
        marker, col = style[r.decision]
        ax.scatter(node_x(v), v["height"], s=120, marker=marker, color=col, edgecolor="white", lw=1, zorder=5)
        ax.annotate(f"{r.node} (#{r.dDI_rank})\nΔ{r.dDI:+.2f}", (node_x(v), v["height"]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7.5, color="#2b2b28" if r.decision != "not reached" else "#9a9990")
    ax.set_ylim(0, max(1.0, max(v["height"] for v in info.values())) * 1.13)
    ax.set_ylabel("cophenetic height (StructureMap)")
    ax.set_title(f"ΔDI (and rank #) of every branch point; {mode_text}\n"
                 "● red = split   ✕ = kept together   grey = inside a kept branch; leaf colour = final structure",
                 loc="left", fontsize=9.5, pad=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    path = Path(out_dir) / "autosplit_dendrogram.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_before_after(out_dir, clusters: pd.DataFrame) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = clusters.copy()
    t["_sid"] = t.final_structure.str.extract(r"(\d+)$")[0].astype(int)
    t["_ck"] = t.cluster.map(_cluster_sort_key)
    t = t.sort_values(["_sid", "_ck"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(t) + 3), 4.2))
    plot_cluster_di_before_after(ax, t, "DI_before_tissue", "DI_after_final", "final_structure",
                                 "DI of every cluster: original (whole tissue) vs final (inside its split structure); "
                                 "dashed = 0.3")
    fig.tight_layout()
    path = Path(out_dir) / "autosplit_cluster_DI_before_after.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_partition(out_dir, cells, x_col, y_col, assign, final_groups) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(0)
    take = rng.choice(len(cells), min(120000, len(cells)), replace=False)
    colors = np.array(["#dddddd"] + [_PALETTE[i % len(_PALETTE)] for i in range(len(final_groups))])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(cells[x_col].to_numpy()[take], cells[y_col].to_numpy()[take], s=0.2, lw=0,
               c=colors[np.clip(assign[take], 0, len(final_groups))], rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel(f"{x_col} (µm)")
    ax.set_ylabel(f"{y_col} (µm)")
    ax.set_title("HistoSeg partition of the automatically split structures", loc="left", fontsize=10)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=_PALETTE[i % len(_PALETTE)],
                              label=f"Structure {i + 1}: {', '.join(g['clusters'])}") for i, g in enumerate(final_groups)],
              frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    path = Path(out_dir) / "autosplit_partition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
