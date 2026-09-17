from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from histoseg.spatial_pathologist.auto_split import build_tree, run_auto_split

XE = np.linspace(0.0, 200.0, 41)
YE = np.linspace(0.0, 100.0, 21)


def _cells(seed: int = 0) -> pd.DataFrame:
    """A, B uniformly mixed on the left half; C packed top-right; D packed bottom-right."""
    rng = np.random.default_rng(seed)
    a = np.column_stack([rng.uniform(0, 100, 400), rng.uniform(0, 100, 400)])
    b = np.column_stack([rng.uniform(0, 100, 400), rng.uniform(0, 100, 400)])
    c = np.column_stack([rng.uniform(100, 200, 400), rng.uniform(0, 50, 400)])
    d = np.column_stack([rng.uniform(100, 200, 400), rng.uniform(50, 100, 400)])
    xy = np.vstack([a, b, c, d])
    return pd.DataFrame({"x_centroid": xy[:, 0], "y_centroid": xy[:, 1],
                         "cluster": ["A"] * 400 + ["B"] * 400 + ["C"] * 400 + ["D"] * 400})


def _structuremap() -> pd.DataFrame:
    labels = ["A", "B", "C", "D"]
    coph = np.array([[0, .1, 1, 1], [.1, 0, 1, 1], [1, 1, 0, .4], [1, 1, .4, 0]], dtype=float)
    return pd.DataFrame(coph, index=labels, columns=labels)


def _partition_fn(cells):
    """Synthetic 'HistoSeg': each group owns the half/quadrant where its clusters live."""
    xs, ys = np.meshgrid(0.5 * (XE[:-1] + XE[1:]), 0.5 * (YE[:-1] + YE[1:]))
    region = {"A": (xs < 100) & (ys < 50), "B": (xs < 100) & (ys >= 50),
              "C": (xs >= 100) & (ys < 50), "D": (xs >= 100) & (ys >= 50)}

    def fn(groups):
        labels = np.zeros(xs.shape, dtype=np.int32)
        for gid, g in enumerate(groups, 1):
            for c in g:
                labels[region[c]] = gid
        ix = np.clip(np.searchsorted(XE, cells.x_centroid, side="right") - 1, 0, 39)
        iy = np.clip(np.searchsorted(YE, cells.y_centroid, side="right") - 1, 0, 19)
        return labels, XE, YE, labels[iy, ix]

    return fn


QUICK = dict(n_sim=19, r_min=10.0, r_max=30.0, r_step=5.0, max_points=5000)


@pytest.mark.parametrize("rule", ["extract_top_branch", "split_parent", "stop"])
def test_auto_split_splits_segregated_branches_only(tmp_path, rule):
    cells = _cells()
    result = run_auto_split(cells, _structuremap(), _partition_fn(cells), tmp_path / rule, min_ddi=0.3,
                            descendant_rule=rule, params=QUICK, workers=1, progress=lambda f, m: None)
    ddi = result.nodes.set_index("node").dDI
    assert ddi["N1"] > 1.0          # {A,B} vs {C,D}: strong segregation
    assert ddi["N2"] > 0.3          # C vs D: top/bottom segregation
    assert ddi["N3"] < 0.3          # A vs B: uniformly mixed, splitting does not help
    lines = sorted(result.structure_lines.splitlines())
    assert lines == ["A,B", "C", "D"] or lines == ["B,A", "C", "D"]
    assert result.sum_di_final < result.sum_di_baseline
    assert set(result.clusters.columns) >= {"DI_before_tissue", "DI_after_final", "dDI"}
    assert set(result.node_cluster_di.node) == {"N1", "N2", "N3"}
    for f in result.files:
        assert f.exists()
    assert "ΔDI of every branch point" in result.report_md.read_text(encoding="utf-8")


def test_build_tree_rejects_non_ultrametric():
    M = pd.DataFrame([[0, 1, 2], [1, 0, 5], [2, 5, 0]], index=list("abc"), columns=list("abc"), dtype=float)
    with pytest.raises(ValueError, match="ultrametric"):
        build_tree(M)
