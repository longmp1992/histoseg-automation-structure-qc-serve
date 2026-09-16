"""Per-gland 3D quality-control atlas for aligned HistoSeg contour stacks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon, shape


PathLike = Union[str, Path]


@dataclass(frozen=True)
class GlandQCAtlasConfig:
    """Configuration for a local, component-tracked gland QC atlas."""

    stack_root: PathLike
    out_dir: PathLike
    aligned_cells_parquet: PathLike | None = None
    structures: Sequence[str] = ()
    group_property: str = "structure"
    pixel_size_um: float = 0.2125
    padding_um: float = 250.0
    z_visual_scale: float = 8.0
    max_cells_per_gland: int = 50000
    max_gland_pages: int | None = None
    label_column: str = "leiden_1_0"
    random_state: int = 0
    min_overlap_fraction: float = 0.05
    good_overlap_fraction: float = 0.20
    centroid_fallback_um: float = 160.0
    max_area_ratio_for_fallback: float = 4.0
    centroid_jump_review_um: float = 350.0
    area_cv_review_threshold: float = 1.0


@dataclass
class GlandQCAtlasResult:
    out_dir: Path
    atlas_html: Path
    gland_tracks_csv: Path
    gland_qc_index_csv: Path
    summary_json: Path
    gland_count: int
    component_count: int
    structure_count: int
    rendered_gland_pages: int


@dataclass(frozen=True)
class _GlandComponent:
    node_id: str
    feature_index: int
    slice_order: int
    sample_id: str
    z_um: float
    structure: str
    structure_id: str
    component_index: str
    polygon_index: str
    feature_name: str
    geometry_um: Any
    area_um2: float
    centroid_x_um: float
    centroid_y_um: float
    min_x_um: float
    min_y_um: float
    max_x_um: float
    max_y_um: float


@dataclass(frozen=True)
class _GlandLink:
    source: str
    target: str
    overlap_fraction: float
    centroid_distance_um: float
    area_ratio: float
    link_type: str
    weak: bool


@dataclass(frozen=True)
class _CellIndex:
    cells: pd.DataFrame
    bins: Mapping[tuple[int, int], np.ndarray]
    bin_size_um: float


def render_gland_qc_atlas(cfg: GlandQCAtlasConfig) -> GlandQCAtlasResult:
    """Render a per-gland 3D QC atlas from an existing aligned contour stack."""

    _validate_config(cfg)
    stack_root = Path(cfg.stack_root).expanduser().resolve()
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gland_dir = out_dir / "glands"
    gland_dir.mkdir(parents=True, exist_ok=True)

    components = _load_gland_components(cfg)
    if not components:
        raise ValueError("No aligned contour components matched the requested structures.")

    candidate_counts, links = _build_neighbor_links(components, cfg)
    gland_ids = _assign_gland_ids(components, links)
    tracks = _build_tracks_table(components, links, gland_ids, candidate_counts)
    index = _build_gland_index(tracks, links, cfg)

    tracks_csv = out_dir / "gland_tracks.csv"
    index_csv = out_dir / "gland_qc_index.csv"
    tracks.to_csv(tracks_csv, index=False)

    cells = _load_optional_cells(cfg)
    components_by_id = {component.node_id: component for component in components}
    rendered_pages = 0
    page_index = index.sort_values(["review_priority", "gland_id"])
    if cfg.max_gland_pages is not None:
        page_index = page_index.head(int(cfg.max_gland_pages))
    rendered_gland_ids = set(page_index["gland_id"].astype(str).tolist())
    for _, gland in page_index.iterrows():
        gland_components = [
            components_by_id[node_id]
            for node_id in tracks.loc[tracks["gland_id"] == gland["gland_id"], "node_id"]
            if node_id in components_by_id
        ]
        if not gland_components:
            continue
        page = gland_dir / f"{gland['gland_id']}.html"
        _write_gland_page(
            page,
            gland,
            gland_components,
            components,
            tracks,
            cells,
            cfg,
        )
        rendered_pages += 1

    index = index.copy()
    index["page_rendered"] = index["gland_id"].astype(str).isin(rendered_gland_ids)
    index.loc[~index["page_rendered"], "page"] = ""
    index.to_csv(index_csv, index=False)

    atlas_html = out_dir / "gland_qc_atlas.html"
    _write_atlas_page(atlas_html, index, cfg)

    summary = {
        "config": asdict(cfg),
        "stack_root": str(stack_root),
        "out_dir": str(out_dir),
        "component_count": int(len(components)),
        "gland_count": int(len(index)),
        "structure_count": int(index["structure"].nunique()) if not index.empty else 0,
        "rendered_gland_pages": int(rendered_pages),
        "max_gland_pages": cfg.max_gland_pages,
        "outputs": {
            "atlas_html": str(atlas_html),
            "gland_tracks_csv": str(tracks_csv),
            "gland_qc_index_csv": str(index_csv),
        },
    }
    summary_json = out_dir / "gland_qc_summary.json"
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return GlandQCAtlasResult(
        out_dir=out_dir,
        atlas_html=atlas_html,
        gland_tracks_csv=tracks_csv,
        gland_qc_index_csv=index_csv,
        summary_json=summary_json,
        gland_count=int(len(index)),
        component_count=int(len(components)),
        structure_count=int(index["structure"].nunique()) if not index.empty else 0,
        rendered_gland_pages=int(rendered_pages),
    )


def _load_gland_components(cfg: GlandQCAtlasConfig) -> list[_GlandComponent]:
    stack_root = Path(cfg.stack_root).expanduser().resolve()
    manifest_path = stack_root / "aligned_slice_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    required = {"order", "sample_id", "z_um", "aligned_geojson"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {missing}")

    requested = {str(item) for item in cfg.structures} if cfg.structures else None
    components: list[_GlandComponent] = []
    for _, row in manifest.sort_values("order").iterrows():
        order = int(row["order"])
        sample_id = str(row["sample_id"])
        z_um = float(row["z_um"])
        geojson_path = _resolve_path(row["aligned_geojson"], base=stack_root)
        payload = json.loads(geojson_path.read_text(encoding="utf-8"))
        for feature_index, feature in enumerate(payload.get("features", [])):
            props = feature.get("properties") or {}
            structure = _feature_structure(props, cfg.group_property)
            if requested is not None and structure not in requested:
                continue
            geom_payload = feature.get("geometry")
            if geom_payload is None:
                continue
            geom = shape(geom_payload)
            if geom.is_empty:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                continue
            geom_um = affinity.scale(
                geom,
                xfact=float(cfg.pixel_size_um),
                yfact=float(cfg.pixel_size_um),
                origin=(0.0, 0.0),
            )
            if geom_um.is_empty or geom_um.area <= 0:
                continue
            centroid = geom_um.centroid
            min_x, min_y, max_x, max_y = geom_um.bounds
            components.append(
                _GlandComponent(
                    node_id=f"{order:03d}_{feature_index:05d}",
                    feature_index=int(feature_index),
                    slice_order=order,
                    sample_id=sample_id,
                    z_um=z_um,
                    structure=structure,
                    structure_id=str(props.get("structure_id", "")),
                    component_index=str(props.get("component_index", "")),
                    polygon_index=str(props.get("polygon_index", "")),
                    feature_name=str(props.get("name", "")),
                    geometry_um=geom_um,
                    area_um2=float(geom_um.area),
                    centroid_x_um=float(centroid.x),
                    centroid_y_um=float(centroid.y),
                    min_x_um=float(min_x),
                    min_y_um=float(min_y),
                    max_x_um=float(max_x),
                    max_y_um=float(max_y),
                )
            )
    return components


def _build_neighbor_links(
    components: Sequence[_GlandComponent],
    cfg: GlandQCAtlasConfig,
) -> tuple[dict[str, dict[str, int]], list[_GlandLink]]:
    by_order_structure: dict[tuple[int, str], list[_GlandComponent]] = {}
    for component in components:
        by_order_structure.setdefault((component.slice_order, component.structure), []).append(component)

    candidate_counts: dict[str, dict[str, int]] = {
        component.node_id: {"next": 0, "prev": 0}
        for component in components
    }
    all_candidates: list[_GlandLink] = []
    orders = sorted({component.slice_order for component in components})
    for order in orders:
        next_order = order + 1
        structures = sorted(
            {
                structure
                for slice_order, structure in by_order_structure
                if slice_order in {order, next_order}
            }
        )
        for structure in structures:
            left = by_order_structure.get((order, structure), [])
            right = by_order_structure.get((next_order, structure), [])
            if not left or not right:
                continue
            overlap_sources: set[str] = set()
            overlap_targets: set[str] = set()
            for source in left:
                for target in right:
                    candidate = _candidate_link(source, target, cfg, allow_fallback=False)
                    if candidate is None:
                        continue
                    all_candidates.append(candidate)
                    overlap_sources.add(source.node_id)
                    overlap_targets.add(target.node_id)
                    candidate_counts[source.node_id]["next"] += 1
                    candidate_counts[target.node_id]["prev"] += 1
            for source in left:
                if source.node_id in overlap_sources:
                    continue
                for target in right:
                    if target.node_id in overlap_targets:
                        continue
                    candidate = _candidate_link(source, target, cfg, allow_fallback=True)
                    if candidate is None:
                        continue
                    all_candidates.append(candidate)
                    candidate_counts[source.node_id]["next"] += 1
                    candidate_counts[target.node_id]["prev"] += 1

    best_out: dict[str, _GlandLink] = {}
    best_in: dict[str, _GlandLink] = {}
    for candidate in all_candidates:
        if _link_better(candidate, best_out.get(candidate.source)):
            best_out[candidate.source] = candidate
        if _link_better(candidate, best_in.get(candidate.target)):
            best_in[candidate.target] = candidate

    links = [
        candidate
        for candidate in all_candidates
        if best_out.get(candidate.source) == candidate and best_in.get(candidate.target) == candidate
    ]
    return candidate_counts, links


def _candidate_link(
    source: _GlandComponent,
    target: _GlandComponent,
    cfg: GlandQCAtlasConfig,
    *,
    allow_fallback: bool,
) -> _GlandLink | None:
    intersection_area = float(source.geometry_um.intersection(target.geometry_um).area)
    min_area = max(min(source.area_um2, target.area_um2), 1e-9)
    overlap_fraction = intersection_area / min_area
    dx = source.centroid_x_um - target.centroid_x_um
    dy = source.centroid_y_um - target.centroid_y_um
    centroid_distance = math.sqrt(dx * dx + dy * dy)
    area_ratio = max(source.area_um2, target.area_um2) / min_area

    if overlap_fraction >= cfg.min_overlap_fraction:
        return _GlandLink(
            source=source.node_id,
            target=target.node_id,
            overlap_fraction=float(overlap_fraction),
            centroid_distance_um=float(centroid_distance),
            area_ratio=float(area_ratio),
            link_type="overlap",
            weak=bool(overlap_fraction < cfg.good_overlap_fraction),
        )
    if (
        allow_fallback
        and overlap_fraction <= 0
        and centroid_distance <= cfg.centroid_fallback_um
        and area_ratio <= cfg.max_area_ratio_for_fallback
    ):
        return _GlandLink(
            source=source.node_id,
            target=target.node_id,
            overlap_fraction=float(overlap_fraction),
            centroid_distance_um=float(centroid_distance),
            area_ratio=float(area_ratio),
            link_type="centroid_fallback",
            weak=True,
        )
    return None


def _link_better(candidate: _GlandLink, current: _GlandLink | None) -> bool:
    if current is None:
        return True
    key_candidate = (
        candidate.overlap_fraction,
        -candidate.centroid_distance_um,
        -candidate.area_ratio,
    )
    key_current = (
        current.overlap_fraction,
        -current.centroid_distance_um,
        -current.area_ratio,
    )
    return key_candidate > key_current


def _assign_gland_ids(
    components: Sequence[_GlandComponent],
    links: Sequence[_GlandLink],
) -> dict[str, str]:
    parent = {component.node_id: component.node_id for component in components}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for link in links:
        union(link.source, link.target)

    by_root: dict[str, list[_GlandComponent]] = {}
    by_node = {component.node_id: component for component in components}
    for component in components:
        by_root.setdefault(find(component.node_id), []).append(component)

    sorted_roots = sorted(
        by_root,
        key=lambda root: (
            by_root[root][0].structure,
            min(item.slice_order for item in by_root[root]),
            np.median([item.centroid_x_um for item in by_root[root]]),
            np.median([item.centroid_y_um for item in by_root[root]]),
        ),
    )
    gland_by_root = {
        root: f"gland_{index:04d}"
        for index, root in enumerate(sorted_roots, start=1)
    }
    return {
        node_id: gland_by_root[find(node_id)]
        for node_id in by_node
    }


def _build_tracks_table(
    components: Sequence[_GlandComponent],
    links: Sequence[_GlandLink],
    gland_ids: Mapping[str, str],
    candidate_counts: Mapping[str, Mapping[str, int]],
) -> pd.DataFrame:
    prev_links = {link.target: link for link in links}
    next_links = {link.source: link for link in links}
    rows: list[dict[str, Any]] = []
    for component in sorted(components, key=lambda item: (gland_ids[item.node_id], item.slice_order, item.node_id)):
        prev_link = prev_links.get(component.node_id)
        next_link = next_links.get(component.node_id)
        rows.append(
            {
                "gland_id": gland_ids[component.node_id],
                "node_id": component.node_id,
                "slice_order": component.slice_order,
                "sample_id": component.sample_id,
                "z_um": component.z_um,
                "structure": component.structure,
                "structure_id": component.structure_id,
                "feature_index": component.feature_index,
                "component_index": component.component_index,
                "polygon_index": component.polygon_index,
                "feature_name": component.feature_name,
                "area_um2": component.area_um2,
                "centroid_x_um": component.centroid_x_um,
                "centroid_y_um": component.centroid_y_um,
                "min_x_um": component.min_x_um,
                "min_y_um": component.min_y_um,
                "max_x_um": component.max_x_um,
                "max_y_um": component.max_y_um,
                "prev_candidate_count": int(candidate_counts[component.node_id]["prev"]),
                "next_candidate_count": int(candidate_counts[component.node_id]["next"]),
                "prev_link_node_id": prev_link.source if prev_link else "",
                "next_link_node_id": next_link.target if next_link else "",
                "prev_link_overlap_fraction": prev_link.overlap_fraction if prev_link else np.nan,
                "next_link_overlap_fraction": next_link.overlap_fraction if next_link else np.nan,
                "prev_link_centroid_distance_um": prev_link.centroid_distance_um if prev_link else np.nan,
                "next_link_centroid_distance_um": next_link.centroid_distance_um if next_link else np.nan,
                "prev_link_type": prev_link.link_type if prev_link else "",
                "next_link_type": next_link.link_type if next_link else "",
                "prev_link_weak": bool(prev_link.weak) if prev_link else False,
                "next_link_weak": bool(next_link.weak) if next_link else False,
            }
        )
    return pd.DataFrame(rows)


def _build_gland_index(
    tracks: pd.DataFrame,
    links: Sequence[_GlandLink],
    cfg: GlandQCAtlasConfig,
) -> pd.DataFrame:
    link_by_pair = {(link.source, link.target): link for link in links}
    rows: list[dict[str, Any]] = []
    for gland_id, group in tracks.groupby("gland_id", sort=True):
        group = group.sort_values("slice_order")
        areas = group["area_um2"].to_numpy(dtype=float)
        area_mean = float(np.mean(areas)) if len(areas) else math.nan
        area_cv = float(np.std(areas) / area_mean) if area_mean > 0 else math.nan
        unique_slices = sorted(group["slice_order"].astype(int).unique().tolist())
        overlaps: list[float] = []
        distances: list[float] = []
        weak_links = 0
        for _, row in group.iterrows():
            next_node = str(row.get("next_link_node_id", ""))
            if not next_node:
                continue
            link = link_by_pair.get((str(row["node_id"]), next_node))
            if link is None:
                continue
            overlaps.append(float(link.overlap_fraction))
            distances.append(float(link.centroid_distance_um))
            weak_links += int(link.weak)
        branch_merge_count = int(
            ((group["prev_candidate_count"].astype(int) > 1)
             | (group["next_candidate_count"].astype(int) > 1)).sum()
        )
        slice_count = int(len(unique_slices))
        min_overlap = float(np.min(overlaps)) if overlaps else math.nan
        median_overlap = float(np.median(overlaps)) if overlaps else math.nan
        max_jump = float(np.max(distances)) if distances else math.nan
        quality = _quality_flag(
            slice_count=slice_count,
            min_overlap=min_overlap,
            max_centroid_jump=max_jump,
            area_cv=area_cv,
            branch_merge_count=branch_merge_count,
            weak_link_count=weak_links,
            cfg=cfg,
        )
        rows.append(
            {
                "gland_id": gland_id,
                "structure": str(group["structure"].iloc[0]),
                "quality_flag": quality,
                "review_priority": _review_priority(quality, branch_merge_count, min_overlap, slice_count),
                "component_count": int(len(group)),
                "slice_count": slice_count,
                "first_slice_order": int(min(unique_slices)),
                "last_slice_order": int(max(unique_slices)),
                "z_min_um": float(group["z_um"].min()),
                "z_max_um": float(group["z_um"].max()),
                "z_span_um": float(group["z_um"].max() - group["z_um"].min()),
                "area_min_um2": float(np.min(areas)) if len(areas) else math.nan,
                "area_median_um2": float(np.median(areas)) if len(areas) else math.nan,
                "area_max_um2": float(np.max(areas)) if len(areas) else math.nan,
                "area_cv": area_cv,
                "min_adjacent_overlap_fraction": min_overlap,
                "median_adjacent_overlap_fraction": median_overlap,
                "max_centroid_jump_um": max_jump,
                "weak_link_count": int(weak_links),
                "branch_merge_count": branch_merge_count,
                "min_x_um": float(group["min_x_um"].min()),
                "min_y_um": float(group["min_y_um"].min()),
                "max_x_um": float(group["max_x_um"].max()),
                "max_y_um": float(group["max_y_um"].max()),
                "centroid_x_um": float(group["centroid_x_um"].median()),
                "centroid_y_um": float(group["centroid_y_um"].median()),
                "page": f"glands/{gland_id}.html",
            }
        )
    return pd.DataFrame(rows)


def _quality_flag(
    *,
    slice_count: int,
    min_overlap: float,
    max_centroid_jump: float,
    area_cv: float,
    branch_merge_count: int,
    weak_link_count: int,
    cfg: GlandQCAtlasConfig,
) -> str:
    if slice_count <= 1:
        return "isolated"
    if slice_count <= 2:
        return "review"
    if branch_merge_count > 0 or weak_link_count > 0:
        return "review"
    if math.isfinite(min_overlap) and min_overlap < cfg.good_overlap_fraction:
        return "review"
    if math.isfinite(max_centroid_jump) and max_centroid_jump > cfg.centroid_jump_review_um:
        return "review"
    if math.isfinite(area_cv) and area_cv > cfg.area_cv_review_threshold:
        return "review"
    return "good"


def _review_priority(
    quality: str,
    branch_merge_count: int,
    min_overlap: float,
    slice_count: int,
) -> int:
    if quality == "review":
        if branch_merge_count > 0:
            return 0
        if math.isfinite(min_overlap) and min_overlap < 0.05:
            return 1
        return 2
    if quality == "isolated":
        return 3
    if slice_count <= 3:
        return 4
    return 5


def _load_optional_cells(cfg: GlandQCAtlasConfig) -> _CellIndex | None:
    if cfg.aligned_cells_parquet is None:
        return None
    path = Path(cfg.aligned_cells_parquet).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    columns = ["x_3d_um", "y_3d_um", "z_um", cfg.label_column]
    cells = pd.read_parquet(path, columns=columns)
    cells = cells.replace([np.inf, -np.inf], np.nan).dropna(subset=["x_3d_um", "y_3d_um", "z_um"]).reset_index(drop=True)
    bin_size_um = max(float(cfg.padding_um), 250.0)
    bins_x = np.floor(cells["x_3d_um"].to_numpy(dtype=float) / bin_size_um).astype(int)
    bins_y = np.floor(cells["y_3d_um"].to_numpy(dtype=float) / bin_size_um).astype(int)
    tmp = pd.DataFrame({"_index": np.arange(len(cells), dtype=int), "_bx": bins_x, "_by": bins_y})
    bins = {
        (int(key[0]), int(key[1])): group["_index"].to_numpy(dtype=int)
        for key, group in tmp.groupby(["_bx", "_by"], sort=False)
    }
    return _CellIndex(cells=cells, bins=bins, bin_size_um=bin_size_um)


def _write_gland_page(
    path: Path,
    gland: Mapping[str, Any],
    gland_components: Sequence[_GlandComponent],
    all_components: Sequence[_GlandComponent],
    tracks: pd.DataFrame,
    cells: _CellIndex | None,
    cfg: GlandQCAtlasConfig,
) -> None:
    bbox = _padded_bbox(gland_components, cfg.padding_um)
    context = [
        component
        for component in all_components
        if component.structure == gland["structure"] and _intersects_bbox(component, bbox)
    ]
    traces = _plotly_traces(gland_components, context, cells, bbox, gland["gland_id"], cfg)
    metrics_html = _metrics_table(gland)
    strip_html = _slice_strip(gland_components, context, bbox, cfg)
    title = f"{gland['gland_id']} | {gland['structure']} | {gland['quality_flag']}"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; background: #08111f; color: #dbe7f3; font-family: Arial, sans-serif; }}
    header {{ padding: 14px 18px; border-bottom: 1px solid #223247; }}
    a {{ color: #7dd3fc; }}
    #plot {{ width: 100vw; height: 64vh; }}
    .panel {{ padding: 14px 18px; }}
    .grid {{ display: grid; grid-template-columns: minmax(260px, 420px) 1fr; gap: 18px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    td, th {{ border-bottom: 1px solid #223247; padding: 5px 7px; text-align: left; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #1e3a5f; }}
    .slice-strip {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-start; }}
    .slice-card {{ border: 1px solid #223247; background: #0f1b2d; padding: 6px; }}
    .slice-card-title {{ font-size: 11px; color: #9fb3c8; margin-bottom: 4px; }}
  </style>
</head>
<body>
  <header>
    <a href="../gland_qc_atlas.html">Back to atlas</a>
    <h1>{escape(title)}</h1>
    <span class="badge">Z visual scale x{float(cfg.z_visual_scale):g}</span>
  </header>
  <div id="plot"></div>
  <div class="panel grid">
    <section>
      <h2>QC metrics</h2>
      {metrics_html}
    </section>
    <section>
      <h2>Slice strip</h2>
      {strip_html}
    </section>
  </div>
  <script>
    const data = {json.dumps(traces)};
    const layout = {{
      paper_bgcolor: "#08111f",
      plot_bgcolor: "#08111f",
      title: {{text: {json.dumps(title)}, font: {{color: "#dbe7f3"}}}},
      scene: {{
        xaxis: {{title: "X (um)", color: "#b8c7d9", gridcolor: "#26364d"}},
        yaxis: {{title: "Y (um)", color: "#b8c7d9", gridcolor: "#26364d"}},
        zaxis: {{title: "Z (um, visual scale x{float(cfg.z_visual_scale):g})", color: "#b8c7d9", gridcolor: "#26364d"}},
        aspectmode: "data"
      }},
      legend: {{font: {{color: "#dbe7f3"}}}},
      margin: {{l: 0, r: 0, t: 42, b: 0}}
    }};
    Plotly.newPlot("plot", data, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _plotly_traces(
    gland_components: Sequence[_GlandComponent],
    context: Sequence[_GlandComponent],
    cells: _CellIndex | None,
    bbox: tuple[float, float, float, float],
    gland_id: str,
    cfg: GlandQCAtlasConfig,
) -> list[dict[str, Any]]:
    gland_ids = {component.node_id for component in gland_components}
    traces: list[dict[str, Any]] = []
    context_components = [component for component in context if component.node_id not in gland_ids]
    if context_components:
        traces.append(_component_trace(context_components, "nearby same-structure contours", "#64748b", cfg, width=2, opacity=0.35))
    traces.append(_component_trace(gland_components, f"{gland_id} tracked contours", "#facc15", cfg, width=7, opacity=1.0))
    if cells is not None:
        cell_trace = _cell_trace(cells, bbox, cfg)
        if cell_trace is not None:
            traces.append(cell_trace)
    return traces


def _component_trace(
    components: Sequence[_GlandComponent],
    name: str,
    color: str,
    cfg: GlandQCAtlasConfig,
    *,
    width: float,
    opacity: float,
) -> dict[str, Any]:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    hover: list[str | None] = []
    for component in sorted(components, key=lambda item: (item.slice_order, item.node_id)):
        for polygon in _iter_polygons(component.geometry_um):
            coords = list(polygon.exterior.coords)
            x_values.extend(float(x) for x, _ in coords)
            y_values.extend(float(y) for _, y in coords)
            z_values.extend(float(component.z_um * cfg.z_visual_scale) for _ in coords)
            hover.extend(
                [
                    (
                        f"{component.node_id}<br>{component.structure}<br>"
                        f"slice {component.slice_order}<br>z={component.z_um:.2f} um"
                    )
                    for _ in coords
                ]
            )
            x_values.append(None)
            y_values.append(None)
            z_values.append(None)
            hover.append(None)
    return {
        "type": "scatter3d",
        "mode": "lines",
        "name": name,
        "x": x_values,
        "y": y_values,
        "z": z_values,
        "text": hover,
        "hovertemplate": "%{text}<extra></extra>",
        "line": {"color": color, "width": width},
        "opacity": opacity,
    }


def _cell_trace(
    cell_index: _CellIndex,
    bbox: tuple[float, float, float, float],
    cfg: GlandQCAtlasConfig,
) -> dict[str, Any] | None:
    min_x, min_y, max_x, max_y = bbox
    min_bx = int(math.floor(min_x / cell_index.bin_size_um))
    max_bx = int(math.floor(max_x / cell_index.bin_size_um))
    min_by = int(math.floor(min_y / cell_index.bin_size_um))
    max_by = int(math.floor(max_y / cell_index.bin_size_um))
    chunks: list[np.ndarray] = []
    for bx in range(min_bx, max_bx + 1):
        for by in range(min_by, max_by + 1):
            indices = cell_index.bins.get((bx, by))
            if indices is not None:
                chunks.append(indices)
    if not chunks:
        return None
    candidate_indices = np.unique(np.concatenate(chunks))
    candidates = cell_index.cells.iloc[candidate_indices]
    local = candidates.loc[
        (candidates["x_3d_um"] >= min_x)
        & (candidates["x_3d_um"] <= max_x)
        & (candidates["y_3d_um"] >= min_y)
        & (candidates["y_3d_um"] <= max_y)
    ].copy()
    if local.empty:
        return None
    if len(local) > cfg.max_cells_per_gland:
        local = local.sample(
            n=int(cfg.max_cells_per_gland),
            random_state=int(cfg.random_state),
        )
    labels = local[cfg.label_column].astype(str)
    return {
        "type": "scatter3d",
        "mode": "markers",
        "name": f"local cells ({len(local):,})",
        "x": local["x_3d_um"].round(3).tolist(),
        "y": local["y_3d_um"].round(3).tolist(),
        "z": (local["z_um"] * float(cfg.z_visual_scale)).round(3).tolist(),
        "marker": {
            "size": 1.5,
            "opacity": 0.35,
            "color": pd.factorize(labels)[0].tolist(),
            "colorscale": "Viridis",
        },
        "text": labels.tolist(),
        "hovertemplate": f"{escape(cfg.label_column)}=%{{text}}<extra></extra>",
    }


def _metrics_table(gland: Mapping[str, Any]) -> str:
    fields = [
        "quality_flag",
        "structure",
        "slice_count",
        "component_count",
        "z_span_um",
        "min_adjacent_overlap_fraction",
        "median_adjacent_overlap_fraction",
        "max_centroid_jump_um",
        "area_cv",
        "weak_link_count",
        "branch_merge_count",
    ]
    rows = []
    for field in fields:
        value = gland.get(field, "")
        if isinstance(value, float):
            value = f"{value:.4g}" if math.isfinite(value) else "NA"
        rows.append(f"<tr><th>{escape(field)}</th><td>{escape(str(value))}</td></tr>")
    return "<table>" + "\n".join(rows) + "</table>"


def _slice_strip(
    gland_components: Sequence[_GlandComponent],
    context: Sequence[_GlandComponent],
    bbox: tuple[float, float, float, float],
    cfg: GlandQCAtlasConfig,
) -> str:
    gland_ids = {component.node_id for component in gland_components}
    orders = sorted({component.slice_order for component in gland_components})
    cards = []
    for order in orders:
        order_components = [
            component
            for component in context
            if component.slice_order == order
        ]
        svg = _slice_svg(order_components, gland_ids, bbox)
        cards.append(
            f'<div class="slice-card"><div class="slice-card-title">slice {order}</div>{svg}</div>'
        )
    return '<div class="slice-strip">' + "\n".join(cards) + "</div>"


def _slice_svg(
    components: Sequence[_GlandComponent],
    gland_ids: set[str],
    bbox: tuple[float, float, float, float],
) -> str:
    min_x, min_y, max_x, max_y = bbox
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    paths = []
    for component in components:
        is_gland = component.node_id in gland_ids
        stroke = "#facc15" if is_gland else "#64748b"
        fill = "rgba(250, 204, 21, 0.18)" if is_gland else "rgba(100, 116, 139, 0.10)"
        stroke_width = "3" if is_gland else "1"
        for polygon in _iter_polygons(component.geometry_um):
            points = []
            for x, y in polygon.exterior.coords:
                sx = (float(x) - min_x) / width * 180.0
                sy = (float(y) - min_y) / height * 150.0
                points.append(f"{sx:.2f},{sy:.2f}")
            paths.append(
                f'<polygon points="{" ".join(points)}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{stroke_width}" />'
            )
    return (
        '<svg width="180" height="150" viewBox="0 0 180 150" '
        'style="background:#08111f">'
        + "\n".join(paths)
        + "</svg>"
    )


def _write_atlas_page(path: Path, index: pd.DataFrame, cfg: GlandQCAtlasConfig) -> None:
    rows = []
    columns = [
        "gland_id",
        "structure",
        "quality_flag",
        "slice_count",
        "z_span_um",
        "min_adjacent_overlap_fraction",
        "max_centroid_jump_um",
        "area_cv",
        "branch_merge_count",
        "weak_link_count",
        "page_rendered",
    ]
    for _, row in index.sort_values(["review_priority", "gland_id"]).iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = f"{value:.4g}" if math.isfinite(value) else "NA"
            if column == "gland_id":
                if bool(row.get("page_rendered")) and str(row.get("page", "")):
                    value = f'<a href="{escape(str(row["page"]))}">{escape(str(value))}</a>'
                else:
                    value = escape(str(value))
            else:
                value = escape(str(value))
            cells.append(f"<td>{value}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>HistoSeg gland QC atlas</title>
  <style>
    body {{ margin: 0; background: #f7fafc; color: #172033; font-family: Arial, sans-serif; }}
    header {{ padding: 18px 22px; background: #0f172a; color: #e2e8f0; }}
    main {{ padding: 18px 22px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; }}
    th, td {{ border-bottom: 1px solid #d9e2ef; padding: 7px 8px; text-align: left; }}
    th {{ cursor: pointer; background: #e8eef7; position: sticky; top: 0; }}
    a {{ color: #0369a1; }}
    .note {{ color: #526276; }}
  </style>
</head>
<body>
  <header>
    <h1>HistoSeg gland QC atlas</h1>
    <p>Stack: {escape(str(Path(cfg.stack_root).expanduser()))}</p>
  </header>
  <main>
    <p class="note">Click headers to sort. Click a gland ID to open the local 3D zoom.</p>
    <table id="atlas">
      <thead><tr>{"".join(f"<th>{escape(col)}</th>" for col in columns)}</tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </main>
  <script>
    document.querySelectorAll("#atlas th").forEach((th, idx) => {{
      th.addEventListener("click", () => {{
        const tbody = th.closest("table").querySelector("tbody");
        const rows = Array.from(tbody.querySelectorAll("tr"));
        rows.sort((a, b) => {{
          const av = a.children[idx].innerText;
          const bv = b.children[idx].innerText;
          const an = Number(av), bn = Number(bv);
          if (!Number.isNaN(an) && !Number.isNaN(bn)) return an - bn;
          return av.localeCompare(bv);
        }});
        rows.forEach(row => tbody.appendChild(row));
      }});
    }});
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _padded_bbox(
    components: Sequence[_GlandComponent],
    padding_um: float,
) -> tuple[float, float, float, float]:
    return (
        min(component.min_x_um for component in components) - float(padding_um),
        min(component.min_y_um for component in components) - float(padding_um),
        max(component.max_x_um for component in components) + float(padding_um),
        max(component.max_y_um for component in components) + float(padding_um),
    )


def _intersects_bbox(
    component: _GlandComponent,
    bbox: tuple[float, float, float, float],
) -> bool:
    min_x, min_y, max_x, max_y = bbox
    return not (
        component.max_x_um < min_x
        or component.min_x_um > max_x
        or component.max_y_um < min_y
        or component.min_y_um > max_y
    )


def _iter_polygons(geom: Any) -> Iterable[Polygon]:
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        yield from geom.geoms


def _feature_structure(properties: Mapping[str, Any], group_property: str) -> str:
    if properties.get("assigned_structure") is not None:
        return str(properties["assigned_structure"])
    if properties.get(group_property) is not None:
        return str(properties[group_property])
    classification = properties.get("classification")
    if isinstance(classification, Mapping) and classification.get("name") is not None:
        return str(classification["name"])
    if properties.get("name") is not None:
        return str(properties["name"])
    return "unknown"


def _resolve_path(value: Any, *, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def _validate_config(cfg: GlandQCAtlasConfig) -> None:
    if cfg.pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be greater than 0.")
    if cfg.padding_um < 0:
        raise ValueError("padding_um must be non-negative.")
    if cfg.z_visual_scale <= 0:
        raise ValueError("z_visual_scale must be greater than 0.")
    if cfg.max_cells_per_gland <= 0:
        raise ValueError("max_cells_per_gland must be greater than 0.")
    if cfg.max_gland_pages is not None and cfg.max_gland_pages <= 0:
        raise ValueError("max_gland_pages must be greater than 0 when provided.")
    if cfg.min_overlap_fraction < 0:
        raise ValueError("min_overlap_fraction must be non-negative.")
    if cfg.good_overlap_fraction < cfg.min_overlap_fraction:
        raise ValueError("good_overlap_fraction must be >= min_overlap_fraction.")
    if cfg.centroid_fallback_um < 0:
        raise ValueError("centroid_fallback_um must be non-negative.")
    if cfg.max_area_ratio_for_fallback < 1:
        raise ValueError("max_area_ratio_for_fallback must be at least 1.")
