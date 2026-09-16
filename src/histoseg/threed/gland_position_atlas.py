"""Interactive 3D gland family atlas for tracked gland instances."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
import json
import math
import re
from typing import Any, Mapping, Sequence, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]

_PALETTE = (
    "#2f80ed",
    "#27ae60",
    "#f2994a",
    "#9b51e0",
    "#eb5757",
    "#00a6a6",
    "#8e6c3a",
    "#5c6bc0",
    "#c2185b",
    "#607d8b",
)


@dataclass(frozen=True)
class GlandPositionAtlasConfig:
    """Configuration for rendering a global 3D gland family position atlas."""

    gland_instance_dir: PathLike
    out_dir: PathLike | None = None
    biology_dir: PathLike | None = None
    candidate_score_threshold: float = 0.55
    z_visual_scale: float = 8.0
    color_by: str = "gland_family_id"
    max_fission_pages: int = 50


@dataclass
class GlandPositionAtlasResult:
    out_dir: Path
    gland_position_atlas_html: Path
    gland_family_links_csv: Path
    gland_family_index_csv: Path
    fission_events_csv: Path
    fission_page_count: int
    gland_count: int
    family_count: int


def render_gland_position_atlas(
    cfg: GlandPositionAtlasConfig,
) -> GlandPositionAtlasResult:
    """Render an interactive global atlas of tracked glands and family links."""

    _validate_config(cfg)
    gland_dir = Path(cfg.gland_instance_dir).expanduser().resolve()
    out_dir = (
        Path(cfg.out_dir).expanduser().resolve()
        if cfg.out_dir is not None
        else gland_dir / "gland_position_atlas"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_csv(gland_dir / "gland_instance_tracks.csv")
    gland_index_path = gland_dir / "gland_instance_qc_index.csv"
    gland_index = pd.read_csv(gland_index_path) if gland_index_path.exists() else _summarize_tracks(tracks)
    biology = _load_biology_metrics(gland_dir, cfg.biology_dir)

    links = _build_family_links(tracks, candidate_score_threshold=float(cfg.candidate_score_threshold))
    family_map = _assign_family_ids(tracks, links)
    links = _attach_family_ids(links, family_map)
    family_index = _build_family_index(
        tracks=tracks,
        gland_index=gland_index,
        links=links,
        family_map=family_map,
        biology=biology,
    )
    fission_events = _build_fission_events(
        tracks=tracks,
        links=links,
        family_index=family_index,
        family_map=family_map,
    )

    family_links_csv = out_dir / "gland_family_links.csv"
    family_index_csv = out_dir / "gland_family_index.csv"
    fission_events_csv = out_dir / "fission_events.csv"
    links.to_csv(family_links_csv, index=False)
    family_index.to_csv(family_index_csv, index=False)
    fission_events.to_csv(fission_events_csv, index=False)

    fission_page_count = _write_fission_pages(
        out_dir=out_dir,
        tracks=tracks,
        links=links,
        fission_events=fission_events,
        family_index=family_index,
        z_visual_scale=float(cfg.z_visual_scale),
        max_pages=int(cfg.max_fission_pages),
    )
    atlas_html = out_dir / "gland_position_atlas.html"
    _write_position_atlas_html(
        atlas_html,
        tracks=tracks,
        links=links,
        family_index=family_index,
        fission_events=fission_events,
        family_map=family_map,
        z_visual_scale=float(cfg.z_visual_scale),
        color_by=cfg.color_by,
    )

    return GlandPositionAtlasResult(
        out_dir=out_dir,
        gland_position_atlas_html=atlas_html,
        gland_family_links_csv=family_links_csv,
        gland_family_index_csv=family_index_csv,
        fission_events_csv=fission_events_csv,
        fission_page_count=fission_page_count,
        gland_count=int(tracks["gland_id"].nunique()) if not tracks.empty else 0,
        family_count=int(family_index["gland_family_id"].nunique()) if not family_index.empty else 0,
    )


def _build_family_links(tracks: pd.DataFrame, *, candidate_score_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    instance_lookup = {
        str(row["slice_instance_id"]): row.to_dict()
        for _, row in tracks.iterrows()
    }
    accepted_pairs: set[tuple[str, str]] = set()
    for _, row in tracks.iterrows():
        source_id = str(row.get("slice_instance_id", ""))
        target_id = _clean_id(row.get("next_slice_instance_id"))
        if not source_id or not target_id or target_id not in instance_lookup:
            continue
        source = instance_lookup[source_id]
        target = instance_lookup[target_id]
        if int(target["slice_order"]) != int(source["slice_order"]) + 1:
            continue
        pair = (source_id, target_id)
        accepted_pairs.add(pair)
        rows.append(
            _edge_row(
                source,
                target,
                link_type="accepted",
                score=_coalesce_float(row.get("next_link_score"), row.get("link_score"), 1.0),
                included_in_family=True,
                reported_by_slice_instance_id=source_id,
            )
        )

    candidate_seen: set[tuple[str, str]] = set()
    for _, row in tracks.iterrows():
        reporter_id = str(row.get("slice_instance_id", ""))
        if reporter_id not in instance_lookup:
            continue
        reporter = instance_lookup[reporter_id]
        for target_id, score in _parse_branch_candidates(row.get("branch_merge_candidates")):
            if target_id not in instance_lookup or score < candidate_score_threshold:
                continue
            target = instance_lookup[target_id]
            if str(target.get("semantic_structure", "")) != str(reporter.get("semantic_structure", "")):
                continue
            if abs(int(target["slice_order"]) - int(reporter["slice_order"])) != 1:
                continue
            source, dest = _orient_edge(reporter, target)
            pair = (str(source["slice_instance_id"]), str(dest["slice_instance_id"]))
            if pair in accepted_pairs or pair in candidate_seen:
                continue
            candidate_seen.add(pair)
            rows.append(
                _edge_row(
                    source,
                    dest,
                    link_type="candidate_branch_merge",
                    score=score,
                    included_in_family=True,
                    reported_by_slice_instance_id=reporter_id,
                )
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "link_type",
                "source_slice_instance_id",
                "target_slice_instance_id",
                "source_gland_id",
                "target_gland_id",
                "semantic_structure",
                "source_slice_order",
                "target_slice_order",
                "source_z_um",
                "target_z_um",
                "source_centroid_x_um",
                "source_centroid_y_um",
                "target_centroid_x_um",
                "target_centroid_y_um",
                "score",
                "included_in_family",
                "is_cross_gland",
                "reported_by_slice_instance_id",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["source_slice_order", "semantic_structure", "source_slice_instance_id", "target_slice_instance_id"]
    )


def _edge_row(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    link_type: str,
    score: float,
    included_in_family: bool,
    reported_by_slice_instance_id: str,
) -> dict[str, Any]:
    source_id = str(source["slice_instance_id"])
    target_id = str(target["slice_instance_id"])
    source_gland = str(source["gland_id"])
    target_gland = str(target["gland_id"])
    return {
        "link_type": link_type,
        "source_slice_instance_id": source_id,
        "target_slice_instance_id": target_id,
        "source_gland_id": source_gland,
        "target_gland_id": target_gland,
        "semantic_structure": str(source.get("semantic_structure", "")),
        "source_slice_order": int(source["slice_order"]),
        "target_slice_order": int(target["slice_order"]),
        "source_z_um": float(source["z_um"]),
        "target_z_um": float(target["z_um"]),
        "source_centroid_x_um": float(source["centroid_x_um"]),
        "source_centroid_y_um": float(source["centroid_y_um"]),
        "target_centroid_x_um": float(target["centroid_x_um"]),
        "target_centroid_y_um": float(target["centroid_y_um"]),
        "score": float(score),
        "included_in_family": bool(included_in_family),
        "is_cross_gland": source_gland != target_gland,
        "reported_by_slice_instance_id": reported_by_slice_instance_id,
    }


def _assign_family_ids(tracks: pd.DataFrame, links: pd.DataFrame) -> dict[str, str]:
    gland_ids = sorted(str(item) for item in tracks["gland_id"].dropna().unique())
    parent = {gland_id: gland_id for gland_id in gland_ids}

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

    for _, edge in links.iterrows():
        if not bool(edge.get("included_in_family", True)):
            continue
        left = str(edge["source_gland_id"])
        right = str(edge["target_gland_id"])
        if left in parent and right in parent:
            union(left, right)

    roots = sorted({find(gland_id) for gland_id in gland_ids}, key=_natural_key)
    root_to_family = {root: f"family_{index:04d}" for index, root in enumerate(roots, start=1)}
    return {gland_id: root_to_family[find(gland_id)] for gland_id in gland_ids}


def _attach_family_ids(links: pd.DataFrame, family_map: Mapping[str, str]) -> pd.DataFrame:
    if links.empty:
        links = links.copy()
        links["gland_family_id"] = []
        return links
    links = links.copy()
    links["source_family_id"] = links["source_gland_id"].astype(str).map(family_map)
    links["target_family_id"] = links["target_gland_id"].astype(str).map(family_map)
    links["gland_family_id"] = links["source_family_id"]
    return links


def _build_family_index(
    *,
    tracks: pd.DataFrame,
    gland_index: pd.DataFrame,
    links: pd.DataFrame,
    family_map: Mapping[str, str],
    biology: pd.DataFrame,
) -> pd.DataFrame:
    gland_rows = gland_index.copy()
    if not biology.empty:
        gland_rows = gland_rows.merge(biology, on="gland_id", how="left")
    gland_rows["gland_family_id"] = gland_rows["gland_id"].astype(str).map(family_map)
    track_lookup = tracks.copy()
    track_lookup["gland_family_id"] = track_lookup["gland_id"].astype(str).map(family_map)
    rows: list[dict[str, Any]] = []
    for family_id, family_glands in gland_rows.groupby("gland_family_id", sort=True):
        family_tracks = track_lookup.loc[track_lookup["gland_family_id"] == family_id]
        family_links = links.loc[links["gland_family_id"] == family_id] if not links.empty else links
        motif = _classify_motif(family_links)
        member_glands = sorted(family_glands["gland_id"].astype(str).unique(), key=_natural_key)
        row = {
            "gland_family_id": family_id,
            "motif_type": motif,
            "member_gland_count": int(len(member_glands)),
            "member_gland_ids": ";".join(member_glands),
            "semantic_structures": ";".join(sorted(family_tracks["semantic_structure"].astype(str).unique())),
            "slice_count_max": int(family_tracks.groupby("gland_id")["slice_order"].nunique().max()),
            "first_slice_order": int(family_tracks["slice_order"].min()),
            "last_slice_order": int(family_tracks["slice_order"].max()),
            "z_min_um": float(family_tracks["z_um"].min()),
            "z_max_um": float(family_tracks["z_um"].max()),
            "candidate_edge_count": int((family_links["link_type"] == "candidate_branch_merge").sum())
            if not family_links.empty
            else 0,
            "accepted_edge_count": int((family_links["link_type"] == "accepted").sum())
            if not family_links.empty
            else 0,
        }
        for column in [
            "state_entropy",
            "fission_candidate_score",
            "lumen_collapse_score",
            "signature_stromal_niche",
            "signature_stem_wnt",
            "signature_proliferation",
            "boundary_wnt_niche_score",
        ]:
            row[f"max_{column}"] = _max_or_nan(family_glands[column]) if column in family_glands else math.nan
            row[f"mean_{column}"] = _mean_or_nan(family_glands[column]) if column in family_glands else math.nan
        rows.append(row)
    index = pd.DataFrame(rows)
    if index.empty:
        return index
    biology_columns = [
        "max_fission_candidate_score",
        "max_signature_stromal_niche",
        "max_signature_stem_wnt",
        "max_signature_proliferation",
        "max_boundary_wnt_niche_score",
        "max_state_entropy",
        "max_lumen_collapse_score",
    ]
    percentile_columns: list[str] = []
    for column in biology_columns:
        if column not in index:
            continue
        percentile_column = f"{column}_percentile"
        values = pd.to_numeric(index[column], errors="coerce")
        if values.notna().any() and values.nunique(dropna=True) > 1:
            index[percentile_column] = values.rank(pct=True)
        else:
            index[percentile_column] = 0.0
        percentile_columns.append(percentile_column)
    index["biology_support_percentile_score"] = (
        index[percentile_columns].max(axis=1).fillna(0.0) if percentile_columns else 0.0
    )
    index["review_priority"] = [
        _review_priority(motif, slice_count, score)
        for motif, slice_count, score in zip(
            index["motif_type"],
            index["slice_count_max"],
            index["biology_support_percentile_score"],
        )
    ]
    return index.sort_values(
        ["review_priority", "motif_type", "biology_support_percentile_score", "gland_family_id"],
        ascending=[True, True, False, True],
    )


def _build_fission_events(
    *,
    tracks: pd.DataFrame,
    links: pd.DataFrame,
    family_index: pd.DataFrame,
    family_map: Mapping[str, str],
) -> pd.DataFrame:
    candidate = links.loc[links["link_type"] == "candidate_branch_merge"].copy() if not links.empty else links.copy()
    if candidate.empty:
        return pd.DataFrame(
            columns=[
                "fission_event_id",
                "event_type",
                "gland_family_id",
                "source_gland_id",
                "target_gland_id",
                "source_slice_order",
                "target_slice_order",
                "score",
                "page",
            ]
        )
    all_edges = links.loc[links["included_in_family"].astype(bool)].copy()
    outgoing = all_edges.groupby("source_slice_instance_id")["target_slice_instance_id"].nunique()
    incoming = all_edges.groupby("target_slice_instance_id")["source_slice_instance_id"].nunique()
    family_priority = {
        str(row["gland_family_id"]): str(row.get("review_priority", "medium"))
        for _, row in family_index.iterrows()
    }
    family_bio = {
        str(row["gland_family_id"]): float(row.get("biology_support_percentile_score", 0.0) or 0.0)
        for _, row in family_index.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for idx, edge in candidate.sort_values(["score"], ascending=False).reset_index(drop=True).iterrows():
        source_count = int(outgoing.get(edge["source_slice_instance_id"], 0))
        target_count = int(incoming.get(edge["target_slice_instance_id"], 0))
        event_type = _event_type(source_count, target_count)
        family_id = str(edge.get("gland_family_id") or family_map.get(str(edge["source_gland_id"]), ""))
        event_id = f"event_{idx + 1:04d}"
        rows.append(
            {
                "fission_event_id": event_id,
                "event_type": event_type,
                "gland_family_id": family_id,
                "review_priority": family_priority.get(family_id, "medium"),
                "biology_support_percentile_score": family_bio.get(family_id, 0.0),
                "source_slice_instance_id": edge["source_slice_instance_id"],
                "target_slice_instance_id": edge["target_slice_instance_id"],
                "source_gland_id": edge["source_gland_id"],
                "target_gland_id": edge["target_gland_id"],
                "source_slice_order": int(edge["source_slice_order"]),
                "target_slice_order": int(edge["target_slice_order"]),
                "source_centroid_x_um": float(edge["source_centroid_x_um"]),
                "source_centroid_y_um": float(edge["source_centroid_y_um"]),
                "target_centroid_x_um": float(edge["target_centroid_x_um"]),
                "target_centroid_y_um": float(edge["target_centroid_y_um"]),
                "score": float(edge["score"]),
                "event_priority_score": float(edge["score"]) + family_bio.get(family_id, 0.0),
                "page": f"fission_events/{event_id}.html",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["review_priority", "event_priority_score", "fission_event_id"],
        ascending=[True, False, True],
    )


def _write_position_atlas_html(
    path: Path,
    *,
    tracks: pd.DataFrame,
    links: pd.DataFrame,
    family_index: pd.DataFrame,
    fission_events: pd.DataFrame,
    family_map: Mapping[str, str],
    z_visual_scale: float,
    color_by: str,
) -> None:
    gland_payload = _gland_payload(tracks, family_index, family_map, z_visual_scale=z_visual_scale)
    edge_payload = _edge_payload(links, z_visual_scale=z_visual_scale)
    color_maps = _color_maps(gland_payload)
    color_by = color_by if color_by in color_maps else "gland_family_id"
    table_rows = _family_table_rows(family_index)
    event_rows = _event_table_rows(fission_events)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HistoSeg gland position atlas</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #1f2933; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 12px; }}
    .plots {{ display: grid; grid-template-columns: minmax(0, 2fr) minmax(360px, 1fr); gap: 16px; }}
    #plot3d {{ height: 720px; border: 1px solid #d9e2ec; }}
    #plot2d {{ height: 720px; border: 1px solid #d9e2ec; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 16px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 5px 7px; text-align: left; }}
    th {{ background: #f0f4f8; position: sticky; top: 0; }}
    tr {{ cursor: pointer; }}
    tr:hover {{ background: #f7fbff; }}
    input, select, button {{ padding: 7px; }}
    .note {{ color: #52616b; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>HistoSeg 3D Gland Family Atlas</h1>
  <p class="note">Strict gland IDs are preserved. Dashed orange links are branch/merge candidates used only for the derived family layer.</p>
  <div class="controls">
    <label>Color by
      <select id="colorMode" onchange="applyColorMode()">
        {_color_options(color_maps, color_by)}
      </select>
    </label>
    <label>Motif
      <select id="motifFilter" onchange="filterRows()">
        <option value="">all</option><option value="linear">linear</option><option value="candidate_split">candidate_split</option><option value="candidate_merge">candidate_merge</option><option value="complex">complex</option>
      </select>
    </label>
    <label>Priority
      <select id="priorityFilter" onchange="filterRows()">
        <option value="">all</option><option value="high">high</option><option value="medium">medium</option><option value="low">low</option><option value="isolated/review">isolated/review</option>
      </select>
    </label>
    <input id="textFilter" placeholder="Filter family, gland, structure..." onkeyup="filterRows()">
    <button onclick="clearHighlight()">Clear highlight</button>
  </div>
  <div class="plots">
    <div id="plot3d"></div>
    <div id="plot2d"></div>
  </div>
  <h2>Gland families</h2>
  <table id="familyTable">
    <thead><tr><th>Family</th><th>Motif</th><th>Priority</th><th>Glands</th><th>Structure</th><th>Slices</th><th>Candidate edges</th><th>Biology support</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  <h2>Top fission/budding candidate edges</h2>
  <table id="eventTable">
    <thead><tr><th>Event</th><th>Type</th><th>Family</th><th>Priority</th><th>Source</th><th>Target</th><th>Score</th><th>Page</th></tr></thead>
    <tbody>{event_rows}</tbody>
  </table>
  <script>
    const glands = {json.dumps(gland_payload)};
    const edges = {json.dumps(edge_payload)};
    const colorMaps = {json.dumps(color_maps)};
    const zVisualScale = {z_visual_scale:g};
    const glandTraceCount = glands.length;

    function glandTrace3d(g, color) {{
      return {{
        type: 'scatter3d',
        mode: 'lines+markers',
        name: g.gland_id,
        x: g.x,
        y: g.y,
        z: g.z,
        text: g.hover,
        hovertemplate: '%{{text}}<extra></extra>',
        legendgroup: g.gland_family_id,
        meta: g,
        line: {{color: color, width: 4}},
        marker: {{color: color, size: 4}},
        opacity: 0.92
      }};
    }}
    function glandTrace2d(g, color) {{
      return {{
        type: 'scatter',
        mode: 'lines+markers',
        name: g.gland_id,
        x: g.x,
        y: g.y,
        text: g.hover,
        hovertemplate: '%{{text}}<extra></extra>',
        legendgroup: g.gland_family_id,
        meta: g,
        line: {{color: color, width: 3}},
        marker: {{color: color, size: 5}},
        opacity: 0.9,
        showlegend: false
      }};
    }}
    function edgeTrace3d(kind) {{
      const selected = edges.filter(e => e.link_type === kind);
      const xs = [], ys = [], zs = [], text = [];
      for (const e of selected) {{
        xs.push(e.source_x, e.target_x, null);
        ys.push(e.source_y, e.target_y, null);
        zs.push(e.source_z, e.target_z, null);
        const label = `${{e.link_type}}<br>${{e.source_gland_id}} -> ${{e.target_gland_id}}<br>score ${{e.score.toFixed(3)}}`;
        text.push(label, label, null);
      }}
      return {{
        type: 'scatter3d',
        mode: 'lines',
        name: kind,
        x: xs, y: ys, z: zs, text: text,
        hovertemplate: '%{{text}}<extra></extra>',
        line: {{color: kind === 'accepted' ? '#96a4b8' : '#f2994a', width: kind === 'accepted' ? 2 : 5, dash: kind === 'accepted' ? 'solid' : 'dash'}},
        opacity: kind === 'accepted' ? 0.18 : 0.88
      }};
    }}
    function edgeTrace2d(kind) {{
      const selected = edges.filter(e => e.link_type === kind);
      const xs = [], ys = [], text = [];
      for (const e of selected) {{
        xs.push(e.source_x, e.target_x, null);
        ys.push(e.source_y, e.target_y, null);
        const label = `${{e.link_type}}<br>${{e.source_gland_id}} -> ${{e.target_gland_id}}<br>score ${{e.score.toFixed(3)}}`;
        text.push(label, label, null);
      }}
      return {{
        type: 'scatter',
        mode: 'lines',
        name: kind,
        x: xs, y: ys, text: text,
        hovertemplate: '%{{text}}<extra></extra>',
        line: {{color: kind === 'accepted' ? '#96a4b8' : '#f2994a', width: kind === 'accepted' ? 1 : 3, dash: kind === 'accepted' ? 'solid' : 'dash'}},
        opacity: kind === 'accepted' ? 0.16 : 0.86
      }};
    }}
    function baseColors() {{
      return colorMaps[document.getElementById('colorMode').value] || colorMaps.gland_family_id;
    }}
    const colors = baseColors();
    Plotly.newPlot('plot3d',
      glands.map((g, i) => glandTrace3d(g, colors[i])).concat([edgeTrace3d('accepted'), edgeTrace3d('candidate_branch_merge')]),
      {{scene: {{xaxis: {{title: 'x (um)'}}, yaxis: {{title: 'y (um)'}}, zaxis: {{title: `z (visual x${{zVisualScale}})`}}, aspectmode: 'data'}}, margin: {{l:0,r:0,b:0,t:10}}, showlegend: false}},
      {{responsive: true}}
    );
    Plotly.newPlot('plot2d',
      glands.map((g, i) => glandTrace2d(g, colors[i])).concat([edgeTrace2d('accepted'), edgeTrace2d('candidate_branch_merge')]),
      {{xaxis: {{title: 'x (um)', scaleanchor: 'y'}}, yaxis: {{title: 'y (um)', autorange: 'reversed'}}, margin: {{l:45,r:10,b:45,t:10}}, showlegend: false}},
      {{responsive: true}}
    );
    function applyColorMode() {{
      const colors = baseColors();
      for (let i = 0; i < glandTraceCount; i++) {{
        Plotly.restyle('plot3d', {{'line.color': [colors[i]], 'marker.color': [colors[i]], opacity: [0.92], 'line.width': [4]}}, [i]);
        Plotly.restyle('plot2d', {{'line.color': [colors[i]], 'marker.color': [colors[i]], opacity: [0.9], 'line.width': [3]}}, [i]);
      }}
    }}
    function highlightFamily(familyId) {{
      const colors = baseColors();
      for (let i = 0; i < glandTraceCount; i++) {{
        const active = glands[i].gland_family_id === familyId;
        Plotly.restyle('plot3d', {{opacity: [active ? 1.0 : 0.06], 'line.width': [active ? 7 : 2], 'line.color': [active ? colors[i] : '#c7d0dd'], 'marker.color': [active ? colors[i] : '#c7d0dd']}}, [i]);
        Plotly.restyle('plot2d', {{opacity: [active ? 1.0 : 0.06], 'line.width': [active ? 5 : 1], 'line.color': [active ? colors[i] : '#c7d0dd'], 'marker.color': [active ? colors[i] : '#c7d0dd']}}, [i]);
      }}
    }}
    function clearHighlight() {{
      applyColorMode();
    }}
    function filterRows() {{
      const motif = document.getElementById('motifFilter').value;
      const priority = document.getElementById('priorityFilter').value;
      const q = document.getElementById('textFilter').value.toLowerCase();
      for (const row of document.querySelectorAll('#familyTable tbody tr')) {{
        const keepMotif = !motif || row.dataset.motif === motif;
        const keepPriority = !priority || row.dataset.priority === priority;
        const keepText = !q || row.innerText.toLowerCase().includes(q);
        row.style.display = keepMotif && keepPriority && keepText ? '' : 'none';
      }}
    }}
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _write_fission_pages(
    *,
    out_dir: Path,
    tracks: pd.DataFrame,
    links: pd.DataFrame,
    fission_events: pd.DataFrame,
    family_index: pd.DataFrame,
    z_visual_scale: float,
    max_pages: int,
) -> int:
    if fission_events.empty or max_pages <= 0:
        return 0
    event_dir = out_dir / "fission_events"
    event_dir.mkdir(parents=True, exist_ok=True)
    family_members = {
        str(row["gland_family_id"]): str(row.get("member_gland_ids", "")).split(";")
        for _, row in family_index.iterrows()
    }
    rendered = 0
    for _, event in fission_events.head(max_pages).iterrows():
        family_id = str(event["gland_family_id"])
        gland_ids = [item for item in family_members.get(family_id, []) if item]
        if not gland_ids:
            gland_ids = [str(event["source_gland_id"]), str(event["target_gland_id"])]
        local_tracks = tracks.loc[tracks["gland_id"].astype(str).isin(gland_ids)].copy()
        local_links = links.loc[
            links["source_gland_id"].astype(str).isin(gland_ids)
            | links["target_gland_id"].astype(str).isin(gland_ids)
        ].copy()
        traces = _local_event_traces(local_tracks, local_links, z_visual_scale=z_visual_scale)
        rows = "".join(
            f"<tr><th>{escape(str(key))}</th><td>{escape(_fmt(value))}</td></tr>"
            for key, value in event.to_dict().items()
        )
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape(str(event['fission_event_id']))}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #1f2933; }}
    #plot {{ height: 680px; border: 1px solid #d9e2ec; }}
    table {{ border-collapse: collapse; font-size: 13px; margin-top: 14px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 5px 8px; text-align: left; }}
  </style>
</head>
<body>
  <p><a href="../gland_position_atlas.html">Back to gland position atlas</a></p>
  <h1>{escape(str(event['fission_event_id']))}: {escape(str(event['event_type']))}</h1>
  <div id="plot"></div>
  <table>{rows}</table>
  <script>
    Plotly.newPlot('plot', {json.dumps(traces)}, {{
      scene: {{xaxis: {{title: 'x (um)'}}, yaxis: {{title: 'y (um)'}}, zaxis: {{title: 'z (visual x{z_visual_scale:g})'}}, aspectmode: 'data'}},
      margin: {{l:0,r:0,b:0,t:10}}
    }}, {{responsive: true}});
  </script>
</body>
</html>
"""
        (event_dir / f"{event['fission_event_id']}.html").write_text(html, encoding="utf-8")
        rendered += 1
    return rendered


def _local_event_traces(tracks: pd.DataFrame, links: pd.DataFrame, *, z_visual_scale: float) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for index, (gland_id, group) in enumerate(tracks.groupby("gland_id", sort=True)):
        group = group.sort_values("slice_order")
        color = _PALETTE[index % len(_PALETTE)]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines+markers",
                "name": str(gland_id),
                "x": group["centroid_x_um"].astype(float).tolist(),
                "y": group["centroid_y_um"].astype(float).tolist(),
                "z": (group["z_um"].astype(float) * z_visual_scale).tolist(),
                "line": {"color": color, "width": 6},
                "marker": {"color": color, "size": 5},
                "text": [
                    f"{gland_id}<br>{row.slice_instance_id}<br>slice {int(row.slice_order)}<br>true z {float(row.z_um):.2f} um"
                    for row in group.itertuples(index=False)
                ],
                "hovertemplate": "%{text}<extra></extra>",
            }
        )
    for kind in ("accepted", "candidate_branch_merge"):
        subset = links.loc[links["link_type"] == kind]
        if subset.empty:
            continue
        x: list[float | None] = []
        y: list[float | None] = []
        z: list[float | None] = []
        text: list[str | None] = []
        for _, edge in subset.iterrows():
            x.extend([float(edge["source_centroid_x_um"]), float(edge["target_centroid_x_um"]), None])
            y.extend([float(edge["source_centroid_y_um"]), float(edge["target_centroid_y_um"]), None])
            z.extend([float(edge["source_z_um"]) * z_visual_scale, float(edge["target_z_um"]) * z_visual_scale, None])
            label = f"{kind}<br>{edge['source_gland_id']} -> {edge['target_gland_id']}<br>score {float(edge['score']):.3f}"
            text.extend([label, label, None])
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": kind,
                "x": x,
                "y": y,
                "z": z,
                "text": text,
                "hovertemplate": "%{text}<extra></extra>",
                "line": {
                    "color": "#f2994a" if kind == "candidate_branch_merge" else "#96a4b8",
                    "width": 6 if kind == "candidate_branch_merge" else 2,
                    "dash": "dash" if kind == "candidate_branch_merge" else "solid",
                },
                "opacity": 0.9 if kind == "candidate_branch_merge" else 0.25,
            }
        )
    return traces


def _gland_payload(
    tracks: pd.DataFrame,
    family_index: pd.DataFrame,
    family_map: Mapping[str, str],
    *,
    z_visual_scale: float,
) -> list[dict[str, Any]]:
    family_lookup = {
        str(row["gland_family_id"]): row.to_dict()
        for _, row in family_index.iterrows()
    }
    payload: list[dict[str, Any]] = []
    for gland_id, group in tracks.groupby("gland_id", sort=True):
        group = group.sort_values("slice_order")
        family_id = family_map.get(str(gland_id), "")
        family = family_lookup.get(family_id, {})
        hover = [
            (
                f"{gland_id}<br>{row.slice_instance_id}<br>"
                f"family {family_id}<br>slice {int(row.slice_order)}<br>"
                f"true z {float(row.z_um):.2f} um<br>"
                f"x {float(row.centroid_x_um):.1f}, y {float(row.centroid_y_um):.1f}"
            )
            for row in group.itertuples(index=False)
        ]
        payload.append(
            {
                "gland_id": str(gland_id),
                "gland_family_id": family_id,
                "semantic_structure": str(group["semantic_structure"].mode().iloc[0]),
                "qc_flags": _join_flags(group["qc_flags"]) or "clean",
                "motif_type": str(family.get("motif_type", "linear")),
                "review_priority": str(family.get("review_priority", "low")),
                "slice_count": int(group["slice_order"].nunique()),
                "state_entropy": _metric(family, "max_state_entropy"),
                "fission_candidate_score": _metric(family, "max_fission_candidate_score"),
                "boundary_wnt_niche_score": _metric(family, "max_boundary_wnt_niche_score"),
                "lumen_collapse_score": _metric(family, "max_lumen_collapse_score"),
                "x": group["centroid_x_um"].astype(float).tolist(),
                "y": group["centroid_y_um"].astype(float).tolist(),
                "z": (group["z_um"].astype(float) * z_visual_scale).tolist(),
                "hover": hover,
            }
        )
    return payload


def _edge_payload(links: pd.DataFrame, *, z_visual_scale: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, edge in links.iterrows():
        rows.append(
            {
                "link_type": str(edge["link_type"]),
                "source_gland_id": str(edge["source_gland_id"]),
                "target_gland_id": str(edge["target_gland_id"]),
                "gland_family_id": str(edge.get("gland_family_id", "")),
                "source_x": float(edge["source_centroid_x_um"]),
                "source_y": float(edge["source_centroid_y_um"]),
                "source_z": float(edge["source_z_um"]) * z_visual_scale,
                "target_x": float(edge["target_centroid_x_um"]),
                "target_y": float(edge["target_centroid_y_um"]),
                "target_z": float(edge["target_z_um"]) * z_visual_scale,
                "score": float(edge["score"]),
            }
        )
    return rows


def _color_maps(glands: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    if not glands:
        return {}
    fields = {
        "gland_family_id": "categorical",
        "semantic_structure": "categorical",
        "qc_flags": "categorical",
        "slice_count": "continuous",
        "state_entropy": "continuous",
        "fission_candidate_score": "continuous",
        "boundary_wnt_niche_score": "continuous",
        "lumen_collapse_score": "continuous",
    }
    maps: dict[str, list[str]] = {}
    for field, kind in fields.items():
        values = [item.get(field) for item in glands]
        if kind == "categorical":
            maps[field] = _categorical_colors(values)
        else:
            maps[field] = _continuous_colors(values)
    return maps


def _categorical_colors(values: Sequence[Any]) -> list[str]:
    categories = {str(value): index for index, value in enumerate(sorted({str(v) for v in values}, key=_natural_key))}
    return [_PALETTE[categories[str(value)] % len(_PALETTE)] for value in values]


def _continuous_colors(values: Sequence[Any]) -> list[str]:
    numeric = np.asarray([_safe_float(value, math.nan) for value in values], dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0 or float(finite.max()) == float(finite.min()):
        return ["#607d8b" for _ in values]
    low = float(finite.min())
    high = float(finite.max())
    colors = []
    for value in numeric:
        if not np.isfinite(value):
            colors.append("#c7d0dd")
        else:
            t = (float(value) - low) / (high - low)
            colors.append(_interpolate_color(t))
    return colors


def _interpolate_color(t: float) -> str:
    stops = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    t = min(max(float(t), 0.0), 1.0)
    scaled = t * (len(stops) - 1)
    index = min(int(math.floor(scaled)), len(stops) - 2)
    frac = scaled - index
    left = stops[index]
    right = stops[index + 1]
    rgb = tuple(int(round(left[i] + frac * (right[i] - left[i]))) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _family_table_rows(family_index: pd.DataFrame) -> str:
    rows = []
    for _, row in family_index.iterrows():
        family_id = str(row.get("gland_family_id", ""))
        motif = str(row.get("motif_type", ""))
        priority = str(row.get("review_priority", ""))
        rows.append(
            f'<tr data-family="{escape(family_id)}" data-motif="{escape(motif)}" '
            f'data-priority="{escape(priority)}" onclick="highlightFamily(\'{escape(family_id)}\')">'
            f"<td>{escape(family_id)}</td>"
            f"<td>{escape(motif)}</td>"
            f"<td>{escape(priority)}</td>"
            f"<td>{escape(str(row.get('member_gland_ids', '')))}</td>"
            f"<td>{escape(str(row.get('semantic_structures', '')))}</td>"
            f"<td>{escape(str(row.get('first_slice_order', '')))}-{escape(str(row.get('last_slice_order', '')))}</td>"
            f"<td>{escape(str(row.get('candidate_edge_count', '')))}</td>"
            f"<td>{escape(_fmt(row.get('biology_support_percentile_score')))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _event_table_rows(fission_events: pd.DataFrame) -> str:
    rows = []
    for _, row in fission_events.head(80).iterrows():
        page = str(row.get("page", ""))
        page_cell = f'<a href="{escape(page)}">open</a>' if page else ""
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('fission_event_id', '')))}</td>"
            f"<td>{escape(str(row.get('event_type', '')))}</td>"
            f"<td>{escape(str(row.get('gland_family_id', '')))}</td>"
            f"<td>{escape(str(row.get('review_priority', '')))}</td>"
            f"<td>{escape(str(row.get('source_gland_id', '')))}</td>"
            f"<td>{escape(str(row.get('target_gland_id', '')))}</td>"
            f"<td>{escape(_fmt(row.get('score')))}</td>"
            f"<td>{page_cell}</td>"
            "</tr>"
        )
    return "".join(rows)


def _color_options(color_maps: Mapping[str, Sequence[str]], selected: str) -> str:
    labels = {
        "gland_family_id": "family",
        "semantic_structure": "structure",
        "qc_flags": "QC flags",
        "slice_count": "slice count",
        "state_entropy": "entropy",
        "fission_candidate_score": "fission score",
        "boundary_wnt_niche_score": "boundary Wnt/niche",
        "lumen_collapse_score": "lumen collapse",
    }
    return "".join(
        f'<option value="{escape(key)}"{" selected" if key == selected else ""}>{escape(labels.get(key, key))}</option>'
        for key in color_maps
    )


def _load_biology_metrics(gland_dir: Path, biology_dir: PathLike | None) -> pd.DataFrame:
    bio_dir = Path(biology_dir).expanduser().resolve() if biology_dir is not None else _default_biology_dir(gland_dir)
    if bio_dir is None:
        return pd.DataFrame()
    feature_path = bio_dir / "gland_biology_feature_matrix.csv"
    if not feature_path.exists():
        return pd.DataFrame()
    feature = pd.read_csv(feature_path)
    keep = [
        "gland_id",
        "state_entropy",
        "fission_candidate_score",
        "lumen_collapse_score",
        "signature_stromal_niche",
        "signature_stem_wnt",
        "signature_proliferation",
    ]
    result = feature[[column for column in keep if column in feature.columns]].copy()
    communication_path = bio_dir / "gland_boundary_communication.csv"
    if communication_path.exists():
        communication = pd.read_csv(communication_path)
        if not communication.empty:
            niche = communication.loc[
                communication["ligand"].astype(str).isin({"WNT2B", "RSPO3", "GREM1"})
            ].copy()
            if not niche.empty:
                boundary = (
                    niche.groupby("gland_id")["max_boundary_interaction_score"]
                    .max()
                    .reset_index(name="boundary_wnt_niche_score")
                )
                result = result.merge(boundary, on="gland_id", how="left")
    if "boundary_wnt_niche_score" not in result:
        result["boundary_wnt_niche_score"] = math.nan
    return result


def _default_biology_dir(gland_dir: Path) -> Path | None:
    candidates = sorted(
        [path for path in gland_dir.glob("biology_mining*") if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _parse_branch_candidates(value: Any) -> list[tuple[str, float]]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    out: list[tuple[str, float]] = []
    for part in str(value).split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        target, score = part.rsplit(":", 1)
        try:
            out.append((target.strip(), float(score)))
        except ValueError:
            continue
    return out


def _orient_edge(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if int(left["slice_order"]) <= int(right["slice_order"]):
        return left, right
    return right, left


def _classify_motif(links: pd.DataFrame) -> str:
    if links.empty or not (links["link_type"] == "candidate_branch_merge").any():
        return "linear"
    included = links.loc[links["included_in_family"].astype(bool)]
    outgoing = included.groupby("source_slice_instance_id")["target_slice_instance_id"].nunique()
    incoming = included.groupby("target_slice_instance_id")["source_slice_instance_id"].nunique()
    has_split = bool((outgoing > 1).any())
    has_merge = bool((incoming > 1).any())
    if has_split and has_merge:
        return "complex"
    if has_split:
        return "candidate_split"
    if has_merge:
        return "candidate_merge"
    return "complex"


def _event_type(outgoing_count: int, incoming_count: int) -> str:
    has_split = outgoing_count > 1
    has_merge = incoming_count > 1
    if has_split and has_merge:
        return "complex"
    if has_split:
        return "candidate_split"
    if has_merge:
        return "candidate_merge"
    return "candidate_branch"


def _review_priority(motif: str, slice_count: int, biology_support_score: float) -> str:
    if motif != "linear" and (motif == "complex" or biology_support_score >= 0.75):
        return "high"
    if motif != "linear":
        return "medium"
    if slice_count < 3:
        return "isolated/review"
    return "low"


def _summarize_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gland_id, group in tracks.groupby("gland_id", sort=True):
        rows.append(
            {
                "gland_id": gland_id,
                "semantic_structure": str(group["semantic_structure"].mode().iloc[0]),
                "slice_count": int(group["slice_order"].nunique()),
                "first_slice_order": int(group["slice_order"].min()),
                "last_slice_order": int(group["slice_order"].max()),
                "z_min_um": float(group["z_um"].min()),
                "z_max_um": float(group["z_um"].max()),
                "z_span_um": float(group["z_um"].max() - group["z_um"].min()),
                "area_median_um2": float(group["area_um2"].median()),
                "branch_merge_candidate_count": int(
                    group["branch_merge_candidates"].fillna("").astype(str).map(bool).sum()
                ),
                "qc_flags": _join_flags(group["qc_flags"]),
            }
        )
    return pd.DataFrame(rows)


def _join_flags(values: Sequence[Any]) -> str:
    flags = sorted(
        {
            flag
            for value in values
            for flag in str(value if not pd.isna(value) else "").split(";")
            if flag and flag.lower() != "nan"
        }
    )
    return ";".join(flags)


def _clean_id(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    value = str(value).strip()
    return value or None


def _coalesce_float(*values: Any) -> float:
    for value in values:
        out = _safe_float(value, math.nan)
        if math.isfinite(out):
            return out
    return math.nan


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _metric(row: Mapping[str, Any], key: str) -> float:
    return _safe_float(row.get(key), 0.0)


def _max_or_nan(values: Sequence[Any]) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    return float(numeric.max()) if numeric.notna().any() else math.nan


def _mean_or_nan(values: Sequence[Any]) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else math.nan


def _natural_key(value: Any) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def _validate_config(cfg: GlandPositionAtlasConfig) -> None:
    gland_dir = Path(cfg.gland_instance_dir).expanduser()
    if not gland_dir.exists():
        raise FileNotFoundError(gland_dir)
    tracks = gland_dir / "gland_instance_tracks.csv"
    if not tracks.exists():
        raise FileNotFoundError(tracks)
    if cfg.candidate_score_threshold < 0:
        raise ValueError("`candidate_score_threshold` must be non-negative.")
    if cfg.z_visual_scale <= 0:
        raise ValueError("`z_visual_scale` must be positive.")
    if cfg.max_fission_pages < 0:
        raise ValueError("`max_fission_pages` must be non-negative.")


__all__ = [
    "GlandPositionAtlasConfig",
    "GlandPositionAtlasResult",
    "render_gland_position_atlas",
]
