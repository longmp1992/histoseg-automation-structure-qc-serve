"""HTML atlas rendering and chained workflow for gland instance reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
import json
from typing import Any, Sequence

import pandas as pd
from shapely.geometry import shape

from .gland_instances import (
    DEFAULT_EPITHELIAL_MARKERS,
    DEFAULT_EPITHELIAL_STRUCTURES,
    GlandInstanceSegmentationConfig,
    GlandInstanceSegmentationResult,
    segment_gland_instances,
)
from .gland_tracking import (
    GlandInstanceTrackingConfig,
    GlandInstanceTrackingResult,
    track_gland_instances,
)


PathLike = str | Path


@dataclass
class GlandInstanceAtlasResult:
    out_dir: Path
    atlas_html: Path
    gland_instance_qc_index_csv: Path
    gland_page_count: int


@dataclass
class GlandInstanceDetectionResult:
    out_dir: Path
    slice_gland_instances_geojson: Path
    slice_qc_csv: Path
    gland_instance_tracks_csv: Path
    gland_instance_qc_index_csv: Path
    gland_instance_atlas_html: Path
    slice_count: int
    total_gland_instances: int
    gland_count: int


def run_gland_instance_detection(
    *,
    segmentation_config: GlandInstanceSegmentationConfig,
    tracking_config: GlandInstanceTrackingConfig | None = None,
    max_gland_pages: int | None = 250,
    z_visual_scale: float = 8.0,
) -> GlandInstanceDetectionResult:
    """Run segmentation, tracking, and atlas rendering for gland instances."""

    segmentation = segment_gland_instances(segmentation_config)
    if tracking_config is None:
        tracking_config = GlandInstanceTrackingConfig(
            segmentation_result_dir=segmentation.out_dir,
            out_dir=segmentation.out_dir,
        )
    tracking = track_gland_instances(tracking_config)
    atlas = render_gland_instance_atlas(
        tracks_csv=tracking.gland_instance_tracks_csv,
        out_dir=tracking.out_dir,
        instances_geojson=segmentation.slice_gland_instances_geojson,
        qc_index_csv=tracking.gland_instance_qc_index_csv,
        max_gland_pages=max_gland_pages,
        z_visual_scale=z_visual_scale,
    )
    return GlandInstanceDetectionResult(
        out_dir=segmentation.out_dir,
        slice_gland_instances_geojson=segmentation.slice_gland_instances_geojson,
        slice_qc_csv=segmentation.slice_qc_csv,
        gland_instance_tracks_csv=tracking.gland_instance_tracks_csv,
        gland_instance_qc_index_csv=tracking.gland_instance_qc_index_csv,
        gland_instance_atlas_html=atlas.atlas_html,
        slice_count=segmentation.slice_count,
        total_gland_instances=segmentation.total_gland_instances,
        gland_count=tracking.gland_count,
    )


def render_gland_instance_atlas(
    *,
    tracks_csv: PathLike,
    out_dir: PathLike,
    stack_root: PathLike | None = None,
    instances_geojson: PathLike | None = None,
    qc_index_csv: PathLike | None = None,
    max_gland_pages: int | None = 250,
    z_visual_scale: float = 8.0,
) -> GlandInstanceAtlasResult:
    """Render a small, dependency-light HTML atlas for gland instance tracks."""

    del stack_root  # Reserved for future image overlays without changing the API.
    tracks_path = Path(tracks_csv).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    gland_dir = out_path / "glands"
    gland_dir.mkdir(parents=True, exist_ok=True)
    tracks = pd.read_csv(tracks_path)
    if qc_index_csv is None:
        qc_index_path = out_path / "gland_instance_qc_index.csv"
    else:
        qc_index_path = Path(qc_index_csv).expanduser().resolve()
    if qc_index_path.exists():
        index = pd.read_csv(qc_index_path)
    else:
        index = _summarize_tracks(tracks)
        index.to_csv(qc_index_path, index=False)

    geojson_path = (
        Path(instances_geojson).expanduser().resolve()
        if instances_geojson is not None
        else tracks_path.parent / "slice_gland_instances.geojson"
    )
    feature_lookup = _load_feature_lookup(geojson_path) if geojson_path.exists() else {}

    page_index = index.sort_values(
        ["branch_merge_candidate_count", "slice_count", "gland_id"],
        ascending=[False, True, True],
    )
    if max_gland_pages is not None:
        page_index = page_index.head(int(max_gland_pages))
    rendered_ids: set[str] = set()
    for _, gland in page_index.iterrows():
        gland_id = str(gland["gland_id"])
        gland_tracks = tracks.loc[tracks["gland_id"].astype(str) == gland_id].copy()
        if gland_tracks.empty:
            continue
        _write_gland_page(
            gland_dir / f"{gland_id}.html",
            gland=gland.to_dict(),
            tracks=gland_tracks,
            feature_lookup=feature_lookup,
            z_visual_scale=float(z_visual_scale),
        )
        rendered_ids.add(gland_id)

    index = index.copy()
    index["page_rendered"] = index["gland_id"].astype(str).isin(rendered_ids)
    index["page"] = index["gland_id"].astype(str).map(
        lambda value: f"glands/{value}.html" if value in rendered_ids else ""
    )
    index.to_csv(qc_index_path, index=False)

    atlas_html = out_path / "gland_instance_atlas.html"
    _write_index_page(atlas_html, index)
    return GlandInstanceAtlasResult(
        out_dir=out_path,
        atlas_html=atlas_html,
        gland_instance_qc_index_csv=qc_index_path,
        gland_page_count=len(rendered_ids),
    )


def _load_feature_lookup(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[str, dict[str, Any]] = {}
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        instance_id = str(props.get("slice_instance_id", ""))
        if not instance_id:
            continue
        lookup[instance_id] = feature
    return lookup


def _write_index_page(path: Path, index: pd.DataFrame) -> None:
    rows = []
    for _, row in index.iterrows():
        page = str(row.get("page", ""))
        gland_id = escape(str(row.get("gland_id", "")))
        if page:
            gland_cell = f'<a href="{escape(page)}">{gland_id}</a>'
        else:
            gland_cell = gland_id
        rows.append(
            "<tr>"
            f"<td>{gland_cell}</td>"
            f"<td>{escape(str(row.get('semantic_structure', '')))}</td>"
            f"<td>{escape(str(row.get('slice_count', '')))}</td>"
            f"<td>{escape(str(row.get('qc_flags', '')))}</td>"
            f"<td>{escape(str(row.get('branch_merge_candidate_count', '')))}</td>"
            f"<td>{escape(_fmt(row.get('area_median_um2')))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HistoSeg gland instance atlas</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 6px 8px; text-align: left; }}
    th {{ background: #f0f4f8; position: sticky; top: 0; }}
    input {{ margin: 0 0 12px 0; padding: 8px; width: 320px; }}
  </style>
</head>
<body>
  <h1>HistoSeg Gland Instance Atlas</h1>
  <p>Each row is a tracked gland/crypt instance reconstructed from lumen-seeded epithelial support.</p>
  <input id="filter" placeholder="Filter table..." onkeyup="filterRows()">
  <table id="atlas">
    <thead><tr><th>Gland</th><th>Structure</th><th>Slices</th><th>QC flags</th><th>Branch candidates</th><th>Median area um2</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <script>
    function filterRows() {{
      const q = document.getElementById('filter').value.toLowerCase();
      for (const row of document.querySelectorAll('#atlas tbody tr')) {{
        row.style.display = row.innerText.toLowerCase().includes(q) ? '' : 'none';
      }}
    }}
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _write_gland_page(
    path: Path,
    *,
    gland: dict[str, Any],
    tracks: pd.DataFrame,
    feature_lookup: dict[str, dict[str, Any]],
    z_visual_scale: float,
) -> None:
    traces = []
    lumen_x: list[float] = []
    lumen_y: list[float] = []
    lumen_z: list[float] = []
    for _, row in tracks.sort_values("slice_order").iterrows():
        instance_id = str(row["slice_instance_id"])
        feature = feature_lookup.get(instance_id)
        if feature is None:
            continue
        geom = shape(feature.get("geometry"))
        if geom.is_empty:
            continue
        coords = list(geom.exterior.coords) if geom.geom_type == "Polygon" else []
        if coords:
            x = [float(point[0]) for point in coords]
            y = [float(point[1]) for point in coords]
            z = [float(row["z_um"]) * z_visual_scale for _ in coords]
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "name": f"slice {int(row['slice_order'])}",
                    "x": x,
                    "y": y,
                    "z": z,
                    "line": {"width": 4},
                }
            )
        if pd.notna(row.get("lumen_centroid_x_um")) and pd.notna(row.get("lumen_centroid_y_um")):
            lumen_x.append(float(row["lumen_centroid_x_um"]))
            lumen_y.append(float(row["lumen_centroid_y_um"]))
            lumen_z.append(float(row["z_um"]) * z_visual_scale)
    if lumen_x:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": "lumen centers",
                "x": lumen_x,
                "y": lumen_y,
                "z": lumen_z,
                "marker": {"size": 5, "color": "#d1495b"},
            }
        )
    metrics_rows = []
    for key in [
        "gland_id",
        "semantic_structure",
        "slice_count",
        "z_span_um",
        "area_median_um2",
        "median_ring_support_score",
        "median_epithelial_marker_score",
        "qc_flags",
        "branch_merge_candidate_count",
    ]:
        metrics_rows.append(
            f"<tr><th>{escape(key)}</th><td>{escape(_fmt(gland.get(key)))}</td></tr>"
        )
    strip = _slice_strip_svg(tracks)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape(str(gland.get('gland_id', 'gland')))}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #1f2933; }}
    #plot {{ width: 100%; height: 620px; }}
    table {{ border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 5px 8px; text-align: left; }}
    .layout {{ display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 20px; }}
  </style>
</head>
<body>
  <p><a href="../gland_instance_atlas.html">Back to atlas</a></p>
  <h1>{escape(str(gland.get('gland_id', 'gland')))}</h1>
  <div class="layout">
    <div>
      <div id="plot"></div>
      <h2>Slice strip</h2>
      {strip}
    </div>
    <div>
      <h2>QC metrics</h2>
      <table>{''.join(metrics_rows)}</table>
      <h2>Per-slice rows</h2>
      {tracks[['slice_order','slice_instance_id','qc_flags','link_score']].to_html(index=False, escape=True)}
    </div>
  </div>
  <script>
    const traces = {json.dumps(traces)};
    Plotly.newPlot('plot', traces, {{
      scene: {{
        xaxis: {{title: 'x (um)'}},
        yaxis: {{title: 'y (um)'}},
        zaxis: {{title: 'z (visual scale x{z_visual_scale:g})'}},
        aspectmode: 'data'
      }},
      margin: {{l: 0, r: 0, b: 0, t: 10}}
    }}, {{responsive: true}});
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _slice_strip_svg(tracks: pd.DataFrame) -> str:
    rows = tracks.sort_values("slice_order")
    if rows.empty:
        return "<p>No slices.</p>"
    width = max(320, int(len(rows) * 60))
    height = 90
    min_order = int(rows["slice_order"].min())
    max_order = int(rows["slice_order"].max())
    span = max(max_order - min_order, 1)
    circles = []
    for _, row in rows.iterrows():
        x = 30 + (int(row["slice_order"]) - min_order) / span * (width - 60)
        color = "#2f80ed" if not str(row.get("qc_flags", "")) else "#d1495b"
        circles.append(
            f'<circle cx="{x:.1f}" cy="38" r="9" fill="{color}">'
            f'<title>slice {int(row["slice_order"])} {escape(str(row.get("qc_flags", "")))}</title></circle>'
            f'<text x="{x:.1f}" y="68" text-anchor="middle" font-size="11">{int(row["slice_order"])}</text>'
        )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<line x1="30" y1="38" x2="{width - 30}" y2="38" stroke="#bcccdc" stroke-width="2"/>'
        f"{''.join(circles)}</svg>"
    )


def _summarize_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    if tracks.empty:
        return pd.DataFrame()
    rows = []
    for gland_id, group in tracks.groupby("gland_id", sort=True):
        rows.append(
            {
                "gland_id": gland_id,
                "semantic_structure": str(group["semantic_structure"].mode().iloc[0]),
                "slice_count": int(group["slice_order"].nunique()),
                "z_span_um": float(group["z_um"].max() - group["z_um"].min()),
                "area_median_um2": float(group["area_um2"].median()),
                "median_ring_support_score": float(group["ring_support_score"].median()),
                "median_epithelial_marker_score": float(group["epithelial_marker_score"].median()),
                "branch_merge_candidate_count": int(
                    group["branch_merge_candidates"].fillna("").astype(str).map(bool).sum()
                ),
                "qc_flags": ";".join(
                    sorted(
                        {
                            flag
                            for value in group["qc_flags"].fillna("").astype(str)
                            for flag in value.split(";")
                            if flag
                        }
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


__all__ = [
    "DEFAULT_EPITHELIAL_MARKERS",
    "DEFAULT_EPITHELIAL_STRUCTURES",
    "GlandInstanceAtlasResult",
    "GlandInstanceDetectionResult",
    "render_gland_instance_atlas",
    "run_gland_instance_detection",
]
