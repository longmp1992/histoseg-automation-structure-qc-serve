"""SpaceMap-style solid surface rendering for tracked gland instances."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
import json
import math
from typing import Any, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import draw, measure
from shapely.geometry import MultiPolygon, Polygon, shape


PathLike = Union[str, Path]

_PALETTE = (
    "#2f80ed",
    "#27ae60",
    "#f2994a",
    "#9b51e0",
    "#eb5757",
    "#00a6a6",
    "#607d8b",
    "#c2185b",
)


@dataclass(frozen=True)
class GlandSurfaceAtlasConfig:
    """Configuration for rendering gland instances as solid 3D surfaces."""

    gland_instance_dir: PathLike
    out_dir: PathLike
    aligned_cells_parquet: PathLike | None = None
    biology_dir: PathLike | None = None
    gland_ids: Sequence[str] = ()
    family_ids: Sequence[str] = ()
    max_glands: int = 30
    padding_um: float = 120.0
    voxel_size_xy_um: float = 6.0
    voxel_size_z_um: float | None = None
    z_interpolation_factor: int = 3
    z_visual_scale: float = 8.0
    surface_smoothing_iterations: int = 10
    surface_smoothing_lambda: float = 0.35
    max_faces_per_mesh: int = 50000
    max_vertices_per_mesh: int = 30000
    max_cells_per_view: int = 25000
    cell_color_by: str = "leiden_1_0"
    preset: str = "publication"
    export_meshes: bool = False
    transparent_shell: bool = False
    random_state: int = 0


@dataclass
class GlandSurfaceAtlasResult:
    out_dir: Path
    gland_surface_atlas_html: Path
    gland_surface_mesh_manifest_csv: Path
    rendered_gland_count: int
    figure_count: int
    mesh_export_count: int


@dataclass(frozen=True)
class _Mesh:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class _VolumeBuild:
    shell_volume: np.ndarray
    lumen_volume: np.ndarray
    origin_xyz_um: tuple[float, float, float]
    spacing_zyx_um: tuple[float, float, float]
    warnings: tuple[str, ...]


def render_gland_surface_atlas(cfg: GlandSurfaceAtlasConfig) -> GlandSurfaceAtlasResult:
    """Render tracked gland instances as smooth, browser-safe 3D meshes."""

    _validate_config(cfg)
    gland_dir = Path(cfg.gland_instance_dir).expanduser().resolve()
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gland_page_dir = out_dir / "glands"
    figure_dir = out_dir / "figures"
    mesh_dir = out_dir / "meshes"
    gland_page_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if cfg.export_meshes:
        mesh_dir.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_csv(gland_dir / "gland_instance_tracks.csv")
    gland_index_path = gland_dir / "gland_instance_qc_index.csv"
    gland_index = pd.read_csv(gland_index_path) if gland_index_path.exists() else _summarize_tracks(tracks)
    feature_lookup = _load_feature_lookup(gland_dir / "slice_gland_instances.geojson")
    family_index = _load_family_index(gland_dir, cfg.biology_dir)
    selected_glands = _select_gland_ids(
        tracks=tracks,
        gland_index=gland_index,
        family_index=family_index,
        gland_ids=cfg.gland_ids,
        family_ids=cfg.family_ids,
        max_glands=int(cfg.max_glands),
    )
    cells = _load_cells(cfg.aligned_cells_parquet)
    manifest_rows: list[dict[str, Any]] = []
    rendered_pages: list[dict[str, Any]] = []
    figure_count = 0
    mesh_export_count = 0
    for gland_id in selected_glands:
        gland_tracks = tracks.loc[tracks["gland_id"].astype(str) == gland_id].copy()
        if gland_tracks.empty:
            continue
        try:
            volume = _build_gland_volume(gland_tracks, feature_lookup, cfg)
            shell_mesh = _mesh_from_volume(
                volume.shell_volume,
                origin_xyz_um=volume.origin_xyz_um,
                spacing_zyx_um=volume.spacing_zyx_um,
                smoothing_iterations=int(cfg.surface_smoothing_iterations),
                smoothing_lambda=float(cfg.surface_smoothing_lambda),
                max_faces=int(cfg.max_faces_per_mesh),
                max_vertices=int(cfg.max_vertices_per_mesh),
            )
            lumen_mesh = (
                _mesh_from_volume(
                    volume.lumen_volume,
                    origin_xyz_um=volume.origin_xyz_um,
                    spacing_zyx_um=volume.spacing_zyx_um,
                    smoothing_iterations=max(1, int(cfg.surface_smoothing_iterations) // 2),
                    smoothing_lambda=float(cfg.surface_smoothing_lambda),
                    max_faces=max(2000, int(cfg.max_faces_per_mesh) // 3),
                    max_vertices=max(1000, int(cfg.max_vertices_per_mesh) // 3),
                )
                if volume.lumen_volume.any()
                else None
            )
            warnings = list(volume.warnings)
            if shell_mesh.faces.size == 0:
                warnings.append("empty_shell_mesh")
        except Exception as exc:  # pragma: no cover - defensive manifest path
            manifest_rows.append(_failed_manifest_row(gland_id, gland_tracks, str(exc)))
            continue

        local_cells = _local_cells_for_gland(cells, gland_tracks, cfg) if cells is not None else pd.DataFrame()
        page_path = gland_page_dir / f"{gland_id}_surface.html"
        figure_path = figure_dir / f"{gland_id}_surface.png"
        figure_written = _write_static_png(
            figure_path,
            gland_id=gland_id,
            shell_mesh=shell_mesh,
            lumen_mesh=lumen_mesh,
            tracks=gland_tracks,
            cfg=cfg,
        )
        if figure_written:
            figure_count += 1
        mesh_path = None
        if cfg.export_meshes and shell_mesh.faces.size:
            mesh_path = mesh_dir / f"{gland_id}_shell.ply"
            _write_ply(mesh_path, shell_mesh)
            mesh_export_count += 1
        _write_gland_surface_page(
            page_path,
            gland_id=gland_id,
            shell_mesh=shell_mesh,
            lumen_mesh=lumen_mesh,
            tracks=gland_tracks,
            cells=local_cells,
            figure_path=figure_path if figure_written else None,
            mesh_path=mesh_path,
            warnings=warnings,
            cfg=cfg,
        )
        manifest_rows.append(
            {
                "gland_id": gland_id,
                "semantic_structure": str(gland_tracks["semantic_structure"].mode().iloc[0]),
                "slice_count": int(gland_tracks["slice_order"].nunique()),
                "z_min_um": float(gland_tracks["z_um"].min()),
                "z_max_um": float(gland_tracks["z_um"].max()),
                "voxel_shape_zyx": "x".join(str(v) for v in volume.shell_volume.shape),
                "shell_vertex_count": int(len(shell_mesh.vertices)),
                "shell_face_count": int(len(shell_mesh.faces)),
                "lumen_vertex_count": int(len(lumen_mesh.vertices)) if lumen_mesh is not None else 0,
                "lumen_face_count": int(len(lumen_mesh.faces)) if lumen_mesh is not None else 0,
                "lumen_available": bool(lumen_mesh is not None and len(lumen_mesh.faces) > 0),
                "cell_overlay_count": int(len(local_cells)),
                "html": f"glands/{page_path.name}",
                "figure_png": f"figures/{figure_path.name}" if figure_written else "",
                "mesh_ply": f"meshes/{mesh_path.name}" if mesh_path is not None else "",
                "warnings": ";".join(warnings),
                "config_json": json.dumps(asdict(cfg), default=str),
            }
        )
        rendered_pages.append(
            {
                "gland_id": gland_id,
                "page": f"glands/{page_path.name}",
                "figure": f"figures/{figure_path.name}" if figure_written else "",
                "slice_count": int(gland_tracks["slice_order"].nunique()),
                "semantic_structure": str(gland_tracks["semantic_structure"].mode().iloc[0]),
                "warnings": ";".join(warnings),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "gland_surface_mesh_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    atlas_html = out_dir / "gland_surface_atlas.html"
    _write_surface_atlas(atlas_html, rendered_pages, manifest)
    return GlandSurfaceAtlasResult(
        out_dir=out_dir,
        gland_surface_atlas_html=atlas_html,
        gland_surface_mesh_manifest_csv=manifest_path,
        rendered_gland_count=len(rendered_pages),
        figure_count=figure_count,
        mesh_export_count=mesh_export_count,
    )


def _build_gland_volume(
    gland_tracks: pd.DataFrame,
    feature_lookup: Mapping[str, Mapping[str, Any]],
    cfg: GlandSurfaceAtlasConfig,
) -> _VolumeBuild:
    polygons: list[tuple[pd.Series, Polygon | MultiPolygon, Polygon | None]] = []
    for _, row in gland_tracks.sort_values("slice_order").iterrows():
        feature = feature_lookup.get(str(row["slice_instance_id"]))
        if feature is None:
            continue
        geom = shape(feature.get("geometry"))
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        lumen = _parse_lumen_polygon(row.get("lumen_polygon_coordinates"))
        if lumen is None:
            props = feature.get("properties") or {}
            lumen = _parse_lumen_polygon(props.get("lumen_polygon_coordinates"))
        polygons.append((row, geom, lumen))
    if not polygons:
        raise ValueError("No renderable gland polygons found.")

    bounds = [geom.bounds for _, geom, _ in polygons]
    x_min = min(item[0] for item in bounds) - float(cfg.padding_um)
    y_min = min(item[1] for item in bounds) - float(cfg.padding_um)
    x_max = max(item[2] for item in bounds) + float(cfg.padding_um)
    y_max = max(item[3] for item in bounds) + float(cfg.padding_um)
    voxel_xy = float(cfg.voxel_size_xy_um)
    width = max(4, int(math.ceil((x_max - x_min) / voxel_xy)) + 1)
    height = max(4, int(math.ceil((y_max - y_min) / voxel_xy)) + 1)
    z_values = sorted({float(row["z_um"]) for row, _, _ in polygons})
    z_spacing = _infer_z_spacing(z_values, cfg.voxel_size_z_um)
    z_min = min(z_values)
    z_max = max(z_values)
    depth = max(1, int(round((z_max - z_min) / z_spacing)) + 1)
    outer = np.zeros((depth, height, width), dtype=bool)
    lumen_volume = np.zeros_like(outer)
    warnings: list[str] = []
    for row, geom, lumen in polygons:
        z_index = int(round((float(row["z_um"]) - z_min) / z_spacing))
        z_index = min(max(z_index, 0), depth - 1)
        outer[z_index] |= _rasterize_geometry(geom, x_min=x_min, y_min=y_min, pixel_size_um=voxel_xy, shape_yx=(height, width))
        if lumen is not None and not lumen.is_empty:
            lumen_volume[z_index] |= _rasterize_geometry(
                lumen,
                x_min=x_min,
                y_min=y_min,
                pixel_size_um=voxel_xy,
                shape_yx=(height, width),
            )
    if len(z_values) < 3:
        warnings.append("low_z_support")
    if not lumen_volume.any():
        warnings.append("no_lumen_surface")

    factor = max(1, int(cfg.z_interpolation_factor))
    z_spacing_eff = z_spacing
    if factor > 1 and outer.shape[0] > 1:
        outer = ndi.zoom(outer.astype(float), (factor, 1, 1), order=1) >= 0.5
        lumen_volume = ndi.zoom(lumen_volume.astype(float), (factor, 1, 1), order=1) >= 0.5
        z_spacing_eff = z_spacing / factor
    shell = outer & ~lumen_volume
    if shell.any():
        shell = ndi.binary_closing(shell, structure=np.ones((1, 2, 2), dtype=bool))
    return _VolumeBuild(
        shell_volume=shell,
        lumen_volume=lumen_volume,
        origin_xyz_um=(x_min, y_min, z_min),
        spacing_zyx_um=(z_spacing_eff, voxel_xy, voxel_xy),
        warnings=tuple(warnings),
    )


def _mesh_from_volume(
    volume: np.ndarray,
    *,
    origin_xyz_um: tuple[float, float, float],
    spacing_zyx_um: tuple[float, float, float],
    smoothing_iterations: int,
    smoothing_lambda: float,
    max_faces: int,
    max_vertices: int,
) -> _Mesh:
    if not volume.any():
        return _Mesh(vertices=np.empty((0, 3), dtype=float), faces=np.empty((0, 3), dtype=np.int32))
    padded = np.pad(volume.astype(float), 1, mode="constant", constant_values=0.0)
    step_size = max(1, int(math.ceil(float(volume.sum()) / 300000.0)))
    try:
        vertices_zyx, faces, _, _ = measure.marching_cubes(
            padded,
            level=0.5,
            spacing=spacing_zyx_um,
            step_size=step_size,
        )
    except ValueError:
        return _Mesh(vertices=np.empty((0, 3), dtype=float), faces=np.empty((0, 3), dtype=np.int32))
    z_spacing, y_spacing, x_spacing = spacing_zyx_um
    x_origin, y_origin, z_origin = origin_xyz_um
    vertices = np.column_stack(
        [
            vertices_zyx[:, 2] + x_origin - x_spacing,
            vertices_zyx[:, 1] + y_origin - y_spacing,
            vertices_zyx[:, 0] + z_origin - z_spacing,
        ]
    )
    faces = faces.astype(np.int32, copy=False)
    if smoothing_iterations > 0 and len(vertices) > 0 and len(faces) > 0:
        vertices = _laplacian_smooth(vertices, faces, iterations=smoothing_iterations, lam=smoothing_lambda)
    vertices, faces = _simplify_mesh(vertices, faces, max_faces=max_faces, max_vertices=max_vertices)
    return _Mesh(vertices=vertices, faces=faces)


def _laplacian_smooth(vertices: np.ndarray, faces: np.ndarray, *, iterations: int, lam: float) -> np.ndarray:
    vertices = vertices.astype(float, copy=True)
    if faces.size == 0:
        return vertices
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
            faces[:, [1, 0]],
            faces[:, [2, 1]],
            faces[:, [0, 2]],
        ],
        axis=0,
    )
    source = edges[:, 0]
    target = edges[:, 1]
    lam = min(max(float(lam), 0.0), 1.0)
    for _ in range(int(iterations)):
        sums = np.zeros_like(vertices)
        counts = np.zeros(len(vertices), dtype=float)
        np.add.at(sums, source, vertices[target])
        np.add.at(counts, source, 1.0)
        valid = counts > 0
        avg = vertices.copy()
        avg[valid] = sums[valid] / counts[valid, None]
        vertices[valid] = vertices[valid] + lam * (avg[valid] - vertices[valid])
    return vertices


def _simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_faces: int,
    max_vertices: int,
) -> tuple[np.ndarray, np.ndarray]:
    if faces.size == 0 or vertices.size == 0:
        return vertices, faces
    max_faces = max(4, int(max_faces))
    max_vertices = max(4, int(max_vertices))
    if len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
        faces = faces[indices]
    vertices, faces = _compact_mesh(vertices, faces)
    while len(vertices) > max_vertices and len(faces) > 4:
        faces = faces[::2]
        vertices, faces = _compact_mesh(vertices, faces)
    return vertices, faces


def _compact_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(faces.ravel())
    mapping = np.full(len(vertices), -1, dtype=np.int32)
    mapping[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used], mapping[faces]


def _rasterize_geometry(
    geom: Polygon | MultiPolygon,
    *,
    x_min: float,
    y_min: float,
    pixel_size_um: float,
    shape_yx: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape_yx, dtype=bool)
    polygons = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    for polygon in polygons:
        if polygon.is_empty:
            continue
        _fill_ring(mask, polygon.exterior.coords, x_min=x_min, y_min=y_min, pixel_size_um=pixel_size_um)
        for interior in polygon.interiors:
            hole = np.zeros(shape_yx, dtype=bool)
            _fill_ring(hole, interior.coords, x_min=x_min, y_min=y_min, pixel_size_um=pixel_size_um)
            mask &= ~hole
    return mask


def _fill_ring(
    mask: np.ndarray,
    coords: Any,
    *,
    x_min: float,
    y_min: float,
    pixel_size_um: float,
) -> None:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return
    cols = (arr[:, 0] - x_min) / pixel_size_um
    rows = (arr[:, 1] - y_min) / pixel_size_um
    rr, cc = draw.polygon(rows, cols, shape=mask.shape)
    mask[rr, cc] = True


def _write_gland_surface_page(
    path: Path,
    *,
    gland_id: str,
    shell_mesh: _Mesh,
    lumen_mesh: _Mesh | None,
    tracks: pd.DataFrame,
    cells: pd.DataFrame,
    figure_path: Path | None,
    mesh_path: Path | None,
    warnings: Sequence[str],
    cfg: GlandSurfaceAtlasConfig,
) -> None:
    traces = []
    if shell_mesh.faces.size:
        traces.append(
            _mesh_trace(
                shell_mesh,
                name="epithelial shell",
                color="#d9dde3",
                opacity=0.42 if cfg.transparent_shell else 0.88,
                z_visual_scale=float(cfg.z_visual_scale),
            )
        )
    if lumen_mesh is not None and lumen_mesh.faces.size:
        traces.append(
            _mesh_trace(
                lumen_mesh,
                name="lumen surface",
                color="#111827",
                opacity=1.0,
                z_visual_scale=float(cfg.z_visual_scale),
            )
        )
    if cfg.preset == "qc":
        traces.extend(_slice_centroid_traces(tracks, cfg))
    if not cells.empty:
        traces.extend(_cell_traces(cells, cfg))
    links = []
    if figure_path is not None:
        links.append(f'<a href="../figures/{escape(figure_path.name)}">static PNG</a>')
    if mesh_path is not None:
        links.append(f'<a href="../meshes/{escape(mesh_path.name)}">shell PLY</a>')
    metrics = {
        "gland_id": gland_id,
        "slice_count": int(tracks["slice_order"].nunique()),
        "z_span_um": float(tracks["z_um"].max() - tracks["z_um"].min()),
        "shell_vertices": len(shell_mesh.vertices),
        "shell_faces": len(shell_mesh.faces),
        "lumen_faces": len(lumen_mesh.faces) if lumen_mesh is not None else 0,
        "cell_overlay_count": len(cells),
        "warnings": ";".join(warnings),
    }
    metric_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(_fmt(value))}</td></tr>"
        for key, value in metrics.items()
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape(gland_id)} surface</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #1f2933; }}
    #plot {{ height: 760px; border: 1px solid #d9e2ec; }}
    table {{ border-collapse: collapse; font-size: 13px; margin-top: 14px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 5px 8px; text-align: left; }}
    .links a {{ margin-right: 14px; }}
  </style>
</head>
<body>
  <p><a href="../gland_surface_atlas.html">Back to surface atlas</a></p>
  <h1>{escape(gland_id)} solid gland surface</h1>
  <p class="links">{' '.join(links)}</p>
  <div id="plot"></div>
  <table>{metric_rows}</table>
  <script>
    const traces = {json.dumps(traces)};
    Plotly.newPlot('plot', traces, {{
      scene: {{
        xaxis: {{title: 'x (um)', showbackground: false}},
        yaxis: {{title: 'y (um)', showbackground: false}},
        zaxis: {{title: 'z (visual x{float(cfg.z_visual_scale):g})', showbackground: false}},
        aspectmode: 'data'
      }},
      paper_bgcolor: 'white',
      plot_bgcolor: 'white',
      margin: {{l: 0, r: 0, b: 0, t: 10}},
      showlegend: true
    }}, {{responsive: true}});
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _mesh_trace(
    mesh: _Mesh,
    *,
    name: str,
    color: str,
    opacity: float,
    z_visual_scale: float,
) -> dict[str, Any]:
    return {
        "type": "mesh3d",
        "name": name,
        "x": mesh.vertices[:, 0].round(3).tolist(),
        "y": mesh.vertices[:, 1].round(3).tolist(),
        "z": (mesh.vertices[:, 2] * float(z_visual_scale)).round(3).tolist(),
        "i": mesh.faces[:, 0].astype(int).tolist(),
        "j": mesh.faces[:, 1].astype(int).tolist(),
        "k": mesh.faces[:, 2].astype(int).tolist(),
        "color": color,
        "opacity": float(opacity),
        "flatshading": False,
        "lighting": {"ambient": 0.36, "diffuse": 0.72, "specular": 0.28, "roughness": 0.55},
        "hoverinfo": "skip",
    }


def _slice_centroid_traces(tracks: pd.DataFrame, cfg: GlandSurfaceAtlasConfig) -> list[dict[str, Any]]:
    group = tracks.sort_values("slice_order")
    return [
        {
            "type": "scatter3d",
            "mode": "lines+markers",
            "name": "slice centroids",
            "x": group["centroid_x_um"].astype(float).round(3).tolist(),
            "y": group["centroid_y_um"].astype(float).round(3).tolist(),
            "z": (group["z_um"].astype(float) * float(cfg.z_visual_scale)).round(3).tolist(),
            "marker": {"size": 4, "color": "#2f80ed"},
            "line": {"width": 3, "color": "#2f80ed"},
            "text": [f"slice {int(row.slice_order)}<br>{row.slice_instance_id}" for row in group.itertuples(index=False)],
            "hovertemplate": "%{text}<extra></extra>",
        }
    ]


def _cell_traces(cells: pd.DataFrame, cfg: GlandSurfaceAtlasConfig) -> list[dict[str, Any]]:
    if cells.empty:
        return []
    color_col = cfg.cell_color_by if cfg.cell_color_by in cells.columns else None
    if color_col is None:
        return [
            _scatter_cells(cells, name="nearby cells", color="#2f80ed", cfg=cfg)
        ]
    traces: list[dict[str, Any]] = []
    for index, (label, group) in enumerate(cells.groupby(color_col, sort=True)):
        if index >= 24:
            break
        traces.append(_scatter_cells(group, name=str(label), color=_PALETTE[index % len(_PALETTE)], cfg=cfg))
    return traces


def _scatter_cells(cells: pd.DataFrame, *, name: str, color: str, cfg: GlandSurfaceAtlasConfig) -> dict[str, Any]:
    return {
        "type": "scatter3d",
        "mode": "markers",
        "name": name,
        "x": cells["x_3d_um"].astype(float).round(3).tolist(),
        "y": cells["y_3d_um"].astype(float).round(3).tolist(),
        "z": (cells["z_um"].astype(float) * float(cfg.z_visual_scale)).round(3).tolist(),
        "marker": {"size": 2.2 if cfg.preset == "qc" else 1.6, "color": color, "opacity": 0.58},
        "hoverinfo": "skip",
    }


def _write_surface_atlas(path: Path, pages: Sequence[Mapping[str, Any]], manifest: pd.DataFrame) -> None:
    rows = []
    for page in pages:
        figure = str(page.get("figure", ""))
        thumb = f'<img src="{escape(figure)}" style="height:120px">' if figure else ""
        rows.append(
            "<tr>"
            f"<td><a href=\"{escape(str(page.get('page', '')))}\">{escape(str(page.get('gland_id', '')))}</a></td>"
            f"<td>{escape(str(page.get('semantic_structure', '')))}</td>"
            f"<td>{escape(str(page.get('slice_count', '')))}</td>"
            f"<td>{escape(str(page.get('warnings', '')))}</td>"
            f"<td>{thumb}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HistoSeg gland surface atlas</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; position: sticky; top: 0; }}
    input {{ padding: 8px; width: 340px; margin-bottom: 12px; }}
  </style>
</head>
<body>
  <h1>HistoSeg Solid 3D Gland Surface Atlas</h1>
  <p>Geometry-first SpaceMap-style gland rendering. H&E is not required; surfaces are reconstructed from tracked Xenium gland/lumen polygons.</p>
  <p>Rendered glands: {len(pages)}. Manifest rows: {0 if manifest.empty else len(manifest)}.</p>
  <input id="filter" placeholder="Filter glands..." onkeyup="filterRows()">
  <table id="atlas">
    <thead><tr><th>Gland</th><th>Structure</th><th>Slices</th><th>Warnings</th><th>Preview</th></tr></thead>
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


def _write_static_png(
    path: Path,
    *,
    gland_id: str,
    shell_mesh: _Mesh,
    lumen_mesh: _Mesh | None,
    tracks: pd.DataFrame,
    cfg: GlandSurfaceAtlasConfig,
) -> bool:
    if shell_mesh.faces.size == 0:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception:
        return False
    try:
        fig = plt.figure(figsize=(7.0, 5.6), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        shell_display = _scaled_mesh_for_display(shell_mesh, z_visual_scale=float(cfg.z_visual_scale))
        lumen_display = (
            _scaled_mesh_for_display(lumen_mesh, z_visual_scale=float(cfg.z_visual_scale))
            if lumen_mesh is not None
            else None
        )
        _add_mesh_collection(ax, shell_display, facecolor=(0.82, 0.84, 0.88, 0.92), edgecolor=(0.75, 0.78, 0.82, 0.05), Poly3DCollection=Poly3DCollection)
        if lumen_display is not None and lumen_display.faces.size:
            _add_mesh_collection(ax, lumen_display, facecolor=(0.03, 0.04, 0.06, 1.0), edgecolor=(0.03, 0.04, 0.06, 0.1), Poly3DCollection=Poly3DCollection)
        ax.set_title(gland_id)
        ax.set_xlabel("x um")
        ax.set_ylabel("y um")
        ax.set_zlabel(f"z x{float(cfg.z_visual_scale):g}")
        ax.view_init(elev=24, azim=-58)
        _set_axes_equal(ax, shell_display.vertices)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return False


def _add_mesh_collection(ax: Any, mesh: _Mesh, *, facecolor: Any, edgecolor: Any, Poly3DCollection: Any) -> None:
    faces = mesh.faces
    if len(faces) > 8000:
        faces = faces[np.linspace(0, len(faces) - 1, 8000, dtype=int)]
    collection = Poly3DCollection(mesh.vertices[faces], facecolor=facecolor, edgecolor=edgecolor, linewidth=0.02)
    ax.add_collection3d(collection)


def _scaled_mesh_for_display(mesh: _Mesh, *, z_visual_scale: float) -> _Mesh:
    vertices = mesh.vertices.copy()
    vertices[:, 2] *= float(z_visual_scale)
    return _Mesh(vertices=vertices, faces=mesh.faces)


def _set_axes_equal(ax: Any, vertices: np.ndarray) -> None:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    radius = max(radius, 1.0)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _write_ply(path: Path, mesh: _Mesh) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(mesh.vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(mesh.faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for vertex in mesh.vertices:
            handle.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in mesh.faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def _load_feature_lookup(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[str, dict[str, Any]] = {}
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        instance_id = str(props.get("slice_instance_id", ""))
        if instance_id:
            lookup[instance_id] = feature
    return lookup


def _load_family_index(gland_dir: Path, biology_dir: PathLike | None) -> pd.DataFrame:
    candidates = [
        gland_dir / "gland_family_index.csv",
    ]
    if biology_dir is not None:
        candidates.append(Path(biology_dir).expanduser().resolve() / "gland_family_index.csv")
    candidates.extend(
        sorted(
            gland_dir.glob("gland_position_atlas*/gland_family_index.csv"),
            key=lambda path: path.parent.name,
            reverse=True,
        )
    )
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def _select_gland_ids(
    *,
    tracks: pd.DataFrame,
    gland_index: pd.DataFrame,
    family_index: pd.DataFrame,
    gland_ids: Sequence[str],
    family_ids: Sequence[str],
    max_glands: int,
) -> list[str]:
    requested = [str(item) for item in gland_ids if str(item)]
    if family_ids and not family_index.empty and "member_gland_ids" in family_index:
        families = {str(item) for item in family_ids}
        for _, row in family_index.loc[family_index["gland_family_id"].astype(str).isin(families)].iterrows():
            requested.extend([item for item in str(row["member_gland_ids"]).split(";") if item])
    valid = set(tracks["gland_id"].astype(str).unique())
    if requested:
        return [item for item in dict.fromkeys(requested) if item in valid]

    index = gland_index.copy()
    if index.empty:
        return sorted(valid)[:max_glands]
    index["branch_merge_candidate_count"] = pd.to_numeric(
        index.get("branch_merge_candidate_count", 0),
        errors="coerce",
    ).fillna(0)
    index["slice_count"] = pd.to_numeric(index.get("slice_count", 0), errors="coerce").fillna(0)
    high_conf = index.loc[(index["slice_count"] >= 3) & (index["branch_merge_candidate_count"] == 0)].copy()
    high_conf = high_conf.sort_values(["slice_count", "gland_id"], ascending=[False, True])
    fission = index.loc[index["branch_merge_candidate_count"] > 0].copy()
    fission = fission.sort_values(["branch_merge_candidate_count", "slice_count"], ascending=[False, False])
    half = max(1, int(max_glands) // 2)
    selected = list(high_conf["gland_id"].astype(str).head(half))
    selected.extend(list(fission["gland_id"].astype(str).head(max_glands - len(selected))))
    if len(selected) < max_glands:
        selected.extend(list(index["gland_id"].astype(str).head(max_glands - len(selected))))
    return [item for item in dict.fromkeys(selected) if item in valid][:max_glands]


def _load_cells(path: PathLike | None) -> pd.DataFrame | None:
    if path is None:
        return None
    cell_path = Path(path).expanduser().resolve()
    if not cell_path.exists():
        raise FileNotFoundError(cell_path)
    return pd.read_parquet(cell_path)


def _local_cells_for_gland(
    cells: pd.DataFrame,
    tracks: pd.DataFrame,
    cfg: GlandSurfaceAtlasConfig,
) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame()
    x_min = float(tracks["centroid_x_um"].min()) - float(cfg.padding_um) * 2.0
    x_max = float(tracks["centroid_x_um"].max()) + float(cfg.padding_um) * 2.0
    y_min = float(tracks["centroid_y_um"].min()) - float(cfg.padding_um) * 2.0
    y_max = float(tracks["centroid_y_um"].max()) + float(cfg.padding_um) * 2.0
    z_min = float(tracks["z_um"].min()) - 1e-6
    z_max = float(tracks["z_um"].max()) + 1e-6
    local = cells.loc[
        (cells["x_3d_um"].astype(float) >= x_min)
        & (cells["x_3d_um"].astype(float) <= x_max)
        & (cells["y_3d_um"].astype(float) >= y_min)
        & (cells["y_3d_um"].astype(float) <= y_max)
        & (cells["z_um"].astype(float) >= z_min)
        & (cells["z_um"].astype(float) <= z_max)
    ].copy()
    if len(local) > int(cfg.max_cells_per_view):
        local = local.sample(n=int(cfg.max_cells_per_view), random_state=int(cfg.random_state))
    return local


def _parse_lumen_polygon(value: Any) -> Polygon | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        coords = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return None
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return None
    polygon = Polygon(arr)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        return None
    return polygon


def _infer_z_spacing(z_values: Sequence[float], configured: float | None) -> float:
    if configured is not None:
        return float(configured)
    if len(z_values) > 1:
        diffs = np.diff(sorted(z_values))
        positive = diffs[diffs > 0]
        if positive.size:
            return float(np.median(positive))
    return 5.0


def _summarize_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gland_id, group in tracks.groupby("gland_id", sort=True):
        rows.append(
            {
                "gland_id": gland_id,
                "semantic_structure": str(group["semantic_structure"].mode().iloc[0]),
                "slice_count": int(group["slice_order"].nunique()),
                "branch_merge_candidate_count": int(
                    group["branch_merge_candidates"].fillna("").astype(str).map(bool).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _failed_manifest_row(gland_id: str, tracks: pd.DataFrame, message: str) -> dict[str, Any]:
    return {
        "gland_id": gland_id,
        "semantic_structure": str(tracks["semantic_structure"].mode().iloc[0]) if not tracks.empty else "",
        "slice_count": int(tracks["slice_order"].nunique()) if not tracks.empty else 0,
        "z_min_um": float(tracks["z_um"].min()) if not tracks.empty else math.nan,
        "z_max_um": float(tracks["z_um"].max()) if not tracks.empty else math.nan,
        "voxel_shape_zyx": "",
        "shell_vertex_count": 0,
        "shell_face_count": 0,
        "lumen_vertex_count": 0,
        "lumen_face_count": 0,
        "lumen_available": False,
        "cell_overlay_count": 0,
        "html": "",
        "figure_png": "",
        "mesh_ply": "",
        "warnings": f"render_failed:{message}",
        "config_json": "",
    }


def _fmt(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def _validate_config(cfg: GlandSurfaceAtlasConfig) -> None:
    gland_dir = Path(cfg.gland_instance_dir).expanduser()
    if not gland_dir.exists():
        raise FileNotFoundError(gland_dir)
    for name in ("gland_instance_tracks.csv", "slice_gland_instances.geojson"):
        path = gland_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
    if cfg.max_glands <= 0 and not cfg.gland_ids and not cfg.family_ids:
        raise ValueError("`max_glands` must be positive unless explicit gland or family IDs are provided.")
    if cfg.padding_um < 0:
        raise ValueError("`padding_um` must be non-negative.")
    if cfg.voxel_size_xy_um <= 0:
        raise ValueError("`voxel_size_xy_um` must be positive.")
    if cfg.voxel_size_z_um is not None and cfg.voxel_size_z_um <= 0:
        raise ValueError("`voxel_size_z_um` must be positive when provided.")
    if cfg.z_interpolation_factor < 1:
        raise ValueError("`z_interpolation_factor` must be at least 1.")
    if cfg.z_visual_scale <= 0:
        raise ValueError("`z_visual_scale` must be positive.")
    if cfg.max_faces_per_mesh <= 0 or cfg.max_vertices_per_mesh <= 0:
        raise ValueError("Mesh face and vertex limits must be positive.")
    if cfg.preset not in {"publication", "qc"}:
        raise ValueError("`preset` must be 'publication' or 'qc'.")


__all__ = [
    "GlandSurfaceAtlasConfig",
    "GlandSurfaceAtlasResult",
    "render_gland_surface_atlas",
]
