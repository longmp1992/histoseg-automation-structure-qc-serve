from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape

PathLike = Union[str, Path]


@dataclass(frozen=True)
class LabelFreeBeforeAfterFigureConfig:
    fixed_geojson: PathLike
    moving_geojson: PathLike
    aligned_geojson: PathLike
    out_png: PathLike
    anchors_csv: PathLike | None = None
    out_svg: PathLike | None = None
    dpi: int = 300
    fixed_color: str = "#2b7ec8"
    moving_before_color: str = "#e53935"
    moving_after_color: str = "#34a853"
    anchor_color: str = "#8b5cf6"
    line_width: float = 0.7
    anchor_line_width: float = 0.45
    show_titles: bool = False
    invert_y: bool = True


@dataclass
class LabelFreeBeforeAfterFigureResult:
    out_png: Path
    out_svg: Path | None


def render_label_free_before_after_panel(
    cfg: LabelFreeBeforeAfterFigureConfig,
) -> LabelFreeBeforeAfterFigureResult:
    """Render a clean before/after partial-anchor alignment panel.

    The figure intentionally avoids text inside the plotting area so it can be
    used as a manuscript panel without label overlap. Anchor links are drawn
    only when the landmark CSV contains fixed, raw moving, or aligned moving
    centroid columns.
    """

    fixed_payload = _read_geojson(cfg.fixed_geojson)
    moving_payload = _read_geojson(cfg.moving_geojson)
    aligned_payload = _read_geojson(cfg.aligned_geojson)
    anchors = _read_anchor_rows(cfg.anchors_csv)

    out_png = Path(cfg.out_png).expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg = Path(cfg.out_svg).expanduser().resolve() if cfg.out_svg is not None else None
    if out_svg is not None:
        out_svg.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    before_ax, after_ax = axes
    _plot_geojson(before_ax, fixed_payload, color=cfg.fixed_color, linewidth=cfg.line_width)
    _plot_geojson(before_ax, moving_payload, color=cfg.moving_before_color, linewidth=cfg.line_width)
    _plot_anchor_links(
        before_ax,
        anchors,
        source_x="moving_centroid_x",
        source_y="moving_centroid_y",
        target_x="fixed_centroid_x",
        target_y="fixed_centroid_y",
        color=cfg.anchor_color,
        linewidth=cfg.anchor_line_width,
    )

    _plot_geojson(after_ax, fixed_payload, color=cfg.fixed_color, linewidth=cfg.line_width)
    _plot_geojson(after_ax, aligned_payload, color=cfg.moving_after_color, linewidth=cfg.line_width)
    _plot_anchor_links(
        after_ax,
        anchors,
        source_x="aligned_moving_centroid_x",
        source_y="aligned_moving_centroid_y",
        target_x="fixed_centroid_x",
        target_y="fixed_centroid_y",
        color=cfg.anchor_color,
        linewidth=cfg.anchor_line_width,
    )

    if cfg.show_titles:
        before_ax.set_title("Before alignment", fontsize=11)
        after_ax.set_title("After automatic partial-anchor alignment", fontsize=11)
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        if cfg.invert_y:
            ax.invert_yaxis()
    _set_shared_limits(axes, [fixed_payload, moving_payload, aligned_payload])

    fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
    if out_svg is not None:
        fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    return LabelFreeBeforeAfterFigureResult(out_png=out_png, out_svg=out_svg)


def _read_geojson(path: PathLike) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _read_anchor_rows(path: PathLike | None) -> pd.DataFrame | None:
    if path is None:
        return None
    anchor_path = Path(path).expanduser()
    if not anchor_path.exists():
        return None
    try:
        data = pd.read_csv(anchor_path)
    except Exception:
        return None
    return data if not data.empty else None


def _iter_polygons(payload: dict[str, Any]) -> Iterable[Polygon]:
    for feature in payload.get("features", []):
        try:
            geom = shape(feature.get("geometry"))
        except Exception:
            continue
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            yield geom
        elif isinstance(geom, MultiPolygon):
            yield from geom.geoms
        elif isinstance(geom, GeometryCollection):
            for part in geom.geoms:
                if isinstance(part, Polygon):
                    yield part
                elif isinstance(part, MultiPolygon):
                    yield from part.geoms


def _plot_geojson(ax: plt.Axes, payload: dict[str, Any], *, color: str, linewidth: float) -> None:
    for polygon in _iter_polygons(payload):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth, alpha=0.95)
        for interior in polygon.interiors:
            hx, hy = interior.xy
            ax.plot(hx, hy, color=color, linewidth=max(linewidth * 0.65, 0.25), alpha=0.55)


def _plot_anchor_links(
    ax: plt.Axes,
    anchors: pd.DataFrame | None,
    *,
    source_x: str,
    source_y: str,
    target_x: str,
    target_y: str,
    color: str,
    linewidth: float,
) -> None:
    if anchors is None:
        return
    required = {source_x, source_y, target_x, target_y}
    if not required.issubset(anchors.columns):
        return
    rows = anchors
    if "used_for_transform" in rows.columns:
        rows = rows.loc[rows["used_for_transform"].astype(str).str.lower().isin({"true", "1", "yes"})]
    for row in rows.itertuples(index=False):
        sx = getattr(row, source_x)
        sy = getattr(row, source_y)
        tx = getattr(row, target_x)
        ty = getattr(row, target_y)
        try:
            ax.plot([float(sx), float(tx)], [float(sy), float(ty)], color=color, linewidth=linewidth, alpha=0.55)
        except Exception:
            continue


def _set_shared_limits(axes: Iterable[plt.Axes], payloads: Iterable[dict[str, Any]]) -> None:
    bounds: list[tuple[float, float, float, float]] = []
    for payload in payloads:
        for polygon in _iter_polygons(payload):
            bounds.append(polygon.bounds)
    if not bounds:
        return
    minx = min(item[0] for item in bounds)
    miny = min(item[1] for item in bounds)
    maxx = max(item[2] for item in bounds)
    maxy = max(item[3] for item in bounds)
    pad_x = max((maxx - minx) * 0.04, 1.0)
    pad_y = max((maxy - miny) * 0.04, 1.0)
    for ax in axes:
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(maxy + pad_y, miny - pad_y)
