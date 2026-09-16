from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import pandas as pd
from shapely import affinity
from shapely.geometry import shape

from .multislice import _feature_group, _iter_polygons, _read_geojson, _valid_polygonal_geometry

PathLike = Union[str, Path]


@dataclass(frozen=True)
class SpatialData3DExportConfig:
    """Export a HistoSeg 3D stack to SpatialData for napari-spatialdata 3D viewing."""

    stack_root: PathLike
    out_zarr: PathLike | None = None
    aligned_cells_parquet: PathLike | None = None
    group_property: str = "structure"
    xenium_pixel_size_um: float = 0.2125
    include_contour_points: bool = True
    include_cells: bool = False
    max_cells: int | None = 300_000
    cell_x_column: str = "x_aligned_um"
    cell_y_column: str = "y_aligned_um"
    cell_z_column: str = "z_um"
    overwrite: bool = False


@dataclass
class SpatialData3DExportResult:
    out_zarr: Path
    summary_json: Path
    shape_count: int
    contour_point_count: int
    cell_count: int


def export_stack_to_spatialdata_3d(
    cfg: SpatialData3DExportConfig,
) -> SpatialData3DExportResult:
    """Write aligned HistoSeg contours/points as a SpatialData Zarr store.

    The exported shapes are 2.5D GeoDataFrame polygons with a scalar ``z`` column.
    napari-spatialdata PR #393 renders these as 3D vector layers when z is not
    discarded in the plugin UI.
    """

    try:
        import dask

        # spatialdata currently rejects the dask-expr dataframe backend. Set the
        # legacy dataframe backend before importing spatialdata or dask.dataframe.
        dask.config.set({"dataframe.query-planning": False})
        import dask.dataframe as dd
        import geopandas as gpd
        from spatialdata import SpatialData
        from spatialdata.models import PointsModel, ShapesModel
    except Exception as exc:  # pragma: no cover - depends on optional GUI stack.
        raise ImportError(
            "Exporting to SpatialData requires optional packages: spatialdata, "
            "geopandas, and dask. Install napari-spatialdata from a version that "
            "includes scverse/napari-spatialdata#393 to view 3D points and 2.5D shapes."
        ) from exc

    stack_root = Path(cfg.stack_root).expanduser().resolve()
    out_zarr = (
        Path(cfg.out_zarr).expanduser().resolve()
        if cfg.out_zarr is not None
        else stack_root / "spatialdata_3d.zarr"
    )
    if out_zarr.exists() and not cfg.overwrite:
        raise FileExistsError(f"{out_zarr} already exists. Use overwrite=True.")

    aligned_manifest = stack_root / "aligned_slice_manifest.csv"
    if not aligned_manifest.exists():
        raise FileNotFoundError(aligned_manifest)
    manifest = pd.read_csv(aligned_manifest)

    shape_rows = _collect_aligned_contour_shape_rows(
        manifest,
        group_property=cfg.group_property,
        xenium_pixel_size_um=cfg.xenium_pixel_size_um,
    )
    if not shape_rows:
        raise ValueError("No polygonal aligned contours were found for SpatialData export.")

    shapes_gdf = gpd.GeoDataFrame(shape_rows, geometry="geometry")
    shapes = ShapesModel.parse(shapes_gdf)

    points: dict[str, Any] = {}
    contour_point_count = 0
    contour_points_path = stack_root / "aligned_contour_3d_points.csv"
    if cfg.include_contour_points and contour_points_path.exists():
        contour_df = pd.read_csv(contour_points_path)
        if {"x_um", "y_um", "z_um"}.issubset(contour_df.columns):
            contour_df = contour_df.rename(
                columns={"x_um": "x", "y_um": "y", "z_um": "z"}
            )
            contour_point_count = int(len(contour_df))
            points["aligned_contour_points"] = PointsModel.parse(
                dd.from_pandas(contour_df, npartitions=max(1, min(16, len(contour_df) // 50_000 + 1))),
                coordinates={"x": "x", "y": "y", "z": "z"},
            )

    cell_count = 0
    if cfg.include_cells:
        cells_path = Path(cfg.aligned_cells_parquet).expanduser().resolve() if cfg.aligned_cells_parquet else stack_root / "aligned_leiden_3d_cells.parquet"
        if cells_path.exists():
            cell_df = pd.read_parquet(cells_path)
            required = {cfg.cell_x_column, cfg.cell_y_column, cfg.cell_z_column}
            missing = required.difference(cell_df.columns)
            if missing:
                raise ValueError(
                    "Aligned cell parquet is missing required coordinate columns: "
                    + ", ".join(sorted(missing))
                )
            if cfg.max_cells is not None and len(cell_df) > cfg.max_cells:
                cell_df = cell_df.sample(n=int(cfg.max_cells), random_state=0)
            cell_df = cell_df.rename(
                columns={
                    cfg.cell_x_column: "x",
                    cfg.cell_y_column: "y",
                    cfg.cell_z_column: "z",
                }
            )
            cell_count = int(len(cell_df))
            points["aligned_cells"] = PointsModel.parse(
                dd.from_pandas(cell_df, npartitions=max(1, min(16, len(cell_df) // 50_000 + 1))),
                coordinates={"x": "x", "y": "y", "z": "z"},
            )

    sdata = SpatialData(
        shapes={"aligned_contours_2_5d": shapes},
        points=points,
    )
    sdata.write(str(out_zarr), overwrite=cfg.overwrite)

    summary = {
        "stack_root": str(stack_root),
        "out_zarr": str(out_zarr),
        "shape_count": int(len(shape_rows)),
        "contour_point_count": int(contour_point_count),
        "cell_count": int(cell_count),
        "napari_spatialdata_note": (
            "Open this SpatialData store with napari-spatialdata main or a release "
            "including scverse/napari-spatialdata#393. Leave z enabled for 3D "
            "points and 2.5D shapes."
        ),
        "layers": {
            "shapes": ["aligned_contours_2_5d"],
            "points": list(points.keys()),
        },
    }
    summary_json = out_zarr.parent / f"{out_zarr.name}_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return SpatialData3DExportResult(
        out_zarr=out_zarr,
        summary_json=summary_json,
        shape_count=int(len(shape_rows)),
        contour_point_count=contour_point_count,
        cell_count=cell_count,
    )


def _collect_aligned_contour_shape_rows(
    aligned_manifest: pd.DataFrame,
    *,
    group_property: str,
    xenium_pixel_size_um: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, manifest_row in aligned_manifest.iterrows():
        aligned_geojson = manifest_row.get("aligned_geojson")
        if pd.isna(aligned_geojson):
            continue
        payload = _read_geojson(Path(str(aligned_geojson)))
        z_um = float(manifest_row["z_um"])
        order = int(manifest_row["order"])
        sample_id = str(manifest_row["sample_id"])
        for feature_index, feature in enumerate(payload.get("features", [])):
            properties: Mapping[str, Any] = feature.get("properties") or {}
            group = _feature_group(dict(properties), group_property)
            geom = _valid_polygonal_geometry(shape(feature["geometry"]))
            if geom is None:
                continue
            scaled = affinity.scale(
                geom,
                xfact=float(xenium_pixel_size_um),
                yfact=float(xenium_pixel_size_um),
                origin=(0.0, 0.0),
            )
            for polygon_index, polygon in enumerate(_iter_polygons(scaled)):
                rows.append(
                    {
                        "geometry": polygon,
                        "z": z_um,
                        "slice_order": order,
                        "sample_id": sample_id,
                        "structure": None if group is None else str(group),
                        "feature_index": feature_index,
                        "polygon_index": polygon_index,
                    }
                )
    return rows
