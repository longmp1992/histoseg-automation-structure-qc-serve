"""Contour topology analysis for adjacency and enclosure relationships."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

__all__ = [
    "ContourTopologyConfig",
    "ContourTopologyResult",
    "summarize_contour_topology",
]

_GEOMETRY_EPSILON = 1e-9


@dataclass(frozen=True)
class ContourTopologyConfig:
    """Configuration for contour topology analysis."""

    contour_id_col: str = "contour_id"
    geometry_col: str = "geometry"
    groupby: str | None = None
    boundary_tolerance: float = 1.0
    min_shared_boundary: float = 0.0
    enclosure_min_fraction: float = 0.95


@dataclass
class ContourTopologyResult:
    """Topology tables describing contour adjacency and enclosure."""

    boundary_overlap: pd.DataFrame
    enclosure: pd.DataFrame
    contour_summary: pd.DataFrame
    group_boundary_overlap: pd.DataFrame
    group_enclosure: pd.DataFrame


def summarize_contour_topology(
    contours: pd.DataFrame,
    *,
    contour_id_col: str = "contour_id",
    geometry_col: str = "geometry",
    groupby: str | None = None,
    boundary_tolerance: float = 1.0,
    min_shared_boundary: float = 0.0,
    enclosure_min_fraction: float = 0.95,
) -> ContourTopologyResult:
    """
    Compute contour boundary-neighbor and enclosure relationships.

    Parameters
    ----------
    contours
        DataFrame containing one polygonal Shapely geometry per contour.
    contour_id_col
        Column with stable contour identifiers.
    geometry_col
        Column containing Shapely ``Polygon`` or ``MultiPolygon`` geometries.
    groupby
        Optional metadata column used to aggregate contour-pair relationships.
    boundary_tolerance
        Distance tolerance used to treat nearly coincident boundaries as shared.
        Set to ``0`` for exact boundary-only matching.
    min_shared_boundary
        Minimum shared boundary length required for a pair to be considered a
        boundary neighbor.
    enclosure_min_fraction
        Minimum fraction of the inner contour area covered by the outer contour
        to report an enclosure relationship.
    """

    cfg = ContourTopologyConfig(
        contour_id_col=contour_id_col,
        geometry_col=geometry_col,
        groupby=groupby,
        boundary_tolerance=boundary_tolerance,
        min_shared_boundary=min_shared_boundary,
        enclosure_min_fraction=enclosure_min_fraction,
    )
    contour_table = _prepare_contour_table(contours, cfg)
    boundary_overlap, enclosure = _pairwise_topology(contour_table, cfg)
    contour_summary = _summarize_contours(contour_table, boundary_overlap, enclosure, cfg)
    group_boundary_overlap = _summarize_group_boundary(boundary_overlap, cfg)
    group_enclosure = _summarize_group_enclosure(enclosure, cfg)
    return ContourTopologyResult(
        boundary_overlap=boundary_overlap,
        enclosure=enclosure,
        contour_summary=contour_summary,
        group_boundary_overlap=group_boundary_overlap,
        group_enclosure=group_enclosure,
    )


def _prepare_contour_table(
    contours: pd.DataFrame,
    cfg: ContourTopologyConfig,
) -> pd.DataFrame:
    if not isinstance(contours, pd.DataFrame):
        raise TypeError(f"`contours` must be a pandas.DataFrame, got {type(contours)!r}.")

    required = {cfg.contour_id_col, cfg.geometry_col}
    if cfg.groupby is not None:
        required.add(cfg.groupby)
    missing = required.difference(contours.columns)
    if missing:
        raise ValueError(f"`contours` is missing required columns: {sorted(missing)}")

    boundary_tolerance = float(cfg.boundary_tolerance)
    min_shared_boundary = float(cfg.min_shared_boundary)
    enclosure_min_fraction = float(cfg.enclosure_min_fraction)
    if boundary_tolerance < 0:
        raise ValueError("`boundary_tolerance` must be non-negative.")
    if min_shared_boundary < 0:
        raise ValueError("`min_shared_boundary` must be non-negative.")
    if not 0 < enclosure_min_fraction <= 1:
        raise ValueError("`enclosure_min_fraction` must be in the interval (0, 1].")

    if contours[cfg.contour_id_col].duplicated().any():
        duplicated = contours.loc[
            contours[cfg.contour_id_col].duplicated(), cfg.contour_id_col
        ].astype(str)
        raise ValueError(f"`{cfg.contour_id_col}` values must be unique: {sorted(duplicated)}")

    records: list[dict[str, Any]] = []
    for _, row in contours.iterrows():
        geometry = _clean_polygonal_geometry(row[cfg.geometry_col])
        if geometry.is_empty:
            continue
        record = row.to_dict()
        record[cfg.geometry_col] = geometry
        records.append(record)

    if not records:
        raise ValueError("No valid polygonal contour geometries were provided.")
    return pd.DataFrame.from_records(records).reset_index(drop=True)


def _pairwise_topology(
    contour_table: pd.DataFrame,
    cfg: ContourTopologyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    geometries = np.asarray(contour_table[cfg.geometry_col].to_list(), dtype=object)
    boundaries = np.asarray([geometry.boundary for geometry in geometries], dtype=object)
    boundary_lengths = np.asarray([float(boundary.length) for boundary in boundaries], dtype=float)
    areas = np.asarray([float(geometry.area) for geometry in geometries], dtype=float)
    tree = STRtree(geometries)

    boundary_rows: list[dict[str, Any]] = []
    enclosure_rows: list[dict[str, Any]] = []
    metadata_columns = _metadata_columns(contour_table, cfg)
    tolerance = float(cfg.boundary_tolerance)
    min_shared = float(cfg.min_shared_boundary)

    for idx_a, geometry_a in enumerate(geometries):
        query_geometry = geometry_a.buffer(tolerance) if tolerance > 0 else geometry_a
        for idx_b_raw in tree.query(query_geometry):
            idx_b = int(idx_b_raw)
            if idx_b <= idx_a:
                continue

            geometry_b = geometries[idx_b]
            boundary_a = boundaries[idx_a]
            boundary_b = boundaries[idx_b]
            intersection = geometry_a.intersection(geometry_b)
            area_overlap = float(intersection.area) if not intersection.is_empty else 0.0
            min_distance = float(geometry_a.distance(geometry_b))
            exact_shared = float(boundary_a.intersection(boundary_b).length)
            shared = _tolerance_shared_boundary_length(
                boundary_a=boundary_a,
                boundary_b=boundary_b,
                exact_shared=exact_shared,
                area_overlap=area_overlap,
                min_distance=min_distance,
                tolerance=tolerance,
            )
            is_boundary_neighbor = shared > min_shared
            has_area_overlap = area_overlap > _GEOMETRY_EPSILON

            if is_boundary_neighbor or has_area_overlap:
                row_a = contour_table.iloc[idx_a]
                row_b = contour_table.iloc[idx_b]
                boundary_rows.append(
                    {
                        cfg.contour_id_col + "_a": row_a[cfg.contour_id_col],
                        cfg.contour_id_col + "_b": row_b[cfg.contour_id_col],
                        "exact_shared_boundary_length_um": exact_shared,
                        "shared_boundary_length_um": shared,
                        "boundary_length_a_um": boundary_lengths[idx_a],
                        "boundary_length_b_um": boundary_lengths[idx_b],
                        "overlap_fraction_a": _safe_divide(shared, boundary_lengths[idx_a]),
                        "overlap_fraction_b": _safe_divide(shared, boundary_lengths[idx_b]),
                        "boundary_jaccard": _safe_divide(
                            shared,
                            boundary_lengths[idx_a] + boundary_lengths[idx_b] - shared,
                        ),
                        "min_distance_um": min_distance,
                        "area_overlap_um2": area_overlap,
                        "is_boundary_neighbor": bool(is_boundary_neighbor),
                        "has_area_overlap": bool(has_area_overlap),
                        **_pair_metadata(row_a, row_b, metadata_columns),
                    }
                )

            enclosure_rows.extend(
                _enclosure_rows_for_pair(
                    contour_table=contour_table,
                    idx_a=idx_a,
                    idx_b=idx_b,
                    intersection_area=area_overlap,
                    areas=areas,
                    cfg=cfg,
                    metadata_columns=metadata_columns,
                )
            )

    return (
        pd.DataFrame.from_records(boundary_rows, columns=_boundary_columns(cfg, metadata_columns)),
        pd.DataFrame.from_records(enclosure_rows, columns=_enclosure_columns(cfg, metadata_columns)),
    )


def _tolerance_shared_boundary_length(
    *,
    boundary_a: BaseGeometry,
    boundary_b: BaseGeometry,
    exact_shared: float,
    area_overlap: float,
    min_distance: float,
    tolerance: float,
) -> float:
    if exact_shared > _GEOMETRY_EPSILON:
        return float(exact_shared)
    if tolerance <= 0 or min_distance <= _GEOMETRY_EPSILON:
        return 0.0
    if area_overlap > _GEOMETRY_EPSILON:
        return 0.0

    a_to_b = float(boundary_a.intersection(boundary_b.buffer(tolerance)).length)
    b_to_a = float(boundary_b.intersection(boundary_a.buffer(tolerance)).length)
    return max(0.0, min(a_to_b, b_to_a))


def _enclosure_rows_for_pair(
    *,
    contour_table: pd.DataFrame,
    idx_a: int,
    idx_b: int,
    intersection_area: float,
    areas: np.ndarray,
    cfg: ContourTopologyConfig,
    metadata_columns: list[str],
) -> list[dict[str, Any]]:
    if intersection_area <= _GEOMETRY_EPSILON:
        return []
    area_a = float(areas[idx_a])
    area_b = float(areas[idx_b])
    if area_a <= _GEOMETRY_EPSILON or area_b <= _GEOMETRY_EPSILON:
        return []
    if abs(area_a - area_b) <= _GEOMETRY_EPSILON:
        return []

    if area_a > area_b:
        outer_idx, inner_idx = idx_a, idx_b
        outer_area, inner_area = area_a, area_b
    else:
        outer_idx, inner_idx = idx_b, idx_a
        outer_area, inner_area = area_b, area_a

    inner_fraction = _safe_divide(intersection_area, inner_area)
    if inner_fraction < float(cfg.enclosure_min_fraction):
        return []

    outer_row = contour_table.iloc[outer_idx]
    inner_row = contour_table.iloc[inner_idx]
    return [
        {
            "outer_" + cfg.contour_id_col: outer_row[cfg.contour_id_col],
            "inner_" + cfg.contour_id_col: inner_row[cfg.contour_id_col],
            "outer_area_um2": outer_area,
            "inner_area_um2": inner_area,
            "intersection_area_um2": float(intersection_area),
            "inner_area_covered_fraction": inner_fraction,
            "outer_area_occupied_fraction": _safe_divide(intersection_area, outer_area),
            "is_enclosed": True,
            **_enclosure_metadata(outer_row, inner_row, metadata_columns),
        }
    ]


def _summarize_contours(
    contour_table: pd.DataFrame,
    boundary_overlap: pd.DataFrame,
    enclosure: pd.DataFrame,
    cfg: ContourTopologyConfig,
) -> pd.DataFrame:
    metadata_columns = _metadata_columns(contour_table, cfg)
    contour_ids = contour_table[cfg.contour_id_col].to_list()
    neighbor_sets = {contour_id: set() for contour_id in contour_ids}
    shared_lengths = {contour_id: 0.0 for contour_id in contour_ids}
    area_overlap_counts = {contour_id: 0 for contour_id in contour_ids}
    enclosing_counts = {contour_id: 0 for contour_id in contour_ids}
    contained_counts = {contour_id: 0 for contour_id in contour_ids}

    id_a_col = cfg.contour_id_col + "_a"
    id_b_col = cfg.contour_id_col + "_b"
    if not boundary_overlap.empty:
        for _, row in boundary_overlap.iterrows():
            contour_a = row[id_a_col]
            contour_b = row[id_b_col]
            if bool(row["is_boundary_neighbor"]):
                neighbor_sets[contour_a].add(contour_b)
                neighbor_sets[contour_b].add(contour_a)
                shared = float(row["shared_boundary_length_um"])
                shared_lengths[contour_a] += shared
                shared_lengths[contour_b] += shared
            if bool(row["has_area_overlap"]):
                area_overlap_counts[contour_a] += 1
                area_overlap_counts[contour_b] += 1

    outer_col = "outer_" + cfg.contour_id_col
    inner_col = "inner_" + cfg.contour_id_col
    if not enclosure.empty:
        for _, row in enclosure.iterrows():
            contained_counts[row[outer_col]] += 1
            enclosing_counts[row[inner_col]] += 1

    rows: list[dict[str, Any]] = []
    for _, row in contour_table.iterrows():
        contour_id = row[cfg.contour_id_col]
        geometry = row[cfg.geometry_col]
        boundary_length = float(geometry.boundary.length)
        record = {
            cfg.contour_id_col: contour_id,
            "area_um2": float(geometry.area),
            "boundary_length_um": boundary_length,
            "n_boundary_neighbors": len(neighbor_sets[contour_id]),
            "total_shared_boundary_length_um": shared_lengths[contour_id],
            "boundary_contact_fraction": _safe_divide(
                shared_lengths[contour_id], boundary_length
            ),
            "n_area_overlap_contours": area_overlap_counts[contour_id],
            "n_enclosing_contours": enclosing_counts[contour_id],
            "n_contained_contours": contained_counts[contour_id],
        }
        for column in metadata_columns:
            record[column] = row[column]
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def _summarize_group_boundary(
    boundary_overlap: pd.DataFrame,
    cfg: ContourTopologyConfig,
) -> pd.DataFrame:
    columns = [
        "group_a",
        "group_b",
        "n_contour_pairs",
        "n_boundary_pairs",
        "shared_boundary_length_um",
        "area_overlap_um2",
        "mean_overlap_fraction_a",
        "mean_overlap_fraction_b",
    ]
    if cfg.groupby is None or boundary_overlap.empty:
        return pd.DataFrame(columns=columns)

    group_a_col = cfg.groupby + "_a"
    group_b_col = cfg.groupby + "_b"
    if group_a_col not in boundary_overlap.columns or group_b_col not in boundary_overlap.columns:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    working = boundary_overlap.copy()
    working[["group_a", "group_b"]] = working.apply(
        lambda row: pd.Series(_ordered_group_pair(row[group_a_col], row[group_b_col])),
        axis=1,
    )
    for (group_a, group_b), rows in working.groupby(["group_a", "group_b"], dropna=False):
        records.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "n_contour_pairs": int(len(rows)),
                "n_boundary_pairs": int(rows["is_boundary_neighbor"].sum()),
                "shared_boundary_length_um": float(rows["shared_boundary_length_um"].sum()),
                "area_overlap_um2": float(rows["area_overlap_um2"].sum()),
                "mean_overlap_fraction_a": float(rows["overlap_fraction_a"].mean()),
                "mean_overlap_fraction_b": float(rows["overlap_fraction_b"].mean()),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def _summarize_group_enclosure(
    enclosure: pd.DataFrame,
    cfg: ContourTopologyConfig,
) -> pd.DataFrame:
    columns = [
        "outer_group",
        "inner_group",
        "n_enclosures",
        "intersection_area_um2",
        "mean_inner_area_covered_fraction",
        "mean_outer_area_occupied_fraction",
    ]
    if cfg.groupby is None or enclosure.empty:
        return pd.DataFrame(columns=columns)

    outer_group_col = "outer_" + cfg.groupby
    inner_group_col = "inner_" + cfg.groupby
    if outer_group_col not in enclosure.columns or inner_group_col not in enclosure.columns:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    for (outer_group, inner_group), rows in enclosure.groupby(
        [outer_group_col, inner_group_col], dropna=False
    ):
        records.append(
            {
                "outer_group": outer_group,
                "inner_group": inner_group,
                "n_enclosures": int(len(rows)),
                "intersection_area_um2": float(rows["intersection_area_um2"].sum()),
                "mean_inner_area_covered_fraction": float(
                    rows["inner_area_covered_fraction"].mean()
                ),
                "mean_outer_area_occupied_fraction": float(
                    rows["outer_area_occupied_fraction"].mean()
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def _metadata_columns(contour_table: pd.DataFrame, cfg: ContourTopologyConfig) -> list[str]:
    return [
        column
        for column in contour_table.columns
        if column not in {cfg.contour_id_col, cfg.geometry_col}
    ]


def _pair_metadata(
    row_a: pd.Series,
    row_b: pd.Series,
    metadata_columns: list[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for column in metadata_columns:
        payload[column + "_a"] = row_a[column]
        payload[column + "_b"] = row_b[column]
    return payload


def _enclosure_metadata(
    outer_row: pd.Series,
    inner_row: pd.Series,
    metadata_columns: list[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for column in metadata_columns:
        payload["outer_" + column] = outer_row[column]
        payload["inner_" + column] = inner_row[column]
    return payload


def _ordered_group_pair(group_a: Any, group_b: Any) -> tuple[Any, Any]:
    return (group_a, group_b) if str(group_a) <= str(group_b) else (group_b, group_a)


def _boundary_columns(
    cfg: ContourTopologyConfig,
    metadata_columns: list[str],
) -> list[str]:
    return [
        cfg.contour_id_col + "_a",
        cfg.contour_id_col + "_b",
        "exact_shared_boundary_length_um",
        "shared_boundary_length_um",
        "boundary_length_a_um",
        "boundary_length_b_um",
        "overlap_fraction_a",
        "overlap_fraction_b",
        "boundary_jaccard",
        "min_distance_um",
        "area_overlap_um2",
        "is_boundary_neighbor",
        "has_area_overlap",
        *[column + "_a" for column in metadata_columns],
        *[column + "_b" for column in metadata_columns],
    ]


def _enclosure_columns(
    cfg: ContourTopologyConfig,
    metadata_columns: list[str],
) -> list[str]:
    return [
        "outer_" + cfg.contour_id_col,
        "inner_" + cfg.contour_id_col,
        "outer_area_um2",
        "inner_area_um2",
        "intersection_area_um2",
        "inner_area_covered_fraction",
        "outer_area_occupied_fraction",
        "is_enclosed",
        *["outer_" + column for column in metadata_columns],
        *["inner_" + column for column in metadata_columns],
    ]


def _clean_polygonal_geometry(geometry: Any) -> BaseGeometry:
    if not isinstance(geometry, BaseGeometry):
        raise TypeError(
            "`geometry_col` values must be Shapely geometry objects, got "
            f"{type(geometry)!r}."
        )
    if geometry.is_empty:
        return GeometryCollection()

    working = geometry
    if not working.is_valid:
        working = working.buffer(0)
    polygons = _polygon_parts(working)
    if not polygons:
        return GeometryCollection()
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons)


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    if isinstance(geometry, GeometryCollection):
        polygons: list[Polygon] = []
        for part in geometry.geoms:
            polygons.extend(_polygon_parts(part))
        return polygons
    return []


def _safe_divide(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if abs(denominator) <= _GEOMETRY_EPSILON:
        return float("nan")
    return float(numerator) / denominator
