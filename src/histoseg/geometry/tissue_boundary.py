import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Polygon, MultiPolygon
from shapely.ops import unary_union, triangulate


def _alpha_shape(points, alpha):
    """
    Compute an alpha shape (concave hull) from a set of 2D points.

    Parameters
    ----------
    points : np.ndarray of shape (n_points, 2)
    alpha : float

    Returns
    -------
    shapely.geometry.Polygon or MultiPolygon
    """
    if len(points) < 4:
        return MultiPoint(points).convex_hull

    triangles = triangulate(MultiPoint(points))
    kept = []

    for tri in triangles:
        coords = np.array(tri.exterior.coords[:3])
        a, b, c = coords

        side_lengths = [
            np.linalg.norm(a - b),
            np.linalg.norm(b - c),
            np.linalg.norm(c - a),
        ]

        s = sum(side_lengths) / 2.0
        area_sq = (
            s
            * (s - side_lengths[0])
            * (s - side_lengths[1])
            * (s - side_lengths[2])
        )

        if area_sq <= 0:
            continue

        area = np.sqrt(area_sq)
        circumradius = (
            side_lengths[0] * side_lengths[1] * side_lengths[2]
        ) / (4.0 * area)

        if circumradius < 1.0 / alpha:
            kept.append(tri)

    if not kept:
        return MultiPoint(points).convex_hull

    return unary_union(kept)


def generate_tissue_boundary(
    cells_df,
    x_col="x_centroid",
    y_col="y_centroid",
    method="alpha_shape",
    alpha=0.05,
    simplify_tolerance=None,
    output_csv=None,
):
    """
    Generate a tissue boundary polygon from cell spatial coordinates.

    Parameters
    ----------
    cells_df : pandas.DataFrame
        Cell-level table containing spatial coordinates.
    x_col, y_col : str
        Column names for x and y coordinates.
    method : {"alpha_shape", "convex_hull"}
        Geometry method for boundary estimation.
    alpha : float
        Alpha parameter for alpha-shape.
    simplify_tolerance : float or None
        Optional polygon simplification tolerance.
    output_csv : str or Path or None
        If provided, write boundary to CSV.

    Returns
    -------
    pandas.DataFrame
        Columns: ["x", "y", "order"]
    """
    if x_col not in cells_df or y_col not in cells_df:
        raise ValueError(
            f"cells_df must contain columns '{x_col}' and '{y_col}'"
        )

    points = (
        cells_df[[x_col, y_col]]
        .dropna()
        .to_numpy()
    )

    if len(points) == 0:
        raise ValueError("No valid spatial coordinates found.")

    if method == "convex_hull":
        polygon = MultiPoint(points).convex_hull
    elif method == "alpha_shape":
        polygon = _alpha_shape(points, alpha=alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    if simplify_tolerance is not None:
        polygon = polygon.simplify(simplify_tolerance)

    # ------------------------------------------------------------------
    # Normalize geometry:
    # tissue boundary must be a single outer polygon
    # (consistent with tissueboundary.txt behavior)
    # ------------------------------------------------------------------
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda p: p.area)

    if not isinstance(polygon, Polygon):
        raise TypeError(
            f"Expected Polygon after processing, got {type(polygon)}"
        )

    x, y = polygon.exterior.coords.xy

    boundary_df = pd.DataFrame({
        "x": x,
        "y": y,
        "order": np.arange(len(x)),
    })

    if output_csv is not None:
        boundary_df.to_csv(output_csv, index=False)

    return boundary_df
