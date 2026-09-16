"""Mechanistic biology mining for 3D tracked gland instances."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
from typing import Any, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from shapely.geometry import Point, shape
from shapely.prepared import prep
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PathLike = Union[str, Path]


SIGNATURES: dict[str, tuple[str, ...]] = {
    "stem_wnt": ("LGR5", "OLFM4", "ASCL2", "SOX9", "RNF43", "SMOC2", "FZD7", "TCF7L2"),
    "proliferation": ("MKI67", "UBE2C", "PCLAF", "STMN1", "TYMS", "TK1", "RRM2", "PBK", "CDC25C"),
    "goblet_secretory": ("MUC2", "ATOH1", "SPDEF", "REG4", "RETNLB", "CLCA1", "CLCA4", "ITLN1"),
    "absorptive_best4": ("BEST4", "AQP8", "OTOP2", "CA1", "CA2", "GUCA2A", "GUCA2B", "MS4A12"),
    "endocrine_tuft": ("CHGA", "CHGB", "INSM1", "PYY", "GCG", "SST", "TPH1", "POU2F3", "TRPM5", "IL17RB"),
    "stromal_niche": ("GREM1", "GREM2", "RSPO3", "WNT2B", "PDGFRA", "FAP", "COL1A1", "COL1A2", "ACTA2", "TAGLN", "VCAN"),
    "myeloid": ("LYZ", "CD68", "C1QA", "C1QB", "C1QC", "CD14", "CD163", "S100A9", "IL1B", "TYROBP"),
    "t_nk": ("PTPRC", "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "NKG7", "GZMA", "GZMK", "TRAC"),
    "b_plasma": ("MS4A1", "CD79A", "CD79B", "BANK1", "POU2AF1", "SDC1", "TNFRSF17"),
    "mast_dc": ("KIT", "CPA3", "CMA1", "MS4A2", "CLEC9A", "IRF8", "LILRA4"),
    "endothelial_perivascular": ("PECAM1", "CD34", "AQP1", "RGS5", "ACTA2", "TAGLN", "PDGFRA"),
}


BOUNDARY_INTERACTIONS: tuple[tuple[str, str, str], ...] = (
    ("WNT2B", "FZD7", "stromal WNT to epithelial FZD"),
    ("RSPO3", "LGR5", "R-spondin/Wnt-amplified stem niche"),
    ("RSPO3", "RNF43", "R-spondin/RNF43 Wnt-axis modulation"),
    ("GREM1", "BMP6", "BMP-antagonist niche proxy"),
    ("AREG", "EGFR", "EGFR epithelial growth axis"),
    ("CD274", "PDCD1", "immune checkpoint boundary proxy"),
    ("TNFSF13B", "TNFRSF17", "B/plasma survival axis"),
)


@dataclass(frozen=True)
class GlandBiologyMiningConfig:
    """Configuration for gland-level mechanistic biology mining."""

    gland_instance_dir: PathLike
    stack_root: PathLike
    out_dir: PathLike | None = None
    aligned_cells_parquet: PathLike | None = None
    outer_ring_inner_um: float = 10.0
    outer_ring_outer_um: float = 30.0
    max_slices: int | None = None
    high_confidence_min_slice_count: int = 3
    high_confidence_max_branch_candidates: int = 0


@dataclass
class GlandBiologyMiningResult:
    out_dir: Path
    gland_biology_feature_matrix_csv: Path
    gland_fission_microenvironment_csv: Path
    gland_axis_distortion_csv: Path
    gland_entropy_and_plasticity_csv: Path
    gland_boundary_communication_csv: Path
    gland_morphomolecular_latent_space_csv: Path
    gland_cell_assignments_csv: Path
    gland_count: int
    assigned_cell_count: int


def run_gland_biology_mining(cfg: GlandBiologyMiningConfig) -> GlandBiologyMiningResult:
    """Mine gland-level biological mechanisms from reconstructed 3D gland instances."""

    _validate_config(cfg)
    gland_dir = Path(cfg.gland_instance_dir).expanduser().resolve()
    stack_root = Path(cfg.stack_root).expanduser().resolve()
    out_dir = (
        Path(cfg.out_dir).expanduser().resolve()
        if cfg.out_dir is not None
        else gland_dir / "biology_mining"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = pd.read_csv(gland_dir / "gland_instance_tracks.csv")
    gland_index = pd.read_csv(gland_dir / "gland_instance_qc_index.csv")
    if cfg.max_slices is not None:
        tracks = tracks.loc[tracks["slice_order"].astype(int) <= int(cfg.max_slices)].copy()
        gland_index = gland_index.loc[gland_index["gland_id"].isin(set(tracks["gland_id"]))].copy()
    feature_lookup = _load_feature_lookup(gland_dir / "slice_gland_instances.geojson")
    aligned_cells = _load_aligned_cells(cfg, stack_root)
    contexts = _load_xenium_contexts(stack_root)
    gene_names = _load_gene_names(contexts)

    (
        assignments,
        ring_assignments,
        gene_counts,
        slice_instance_gene_counts,
        ring_gene_counts,
        cell_states,
    ) = _process_slices(
        tracks=tracks,
        feature_lookup=feature_lookup,
        aligned_cells=aligned_cells,
        contexts=contexts,
        gene_names=gene_names,
        cfg=cfg,
    )
    feature_matrix = _build_feature_matrix(
        gland_index=gland_index,
        assignments=assignments,
        gene_counts=gene_counts,
        cell_states=cell_states,
        gene_names=gene_names,
        cfg=cfg,
    )
    fission = _build_fission_microenvironment(
        tracks=tracks,
        assignments=assignments,
        ring_assignments=ring_assignments,
        slice_instance_gene_counts=slice_instance_gene_counts,
        ring_gene_counts=ring_gene_counts,
        gene_names=gene_names,
    )
    axis = _build_axis_distortion(assignments, cell_states)
    entropy = _build_entropy_and_plasticity(cell_states, assignments)
    communication = _build_boundary_communication(
        tracks=tracks,
        assignments=assignments,
        ring_assignments=ring_assignments,
        slice_instance_gene_counts=slice_instance_gene_counts,
        ring_gene_counts=ring_gene_counts,
        gene_names=gene_names,
    )
    latent = _build_morphomolecular_latent_space(feature_matrix)

    feature_path = out_dir / "gland_biology_feature_matrix.csv"
    fission_path = out_dir / "gland_fission_microenvironment.csv"
    axis_path = out_dir / "gland_axis_distortion.csv"
    entropy_path = out_dir / "gland_entropy_and_plasticity.csv"
    communication_path = out_dir / "gland_boundary_communication.csv"
    latent_path = out_dir / "gland_morphomolecular_latent_space.csv"
    assignments_path = out_dir / "gland_cell_assignments.csv"
    feature_matrix.to_csv(feature_path, index=False)
    fission.to_csv(fission_path, index=False)
    axis.to_csv(axis_path, index=False)
    entropy.to_csv(entropy_path, index=False)
    communication.to_csv(communication_path, index=False)
    latent.to_csv(latent_path, index=False)
    assignments.to_csv(assignments_path, index=False)

    summary = {
        "config": asdict(cfg),
        "gland_count": int(feature_matrix["gland_id"].nunique()) if not feature_matrix.empty else 0,
        "assigned_cell_count": int(len(assignments)),
        "gene_count": int(len(gene_names)),
        "outputs": {
            "gland_biology_feature_matrix": str(feature_path),
            "gland_fission_microenvironment": str(fission_path),
            "gland_axis_distortion": str(axis_path),
            "gland_entropy_and_plasticity": str(entropy_path),
            "gland_boundary_communication": str(communication_path),
            "gland_morphomolecular_latent_space": str(latent_path),
            "gland_cell_assignments": str(assignments_path),
        },
    }
    (out_dir / "gland_biology_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return GlandBiologyMiningResult(
        out_dir=out_dir,
        gland_biology_feature_matrix_csv=feature_path,
        gland_fission_microenvironment_csv=fission_path,
        gland_axis_distortion_csv=axis_path,
        gland_entropy_and_plasticity_csv=entropy_path,
        gland_boundary_communication_csv=communication_path,
        gland_morphomolecular_latent_space_csv=latent_path,
        gland_cell_assignments_csv=assignments_path,
        gland_count=int(feature_matrix["gland_id"].nunique()) if not feature_matrix.empty else 0,
        assigned_cell_count=int(len(assignments)),
    )


def _process_slices(
    *,
    tracks: pd.DataFrame,
    feature_lookup: Mapping[str, Mapping[str, Any]],
    aligned_cells: pd.DataFrame,
    contexts: Mapping[str, Path],
    gene_names: Sequence[str],
    cfg: GlandBiologyMiningConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignment_frames: list[pd.DataFrame] = []
    ring_frames: list[pd.DataFrame] = []
    gene_frames: list[pd.DataFrame] = []
    slice_gene_frames: list[pd.DataFrame] = []
    ring_gene_frames: list[pd.DataFrame] = []
    state_frames: list[pd.DataFrame] = []
    gene_set = set(gene_names)
    for slice_order, slice_tracks in tracks.groupby("slice_order", sort=True):
        slice_order_int = int(slice_order)
        if cfg.max_slices is not None and slice_order_int > int(cfg.max_slices):
            continue
        slice_cells = aligned_cells.loc[aligned_cells["slice_order"].astype(int) == slice_order_int].copy()
        if slice_cells.empty:
            continue
        slice_assignments, slice_ring = _assign_cells_for_slice(
            slice_tracks=slice_tracks,
            slice_cells=slice_cells,
            feature_lookup=feature_lookup,
            cfg=cfg,
        )
        if slice_assignments.empty:
            continue
        sample_id = str(slice_assignments["sample_id"].iloc[0])
        xenium_dir = contexts.get(sample_id)
        if xenium_dir is None:
            transcript_counts = pd.DataFrame(columns=["cell_id", "feature_name", "count"])
        else:
            cell_ids = set(slice_assignments["barcode"].astype(str))
            if not slice_ring.empty:
                cell_ids.update(slice_ring["barcode"].astype(str))
            transcript_counts = _load_transcript_counts(
                xenium_dir=xenium_dir,
                cell_ids=cell_ids,
                gene_set=gene_set,
            )
        gland_gene = _aggregate_gland_gene_counts(
            transcript_counts=transcript_counts,
            assignments=slice_assignments,
            group_col="gland_id",
        )
        slice_gene = _aggregate_gland_gene_counts(
            transcript_counts=transcript_counts,
            assignments=slice_assignments,
            group_col="slice_instance_id",
        )
        ring_gene = _aggregate_gland_gene_counts(
            transcript_counts=transcript_counts,
            assignments=slice_ring,
            group_col="slice_instance_id",
        )
        cell_states = _cell_state_table(transcript_counts, slice_assignments)
        assignment_frames.append(slice_assignments)
        ring_frames.append(slice_ring)
        gene_frames.append(gland_gene)
        slice_gene_frames.append(slice_gene)
        ring_gene_frames.append(ring_gene)
        state_frames.append(cell_states)
    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    rings = pd.concat(ring_frames, ignore_index=True) if ring_frames else pd.DataFrame()
    gene_counts = pd.concat(gene_frames, ignore_index=True) if gene_frames else pd.DataFrame()
    slice_gene_counts = pd.concat(slice_gene_frames, ignore_index=True) if slice_gene_frames else pd.DataFrame()
    ring_gene_counts = pd.concat(ring_gene_frames, ignore_index=True) if ring_gene_frames else pd.DataFrame()
    if not assignments.empty:
        assignments["longitudinal_coord"] = _longitudinal_coords(assignments)
    cell_states = pd.concat(state_frames, ignore_index=True) if state_frames else pd.DataFrame()
    if not cell_states.empty and not assignments.empty:
        coords = assignments[["sample_id", "barcode", "gland_id", "longitudinal_coord"]].drop_duplicates()
        cell_states = cell_states.drop(columns=["longitudinal_coord"], errors="ignore").merge(
            coords,
            on=["sample_id", "barcode", "gland_id"],
            how="left",
        )
    if not gene_counts.empty:
        gene_counts = gene_counts.groupby(["gland_id", "feature_name"], as_index=False)["count"].sum()
    if not slice_gene_counts.empty:
        slice_gene_counts = slice_gene_counts.groupby(["slice_instance_id", "feature_name"], as_index=False)["count"].sum()
    if not ring_gene_counts.empty:
        ring_gene_counts = ring_gene_counts.groupby(["slice_instance_id", "feature_name"], as_index=False)["count"].sum()
    return assignments, rings, gene_counts, slice_gene_counts, ring_gene_counts, cell_states


def _assign_cells_for_slice(
    *,
    slice_tracks: pd.DataFrame,
    slice_cells: pd.DataFrame,
    feature_lookup: Mapping[str, Mapping[str, Any]],
    cfg: GlandBiologyMiningConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates: list[pd.DataFrame] = []
    rings: list[pd.DataFrame] = []
    for _, row in slice_tracks.iterrows():
        feature = feature_lookup.get(str(row["slice_instance_id"]))
        if feature is None:
            continue
        geom = shape(feature.get("geometry"))
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        min_x, min_y, max_x, max_y = geom.bounds
        area = float(row.get("area_um2", geom.area))
        local = slice_cells.loc[
            (slice_cells["x_3d_um"] >= min_x)
            & (slice_cells["x_3d_um"] <= max_x)
            & (slice_cells["y_3d_um"] >= min_y)
            & (slice_cells["y_3d_um"] <= max_y)
        ].copy()
        if not local.empty:
            inside_mask = _points_in_polygon(local, geom)
            inside = local.loc[inside_mask].copy()
            if not inside.empty:
                inside["slice_instance_id"] = str(row["slice_instance_id"])
                inside["gland_id"] = str(row["gland_id"])
                inside["semantic_structure"] = str(row.get("semantic_structure", ""))
                inside["_assignment_area_um2"] = area
                inside["radial_coord"] = _radial_coords(inside, geom, row)
                inside["longitudinal_coord"] = 0.0
                candidates.append(inside)
        ring = _ring_cells_for_instance(slice_cells, geom, row, cfg)
        if not ring.empty:
            rings.append(ring)
    if candidates:
        assignments = pd.concat(candidates, ignore_index=True)
        assignments["_cell_key"] = assignments["sample_id"].astype(str) + "|" + assignments["barcode"].astype(str)
        assignments = (
            assignments.sort_values("_assignment_area_um2")
            .drop_duplicates("_cell_key", keep="first")
            .drop(columns=["_assignment_area_um2", "_cell_key"])
            .reset_index(drop=True)
        )
        assignments["longitudinal_coord"] = _longitudinal_coords(assignments)
    else:
        assignments = pd.DataFrame()
    rings_df = pd.concat(rings, ignore_index=True) if rings else pd.DataFrame()
    return assignments, rings_df


def _ring_cells_for_instance(
    slice_cells: pd.DataFrame,
    geom: Any,
    row: Mapping[str, Any],
    cfg: GlandBiologyMiningConfig,
) -> pd.DataFrame:
    outer = geom.buffer(float(cfg.outer_ring_outer_um))
    inner = geom.buffer(float(cfg.outer_ring_inner_um))
    if outer.is_empty:
        return pd.DataFrame()
    min_x, min_y, max_x, max_y = outer.bounds
    local = slice_cells.loc[
        (slice_cells["x_3d_um"] >= min_x)
        & (slice_cells["x_3d_um"] <= max_x)
        & (slice_cells["y_3d_um"] >= min_y)
        & (slice_cells["y_3d_um"] <= max_y)
    ].copy()
    if local.empty:
        return local
    outer_prep = prep(outer)
    inner_prep = prep(inner)
    keep = [
        outer_prep.contains(point) and not inner_prep.contains(point)
        for point in (Point(x, y) for x, y in local[["x_3d_um", "y_3d_um"]].to_numpy(dtype=float))
    ]
    ring = local.loc[keep].copy()
    if ring.empty:
        return ring
    ring["slice_instance_id"] = str(row["slice_instance_id"])
    ring["gland_id"] = str(row["gland_id"])
    ring["semantic_structure"] = str(row.get("semantic_structure", ""))
    return ring


def _points_in_polygon(cells: pd.DataFrame, geom: Any) -> np.ndarray:
    if geom.geom_type != "Polygon":
        geom = max(list(geom.geoms), key=lambda item: item.area)
    coords = np.asarray(geom.exterior.coords, dtype=float)
    path = MplPath(coords)
    points = cells[["x_3d_um", "y_3d_um"]].to_numpy(dtype=float)
    return path.contains_points(points, radius=1e-9)


def _radial_coords(cells: pd.DataFrame, geom: Any, row: Mapping[str, Any]) -> np.ndarray:
    lx = float(row.get("lumen_centroid_x_um", math.nan))
    ly = float(row.get("lumen_centroid_y_um", math.nan))
    if not math.isfinite(lx) or not math.isfinite(ly):
        centroid = geom.centroid
        lx, ly = float(centroid.x), float(centroid.y)
    if geom.geom_type == "Polygon":
        boundary = np.asarray(geom.exterior.coords, dtype=float)
    else:
        boundary = np.vstack([np.asarray(part.exterior.coords, dtype=float) for part in geom.geoms])
    denom = float(np.max(np.sqrt((boundary[:, 0] - lx) ** 2 + (boundary[:, 1] - ly) ** 2)))
    denom = max(denom, 1e-9)
    points = cells[["x_3d_um", "y_3d_um"]].to_numpy(dtype=float)
    radial = np.sqrt((points[:, 0] - lx) ** 2 + (points[:, 1] - ly) ** 2) / denom
    return np.clip(radial, 0.0, 1.0)


def _longitudinal_coords(assignments: pd.DataFrame) -> pd.Series:
    result = pd.Series(0.0, index=assignments.index, dtype=float)
    for _, index in assignments.groupby("gland_id").groups.items():
        z = assignments.loc[index, "z_um"].astype(float)
        span = float(z.max() - z.min())
        if span > 0:
            result.loc[index] = (z - z.min()) / span
    return result


def _load_transcript_counts(
    *,
    xenium_dir: Path,
    cell_ids: set[str],
    gene_set: set[str],
) -> pd.DataFrame:
    path = xenium_dir / "transcripts.parquet"
    if not cell_ids:
        return pd.DataFrame(columns=["cell_id", "feature_name", "count"])
    if not path.exists():
        return _load_zarr_transcript_counts(
            xenium_dir=xenium_dir,
            cell_ids=cell_ids,
            gene_set=gene_set,
        )
    transcripts = pd.read_parquet(path, columns=["cell_id", "feature_name"])
    transcripts["cell_id"] = transcripts["cell_id"].astype(str)
    transcripts["feature_name"] = transcripts["feature_name"].astype(str)
    transcripts = transcripts.loc[
        transcripts["cell_id"].isin(cell_ids)
        & transcripts["feature_name"].isin(gene_set)
        & (transcripts["cell_id"] != "UNASSIGNED")
    ]
    if transcripts.empty:
        return pd.DataFrame(columns=["cell_id", "feature_name", "count"])
    return (
        transcripts.groupby(["cell_id", "feature_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )


def _load_zarr_transcript_counts(
    *,
    xenium_dir: Path,
    cell_ids: set[str],
    gene_set: set[str],
    chunk_size: int = 1_000_000,
) -> pd.DataFrame:
    cell_path = xenium_dir / "points" / "transcripts" / "cell_id"
    gene_path = xenium_dir / "points" / "transcripts" / "gene_name"
    if not cell_path.exists() or not gene_path.exists():
        return pd.DataFrame(columns=["cell_id", "feature_name", "count"])
    try:
        import zarr
    except Exception:
        return pd.DataFrame(columns=["cell_id", "feature_name", "count"])

    cell_array = zarr.open(str(cell_path), mode="r")
    gene_array = zarr.open(str(gene_path), mode="r")
    n = int(cell_array.shape[0])
    if int(gene_array.shape[0]) != n:
        raise ValueError(f"Transcript cell/gene arrays have different lengths in {xenium_dir}")

    cell_values = np.asarray(list(cell_ids), dtype=str)
    gene_values = np.asarray(list(gene_set), dtype=str)
    frames: list[pd.DataFrame] = []
    for start in range(0, n, int(chunk_size)):
        stop = min(start + int(chunk_size), n)
        cells = np.asarray(cell_array[start:stop]).astype(str, copy=False)
        genes = np.asarray(gene_array[start:stop]).astype(str, copy=False)
        mask = (
            np.isin(cells, cell_values)
            & np.isin(genes, gene_values)
            & (cells != "UNASSIGNED")
        )
        if not bool(mask.any()):
            continue
        frames.append(pd.DataFrame({"cell_id": cells[mask], "feature_name": genes[mask]}))
    if not frames:
        return pd.DataFrame(columns=["cell_id", "feature_name", "count"])
    transcripts = pd.concat(frames, ignore_index=True)
    return (
        transcripts.groupby(["cell_id", "feature_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )


def _aggregate_gland_gene_counts(
    *,
    transcript_counts: pd.DataFrame,
    assignments: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    if transcript_counts.empty or assignments.empty:
        return pd.DataFrame(columns=[group_col, "feature_name", "count"])
    lookup = assignments[["barcode", group_col]].drop_duplicates()
    merged = transcript_counts.merge(lookup, left_on="cell_id", right_on="barcode", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=[group_col, "feature_name", "count"])
    return merged.groupby([group_col, "feature_name"], as_index=False)["count"].sum()


def _cell_state_table(transcript_counts: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame()
    counts_by_cell = {
        cell_id: group.set_index("feature_name")["count"].to_dict()
        for cell_id, group in transcript_counts.groupby("cell_id")
    } if not transcript_counts.empty else {}
    rows: list[dict[str, Any]] = []
    for _, row in assignments.iterrows():
        counts = counts_by_cell.get(str(row["barcode"]), {})
        scores = {
            signature: float(sum(counts.get(gene, 0.0) for gene in genes))
            for signature, genes in SIGNATURES.items()
        }
        best_state = max(scores, key=scores.get) if scores else "unresolved"
        if scores and scores[best_state] <= 0:
            best_state = "unresolved"
        payload = {
            "sample_id": row["sample_id"],
            "barcode": row["barcode"],
            "slice_order": int(row["slice_order"]),
            "z_um": float(row["z_um"]),
            "gland_id": row["gland_id"],
            "slice_instance_id": row["slice_instance_id"],
            "radial_coord": float(row["radial_coord"]),
            "longitudinal_coord": float(row["longitudinal_coord"]),
            "cell_state": best_state,
        }
        payload.update({f"score_{key}": value for key, value in scores.items()})
        rows.append(payload)
    return pd.DataFrame(rows)


def _build_feature_matrix(
    *,
    gland_index: pd.DataFrame,
    assignments: pd.DataFrame,
    gene_counts: pd.DataFrame,
    cell_states: pd.DataFrame,
    gene_names: Sequence[str],
    cfg: GlandBiologyMiningConfig,
) -> pd.DataFrame:
    feature = gland_index.copy()
    if "gland_id" not in feature.columns:
        return pd.DataFrame()
    cell_counts = assignments.groupby("gland_id").size().rename("assigned_cell_count") if not assignments.empty else pd.Series(dtype=int)
    feature = feature.merge(cell_counts, on="gland_id", how="left")
    feature["assigned_cell_count"] = feature["assigned_cell_count"].fillna(0).astype(int)
    gene_wide = (
        gene_counts.pivot_table(index="gland_id", columns="feature_name", values="count", aggfunc="sum", fill_value=0)
        if not gene_counts.empty
        else pd.DataFrame(index=feature["gland_id"])
    )
    for gene in gene_names:
        if gene not in gene_wide.columns:
            gene_wide[gene] = 0
    gene_wide = gene_wide[list(gene_names)]
    norm = gene_wide.div(feature.set_index("gland_id")["assigned_cell_count"].replace(0, np.nan), axis=0).fillna(0.0)
    norm = norm.add_prefix("gene_").reset_index()
    feature = feature.merge(norm, on="gland_id", how="left")
    for signature, genes in SIGNATURES.items():
        present = [f"gene_{gene}" for gene in genes if f"gene_{gene}" in feature.columns]
        feature[f"signature_{signature}"] = feature[present].sum(axis=1) if present else 0.0
    state_entropy = _state_entropy(cell_states)
    feature = feature.merge(state_entropy, on="gland_id", how="left")
    feature["state_entropy"] = feature["state_entropy"].fillna(0.0)
    feature["high_confidence_subset"] = (
        (feature["slice_count"].fillna(0).astype(int) >= int(cfg.high_confidence_min_slice_count))
        & (
            feature["branch_merge_candidate_count"].fillna(0).astype(int)
            <= int(cfg.high_confidence_max_branch_candidates)
        )
    )
    feature["fission_candidate_score"] = (
        feature["branch_merge_candidate_count"].fillna(0).astype(float)
        / feature["slice_count"].replace(0, np.nan).fillna(1).astype(float)
    )
    feature["lumen_collapse_score"] = 1.0 / np.log1p(feature["area_median_um2"].fillna(0).clip(lower=0) + 1.0)
    return feature


def _build_fission_microenvironment(
    *,
    tracks: pd.DataFrame,
    assignments: pd.DataFrame,
    ring_assignments: pd.DataFrame,
    slice_instance_gene_counts: pd.DataFrame,
    ring_gene_counts: pd.DataFrame,
    gene_names: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    branch_tracks = tracks.loc[tracks["branch_merge_candidates"].fillna("").astype(str).map(bool)].copy()
    for _, row in branch_tracks.iterrows():
        instance_id = str(row["slice_instance_id"])
        gland_id = str(row["gland_id"])
        inside_cells = assignments.loc[assignments["slice_instance_id"].astype(str) == instance_id]
        ring_cells = ring_assignments.loc[ring_assignments["slice_instance_id"].astype(str) == instance_id] if not ring_assignments.empty else pd.DataFrame()
        gene_lookup = _gene_count_lookup(slice_instance_gene_counts, instance_id, key_col="slice_instance_id")
        ring_gene_lookup = _gene_count_lookup(ring_gene_counts, instance_id, key_col="slice_instance_id")
        payload = {
            "gland_id": gland_id,
            "slice_instance_id": instance_id,
            "slice_order": int(row["slice_order"]),
            "z_um": float(row["z_um"]),
            "branch_merge_candidates": row.get("branch_merge_candidates", ""),
            "inside_cell_count": int(len(inside_cells)),
            "outer_ring_cell_count": int(len(ring_cells)),
        }
        for signature, genes in SIGNATURES.items():
            payload[f"junction_signature_{signature}"] = float(sum(gene_lookup.get(gene, 0.0) for gene in genes))
            payload[f"outer_ring_signature_{signature}"] = float(sum(ring_gene_lookup.get(gene, 0.0) for gene in genes))
        rows.append(payload)
    return pd.DataFrame(rows)


def _build_axis_distortion(assignments: pd.DataFrame, cell_states: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty or cell_states.empty:
        return pd.DataFrame()
    merged = assignments[["sample_id", "barcode", "gland_id", "radial_coord", "longitudinal_coord"]].merge(
        cell_states,
        on=["sample_id", "barcode", "gland_id", "radial_coord", "longitudinal_coord"],
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for gland_id, group in merged.groupby("gland_id"):
        payload = {"gland_id": gland_id, "cell_count": int(len(group))}
        for signature in ["stem_wnt", "proliferation", "goblet_secretory", "absorptive_best4"]:
            score_col = f"score_{signature}"
            weights = group[score_col].fillna(0).to_numpy(dtype=float) if score_col in group else np.zeros(len(group))
            payload[f"{signature}_radial_mean"] = _weighted_mean(group["radial_coord"], weights)
            payload[f"{signature}_longitudinal_mean"] = _weighted_mean(group["longitudinal_coord"], weights)
            payload[f"{signature}_longitudinal_spread"] = _weighted_std(group["longitudinal_coord"], weights)
            total = float(weights.sum())
            payload[f"{signature}_apical_quartile_fraction"] = (
                float(weights[group["longitudinal_coord"].to_numpy(dtype=float) >= 0.75].sum() / total)
                if total > 0
                else math.nan
            )
        distortion_values = [
            payload.get("stem_wnt_longitudinal_spread", math.nan),
            payload.get("proliferation_longitudinal_spread", math.nan),
            payload.get("proliferation_apical_quartile_fraction", math.nan),
        ]
        finite = [value for value in distortion_values if math.isfinite(float(value))]
        payload["z_projected_axis_distortion_score"] = float(np.mean(finite)) if finite else math.nan
        rows.append(payload)
    return pd.DataFrame(rows)


def _build_entropy_and_plasticity(cell_states: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    if cell_states.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for gland_id, group in cell_states.groupby("gland_id"):
        counts = group["cell_state"].value_counts()
        entropy = _shannon_entropy(counts.to_numpy(dtype=float))
        payload = {
            "gland_id": gland_id,
            "cell_count": int(len(group)),
            "state_entropy": entropy,
            "dominant_state": str(counts.index[0]) if len(counts) else "unresolved",
            "dominant_state_fraction": float(counts.iloc[0] / counts.sum()) if counts.sum() else math.nan,
            "radial_state_variance": float(group["radial_coord"].var()) if len(group) > 1 else 0.0,
        }
        for state, value in counts.items():
            payload[f"state_fraction_{state}"] = float(value / counts.sum())
        rows.append(payload)
    return pd.DataFrame(rows)


def _build_boundary_communication(
    *,
    tracks: pd.DataFrame,
    assignments: pd.DataFrame,
    ring_assignments: pd.DataFrame,
    slice_instance_gene_counts: pd.DataFrame,
    ring_gene_counts: pd.DataFrame,
    gene_names: Sequence[str],
) -> pd.DataFrame:
    gene_set = set(gene_names)
    pairs = [pair for pair in BOUNDARY_INTERACTIONS if pair[0] in gene_set and pair[1] in gene_set]
    if not pairs or assignments.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in tracks.iterrows():
        instance_id = str(row["slice_instance_id"])
        gland_id = str(row["gland_id"])
        inside_cells = assignments.loc[assignments["slice_instance_id"].astype(str) == instance_id]
        ring_cells = ring_assignments.loc[ring_assignments["slice_instance_id"].astype(str) == instance_id] if not ring_assignments.empty else pd.DataFrame()
        inside_n = max(int(len(inside_cells)), 1)
        ring_n = max(int(len(ring_cells)), 1)
        gene_lookup = _gene_count_lookup(slice_instance_gene_counts, instance_id, key_col="slice_instance_id")
        ring_gene_lookup = _gene_count_lookup(ring_gene_counts, instance_id, key_col="slice_instance_id")
        for ligand, receptor, description in pairs:
            receptor_per_cell = float(gene_lookup.get(receptor, 0.0) / inside_n)
            ligand_per_ring_cell = float(ring_gene_lookup.get(ligand, 0.0) / ring_n)
            ring_density = float(len(ring_cells) / inside_n)
            score = math.log1p(receptor_per_cell) * math.log1p(ligand_per_ring_cell) * math.log1p(ring_density)
            rows.append(
                {
                    "gland_id": gland_id,
                    "slice_instance_id": instance_id,
                    "slice_order": int(row["slice_order"]),
                    "ligand": ligand,
                    "receptor": receptor,
                    "interaction": description,
                    "inside_cell_count": int(len(inside_cells)),
                    "outer_ring_cell_count": int(len(ring_cells)),
                    "receptor_per_gland_cell": receptor_per_cell,
                    "ligand_per_outer_ring_cell": ligand_per_ring_cell,
                    "boundary_interaction_score": score,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return (
        table.groupby(["gland_id", "ligand", "receptor", "interaction"], as_index=False)
        .agg(
            mean_boundary_interaction_score=("boundary_interaction_score", "mean"),
            max_boundary_interaction_score=("boundary_interaction_score", "max"),
            total_outer_ring_cell_count=("outer_ring_cell_count", "sum"),
            total_inside_cell_count=("inside_cell_count", "sum"),
        )
    )


def _build_morphomolecular_latent_space(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    if feature_matrix.empty:
        return pd.DataFrame()
    candidate_cols = [
        col
        for col in feature_matrix.columns
        if col.startswith("signature_")
        or col
        in {
            "slice_count",
            "area_median_um2",
            "area_cv",
            "median_ring_support_score",
            "branch_merge_candidate_count",
            "state_entropy",
            "fission_candidate_score",
            "lumen_collapse_score",
        }
    ]
    data = feature_matrix[candidate_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if data.shape[0] < 2 or data.shape[1] < 2:
        return feature_matrix[["gland_id"]].copy()
    scaled = StandardScaler().fit_transform(data)
    n_components = min(3, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(scaled)
    out = feature_matrix[["gland_id"]].copy()
    for idx in range(n_components):
        out[f"morphomolecular_pc{idx + 1}"] = pcs[:, idx]
        out[f"pc{idx + 1}_explained_variance_ratio"] = float(pca.explained_variance_ratio_[idx])
    return out


def _state_entropy(cell_states: pd.DataFrame) -> pd.DataFrame:
    if cell_states.empty:
        return pd.DataFrame(columns=["gland_id", "state_entropy"])
    rows = []
    for gland_id, group in cell_states.groupby("gland_id"):
        counts = group["cell_state"].value_counts().to_numpy(dtype=float)
        rows.append({"gland_id": gland_id, "state_entropy": _shannon_entropy(counts)})
    return pd.DataFrame(rows)


def _gene_count_lookup(table: pd.DataFrame, key: str, *, key_col: str) -> dict[str, float]:
    if table.empty:
        return {}
    subset = table.loc[table[key_col].astype(str) == str(key)]
    if subset.empty:
        return {}
    return subset.set_index("feature_name")["count"].astype(float).to_dict()


def _weighted_mean(values: pd.Series, weights: np.ndarray) -> float:
    arr = values.to_numpy(dtype=float)
    total = float(np.nansum(weights))
    if total <= 0:
        return math.nan
    return float(np.nansum(arr * weights) / total)


def _weighted_std(values: pd.Series, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    if not math.isfinite(mean):
        return math.nan
    arr = values.to_numpy(dtype=float)
    total = float(np.nansum(weights))
    if total <= 0:
        return math.nan
    return float(np.sqrt(np.nansum(weights * (arr - mean) ** 2) / total))


def _shannon_entropy(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _load_feature_lookup(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[str, Mapping[str, Any]] = {}
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        instance_id = str(props.get("slice_instance_id", ""))
        if instance_id:
            lookup[instance_id] = feature
    return lookup


def _load_aligned_cells(cfg: GlandBiologyMiningConfig, stack_root: Path) -> pd.DataFrame:
    path = (
        Path(cfg.aligned_cells_parquet).expanduser().resolve()
        if cfg.aligned_cells_parquet is not None
        else stack_root / "aligned_leiden_3d_cells.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    columns = ["sample_id", "barcode", "slice_order", "x_3d_um", "y_3d_um", "z_um"]
    cells = pd.read_parquet(path, columns=columns)
    return cells.replace([np.inf, -np.inf], np.nan).dropna(subset=["x_3d_um", "y_3d_um"]).reset_index(drop=True)


def _load_xenium_contexts(stack_root: Path) -> dict[str, Path]:
    path = stack_root / "xenium_slice_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = pd.read_csv(path)
    required = {"sample_id", "xenium_dir"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return {
        str(row["sample_id"]): Path(str(row["xenium_dir"]))
        for _, row in manifest.iterrows()
    }


def _load_gene_names(contexts: Mapping[str, Path]) -> list[str]:
    for xenium_dir in contexts.values():
        features = xenium_dir / "cell_feature_matrix" / "features.tsv.gz"
        if features.exists():
            df = pd.read_csv(features, sep="\t", header=None, names=["id", "name", "type"])
            if "type" in df:
                df = df.loc[df["type"].astype(str).eq("Gene Expression")]
            return df["name"].astype(str).tolist()
    for xenium_dir in contexts.values():
        transcripts = xenium_dir / "transcripts.parquet"
        if not transcripts.exists():
            continue
        df = pd.read_parquet(transcripts, columns=["feature_name"])
        names = (
            df["feature_name"]
            .dropna()
            .astype(str)
            .loc[lambda values: values.ne("")]
            .drop_duplicates()
            .tolist()
        )
        if names:
            return sorted(names)
    for xenium_dir in contexts.values():
        zarr_names = xenium_dir / "tables" / "cells" / "var" / "name"
        if not zarr_names.exists():
            continue
        try:
            import zarr
        except Exception:
            continue
        values = zarr.open(str(zarr_names), mode="r")[:]
        names = sorted({str(value) for value in values if str(value)})
        if names:
            return names
    raise FileNotFoundError(
        "Could not find gene names in cell_feature_matrix/features.tsv.gz "
        "or pyXeniumSlide zarr arrays in any Xenium directory."
    )


def _validate_config(cfg: GlandBiologyMiningConfig) -> None:
    if cfg.outer_ring_inner_um < 0:
        raise ValueError("`outer_ring_inner_um` must be non-negative.")
    if cfg.outer_ring_outer_um <= cfg.outer_ring_inner_um:
        raise ValueError("`outer_ring_outer_um` must exceed `outer_ring_inner_um`.")


__all__ = [
    "BOUNDARY_INTERACTIONS",
    "GlandBiologyMiningConfig",
    "GlandBiologyMiningResult",
    "SIGNATURES",
    "run_gland_biology_mining",
]
