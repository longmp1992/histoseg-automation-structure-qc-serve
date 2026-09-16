from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from skimage import measure

try:
    from shapely import affinity, contains_xy
except Exception:  # pragma: no cover - older Shapely fallback.
    affinity = None
    contains_xy = None

PathLike = Union[str, Path]


@dataclass(frozen=True)
class SpatialModuleDiscoveryConfig:
    """Configuration for 3D gene spatial module discovery.

    The workflow consumes an aligned 3D cell table produced by HistoSeg's
    multi-slice reconstruction and a merged AnnData object containing gene
    expression. For each gene it builds a normalized 3D enrichment field,
    extracts nested Marching Cubes hotspot surfaces, quantifies structure
    overlap/SDF distances, and writes cross-gene matrices.
    """

    h5ad: PathLike
    aligned_cells_parquet: PathLike
    stack_root: PathLike
    genes: Sequence[str]
    out_dir: PathLike | None = None
    gene_file: PathLike | None = None
    sample_column: str = "sample_id"
    barcode_column: str = "barcode"
    skip_order_check: bool = False
    template_density_summary: PathLike | None = None
    xy_voxel_size_um: float = 80.0
    z_voxel_size_um: float = 5.0
    smoothing_sigma_xy_um: float = 120.0
    smoothing_sigma_z_um: float = 10.0
    surface_smoothing_sigma_voxels_zyx: Sequence[float] = (1.0, 0.9, 0.9)
    valid_min_cell_count: float = 8.0
    min_positive_cells: int = 200
    min_expression_sum: float = 0.0
    min_surface_voxels: int = 20
    mesh_export_formats: Sequence[str] = ("ply", "obj")
    structures: Sequence[str] = (
        "Structure 1",
        "Structure 2",
        "Structure 3",
        "Structure 4",
        "Structure 5",
    )
    group_property: str = "structure"
    pixel_size_um: float = 0.2125
    force_rebuild_masks: bool = False


@dataclass(frozen=True)
class SpatialModuleDiscoveryResult:
    out_dir: Path
    status_csv: Path
    summary_json: Path
    completed_genes: tuple[str, ...]
    overlap_fraction_matrix_csv: Path | None
    signed_distance_matrix_csv: Path | None
    fraction_inside_matrix_csv: Path | None


@dataclass(frozen=True)
class GeneStructureQuantificationConfig:
    stack_root: PathLike
    gene_density_dir: PathLike
    gene: str
    structures: Sequence[str] = (
        "Structure 1",
        "Structure 2",
        "Structure 3",
        "Structure 4",
        "Structure 5",
    )
    group_property: str = "structure"
    pixel_size_um: float = 0.2125
    force_rebuild_masks: bool = False


@dataclass(frozen=True)
class GeneStructureQuantificationResult:
    out_dir: Path
    overlap_metrics_csv: Path
    distance_metrics_csv: Path
    overlap_heatmap_png: Path
    summary_json: Path
    structure_mask_npz: Path


@dataclass(frozen=True)
class SpatialModulePlotConfig:
    batch_dir: PathLike
    hotspot: str = "top05"
    matrix: str = "fraction_inside"
    out_png: PathLike | None = None
    row_zscore: bool = True
    cmap: str | None = None


@dataclass(frozen=True)
class SpatialModulePlotResult:
    out_png: Path


@dataclass(frozen=True)
class _GridSpec:
    shape_zyx: tuple[int, int, int]
    x_min_um: float
    x_max_um: float
    y_min_um: float
    y_max_um: float
    z_min_um: float
    z_max_um: float
    xy_voxel_size_um: float
    z_voxel_size_um: float

    @property
    def spacing_zyx_um(self) -> tuple[float, float, float]:
        return (self.z_voxel_size_um, self.xy_voxel_size_um, self.xy_voxel_size_um)

    @property
    def voxel_volume_um3(self) -> float:
        return self.z_voxel_size_um * self.xy_voxel_size_um * self.xy_voxel_size_um

    @property
    def voxel_count(self) -> int:
        return int(np.prod(self.shape_zyx))


def compute_hotspot_overlap_metrics(
    gene_mask: np.ndarray,
    structure_mask: np.ndarray,
    voxel_volume_um3: float,
) -> dict[str, float | int]:
    """Compute pure voxel overlap metrics for one gene hotspot and structure mask."""

    gene_mask = np.asarray(gene_mask, dtype=bool)
    structure_mask = np.asarray(structure_mask, dtype=bool)
    _validate_matching_masks(gene_mask, structure_mask)
    if voxel_volume_um3 <= 0:
        raise ValueError("voxel_volume_um3 must be greater than 0.")

    overlap = gene_mask & structure_mask
    gene_voxels = int(gene_mask.sum())
    structure_voxels = int(structure_mask.sum())
    overlap_voxels = int(overlap.sum())
    return {
        "gene_hotspot_voxels": gene_voxels,
        "gene_hotspot_volume_um3": float(gene_voxels * voxel_volume_um3),
        "structure_voxels": structure_voxels,
        "structure_volume_um3": float(structure_voxels * voxel_volume_um3),
        "overlap_voxels": overlap_voxels,
        "overlap_volume_um3": float(overlap_voxels * voxel_volume_um3),
        "fraction_of_gene_in_structure": float(overlap_voxels / gene_voxels)
        if gene_voxels
        else 0.0,
        "fraction_of_structure_covered_by_gene": float(overlap_voxels / structure_voxels)
        if structure_voxels
        else 0.0,
    }


def compute_hotspot_sdf_metrics(
    gene_mask: np.ndarray,
    structure_mask: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
) -> dict[str, float | int]:
    """Compute anisotropic signed-distance metrics for one hotspot and structure.

    Arrays are interpreted in ``zyx`` order. Distances are physical microns:
    ``spacing_zyx_um`` is passed directly to SciPy's Euclidean distance
    transform. Signed distance is negative inside ``structure_mask`` and
    positive outside. Boundary voxels are not clamped to zero; an isolated
    inside voxel therefore has distance ``-min(spacing_zyx_um)``.
    """

    gene_mask = np.asarray(gene_mask, dtype=bool)
    structure_mask = np.asarray(structure_mask, dtype=bool)
    _validate_matching_masks(gene_mask, structure_mask)
    _validate_spacing_zyx(spacing_zyx_um)

    n_hotspot_voxels = int(gene_mask.sum())
    if n_hotspot_voxels == 0 or not bool(structure_mask.any()):
        return {
            "n_hotspot_voxels": n_hotspot_voxels,
            **_empty_sdf_distribution(),
            "fraction_inside_structure": 0.0,
            "fraction_touching_or_inside_structure": 0.0,
        }

    outside = distance_transform_edt(~structure_mask, sampling=spacing_zyx_um)
    inside = distance_transform_edt(structure_mask, sampling=spacing_zyx_um)
    signed = outside.astype(np.float32)
    signed[structure_mask] = -inside[structure_mask]
    signed_values = signed[gene_mask]
    unsigned_values = np.maximum(signed_values, 0.0)
    return {
        "n_hotspot_voxels": n_hotspot_voxels,
        "min_unsigned_distance_um": float(np.min(unsigned_values)),
        "median_unsigned_distance_um": float(np.median(unsigned_values)),
        "mean_unsigned_distance_um": float(np.mean(unsigned_values)),
        "max_unsigned_distance_um": float(np.max(unsigned_values)),
        "p95_unsigned_distance_um": float(np.percentile(unsigned_values, 95)),
        "min_signed_distance_um": float(np.min(signed_values)),
        "median_signed_distance_um": float(np.median(signed_values)),
        "mean_signed_distance_um": float(np.mean(signed_values)),
        "max_signed_distance_um": float(np.max(signed_values)),
        "p05_signed_distance_um": float(np.percentile(signed_values, 5)),
        "p95_signed_distance_um": float(np.percentile(signed_values, 95)),
        "fraction_inside_structure": float(np.mean(signed_values < 0)),
        "fraction_touching_or_inside_structure": float(np.mean(signed_values <= 0)),
    }


def run_spatial_module_discovery(
    cfg: SpatialModuleDiscoveryConfig,
) -> SpatialModuleDiscoveryResult:
    """Run end-to-end 3D gene spatial module discovery."""

    genes = _collect_genes(cfg.genes, cfg.gene_file)
    stack_root = Path(cfg.stack_root).expanduser()
    out_dir = (
        Path(cfg.out_dir).expanduser()
        if cfg.out_dir is not None
        else stack_root / "gene_overlays" / "batch_3d_genes"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    aligned_cells = pd.read_parquet(Path(cfg.aligned_cells_parquet).expanduser())
    template_summary = _resolve_template_summary(cfg, stack_root)
    grid = _load_or_infer_grid(
        aligned_cells=aligned_cells,
        template_summary=template_summary,
        xy_voxel_size_um=cfg.xy_voxel_size_um,
        z_voxel_size_um=cfg.z_voxel_size_um,
    )
    linear_indices, in_grid = _cell_linear_indices(aligned_cells, grid)
    dropped_cells = int((~in_grid).sum())
    flat_size = grid.voxel_count
    cell_count_raw = np.bincount(linear_indices, minlength=flat_size).reshape(grid.shape_zyx).astype(np.float32)

    density_sigma = (
        cfg.smoothing_sigma_z_um / grid.z_voxel_size_um,
        cfg.smoothing_sigma_xy_um / grid.xy_voxel_size_um,
        cfg.smoothing_sigma_xy_um / grid.xy_voxel_size_um,
    )
    smoothed_cell_count = gaussian_filter(cell_count_raw, sigma=density_sigma, mode="nearest")
    valid = smoothed_cell_count >= cfg.valid_min_cell_count
    surface_sigma = _as_zyx_tuple(cfg.surface_smoothing_sigma_voxels_zyx)
    export_formats = tuple(fmt.strip().lower() for fmt in cfg.mesh_export_formats if fmt.strip())

    try:
        import anndata as ad
    except Exception as exc:  # pragma: no cover - dependency availability.
        raise ImportError("anndata is required for 3D spatial module discovery.") from exc

    adata = ad.read_h5ad(Path(cfg.h5ad).expanduser(), backed="r")
    status_rows: list[dict] = []
    completed_genes: list[str] = []
    try:
        if not cfg.skip_order_check:
            _check_cell_order(adata, aligned_cells, cfg.sample_column, cfg.barcode_column)
        available_genes = set(map(str, adata.var_names))
        for gene in genes:
            gene_dir = out_dir / f"{gene}_density"
            gene_dir.mkdir(parents=True, exist_ok=True)
            status = {
                "gene": gene,
                "status": "pending",
                "raw_nonzero_cell_count": 0,
                "raw_expression_sum": 0.0,
                "valid_voxel_count": int(valid.sum()),
            }
            if gene not in available_genes:
                status["status"] = "missing_gene"
                _write_json(gene_dir / f"{gene}_3d_enrichment_summary.json", status)
                status_rows.append(status)
                continue

            expression = _expression_vector(adata, gene)[in_grid]
            expression = np.where(np.isfinite(expression), expression, 0.0).astype(np.float32)
            positive_count = int(np.count_nonzero(expression > 0))
            expression_total = float(np.sum(expression))
            status["raw_nonzero_cell_count"] = positive_count
            status["raw_expression_sum"] = expression_total
            if positive_count < cfg.min_positive_cells or expression_total <= cfg.min_expression_sum:
                status["status"] = "skipped_low_expression"
                _write_json(gene_dir / f"{gene}_3d_enrichment_summary.json", status)
                status_rows.append(status)
                continue

            expression_raw = np.bincount(linear_indices, weights=expression, minlength=flat_size).reshape(grid.shape_zyx).astype(np.float32)
            positive_raw = np.bincount(
                linear_indices,
                weights=(expression > 0).astype(np.float32),
                minlength=flat_size,
            ).reshape(grid.shape_zyx).astype(np.float32)
            smoothed_expression = gaussian_filter(expression_raw, sigma=density_sigma, mode="nearest")
            smoothed_positive = gaussian_filter(positive_raw, sigma=density_sigma, mode="nearest")
            enrichment = np.divide(
                smoothed_expression,
                smoothed_cell_count,
                out=np.zeros_like(smoothed_expression),
                where=smoothed_cell_count > 0,
            )
            valid_values = enrichment[valid]
            display_threshold = float(np.quantile(valid_values, 0.75)) if valid_values.size else float("nan")
            displayed = valid & (enrichment >= display_threshold)
            _write_gene_density_outputs(
                gene=gene,
                gene_dir=gene_dir,
                grid=grid,
                cell_count_grid=smoothed_cell_count,
                expression_grid=smoothed_expression,
                positive_grid=smoothed_positive,
                enrichment=enrichment,
                valid=valid,
                displayed=displayed,
                raw_positive_count=positive_count,
                raw_expression_sum=expression_total,
                dropped_cells=dropped_cells,
                density_sigma_voxels=density_sigma,
                valid_min_cell_count=cfg.valid_min_cell_count,
                display_threshold=display_threshold,
            )
            iso_summary = _write_isosurfaces(
                gene=gene,
                gene_dir=gene_dir,
                grid=grid,
                enrichment=enrichment,
                valid=valid,
                surface_sigma_voxels=surface_sigma,
                min_surface_voxels=cfg.min_surface_voxels,
                export_formats=export_formats,
            )
            ok_surfaces = [row for row in iso_summary["mesh_manifest"] if row.get("status") == "ok"]
            if not ok_surfaces:
                status["status"] = "skipped_no_isosurface"
                status_rows.append(status)
                continue

            quantify_gene_structure_relationships(
                GeneStructureQuantificationConfig(
                    stack_root=stack_root,
                    gene_density_dir=gene_dir,
                    gene=gene,
                    structures=cfg.structures,
                    group_property=cfg.group_property,
                    pixel_size_um=cfg.pixel_size_um,
                    force_rebuild_masks=cfg.force_rebuild_masks,
                )
            )
            status["status"] = "ok"
            status["ok_surface_count"] = len(ok_surfaces)
            completed_genes.append(gene)
            status_rows.append(status)
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()

    status_csv = out_dir / "batch_gene_status.csv"
    pd.DataFrame(status_rows).to_csv(status_csv, index=False)
    matrix_paths = _write_cross_gene_matrices(out_dir, completed_genes)
    summary_json = out_dir / "batch_3d_gene_spatial_mapping_summary.json"
    _write_json(
        summary_json,
        {
            "stack_root": str(stack_root),
            "h5ad": str(Path(cfg.h5ad).expanduser()),
            "aligned_cells_parquet": str(Path(cfg.aligned_cells_parquet).expanduser()),
            "out_dir": str(out_dir),
            "requested_genes": genes,
            "completed_genes": completed_genes,
            "status_csv": str(status_csv),
            "grid": asdict(grid),
        },
    )
    return SpatialModuleDiscoveryResult(
        out_dir=out_dir,
        status_csv=status_csv,
        summary_json=summary_json,
        completed_genes=tuple(completed_genes),
        overlap_fraction_matrix_csv=matrix_paths.get("overlap"),
        signed_distance_matrix_csv=matrix_paths.get("signed"),
        fraction_inside_matrix_csv=matrix_paths.get("inside"),
    )


def quantify_gene_structure_relationships(
    cfg: GeneStructureQuantificationConfig,
) -> GeneStructureQuantificationResult:
    """Quantify gene hotspot overlap and SDF distance to structure masks."""

    stack_root = Path(cfg.stack_root).expanduser()
    gene_density_dir = Path(cfg.gene_density_dir).expanduser()
    out_dir = gene_density_dir / "structure_relationships"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = stack_root / "structure_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    density_summary = _load_json(gene_density_dir / f"{cfg.gene}_3d_enrichment_summary.json")
    iso_summary = _load_json(gene_density_dir / "isosurfaces" / f"{cfg.gene}_3d_enrichment_isosurfaces_summary.json")
    grid = _grid_from_density_summary(density_summary)
    field, valid, cell_count, expression_sum, positive_count = _load_enrichment_field(
        gene_density_dir / f"{cfg.gene}_3d_enrichment_voxels.csv",
        grid,
        cfg.gene,
    )
    sigma = _as_zyx_tuple(iso_summary.get("surface_smoothing_sigma_voxels_zyx", (1.0, 0.9, 0.9)))
    smooth_field, smooth_weight = _smooth_field(field, valid, sigma)
    quant_valid = valid & (smooth_weight > 1e-6)

    structures = tuple(cfg.structures)
    structure_masks, mask_npz, _ = _rasterize_structure_masks(
        stack_root=stack_root,
        grid=grid,
        structures=structures,
        group_property=cfg.group_property,
        pixel_size_um=cfg.pixel_size_um,
        out_dir=mask_dir,
        force_rebuild=cfg.force_rebuild_masks,
    )

    thresholds = {name: float(value) for name, value in iso_summary["thresholds"].items()}
    hotspot_order = [level for level in ("top15", "top10", "top05") if level in thresholds]
    hotspot_masks = {level: quant_valid & (smooth_field >= thresholds[level]) for level in hotspot_order}

    overlap_rows: list[dict] = []
    distance_rows: list[dict] = []
    hotspot_stats: dict[str, dict] = {}
    for level, gene_mask in hotspot_masks.items():
        gene_voxels = int(gene_mask.sum())
        hotspot_stats[level] = {
            "threshold": thresholds[level],
            "voxel_count": gene_voxels,
            "volume_um3": float(gene_voxels * grid.voxel_volume_um3),
            "cell_count": float(cell_count[gene_mask].sum()),
            "gene_positive_cell_count": float(positive_count[gene_mask].sum()),
            "gene_expression_sum": float(expression_sum[gene_mask].sum()),
        }
        for structure_name in structures:
            structure_mask = structure_masks[structure_name]
            overlap_metrics = compute_hotspot_overlap_metrics(
                gene_mask,
                structure_mask,
                grid.voxel_volume_um3,
            )
            overlap = gene_mask & structure_mask
            overlap_rows.append(
                {
                    "hotspot": level,
                    "structure": structure_name,
                    "threshold_enrichment": thresholds[level],
                    **overlap_metrics,
                    "overlap_cell_count": float(cell_count[overlap].sum()),
                    "overlap_gene_positive_cell_count": float(positive_count[overlap].sum()),
                    "overlap_gene_expression_sum": float(expression_sum[overlap].sum()),
                }
            )
            distance_rows.append(_distance_row(level, structure_name, gene_mask, structure_mask, grid.spacing_zyx_um))

    overlap_df = pd.DataFrame(overlap_rows)
    distance_df = pd.DataFrame(distance_rows)
    overlap_csv = out_dir / f"{cfg.gene}_structure_3d_overlap_metrics.csv"
    distance_csv = out_dir / f"{cfg.gene}_structure_3d_distance_metrics.csv"
    overlap_df.to_csv(overlap_csv, index=False)
    distance_df.to_csv(distance_csv, index=False)
    heatmap_png = out_dir / f"{cfg.gene}_structure_3d_overlap_heatmap.png"
    _plot_overlap_heatmap(overlap_df, hotspot_order, structures, heatmap_png, cfg.gene)

    summary_json = out_dir / f"{cfg.gene}_structure_3d_relationship_summary.json"
    _write_json(
        summary_json,
        {
            "gene": cfg.gene,
            "method": "voxel mask overlap plus anisotropic signed distance fields",
            "outputs": {
                "overlap_metrics_csv": str(overlap_csv),
                "distance_metrics_csv": str(distance_csv),
                "overlap_heatmap_png": str(heatmap_png),
                "structure_mask_npz": str(mask_npz),
            },
            "grid": {
                "shape_zyx": list(grid.shape_zyx),
                "spacing_zyx_um": list(grid.spacing_zyx_um),
                "voxel_volume_um3": grid.voxel_volume_um3,
                "surface_smoothing_sigma_voxels_zyx": list(sigma),
            },
            "hotspots": hotspot_stats,
            "structure_masks": {
                name: {
                    "voxel_count": int(mask.sum()),
                    "volume_um3": float(mask.sum() * grid.voxel_volume_um3),
                }
                for name, mask in structure_masks.items()
            },
            "sdf_definition": {
                "outside_um": "distance_transform_edt(~structure_mask, sampling=(z_um, y_um, x_um))",
                "inside_um": "distance_transform_edt(structure_mask, sampling=(z_um, y_um, x_um))",
                "signed_distance_um": "outside_um outside the structure; -inside_um inside the structure",
                "interpretation": "negative means embedded inside the structure, positive means outside distance to boundary",
            },
        },
    )
    return GeneStructureQuantificationResult(
        out_dir=out_dir,
        overlap_metrics_csv=overlap_csv,
        distance_metrics_csv=distance_csv,
        overlap_heatmap_png=heatmap_png,
        summary_json=summary_json,
        structure_mask_npz=mask_npz,
    )


def plot_spatial_module_clustermap(cfg: SpatialModulePlotConfig) -> SpatialModulePlotResult:
    """Plot a clustered gene-by-structure heatmap from batch matrices."""

    matrix_files = {
        "fraction_inside": "gene_structure_fraction_inside_matrix.csv",
        "overlap_fraction": "gene_structure_overlap_fraction_matrix.csv",
        "signed_distance": "gene_structure_signed_distance_matrix.csv",
    }
    cmaps = {
        "fraction_inside": "mako",
        "overlap_fraction": "magma",
        "signed_distance": "vlag",
    }
    if cfg.matrix not in matrix_files:
        raise ValueError(f"Unknown matrix {cfg.matrix!r}; choose one of {sorted(matrix_files)}")
    batch_dir = Path(cfg.batch_dir).expanduser()
    matrix_path = batch_dir / matrix_files[cfg.matrix]
    out_png = (
        Path(cfg.out_png).expanduser()
        if cfg.out_png is not None
        else batch_dir / f"{cfg.matrix}_{cfg.hotspot}_spatial_clustermap.png"
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    matrix = _load_hotspot_matrix(matrix_path, cfg.hotspot)
    plot_df = _zscore_rows(matrix) if cfg.row_zscore else matrix
    try:
        import seaborn as sns
    except Exception as exc:  # pragma: no cover - dependency availability.
        raise ImportError("seaborn is required to plot spatial module clustermaps.") from exc

    sns.set_theme(context="paper", style="white", font_scale=1.0)
    grid = sns.clustermap(
        plot_df,
        method="average",
        metric="euclidean",
        cmap=cfg.cmap or cmaps[cfg.matrix],
        linewidths=0.35,
        linecolor="white",
        figsize=(8.5, max(6.0, 0.32 * len(plot_df) + 2.0)),
        row_cluster=True,
        col_cluster=True,
        cbar_kws={"label": "row z-score" if cfg.row_zscore else "value"},
    )
    grid.ax_heatmap.set_xlabel("3D structure")
    grid.ax_heatmap.set_ylabel("Gene")
    grid.fig.suptitle(
        f"3D spatial gene modules | {cfg.matrix.replace('_', ' ')} | {cfg.hotspot}",
        y=1.02,
    )
    grid.fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)
    return SpatialModulePlotResult(out_png=out_png)


def _collect_genes(genes: Sequence[str], gene_file: PathLike | None) -> list[str]:
    collected: list[str] = []
    for value in genes:
        collected.extend(part.strip() for part in str(value).split(",") if part.strip())
    if gene_file is not None:
        for line in Path(gene_file).expanduser().read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                collected.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for gene in collected:
        key = gene.upper()
        if key not in seen:
            deduped.append(gene)
            seen.add(key)
    if not deduped:
        raise ValueError("Provide at least one gene.")
    return deduped


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_template_summary(cfg: SpatialModuleDiscoveryConfig, stack_root: Path) -> Path | None:
    if cfg.template_density_summary is not None:
        return Path(cfg.template_density_summary).expanduser()
    candidate = stack_root / "gene_overlays" / "GREM1_density" / "GREM1_3d_enrichment_summary.json"
    return candidate if candidate.exists() else None


def _as_zyx_tuple(values: Sequence[float]) -> tuple[float, float, float]:
    if isinstance(values, str):
        parts = [float(part.strip()) for part in values.split(",") if part.strip()]
    else:
        parts = [float(value) for value in values]
    if len(parts) != 3:
        raise ValueError("Expected exactly three Z,Y,X sigma values.")
    return (parts[0], parts[1], parts[2])


def _load_or_infer_grid(
    *,
    aligned_cells: pd.DataFrame,
    template_summary: Path | None,
    xy_voxel_size_um: float,
    z_voxel_size_um: float,
) -> _GridSpec:
    if template_summary is not None and template_summary.exists():
        summary = _load_json(template_summary)
        voxelization = summary["voxelization"]
        shape = tuple(int(value) for value in voxelization["grid_shape_zyx"])
        return _GridSpec(
            shape_zyx=shape,  # type: ignore[arg-type]
            x_min_um=float(voxelization["x_min_um"]),
            x_max_um=float(voxelization["x_max_um"]),
            y_min_um=float(voxelization["y_min_um"]),
            y_max_um=float(voxelization["y_max_um"]),
            z_min_um=float(voxelization["z_min_um"]),
            z_max_um=float(voxelization["z_max_um"]),
            xy_voxel_size_um=float(voxelization["xy_voxel_size_um"]),
            z_voxel_size_um=float(voxelization["z_voxel_size_um"]),
        )

    x_min = math.floor(float(aligned_cells["x_3d_um"].min()) / xy_voxel_size_um) * xy_voxel_size_um
    x_max = math.ceil(float(aligned_cells["x_3d_um"].max()) / xy_voxel_size_um) * xy_voxel_size_um
    y_min = math.floor(float(aligned_cells["y_3d_um"].min()) / xy_voxel_size_um) * xy_voxel_size_um
    y_max = math.ceil(float(aligned_cells["y_3d_um"].max()) / xy_voxel_size_um) * xy_voxel_size_um
    z_min = 0.0
    z_max = math.ceil(float(aligned_cells["z_um"].max()) / z_voxel_size_um) * z_voxel_size_um
    x_bins = int(round((x_max - x_min) / xy_voxel_size_um)) + 1
    y_bins = int(round((y_max - y_min) / xy_voxel_size_um)) + 1
    z_bins = int(round((z_max - z_min) / z_voxel_size_um)) + 1
    return _GridSpec(
        shape_zyx=(z_bins, y_bins, x_bins),
        x_min_um=x_min,
        x_max_um=x_min + x_bins * xy_voxel_size_um,
        y_min_um=y_min,
        y_max_um=y_min + y_bins * xy_voxel_size_um,
        z_min_um=z_min,
        z_max_um=z_min + (z_bins - 1) * z_voxel_size_um,
        xy_voxel_size_um=xy_voxel_size_um,
        z_voxel_size_um=z_voxel_size_um,
    )


def _grid_from_density_summary(summary: dict) -> _GridSpec:
    voxelization = summary["voxelization"]
    shape = tuple(int(value) for value in voxelization["grid_shape_zyx"])
    return _GridSpec(
        shape_zyx=shape,  # type: ignore[arg-type]
        x_min_um=float(voxelization["x_min_um"]),
        x_max_um=float(voxelization["x_max_um"]),
        y_min_um=float(voxelization["y_min_um"]),
        y_max_um=float(voxelization["y_max_um"]),
        z_min_um=float(voxelization["z_min_um"]),
        z_max_um=float(voxelization["z_max_um"]),
        xy_voxel_size_um=float(voxelization["xy_voxel_size_um"]),
        z_voxel_size_um=float(voxelization["z_voxel_size_um"]),
    )


def _cell_linear_indices(aligned_cells: pd.DataFrame, grid: _GridSpec) -> tuple[np.ndarray, np.ndarray]:
    z_bins, y_bins, x_bins = grid.shape_zyx
    xi = np.floor((aligned_cells["x_3d_um"].to_numpy(dtype=float) - grid.x_min_um) / grid.xy_voxel_size_um).astype(int)
    yi = np.floor((aligned_cells["y_3d_um"].to_numpy(dtype=float) - grid.y_min_um) / grid.xy_voxel_size_um).astype(int)
    zi = np.floor((aligned_cells["z_um"].to_numpy(dtype=float) - grid.z_min_um) / grid.z_voxel_size_um).astype(int)
    in_grid = (xi >= 0) & (xi < x_bins) & (yi >= 0) & (yi < y_bins) & (zi >= 0) & (zi < z_bins)
    linear = zi[in_grid] * y_bins * x_bins + yi[in_grid] * x_bins + xi[in_grid]
    return linear.astype(np.int64), in_grid


def _check_cell_order(adata, aligned_cells: pd.DataFrame, sample_column: str, barcode_column: str) -> None:
    if len(adata.obs) != len(aligned_cells):
        raise ValueError(f"AnnData has {len(adata.obs)} cells but aligned table has {len(aligned_cells)} rows.")
    obs_sample = adata.obs[sample_column].astype(str).to_numpy()
    obs_barcode = adata.obs[barcode_column].astype(str).to_numpy()
    cell_sample = aligned_cells[sample_column].astype(str).to_numpy()
    cell_barcode = aligned_cells[barcode_column].astype(str).to_numpy()
    if not (np.array_equal(obs_sample, cell_sample) and np.array_equal(obs_barcode, cell_barcode)):
        raise ValueError("AnnData obs order does not match aligned cell parquet order.")


def _expression_vector(adata, gene: str) -> np.ndarray:
    index = adata.var_names.get_loc(gene)
    matrix = adata[:, index].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32).reshape(-1)


def _write_gene_density_outputs(
    *,
    gene: str,
    gene_dir: Path,
    grid: _GridSpec,
    cell_count_grid: np.ndarray,
    expression_grid: np.ndarray,
    positive_grid: np.ndarray,
    enrichment: np.ndarray,
    valid: np.ndarray,
    displayed: np.ndarray,
    raw_positive_count: int,
    raw_expression_sum: float,
    dropped_cells: int,
    density_sigma_voxels: tuple[float, float, float],
    valid_min_cell_count: float,
    display_threshold: float,
) -> None:
    lower = gene.lower()
    z_idx, y_idx, x_idx = np.where(valid)
    pd.DataFrame(
        {
            "x_um": grid.x_min_um + (x_idx + 0.5) * grid.xy_voxel_size_um,
            "y_um": grid.y_min_um + (y_idx + 0.5) * grid.xy_voxel_size_um,
            "z_um": grid.z_min_um + z_idx * grid.z_voxel_size_um,
            "cell_count": cell_count_grid[valid],
            f"{lower}_positive_cell_count": positive_grid[valid],
            f"{lower}_expression_sum": expression_grid[valid],
            f"{lower}_enrichment": enrichment[valid],
            "displayed": displayed[valid],
        }
    ).to_csv(gene_dir / f"{gene}_3d_enrichment_voxels.csv", index=False)
    values = enrichment[valid]
    _write_json(
        gene_dir / f"{gene}_3d_enrichment_summary.json",
        {
            "gene": gene,
            "denominator_uses_all_cells": True,
            "raw_total_cell_count": int(cell_count_grid.sum()),
            "raw_gene_nonzero_cell_count": int(raw_positive_count),
            "raw_gene_expression_sum": float(raw_expression_sum),
            "dropped_nonfinite_or_out_of_grid_cells": int(dropped_cells),
            "voxelization": {
                "xy_voxel_size_um": grid.xy_voxel_size_um,
                "z_voxel_size_um": grid.z_voxel_size_um,
                "grid_shape_zyx": list(grid.shape_zyx),
                "x_min_um": grid.x_min_um,
                "x_max_um": grid.x_max_um,
                "y_min_um": grid.y_min_um,
                "y_max_um": grid.y_max_um,
                "z_min_um": grid.z_min_um,
                "z_max_um": grid.z_max_um,
            },
            "smoothing": {"sigma_voxels_zyx": list(density_sigma_voxels)},
            "metric": {
                "numerator": f"smoothed_sum_{gene}_expression",
                "denominator": "smoothed_total_cell_count",
                "value": f"{gene}_enrichment = numerator / denominator",
                "valid_voxel_min_smoothed_cell_count": valid_min_cell_count,
                "display_top_fraction": 0.25,
                "display_threshold": display_threshold,
                "color_min_p01_valid": float(np.percentile(values, 1)) if values.size else 0.0,
                "color_max_p99_valid": float(np.percentile(values, 99)) if values.size else 0.0,
            },
            "voxel_counts": {
                "total_grid_voxels": grid.voxel_count,
                "valid_voxel_count": int(valid.sum()),
                "displayed_voxel_count": int(displayed.sum()),
            },
        },
    )


def _write_isosurfaces(
    *,
    gene: str,
    gene_dir: Path,
    grid: _GridSpec,
    enrichment: np.ndarray,
    valid: np.ndarray,
    surface_sigma_voxels: tuple[float, float, float],
    min_surface_voxels: int,
    export_formats: Iterable[str],
) -> dict:
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - dependency availability.
        raise ImportError("trimesh is required for 3D gene isosurface export.") from exc

    iso_dir = gene_dir / "isosurfaces"
    iso_dir.mkdir(parents=True, exist_ok=True)
    smooth, weight = _smooth_field(enrichment, valid, surface_sigma_voxels)
    quant_valid = valid & (weight > 1e-6)
    valid_values = smooth[quant_valid]
    levels = {"top15": 0.15, "top10": 0.10, "top05": 0.05}
    thresholds = {
        name: float(np.quantile(valid_values, 1.0 - fraction)) if valid_values.size else float("nan")
        for name, fraction in levels.items()
    }
    manifest: list[dict] = []
    for name, threshold in thresholds.items():
        mask = quant_valid & (smooth >= threshold)
        if not np.isfinite(threshold) or threshold <= 0 or int(mask.sum()) < min_surface_voxels:
            manifest.append(
                {
                    "surface": name,
                    "threshold": threshold,
                    "status": "skipped_low_or_empty_hotspot",
                    "voxel_count": int(mask.sum()),
                }
            )
            continue
        volume = np.where(quant_valid, smooth, 0.0).astype(np.float32)
        padded = np.pad(volume, 1, mode="constant", constant_values=0.0)
        if float(padded.max()) <= threshold:
            manifest.append(
                {
                    "surface": name,
                    "threshold": threshold,
                    "status": "skipped_threshold_above_field",
                    "voxel_count": int(mask.sum()),
                }
            )
            continue
        verts_zyx, faces, _, _ = measure.marching_cubes(
            padded,
            level=threshold,
            spacing=(grid.z_voxel_size_um, grid.xy_voxel_size_um, grid.xy_voxel_size_um),
        )
        verts_zyx[:, 0] -= grid.z_voxel_size_um
        verts_zyx[:, 1] -= grid.xy_voxel_size_um
        verts_zyx[:, 2] -= grid.xy_voxel_size_um
        vertices = np.column_stack(
            [
                grid.x_min_um + verts_zyx[:, 2],
                grid.y_min_um + verts_zyx[:, 1],
                grid.z_min_um + verts_zyx[:, 0],
            ]
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        _cleanup_mesh(mesh)
        row = {
            "surface": name,
            "threshold": threshold,
            "status": "ok",
            "voxel_count": int(mask.sum()),
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "surface_area_um2": float(mesh.area),
            "volume_um3": float(abs(mesh.volume)),
            "is_watertight": bool(mesh.is_watertight),
            "component_count": int(len(mesh.split(only_watertight=False))),
        }
        for fmt in export_formats:
            output = iso_dir / f"{gene}_enrichment_{name}_isosurface.{fmt}"
            mesh.export(output)
            row[fmt] = str(output)
        manifest.append(row)
    manifest_csv = iso_dir / f"{gene}_enrichment_isosurface_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_csv, index=False)
    summary = {
        "gene": gene,
        "method": "smoothed_enrichment_field_marching_cubes_nested_isosurfaces",
        "surface_smoothing_sigma_voxels_zyx": list(surface_sigma_voxels),
        "thresholds": thresholds,
        "outputs": {
            "manifest_csv": str(manifest_csv),
            "ply_obj_dir": str(iso_dir),
            "summary_json": str(iso_dir / f"{gene}_3d_enrichment_isosurfaces_summary.json"),
        },
        "mesh_manifest": manifest,
    }
    _write_json(iso_dir / f"{gene}_3d_enrichment_isosurfaces_summary.json", summary)
    return summary


def _cleanup_mesh(mesh) -> None:
    for name in ("remove_degenerate_faces", "remove_duplicate_faces", "remove_infinite_values", "merge_vertices"):
        func = getattr(mesh, name, None)
        if callable(func):
            try:
                func()
            except Exception:
                pass


def _load_enrichment_field(
    voxels_csv: Path,
    grid: _GridSpec,
    gene: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_bins, y_bins, x_bins = grid.shape_zyx
    voxel_df = pd.read_csv(voxels_csv)
    field = np.zeros(grid.shape_zyx, dtype=np.float32)
    valid = np.zeros(grid.shape_zyx, dtype=bool)
    cell_count = np.zeros(grid.shape_zyx, dtype=np.float32)
    expression_sum = np.zeros(grid.shape_zyx, dtype=np.float32)
    positive_count = np.zeros(grid.shape_zyx, dtype=np.float32)
    xi = np.floor((voxel_df["x_um"].to_numpy(dtype=float) - grid.x_min_um) / grid.xy_voxel_size_um).astype(int)
    yi = np.floor((voxel_df["y_um"].to_numpy(dtype=float) - grid.y_min_um) / grid.xy_voxel_size_um).astype(int)
    zi = np.floor((voxel_df["z_um"].to_numpy(dtype=float) - grid.z_min_um) / grid.z_voxel_size_um).astype(int)
    in_grid = (xi >= 0) & (xi < x_bins) & (yi >= 0) & (yi < y_bins) & (zi >= 0) & (zi < z_bins)
    rows = voxel_df.loc[in_grid]
    xi, yi, zi = xi[in_grid], yi[in_grid], zi[in_grid]
    prefix = gene.lower()
    field[zi, yi, xi] = rows[f"{prefix}_enrichment"].to_numpy(dtype=float)
    valid[zi, yi, xi] = True
    cell_count[zi, yi, xi] = rows["cell_count"].to_numpy(dtype=float)
    expression_sum[zi, yi, xi] = rows[f"{prefix}_expression_sum"].to_numpy(dtype=float)
    positive_count[zi, yi, xi] = rows[f"{prefix}_positive_cell_count"].to_numpy(dtype=float)
    return field, valid, cell_count, expression_sum, positive_count


def _smooth_field(field: np.ndarray, valid: np.ndarray, sigma_zyx: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    weighted = gaussian_filter(field * valid.astype(np.float32), sigma=sigma_zyx, mode="nearest")
    weight = gaussian_filter(valid.astype(np.float32), sigma=sigma_zyx, mode="nearest")
    smooth = np.divide(weighted, weight, out=np.zeros_like(weighted), where=weight > 1e-6)
    return smooth, weight


def _feature_structure(feature: dict, group_property: str) -> str | None:
    props = feature.get("properties", {})
    value = props.get(group_property)
    if value:
        return str(value)
    classification = props.get("classification")
    if isinstance(classification, dict) and classification.get("name"):
        return str(classification["name"])
    if props.get("name"):
        return str(props["name"])
    return None


def _rasterize_structure_masks(
    *,
    stack_root: Path,
    grid: _GridSpec,
    structures: Sequence[str],
    group_property: str,
    pixel_size_um: float,
    out_dir: Path,
    force_rebuild: bool,
) -> tuple[dict[str, np.ndarray], Path, Path]:
    mask_npz = out_dir / f"structure_masks_zyx_{int(grid.xy_voxel_size_um)}x{int(grid.xy_voxel_size_um)}x{grid.z_voxel_size_um:g}um.npz"
    meta_path = out_dir / f"structure_masks_zyx_{int(grid.xy_voxel_size_um)}x{int(grid.xy_voxel_size_um)}x{grid.z_voxel_size_um:g}um_summary.json"
    if mask_npz.exists() and not force_rebuild:
        archive = np.load(mask_npz)
        return {name: archive[name].astype(bool) for name in structures}, mask_npz, meta_path
    if affinity is None:
        raise RuntimeError("shapely.affinity is required to rasterize structure masks.")

    manifest = pd.read_csv(stack_root / "aligned_slice_manifest.csv")
    x_centers = grid.x_min_um + (np.arange(grid.shape_zyx[2]) + 0.5) * grid.xy_voxel_size_um
    y_centers = grid.y_min_um + (np.arange(grid.shape_zyx[1]) + 0.5) * grid.xy_voxel_size_um
    xx, yy = np.meshgrid(x_centers, y_centers)
    masks = {name: np.zeros(grid.shape_zyx, dtype=bool) for name in structures}

    for _, row in manifest.sort_values("order" if "order" in manifest.columns else "slice_order").iterrows():
        z_index = int(row["order"] if "order" in row.index else row["slice_order"]) - 1
        data = _load_json(Path(str(row["aligned_geojson"])))
        for structure_name in structures:
            geoms = [
                shape(feature["geometry"])
                for feature in data.get("features", [])
                if _feature_structure(feature, group_property) == structure_name
            ]
            if not geoms:
                continue
            scaled = affinity.scale(unary_union(geoms), xfact=pixel_size_um, yfact=pixel_size_um, origin=(0, 0))
            minx, miny, maxx, maxy = scaled.bounds
            x0 = max(0, int(math.floor((minx - grid.x_min_um) / grid.xy_voxel_size_um)) - 1)
            x1 = min(grid.shape_zyx[2], int(math.ceil((maxx - grid.x_min_um) / grid.xy_voxel_size_um)) + 1)
            y0 = max(0, int(math.floor((miny - grid.y_min_um) / grid.xy_voxel_size_um)) - 1)
            y1 = min(grid.shape_zyx[1], int(math.ceil((maxy - grid.y_min_um) / grid.xy_voxel_size_um)) + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            sub_xx = xx[y0:y1, x0:x1]
            sub_yy = yy[y0:y1, x0:x1]
            if contains_xy is not None:
                sub_mask = contains_xy(scaled, sub_xx, sub_yy)
            else:
                sub_mask = np.zeros_like(sub_xx, dtype=bool)
                for row_idx in range(sub_xx.shape[0]):
                    for col_idx in range(sub_xx.shape[1]):
                        sub_mask[row_idx, col_idx] = scaled.contains(
                            Point(float(sub_xx[row_idx, col_idx]), float(sub_yy[row_idx, col_idx]))
                        )
            masks[structure_name][z_index, y0:y1, x0:x1] |= sub_mask

    np.savez_compressed(mask_npz, **{name: mask.astype(np.uint8) for name, mask in masks.items()})
    _write_json(
        meta_path,
        {
            "mask_npz": str(mask_npz),
            "shape_zyx": list(grid.shape_zyx),
            "spacing_zyx_um": list(grid.spacing_zyx_um),
            "voxel_volume_um3": grid.voxel_volume_um3,
            "structures": {
                name: {"voxel_count": int(mask.sum()), "volume_um3": float(mask.sum() * grid.voxel_volume_um3)}
                for name, mask in masks.items()
            },
        },
    )
    return masks, mask_npz, meta_path


def _distance_row(
    hotspot: str,
    structure: str,
    gene_mask: np.ndarray,
    structure_mask: np.ndarray,
    spacing_zyx_um: tuple[float, float, float],
) -> dict:
    return {
        "hotspot": hotspot,
        "structure": structure,
        **compute_hotspot_sdf_metrics(gene_mask, structure_mask, spacing_zyx_um),
    }


def _validate_matching_masks(gene_mask: np.ndarray, structure_mask: np.ndarray) -> None:
    if gene_mask.shape != structure_mask.shape:
        raise ValueError(
            "gene_mask and structure_mask must have identical zyx shapes, got "
            f"{gene_mask.shape} and {structure_mask.shape}."
        )


def _validate_spacing_zyx(spacing_zyx_um: tuple[float, float, float]) -> None:
    if len(spacing_zyx_um) != 3:
        raise ValueError("spacing_zyx_um must contain exactly three values.")
    if any(float(value) <= 0 for value in spacing_zyx_um):
        raise ValueError("spacing_zyx_um values must be greater than 0.")


def _empty_sdf_distribution() -> dict[str, float]:
    return {
        "min_unsigned_distance_um": math.nan,
        "median_unsigned_distance_um": math.nan,
        "mean_unsigned_distance_um": math.nan,
        "max_unsigned_distance_um": math.nan,
        "p95_unsigned_distance_um": math.nan,
        "min_signed_distance_um": math.nan,
        "median_signed_distance_um": math.nan,
        "mean_signed_distance_um": math.nan,
        "max_signed_distance_um": math.nan,
        "p05_signed_distance_um": math.nan,
        "p95_signed_distance_um": math.nan,
    }


def _plot_overlap_heatmap(
    overlap_df: pd.DataFrame,
    hotspot_order: Sequence[str],
    structures: Sequence[str],
    out_path: Path,
    gene: str,
) -> None:
    heat = overlap_df.pivot(index="hotspot", columns="structure", values="fraction_of_gene_in_structure").loc[
        list(hotspot_order), list(structures)
    ]
    volume = overlap_df.pivot(index="hotspot", columns="structure", values="overlap_volume_um3").loc[
        list(hotspot_order), list(structures)
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    image = ax.imshow(heat.to_numpy(), cmap="magma", vmin=0, vmax=max(0.001, float(np.nanmax(heat.to_numpy()))))
    ax.set_xticks(np.arange(len(structures)), labels=list(structures), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(hotspot_order)), labels=list(hotspot_order))
    ax.set_title(f"{gene} 3D hotspot overlap with structure masks")
    ax.set_xlabel("Structure")
    ax.set_ylabel(f"{gene} enrichment isosurface level")
    for row_idx, hotspot in enumerate(hotspot_order):
        for col_idx, structure in enumerate(structures):
            fraction = float(heat.loc[hotspot, structure])
            volume_million = float(volume.loc[hotspot, structure]) / 1e6
            ax.text(
                col_idx,
                row_idx,
                f"{fraction:.2f}\n{volume_million:.1f}M",
                ha="center",
                va="center",
                color="white" if fraction > 0.25 else "black",
                fontsize=9,
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Fraction of {gene} hotspot volume inside structure")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _write_cross_gene_matrices(out_dir: Path, genes: Sequence[str]) -> dict[str, Path]:
    rows_overlap: dict[str, dict[str, float]] = {}
    rows_signed: dict[str, dict[str, float]] = {}
    rows_inside: dict[str, dict[str, float]] = {}
    for gene in genes:
        rel_dir = out_dir / f"{gene}_density" / "structure_relationships"
        overlap_path = rel_dir / f"{gene}_structure_3d_overlap_metrics.csv"
        distance_path = rel_dir / f"{gene}_structure_3d_distance_metrics.csv"
        if overlap_path.exists():
            overlap_df = pd.read_csv(overlap_path)
            rows_overlap[gene] = {
                f"{record['hotspot']}|{record['structure']}": float(record["fraction_of_gene_in_structure"])
                for _, record in overlap_df.iterrows()
            }
        if distance_path.exists():
            distance_df = pd.read_csv(distance_path)
            rows_signed[gene] = {
                f"{record['hotspot']}|{record['structure']}": float(record["median_signed_distance_um"])
                for _, record in distance_df.iterrows()
            }
            rows_inside[gene] = {
                f"{record['hotspot']}|{record['structure']}": float(record["fraction_inside_structure"])
                for _, record in distance_df.iterrows()
            }
    paths: dict[str, Path] = {}
    if rows_overlap:
        path = out_dir / "gene_structure_overlap_fraction_matrix.csv"
        pd.DataFrame.from_dict(rows_overlap, orient="index").sort_index(axis=1).to_csv(path)
        paths["overlap"] = path
    if rows_signed:
        path = out_dir / "gene_structure_signed_distance_matrix.csv"
        pd.DataFrame.from_dict(rows_signed, orient="index").sort_index(axis=1).to_csv(path)
        paths["signed"] = path
    if rows_inside:
        path = out_dir / "gene_structure_fraction_inside_matrix.csv"
        pd.DataFrame.from_dict(rows_inside, orient="index").sort_index(axis=1).to_csv(path)
        paths["inside"] = path
    return paths


def _load_hotspot_matrix(path: Path, hotspot: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    columns = [column for column in df.columns if str(column).startswith(f"{hotspot}|")]
    if not columns:
        raise ValueError(f"No {hotspot!r} columns were found in {path}")
    subset = df.loc[:, columns].copy()
    subset.columns = [str(column).split("|", 1)[1] for column in subset.columns]
    return subset


def _zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    values = df.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=1, keepdims=True)
    std = np.nanstd(values, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return pd.DataFrame((values - mean) / std, index=df.index, columns=df.columns)
