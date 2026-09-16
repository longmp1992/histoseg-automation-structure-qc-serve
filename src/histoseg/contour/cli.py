from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .auto_structure import AutoStructureDiscoveryConfig, discover_auto_structures
from .boundary_network import BoundaryNetworkConfig, run_group_boundary_network
from .contour_adjacency import ContourAdjacencyConfig, run_contour_adjacency
from .gene_isoline import GeneTranscriptIsolineConfig, run_gene_transcript_isoline
from .multi_structure import MultiStructureContourConfig, run_multi_structure_contours
from .pattern1_isoline import Pattern1IsolineConfig, run_pattern1_isoline


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run HistoSeg contour analysis workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pattern1 = subparsers.add_parser("pattern1", help="Generate Pattern1 isoline contours.")
    pattern1.add_argument("--clusters-csv", required=True, help="GraphClust clusters.csv path.")
    pattern1.add_argument("--cells-parquet", required=True, help="Cell table parquet path.")
    pattern1.add_argument("--out-dir", required=True, help="Output directory.")
    pattern1.add_argument(
        "--pattern1-clusters",
        required=True,
        help="Comma-separated Pattern1 cluster IDs, for example 10,23,19.",
    )
    pattern1.add_argument("--tissue-boundary-csv", default=None, help="Optional tissue boundary CSV.")
    pattern1.add_argument("--grid-n", type=int, default=1200, help="Square grid size.")
    pattern1.add_argument("--knn-k", type=int, default=30, help="KNN neighbors.")
    pattern1.add_argument("--smooth-sigma", type=float, default=5.0, help="Gaussian smoothing sigma.")
    pattern1.add_argument("--min-cells-inside", type=int, default=10, help="Minimum target cells per loop.")

    gene = subparsers.add_parser(
        "gene-isoline",
        help="Generate gene/transcript isoline contours from Xenium transcript tables.",
    )
    gene.add_argument("--xenium-root", required=True, help="Folder containing Xenium sample folders.")
    gene.add_argument("--out-dir", required=True, help="Output directory.")
    gene.add_argument("--genes", required=True, help="Comma-separated gene/feature names, for example GREM1,COL1A1.")
    gene.add_argument("--sample-glob", default="*", help="Glob used to discover sample folders.")
    gene.add_argument("--xenium-output-glob", default="output-*", help="Glob for Xenium output folders below each sample.")
    gene.add_argument("--qv-min", type=float, default=20, help="Minimum transcript QV.")
    gene.add_argument("--min-transcripts", type=int, default=10, help="Minimum transcripts required for a gene.")
    gene.add_argument("--grid-n", type=int, default=1200, help="Square grid size.")
    gene.add_argument("--knn-k", type=int, default=30, help="KNN neighbors.")
    gene.add_argument("--smooth-sigma", type=float, default=5.0, help="Gaussian smoothing sigma.")
    gene.add_argument("--min-cells-inside", type=int, default=10, help="Minimum target transcripts per loop.")
    gene.add_argument("--alpha", type=float, default=0.05, help="Alpha-shape parameter for tissue boundary generation.")
    gene.add_argument("--xenium-pixel-size-um", type=float, default=0.2125, help="Microns per Xenium Explorer pixel.")
    gene.add_argument(
        "--compute-confidence-score",
        action="store_true",
        help="Compute the pseudo-cluster confidence score for each gene.",
    )
    gene.add_argument(
        "--keep-prepared-inputs",
        action="store_true",
        help="Keep generated pseudo cells.parquet, clusters.csv, and tissue_boundary.csv inputs.",
    )
    gene.add_argument(
        "--no-synth-bg",
        action="store_true",
        help="Disable synthetic background points in the underlying Pattern1 run.",
    )
    gene.add_argument("--fail-fast", action="store_true", help="Stop on the first sample/gene error.")

    auto = subparsers.add_parser(
        "auto-structure",
        help="Automatically discover coarse structure groups from a Xenium output folder.",
    )
    auto.add_argument("--xenium-output", default=None, help="Xenium outs folder, or parent containing one *_outs folder.")
    auto.add_argument("--clusters-csv", default=None, help="GraphClust clusters.csv path.")
    auto.add_argument("--cells-parquet", default=None, help="Cell table parquet path.")
    auto.add_argument("--out-dir", required=True, help="Output directory.")
    auto.add_argument(
        "--cluster-count",
        default="auto",
        help=(
            "Number of structures to discover, 'auto' to select between min/max counts, "
            "or 'leaf-balanced' to merge StructureMap leaves bottom-up."
        ),
    )
    auto.add_argument("--min-structure-count", type=int, default=3, help="Minimum auto-selected structure count.")
    auto.add_argument("--max-structure-count", type=int, default=8, help="Maximum auto-selected structure count.")
    auto.add_argument(
        "--max-leaf-clusters-per-structure",
        type=int,
        default=5,
        help=(
            "For --cluster-count leaf-balanced, normal maximum original cluster "
            "leaves per structure before tight-branch extension is considered."
        ),
    )
    auto.add_argument(
        "--min-leaf-clusters-per-structure",
        type=int,
        default=1,
        help="For --cluster-count leaf-balanced, merge tiny leaf groups when possible. Default keeps singleton leaves.",
    )
    auto.add_argument(
        "--extended-leaf-clusters-per-structure",
        type=int,
        default=10,
        help=(
            "For --cluster-count leaf-balanced, allow a tight StructureMap branch "
            "to grow up to this many graphclust leaves."
        ),
    )
    auto.add_argument(
        "--extended-leaf-merge-distance-threshold",
        type=float,
        default=0.25,
        help=(
            "Maximum normalized StructureMap distance inside a branch for extending "
            "beyond --max-leaf-clusters-per-structure."
        ),
    )
    auto.add_argument(
        "--min-structure-cell-fraction",
        type=float,
        default=0.01,
        help="Minimum case-cell fraction preferred for each auto-selected structure.",
    )
    auto.add_argument("--barcode-col", default="Barcode", help="Barcode column in clusters.csv.")
    auto.add_argument("--cluster-col", default="Cluster", help="Cluster-label column in clusters.csv.")
    auto.add_argument("--linkage-method", default="average", help="Agglomerative linkage method.")
    auto.add_argument("--no-cophenetic", action="store_true", help="Cluster directly on spatial distance matrix.")
    auto.add_argument("--dry-run", action="store_true", help="Resolve inputs and compute the plan without writing files.")
    auto.add_argument("--overwrite", action="store_true", help="Overwrite existing auto-structure outputs.")

    multi = subparsers.add_parser("multi-structure", help="Generate multi-structure contours.")
    multi.add_argument("--clusters-csv", required=True, help="GraphClust clusters.csv path.")
    multi.add_argument("--cells-parquet", required=True, help="Cell table parquet path.")
    multi.add_argument("--out-dir", required=True, help="Output directory.")
    multi.add_argument(
        "--structures-json",
        required=True,
        help="JSON file containing a list of structure specs with structure_name and cluster_ids.",
    )
    multi.add_argument("--bins-x", type=int, default=900, help="Rasterization bins along x.")
    multi.add_argument("--bins-y", type=int, default=700, help="Rasterization bins along y.")
    multi.add_argument("--gaussian-sigma", type=float, default=2.25, help="Density smoothing sigma.")
    multi.add_argument("--min-cells", type=int, default=500, help="Minimum assigned cells per contour.")

    network = subparsers.add_parser(
        "boundary-network",
        help="Render a group boundary-overlap network from a CSV edge table.",
    )
    network.add_argument("--boundary-csv", required=True, help="Group boundary-overlap CSV path.")
    network.add_argument("--out-dir", required=True, help="Output directory.")
    network.add_argument(
        "--drop-structures",
        default="",
        help="Comma-separated structure labels or IDs to omit, for example 1,6.",
    )
    network.add_argument(
        "--min-shared-boundary",
        type=float,
        default=0.0,
        help="Minimum shared boundary length to keep an edge.",
    )
    network.add_argument(
        "--min-boundary-pairs",
        type=int,
        default=0,
        help="Minimum boundary pair count to keep an edge.",
    )
    network.add_argument(
        "--top-n-edges",
        type=int,
        default=None,
        help="Keep only the top N edges after filtering.",
    )
    network.add_argument("--dpi", type=int, default=200, help="Preview PNG resolution.")
    network.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip writing group_boundary_network.png.",
    )

    adjacency = subparsers.add_parser(
        "adjacency",
        help="Compute contour-type adjacency from contour geometries.",
    )
    adjacency.add_argument("--contours-csv", required=True, help="Contour CSV path.")
    adjacency.add_argument("--groupby", required=True, help="Contour type/group column.")
    adjacency.add_argument("--out-dir", required=True, help="Output directory.")
    adjacency.add_argument(
        "--contour-id-col",
        default="contour_id",
        help="Column containing stable contour IDs.",
    )
    adjacency.add_argument(
        "--geometry-col",
        default="geometry",
        help="Column containing WKT, WKB hex, or GeoJSON geometries.",
    )
    adjacency.add_argument(
        "--boundary-tolerance",
        type=float,
        default=1.0,
        help="Distance tolerance for nearly coincident boundaries.",
    )
    adjacency.add_argument(
        "--min-shared-boundary",
        type=float,
        default=0.0,
        help="Minimum shared boundary length for boundary-neighbor pairs.",
    )
    adjacency.add_argument(
        "--enclosure-min-fraction",
        type=float,
        default=0.95,
        help="Minimum inner area covered fraction for enclosure pairs.",
    )
    adjacency.add_argument("--dpi", type=int, default=200, help="Preview PNG resolution.")
    adjacency.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip writing adjacency network and heatmap PNGs.",
    )

    args = parser.parse_args(argv)
    if args.command == "pattern1":
        result = run_pattern1_isoline(
            Pattern1IsolineConfig(
                clusters_csv=args.clusters_csv,
                cells_parquet=args.cells_parquet,
                tissue_boundary_csv=args.tissue_boundary_csv,
                out_dir=args.out_dir,
                pattern1_clusters=_parse_cluster_list(args.pattern1_clusters),
                grid_n=args.grid_n,
                knn_k=args.knn_k,
                smooth_sigma=args.smooth_sigma,
                min_cells_inside=args.min_cells_inside,
            )
        )
    elif args.command == "gene-isoline":
        result = run_gene_transcript_isoline(
            GeneTranscriptIsolineConfig(
                xenium_root=args.xenium_root,
                out_dir=args.out_dir,
                genes=_parse_csv_list(args.genes),
                sample_glob=args.sample_glob,
                xenium_output_glob=args.xenium_output_glob,
                qv_min=args.qv_min,
                min_transcripts=args.min_transcripts,
                grid_n=args.grid_n,
                knn_k=args.knn_k,
                smooth_sigma=args.smooth_sigma,
                min_cells_inside=args.min_cells_inside,
                use_synth_bg=not args.no_synth_bg,
                alpha=args.alpha,
                xenium_pixel_size_um=args.xenium_pixel_size_um,
                compute_confidence_score=args.compute_confidence_score,
                keep_prepared_inputs=args.keep_prepared_inputs,
                fail_fast=args.fail_fast,
            )
        )
    elif args.command == "auto-structure":
        result = discover_auto_structures(
            AutoStructureDiscoveryConfig(
                xenium_output=args.xenium_output,
                clusters_csv=args.clusters_csv,
                cells_parquet=args.cells_parquet,
                out_dir=args.out_dir,
                barcode_col=args.barcode_col,
                cluster_col=args.cluster_col,
                cluster_count=_parse_auto_cluster_count(args.cluster_count),
                min_structure_count=args.min_structure_count,
                max_structure_count=args.max_structure_count,
                max_leaf_clusters_per_structure=args.max_leaf_clusters_per_structure,
                min_leaf_clusters_per_structure=args.min_leaf_clusters_per_structure,
                extended_leaf_clusters_per_structure=args.extended_leaf_clusters_per_structure,
                extended_leaf_merge_distance_threshold=args.extended_leaf_merge_distance_threshold,
                min_structure_cell_fraction=args.min_structure_cell_fraction,
                linkage_method=args.linkage_method,
                use_cophenetic=not args.no_cophenetic,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
        )
    elif args.command == "multi-structure":
        structures = json.loads(Path(args.structures_json).read_text(encoding="utf-8"))
        result = run_multi_structure_contours(
            MultiStructureContourConfig(
                clusters_csv=args.clusters_csv,
                cells_parquet=args.cells_parquet,
                out_dir=args.out_dir,
                structures=structures,
                bins_x=args.bins_x,
                bins_y=args.bins_y,
                gaussian_sigma=args.gaussian_sigma,
                min_cells=args.min_cells,
            )
        )
    elif args.command == "boundary-network":
        result = run_group_boundary_network(
            BoundaryNetworkConfig(
                boundary_csv=args.boundary_csv,
                out_dir=args.out_dir,
                drop_structures=_parse_csv_list(args.drop_structures),
                min_shared_boundary=args.min_shared_boundary,
                min_boundary_pairs=args.min_boundary_pairs,
                top_n_edges=args.top_n_edges,
                dpi=args.dpi,
                save_preview_png=not args.no_preview,
            )
        )
    else:
        result = run_contour_adjacency(
            ContourAdjacencyConfig(
                contours=args.contours_csv,
                out_dir=args.out_dir,
                groupby=args.groupby,
                contour_id_col=args.contour_id_col,
                geometry_col=args.geometry_col,
                boundary_tolerance=args.boundary_tolerance,
                min_shared_boundary=args.min_shared_boundary,
                enclosure_min_fraction=args.enclosure_min_fraction,
                dpi=args.dpi,
                save_preview_png=not args.no_preview,
            )
        )

    print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))


def _parse_cluster_list(value: str) -> list[str]:
    clusters = [part.strip() for part in value.split(",") if part.strip()]
    if not clusters:
        raise SystemExit("--pattern1-clusters must contain at least one cluster ID.")
    return clusters


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_auto_cluster_count(value: str) -> int | str:
    text = str(value).strip().lower()
    if text in {"auto", "leaf", "leaf-balanced", "leaf_balanced", "bottom-up", "bottom_up"}:
        return text
    try:
        return int(text)
    except ValueError as exc:
        raise SystemExit("--cluster-count must be an integer, 'auto', or 'leaf-balanced'.") from exc


if __name__ == "__main__":
    main()
