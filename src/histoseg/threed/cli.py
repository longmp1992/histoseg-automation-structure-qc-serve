from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from . import (
    CODA_METHOD_REFERENCE_DOI,
    CellCloudProjectionConfig,
    CellCloudRenderConfig,
    ContourTearClosureConfig,
    GeneStructureQuantificationConfig,
    GlandBiologyMiningConfig,
    GlandInstanceSegmentationConfig,
    GlandInstanceTrackingConfig,
    GlandPositionAtlasConfig,
    GlandQCAtlasConfig,
    GlandSurfaceAtlasConfig,
    LabelFreeContourAlignmentConfig,
    LocalZOrientationConfig,
    SpatialModuleDiscoveryConfig,
    SpatialModulePlotConfig,
    SpatialData3DExportConfig,
    ThreeDContourReconstructionConfig,
    ThreeDStackReconstructionConfig,
    export_stack_to_spatialdata_3d,
    plot_spatial_module_clustermap,
    quantify_gene_structure_relationships,
    render_cell_cloud_html,
    render_gland_qc_atlas,
    render_gland_position_atlas,
    align_contours_label_free,
    run_contour_tear_closure,
    run_3d_contour_reconstruction,
    run_3d_stack_reconstruction,
    run_cell_cloud_projection,
    run_gland_instance_detection,
    run_gland_biology_mining,
    run_local_z_orientation_correction,
    render_gland_surface_atlas,
    run_spatial_module_discovery,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run HistoSeg 3D Analysis workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reconstruct = subparsers.add_parser(
        "reconstruct",
        help="Soft-align a hard-aligned moving contour GeoJSON to a fixed slice.",
    )
    reconstruct.add_argument(
        "--fixed-geojson",
        required=True,
        help="Reference contour GeoJSON path.",
    )
    reconstruct.add_argument(
        "--moving-hard-aligned-geojson",
        required=True,
        help="Moving contour GeoJSON after hard/similarity alignment to the reference.",
    )
    reconstruct.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for soft-alignment artifacts.",
    )
    reconstruct.add_argument(
        "--group-property",
        default="structure",
        help="Feature property used to match contour structures.",
    )
    reconstruct.add_argument(
        "--sampling-distance-um",
        type=float,
        default=50.0,
        help="Boundary landmark sampling interval in micrometers.",
    )
    reconstruct.add_argument(
        "--max-landmark-distance-um",
        type=float,
        default=180.0,
        help="Reject source-target landmark pairs farther apart than this distance.",
    )
    reconstruct.add_argument(
        "--landmarks-per-structure",
        type=int,
        default=260,
        help="Conservative cap for boundary landmarks per structure. Use 0 for no cap.",
    )
    reconstruct.add_argument(
        "--diagnostic-structure-landmarks",
        type=int,
        default=620,
        help="Landmark cap for the diagnostic structure. Use 0 to use the regular cap.",
    )
    reconstruct.add_argument(
        "--landmark-candidate-count",
        type=int,
        default=8,
        help="K nearest fixed-boundary candidates for normal-aware landmark matching.",
    )
    reconstruct.add_argument(
        "--landmark-candidate-spacing-um",
        type=float,
        default=None,
        help="Fixed-boundary candidate sampling interval. Defaults to a conservative auto value.",
    )
    reconstruct.add_argument(
        "--landmark-normal-weight-um",
        type=float,
        default=0.0,
        help="Normal-alignment penalty weight. Use 0 for legacy nearest-projection matching.",
    )
    reconstruct.add_argument(
        "--landmark-normal-step-um",
        type=float,
        default=None,
        help="Arc-length step used to estimate boundary normals. Defaults to an auto value.",
    )
    reconstruct.add_argument(
        "--rbf-kernel",
        default="thin_plate_spline",
        help="SciPy RBFInterpolator kernel.",
    )
    reconstruct.add_argument(
        "--rbf-neighbors",
        type=int,
        default=96,
        help="Local RBF neighbor count. Use 0 for a global interpolator.",
    )
    reconstruct.add_argument(
        "--rbf-smoothing",
        default="1e-4",
        help="RBF smoothing parameter, or 'auto' for k-fold CV selection.",
    )
    reconstruct.add_argument(
        "--topology-grid-size",
        type=int,
        default=24,
        help="Sparse grid size for TPS topology checks. Use 0 to disable.",
    )
    reconstruct.add_argument(
        "--topology-min-area-ratio",
        type=float,
        default=0.5,
        help="Reject soft TPS when any checked grid cell shrinks below this area ratio.",
    )
    reconstruct.add_argument(
        "--topology-max-area-ratio",
        type=float,
        default=2.0,
        help="Reject soft TPS when any checked grid cell expands above this area ratio.",
    )
    reconstruct.add_argument(
        "--diagnostic-structure",
        default="Structure 5",
        help="Structure label for the local diagnostic zoom. Use '' to show all structures.",
    )
    reconstruct.add_argument("--dpi", type=int, default=200, help="Preview PNG resolution.")
    reconstruct.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip writing overlay and diagnostic PNGs.",
    )
    reconstruct.add_argument(
        "--curvature-landmark-weight",
        type=float,
        default=0.5,
        help="Fraction of boundary length to oversample at high-curvature regions (0–1).",
    )
    reconstruct.add_argument(
        "--no-mutual-nn-check",
        action="store_true",
        help="Disable mutual nearest-neighbour consistency filter for landmarks.",
    )
    reconstruct.add_argument(
        "--icp-iterations",
        type=int,
        default=2,
        help="Number of ICP-style iterative TPS refinement passes.",
    )
    reconstruct.add_argument(
        "--zero-anchor-count",
        type=int,
        default=16,
        help="Number of zero-displacement anchors placed on the convex-hull perimeter.",
    )
    reconstruct.add_argument(
        "--landmark-outlier-mad-threshold",
        type=float,
        default=3.5,
        help="MAD threshold for landmark outlier rejection. Use 0 to disable.",
    )
    reconstruct.add_argument(
        "--no-per-structure-soft-acceptance",
        action="store_true",
        help="Disable per-structure soft/hard acceptance mixing.",
    )

    label_free = subparsers.add_parser(
        "align-contours-label-free",
        help="Align two contour GeoJSON files without relying on structure labels.",
    )
    label_free.add_argument("--fixed-geojson", required=True, help="Reference contour GeoJSON.")
    label_free.add_argument("--moving-geojson", required=True, help="Moving contour GeoJSON.")
    label_free.add_argument("--out-dir", required=True, help="Output directory.")
    label_free.add_argument(
        "--maxiter",
        type=int,
        default=80,
        help="Maximum Nelder-Mead iterations for hard alignment.",
    )
    label_free.add_argument(
        "--no-multistart",
        action="store_true",
        help="Disable 0/90/180/270 degree multi-start similarity seeds.",
    )
    label_free.add_argument(
        "--affine-fallback-iou-threshold",
        type=float,
        default=0.15,
        help="Try affine fallback when label-free hard-alignment IoU is below this value.",
    )
    label_free.add_argument(
        "--no-soft-tps",
        action="store_true",
        help="Skip unlabeled boundary-landmark TPS refinement.",
    )
    label_free.add_argument(
        "--sampling-distance-um",
        type=float,
        default=80.0,
        help="Boundary sampling interval for moving landmarks.",
    )
    label_free.add_argument(
        "--max-landmark-distance-um",
        type=float,
        default=250.0,
        help="Reject unlabeled landmark pairs farther than this distance.",
    )
    label_free.add_argument(
        "--landmark-candidate-count",
        type=int,
        default=8,
        help="K nearest fixed-boundary candidates for each moving landmark.",
    )
    label_free.add_argument(
        "--landmark-candidate-spacing-um",
        type=float,
        default=None,
        help="Fixed-boundary candidate sampling interval. Defaults to an auto value.",
    )
    label_free.add_argument(
        "--landmark-normal-weight-um",
        type=float,
        default=0.0,
        help="Normal-alignment penalty weight for unlabeled landmark matching.",
    )
    label_free.add_argument(
        "--landmark-normal-step-um",
        type=float,
        default=None,
        help="Arc-length step used to estimate boundary normals.",
    )
    label_free.add_argument("--rbf-kernel", default="thin_plate_spline")
    label_free.add_argument(
        "--rbf-neighbors",
        type=int,
        default=96,
        help="Local RBF neighbor count. Use 0 for a global interpolator.",
    )
    label_free.add_argument("--rbf-smoothing", type=float, default=1e-4)
    label_free.add_argument("--topology-grid-size", type=int, default=24)
    label_free.add_argument("--topology-min-area-ratio", type=float, default=0.5)
    label_free.add_argument("--topology-max-area-ratio", type=float, default=2.0)
    label_free.add_argument(
        "--min-component-area-um2",
        type=float,
        default=0.0,
        help="Ignore contour components smaller than this area.",
    )
    label_free.add_argument(
        "--max-component-weight",
        type=float,
        default=0.08,
        help="Cap each component's contribution to the layout objective.",
    )
    label_free.add_argument(
        "--boundary-sample-count",
        type=int,
        default=6000,
        help="Maximum sampled boundary points used in the label-free objective.",
    )
    label_free.add_argument(
        "--component-sample-count",
        type=int,
        default=800,
        help="Maximum area-ranked components used in the layout objective.",
    )
    label_free.add_argument(
        "--partial-correspondence",
        action="store_true",
        help="Run v2 partial-correspondence candidate matching instead of warping.",
    )
    label_free.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="With --partial-correspondence, write diagnostics only and do not warp coordinates.",
    )
    label_free.add_argument(
        "--search-window",
        type=float,
        default=800.0,
        help="Centroid search window for partial-correspondence candidates.",
    )
    label_free.add_argument(
        "--knn-neighbors",
        type=int,
        default=6,
        help="Local neighbor count for partial-correspondence topology descriptors.",
    )
    label_free.add_argument(
        "--min-anchor-score",
        type=float,
        default=0.72,
        help="Minimum score for a mutual candidate to become a matched anchor.",
    )
    label_free.add_argument(
        "--min-review-score",
        type=float,
        default=0.55,
        help="Minimum score for a plausible review match.",
    )
    label_free.add_argument(
        "--min-anchor-count",
        type=int,
        default=8,
        help="Warn when fewer than this many anchors are found.",
    )
    label_free.add_argument(
        "--overlap-ransac",
        action="store_true",
        help=(
            "With --partial-correspondence, estimate alignment from a descriptor-first "
            "RANSAC overlap subset instead of current-coordinate anchors."
        ),
    )
    label_free.add_argument(
        "--overlap-candidate-count",
        type=int,
        default=8,
        help="Descriptor candidates retained per fixed contour for overlap RANSAC.",
    )
    label_free.add_argument(
        "--overlap-ransac-trials",
        type=int,
        default=20000,
        help="Maximum deterministic candidate-pair trials for overlap RANSAC.",
    )
    label_free.add_argument(
        "--overlap-min-descriptor-score",
        type=float,
        default=0.42,
        help="Minimum descriptor-only score for overlap RANSAC candidate pairs.",
    )
    label_free.add_argument(
        "--overlap-allow-scale",
        action="store_true",
        help="Allow overlap RANSAC to estimate scale. By default it is rigid rotation/translation.",
    )
    label_free.add_argument(
        "--group-correspondence",
        action="store_true",
        help=(
            "With --partial-correspondence, match assigned_structure groups across slices "
            "before estimating an overlap transform. This allows fixed Structure 2 to "
            "match moving Structure 3, for example."
        ),
    )
    label_free.add_argument(
        "--group-candidate-count",
        type=int,
        default=12,
        help="Descriptor candidates retained per fixed contour inside each group-pair.",
    )
    label_free.add_argument(
        "--group-ransac-trials",
        type=int,
        default=15000,
        help="Maximum RANSAC trials per fixed-group/moving-group pair.",
    )
    label_free.add_argument(
        "--group-min-descriptor-score",
        type=float,
        default=0.35,
        help="Minimum descriptor-only score for group correspondence candidates.",
    )
    label_free.add_argument(
        "--group-residual-limit-um",
        type=float,
        default=900.0,
        help="Maximum centroid residual for group correspondence inliers.",
    )
    label_free.add_argument(
        "--group-min-component-area-um2",
        type=float,
        default=5000.0,
        help="Minimum contour area included in group-correspondence constellation matching.",
    )
    label_free.add_argument("--dpi", type=int, default=180, help="Preview PNG resolution.")
    label_free.add_argument("--no-preview", action="store_true", help="Skip overlay PNGs.")
    label_free.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")

    tear_close = subparsers.add_parser(
        "close-contour-tear",
        help="Close tear-like moving-slice gaps using same-celltype contour correspondences.",
    )
    tear_close.add_argument("--fixed-geojson", required=True, help="Reference contour GeoJSON.")
    tear_close.add_argument(
        "--moving-aligned-geojson",
        required=True,
        help="Moving contour GeoJSON already aligned into the fixed coordinate system.",
    )
    tear_close.add_argument(
        "--moving-cells-csv",
        required=True,
        help="Moving cell table already aligned into the fixed coordinate system.",
    )
    tear_close.add_argument(
        "--fixed-cells-csv",
        default=None,
        help="Optional fixed cell table recorded in the summary for provenance.",
    )
    tear_close.add_argument("--out-dir", required=True, help="Output directory.")
    tear_close.add_argument(
        "--group-property",
        default="assigned_structure",
        help="GeoJSON feature property used for same-type contour matching.",
    )
    tear_close.add_argument(
        "--moving-x-column",
        default="x_aligned_um",
        help="Moving cell x-coordinate column to warp.",
    )
    tear_close.add_argument(
        "--moving-y-column",
        default="y_aligned_um",
        help="Moving cell y-coordinate column to warp.",
    )
    tear_close.add_argument(
        "--fixed-x-column",
        default="x_centroid",
        help="Fixed cell x-coordinate column recorded for provenance.",
    )
    tear_close.add_argument(
        "--fixed-y-column",
        default="y_centroid",
        help="Fixed cell y-coordinate column recorded for provenance.",
    )
    tear_close.add_argument("--cluster-column", default="cluster")
    tear_close.add_argument(
        "--output-coordinate-prefix",
        default="contour_tear_closure",
        help="Prefix for warped moving cell coordinate/displacement columns.",
    )
    tear_close.add_argument("--centroid-search-radius-um", type=float, default=650.0)
    tear_close.add_argument("--area-ratio-min", type=float, default=0.10)
    tear_close.add_argument("--area-ratio-max", type=float, default=10.0)
    tear_close.add_argument("--min-pair-score", type=float, default=0.0)
    tear_close.add_argument("--min-motion-um", type=float, default=0.0)
    tear_close.add_argument("--tear-motion-threshold-um", type=float, default=25.0)
    tear_close.add_argument("--influence-radius-um", type=float, default=650.0)
    tear_close.add_argument("--max-neighbors", type=int, default=12)
    tear_close.add_argument("--max-displacement-um", type=float, default=200.0)
    tear_close.add_argument("--dpi", type=int, default=180)
    tear_close.add_argument("--no-preview", action="store_true", help="Skip landmark review PNG.")
    tear_close.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")

    stack = subparsers.add_parser(
        "reconstruct-stack",
        help="Build and align a same-sample multi-slice Xenium contour stack.",
    )
    stack.add_argument(
        "--xenium-root",
        default=None,
        help=(
            "Folder containing ordered Xenium slice folders. Required unless "
            "--contour-manifest supplies precomputed per-slice GeoJSON contours."
        ),
    )
    stack.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for the 3D contour stack artifacts.",
    )
    stack.add_argument(
        "--segmentation-strategy",
        default=None,
        help="Text file with one comma-separated cluster list per Structure N.",
    )
    stack.add_argument(
        "--contour-manifest",
        default=None,
        help=(
            "CSV manifest of precomputed slice-local contour GeoJSONs with columns "
            "order,sample_id,z_um,geojson. When provided, reconstruct-stack skips "
            "Xenium contour generation and preserves each GeoJSON's existing labels."
        ),
    )
    stack.add_argument(
        "--sample-glob",
        default="*",
        help="Glob used to discover slice folders below --xenium-root.",
    )
    stack.add_argument(
        "--z-spacing-um",
        type=float,
        default=5.0,
        help="Physical spacing between consecutive slices.",
    )
    stack.add_argument(
        "--xenium-pixel-size-um",
        type=float,
        default=0.2125,
        help="Microns per Xenium Explorer annotation pixel.",
    )
    stack.add_argument(
        "--group-property",
        default="structure",
        help="GeoJSON property used for structure labels.",
    )
    stack.add_argument(
        "--cluster-column-name",
        default="cluster",
        help="Cluster column name requested from pyXenium read_xenium.",
    )
    stack.add_argument(
        "--clusters-relpath",
        default="analysis/clustering/gene_expression_graphclust/clusters.csv",
        help="Cluster CSV path relative to each Xenium output directory.",
    )
    stack.add_argument(
        "--merged-h5ad",
        default=None,
        help="Optional merged AnnData with unified per-cell clusters for all slices.",
    )
    stack.add_argument(
        "--merged-cluster-column",
        default=None,
        help="Cluster column in merged AnnData obs. Defaults to leiden_1_0 or first leiden* column.",
    )
    stack.add_argument("--merged-sample-column", default="sample_id")
    stack.add_argument("--merged-barcode-column", default="barcode")
    stack.add_argument("--bins-x", type=int, default=900, help="Contour raster bins in X.")
    stack.add_argument("--bins-y", type=int, default=700, help="Contour raster bins in Y.")
    stack.add_argument("--gaussian-sigma", type=float, default=2.25)
    stack.add_argument("--min-cells", type=int, default=500)
    stack.add_argument("--min-component-pixels", type=int, default=180)
    stack.add_argument(
        "--registration-backend",
        default="auto",
        choices=["auto", "contour-tps", "label-free-group", "coda-image"],
        help=(
            "Hard-alignment seed backend. 'auto' compares same-label contour alignment "
            "with label-free group correspondence and promotes the reliable seed. "
            "'contour-tps' preserves the existing contour similarity seed. "
            "'label-free-group' forces cross-group geometry matching. 'coda-image' uses "
            f"a CODA-inspired tissue-mask seed (DOI: {CODA_METHOD_REFERENCE_DOI})."
        ),
    )
    stack.add_argument("--hard-alignment-maxiter", type=int, default=80)
    stack.add_argument(
        "--coda-raster-size",
        type=int,
        default=512,
        help="Raster size for the CODA-inspired contour tissue-mask proxy.",
    )
    stack.add_argument(
        "--coda-angle-step",
        type=float,
        default=1.0,
        help="Radon angle grid step, in degrees, for --registration-backend coda-image.",
    )
    stack.add_argument(
        "--coda-phase-upsample-factor",
        type=int,
        default=1,
        help="Phase-correlation upsample factor for --registration-backend coda-image.",
    )
    stack.add_argument(
        "--coda-mask-padding-fraction",
        type=float,
        default=0.05,
        help="Fractional square-bounds padding for CODA-inspired mask rasterization.",
    )
    stack.add_argument("--label-free-search-window", type=float, default=800.0)
    stack.add_argument("--label-free-knn-neighbors", type=int, default=6)
    stack.add_argument("--label-free-min-anchor-count", type=int, default=8)
    stack.add_argument("--label-free-group-candidate-count", type=int, default=12)
    stack.add_argument("--label-free-group-ransac-trials", type=int, default=15000)
    stack.add_argument("--label-free-group-min-descriptor-score", type=float, default=0.35)
    stack.add_argument("--label-free-group-residual-limit-um", type=float, default=900.0)
    stack.add_argument("--label-free-group-min-component-area-um2", type=float, default=5000.0)
    stack.add_argument(
        "--no-label-free-guarded-alignment",
        action="store_true",
        help="Disable conservative label-free group transform acceptance guards.",
    )
    stack.add_argument("--label-free-min-similarity-anchor-count", type=int, default=6)
    stack.add_argument("--label-free-max-rotation-degrees", type=float, default=15.0)
    stack.add_argument("--label-free-high-rotation-min-anchor-count", type=int, default=8)
    stack.add_argument("--label-free-high-rotation-residual-limit-um", type=float, default=120.0)
    stack.add_argument("--label-free-min-anchor-coverage", type=float, default=0.05)
    stack.add_argument("--label-free-min-anchor-quadrants", type=int, default=2)
    stack.add_argument("--label-free-near-180-rotation-degrees", type=float, default=150.0)
    stack.add_argument(
        "--sampling-distance-um",
        type=float,
        default=50.0,
        help="TPS boundary sampling interval in GeoJSON coordinate units.",
    )
    stack.add_argument(
        "--max-landmark-distance-um",
        type=float,
        default=180.0,
        help="TPS nearest-boundary rejection distance in GeoJSON coordinate units.",
    )
    stack.add_argument("--landmarks-per-structure", type=int, default=260)
    stack.add_argument("--diagnostic-structure-landmarks", type=int, default=620)
    stack.add_argument("--landmark-candidate-count", type=int, default=8)
    stack.add_argument("--landmark-candidate-spacing-um", type=float, default=None)
    stack.add_argument("--landmark-normal-weight-um", type=float, default=0.0)
    stack.add_argument("--landmark-normal-step-um", type=float, default=None)
    stack.add_argument("--rbf-neighbors", type=int, default=96)
    stack.add_argument("--rbf-smoothing", default="1e-4", help="RBF smoothing or 'auto'.")
    stack.add_argument("--topology-grid-size", type=int, default=24)
    stack.add_argument("--topology-min-area-ratio", type=float, default=0.5)
    stack.add_argument("--topology-max-area-ratio", type=float, default=2.0)
    stack.add_argument("--diagnostic-structure", default="Structure 5")
    stack.add_argument("--point-sample-distance-um", type=float, default=80.0)
    stack.add_argument("--voxel-size-um", type=float, default=80.0)
    stack.add_argument("--mesh-method", default="marching_cubes")
    stack.add_argument(
        "--mesh-smoothing-sigma-um",
        type=float,
        default=40.0,
        help="3D Gaussian smoothing sigma in microns. Use 0 to disable smoothing.",
    )
    stack.add_argument("--mesh-level", type=float, default=0.5)
    stack.add_argument(
        "--mesh-export-formats",
        default="ply,obj",
        help="Comma-separated mesh export formats. Supported: ply,obj.",
    )
    stack.add_argument(
        "--no-mesh-cleanup",
        action="store_true",
        help="Skip trimesh post-processing before writing PLY/OBJ.",
    )
    stack.add_argument(
        "--min-mesh-component-volume-um3",
        type=float,
        default=None,
        help="Drop disconnected volume components smaller than this volume.",
    )
    stack.add_argument("--mesh-max-faces-for-html", type=int, default=25000)
    stack.add_argument("--dpi", type=int, default=180)
    stack.add_argument(
        "--no-soft",
        action="store_true",
        help="Use only hard alignment between slices. Equivalent to --soft-alignment-mode none.",
    )
    stack.add_argument(
        "--soft-alignment-mode",
        default="auto",
        choices=["auto", "semantic", "anchor-only", "none"],
        help=(
            "Soft alignment strategy. auto uses anchor-only residual TPS for "
            "label-free-group hard seeds and semantic TPS for traditional hard seeds."
        ),
    )
    stack.add_argument("--anchor-only-bbox-padding-fraction", type=float, default=0.10)
    stack.add_argument("--anchor-only-identity-padding-count", type=int, default=16)
    stack.add_argument("--anchor-only-rbf-smoothing", type=float, default=1e-4)
    stack.add_argument("--anchor-only-jacobian-grid-size", type=int, default=50)
    stack.add_argument("--anchor-only-max-negative-jacobian-ratio", type=float, default=0.001)
    stack.add_argument(
        "--no-iterative-contour-refinement",
        action="store_true",
        help=(
            "Disable the default post-anchor contour refinement pass for label-free "
            "anchor-only soft alignment."
        ),
    )
    stack.add_argument("--iterative-contour-refinement-rounds", type=int, default=1)
    stack.add_argument("--iterative-contour-min-pair-count", type=int, default=3)
    stack.add_argument("--iterative-contour-min-anchor-count", type=int, default=30)
    stack.add_argument("--iterative-contour-min-pair-score", type=float, default=0.52)
    stack.add_argument("--iterative-contour-min-overlap", type=float, default=0.12)
    stack.add_argument("--iterative-contour-centroid-search-radius-um", type=float, default=350.0)
    stack.add_argument("--iterative-contour-accepted-centroid-radius-um", type=float, default=260.0)
    stack.add_argument("--iterative-contour-max-anchor-distance-um", type=float, default=180.0)
    stack.add_argument("--iterative-contour-residual-limit-um", type=float, default=220.0)
    stack.add_argument("--iterative-contour-max-negative-jacobian-ratio", type=float, default=0.002)
    stack.add_argument(
        "--save-slice-preview",
        action="store_true",
        help="Write the per-slice 2D contour preview PNGs.",
    )
    stack.add_argument(
        "--no-alignment-preview",
        action="store_true",
        help="Skip per-pair soft-alignment preview PNGs.",
    )
    stack.add_argument("--overwrite", action="store_true", help="Recompute existing artifacts.")
    stack.add_argument(
        "--no-multistart",
        action="store_true",
        help="Disable multi-start similarity alignment (single PCA-seeded run only).",
    )
    stack.add_argument(
        "--affine-fallback-iou-threshold",
        type=float,
        default=0.0,
        help="Try 6-DOF affine fallback when similarity IoU is below this value. 0 = disabled.",
    )
    stack.add_argument(
        "--global-drift-correction",
        action="store_true",
        help="Apply linear centroid drift correction after the full alignment chain.",
    )
    stack.add_argument(
        "--curvature-landmark-weight",
        type=float,
        default=0.5,
        help="Fraction of boundary to oversample at high-curvature regions (0–1).",
    )
    stack.add_argument(
        "--no-mutual-nn-check",
        action="store_true",
        help="Disable mutual nearest-neighbour landmark consistency filter.",
    )
    stack.add_argument("--icp-iterations", type=int, default=2)
    stack.add_argument("--zero-anchor-count", type=int, default=16)
    stack.add_argument("--landmark-outlier-mad-threshold", type=float, default=3.5)
    stack.add_argument(
        "--no-per-structure-soft-acceptance",
        action="store_true",
        help="Use global soft/hard acceptance instead of per-structure acceptance.",
    )
    stack.add_argument(
        "--local-z-orientation",
        default="off",
        choices=["off", "auto"],
        help="Optionally infer transcript-level local-z preserve/reverse states after stack reconstruction.",
    )
    stack.add_argument(
        "--vertical-qc-backend",
        default="none",
        choices=["none", "ovrlpy"],
        help="Vertical transcript QC backend used by --local-z-orientation auto.",
    )
    stack.add_argument(
        "--apply-local-z-flip",
        action="store_true",
        help="Apply inferred local-z reverse states to aligned_transcripts_3d.parquet.",
    )
    stack.add_argument(
        "--transcript-relpath",
        default="transcripts.parquet",
        help="Transcript table path relative to each Xenium output directory.",
    )
    stack.add_argument(
        "--orientation-spatial-unit",
        default="auto",
        choices=["auto", "global", "contour"],
        help="Spatial unit used by local-z orientation scoring.",
    )
    stack.add_argument("--contour-group-property", default="structure")
    stack.add_argument("--contour-min-transcripts", type=int, default=50)
    stack.add_argument("--contour-match-min-iou", type=float, default=0.01)
    stack.add_argument("--contour-match-max-distance-um", type=float, default=120.0)
    stack.add_argument("--orientation-bootstrap-iterations", type=int, default=100)
    stack.add_argument("--orientation-bootstrap-seed", type=int, default=0)
    stack.add_argument("--ovrlpy-n-components", type=int, default=20)
    stack.add_argument("--ovrlpy-n-workers", type=int, default=1)
    stack.add_argument(
        "--ovrlpy-fit-umap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit Ovrlpy UMAP embeddings during vertical QC.",
    )
    stack.add_argument("--ovrlpy-min-transcripts", type=float, default=10.0)

    spatialdata_export = subparsers.add_parser(
        "export-spatialdata-3d",
        help="Export an aligned HistoSeg stack as SpatialData for napari-spatialdata 3D viewing.",
    )
    spatialdata_export.add_argument("--stack-root", required=True)
    spatialdata_export.add_argument("--out-zarr", default=None)
    spatialdata_export.add_argument("--aligned-cells-parquet", default=None)
    spatialdata_export.add_argument("--group-property", default="structure")
    spatialdata_export.add_argument("--xenium-pixel-size-um", type=float, default=0.2125)
    spatialdata_export.add_argument(
        "--no-contour-points",
        action="store_true",
        help="Only export 2.5D contour shapes, not sampled contour points.",
    )
    spatialdata_export.add_argument(
        "--include-cells",
        action="store_true",
        help="Also export aligned cell points from --aligned-cells-parquet.",
    )
    spatialdata_export.add_argument("--max-cells", type=int, default=300000)
    spatialdata_export.add_argument("--cell-x-column", default="x_aligned_um")
    spatialdata_export.add_argument("--cell-y-column", default="y_aligned_um")
    spatialdata_export.add_argument("--cell-z-column", default="z_um")
    spatialdata_export.add_argument("--overwrite", action="store_true")

    local_z = subparsers.add_parser(
        "infer-local-z-orientation",
        help="Infer and apply transcript-level local-z orientation for an existing stack.",
    )
    local_z.add_argument("--stack-root", required=True, help="HistoSeg 3D reconstruction output directory.")
    local_z.add_argument("--xenium-root", required=True, help="Folder containing Xenium slice folders.")
    local_z.add_argument("--out-dir", default=None, help="Output directory. Defaults to --stack-root.")
    local_z.add_argument("--sample-glob", default="*")
    local_z.add_argument("--transcript-relpath", default="transcripts.parquet")
    local_z.add_argument("--gene-column", default=None)
    local_z.add_argument("--x-column", default=None)
    local_z.add_argument("--y-column", default=None)
    local_z.add_argument("--z-column", default=None)
    local_z.add_argument("--transcript-id-column", default=None)
    local_z.add_argument("--structure-column", default=None)
    local_z.add_argument("--pixel-size-um", type=float, default=0.2125)
    local_z.add_argument("--z-band-fraction", type=float, default=0.25)
    local_z.add_argument("--min-band-transcripts", type=int, default=10)
    local_z.add_argument("--max-signature-genes", type=int, default=256)
    local_z.add_argument("--low-confidence-margin", type=float, default=0.02)
    local_z.add_argument(
        "--orientation-spatial-unit",
        default="auto",
        choices=["auto", "global", "contour"],
        help="Use contour-aware scoring when available, force global scoring, or force contour scoring.",
    )
    local_z.add_argument("--contour-group-property", default="structure")
    local_z.add_argument("--contour-min-transcripts", type=int, default=50)
    local_z.add_argument("--contour-match-min-iou", type=float, default=0.01)
    local_z.add_argument("--contour-match-max-distance-um", type=float, default=120.0)
    local_z.add_argument("--orientation-bootstrap-iterations", type=int, default=100)
    local_z.add_argument("--orientation-bootstrap-seed", type=int, default=0)
    local_z.add_argument(
        "--vertical-qc-backend",
        default="ovrlpy",
        choices=["none", "ovrlpy"],
    )
    local_z.add_argument(
        "--apply-local-z-flip",
        action="store_true",
        help="Write reversed local-z values for slices inferred as reverse.",
    )
    local_z.add_argument("--ovrlpy-kde-bandwidth", type=float, default=2.5)
    local_z.add_argument("--ovrlpy-n-components", type=int, default=20)
    local_z.add_argument("--ovrlpy-n-workers", type=int, default=1)
    local_z.add_argument(
        "--ovrlpy-fit-umap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit Ovrlpy UMAP embeddings during vertical QC.",
    )
    local_z.add_argument("--ovrlpy-min-transcripts", type=float, default=10.0)
    local_z.add_argument("--doublet-min-signal", type=float, default=4.0)
    local_z.add_argument("--doublet-integrity-sigma", type=float, default=1.0)
    local_z.add_argument("--doublet-exclusion-radius-um", type=float, default=20.0)
    local_z.add_argument("--chunk-size", type=int, default=100000)

    project_cells = subparsers.add_parser(
        "project-cells",
        help="Project merged AnnData cells into a HistoSeg 3D reconstruction.",
    )
    project_cells.add_argument("--h5ad", required=True, help="Merged AnnData .h5ad path.")
    project_cells.add_argument("--stack-root", required=True, help="HistoSeg 3D reconstruction output directory.")
    project_cells.add_argument("--out-parquet", required=True, help="Output aligned 3D cell Parquet path.")
    project_cells.add_argument("--sample-column", default="sample_id")
    project_cells.add_argument("--barcode-column", default="barcode")
    project_cells.add_argument("--x-column", default="x_centroid")
    project_cells.add_argument("--y-column", default="y_centroid")
    project_cells.add_argument(
        "--label-columns",
        nargs="*",
        default=(),
        help="Optional obs columns to carry into the output table. Comma-separated chunks are accepted.",
    )
    project_cells.add_argument("--pixel-size-um", type=float, default=0.2125)
    project_cells.add_argument("--chunk-size", type=int, default=100000)
    project_cells.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Ignore AnnData HistoSeg coordinate cache and recompute coordinates.",
    )
    project_cells.add_argument(
        "--fail-on-stale-cache",
        action="store_true",
        help="Raise an error instead of recomputing when cached coordinates have a stale alignment hash.",
    )
    project_cells.add_argument(
        "--write-cache",
        action="store_true",
        help="Write HistoSeg 3D coordinates and provenance back to AnnData.",
    )
    project_cells.add_argument(
        "--cache-h5ad",
        default=None,
        help="Optional output .h5ad for --write-cache. Defaults to overwriting --h5ad.",
    )
    project_cells.add_argument(
        "--write-scanpy-spatial",
        action="store_true",
        help="With --write-cache, also write aligned XY to .obsm['spatial'].",
    )

    render_cells = subparsers.add_parser(
        "render-cell-cloud",
        help="Render an aligned 3D cell cloud as an interactive Plotly HTML.",
    )
    render_cells.add_argument("--stack-root", required=True, help="HistoSeg 3D reconstruction output directory.")
    render_cells.add_argument("--out-html", required=True, help="Output interactive Plotly HTML path.")
    render_cells.add_argument(
        "--aligned-cells-parquet",
        default=None,
        help="Existing aligned 3D cell Parquet. If omitted, --h5ad and --out-parquet are required.",
    )
    render_cells.add_argument("--h5ad", default=None, help="Merged AnnData .h5ad path to project before rendering.")
    render_cells.add_argument(
        "--out-parquet",
        default=None,
        help="Aligned 3D cell Parquet to write when rendering from --h5ad.",
    )
    render_cells.add_argument("--sample-column", default="sample_id")
    render_cells.add_argument("--barcode-column", default="barcode")
    render_cells.add_argument("--x-column", default="x_centroid")
    render_cells.add_argument("--y-column", default="y_centroid")
    render_cells.add_argument("--label-column", default="leiden_1_0")
    render_cells.add_argument(
        "--label-columns",
        nargs="*",
        default=(),
        help="Extra AnnData obs columns to carry when projecting from --h5ad.",
    )
    render_cells.add_argument("--pixel-size-um", type=float, default=0.2125)
    render_cells.add_argument("--chunk-size", type=int, default=100000)
    render_cells.add_argument("--ignore-cache", action="store_true")
    render_cells.add_argument("--fail-on-stale-cache", action="store_true")
    render_cells.add_argument("--write-cache", action="store_true")
    render_cells.add_argument("--cache-h5ad", default=None)
    render_cells.add_argument("--write-scanpy-spatial", action="store_true")
    render_cells.add_argument("--max-points", type=int, default=300000)
    render_cells.add_argument("--random-state", type=int, default=0)
    render_cells.add_argument("--z-visual-scale", type=float, default=8.0)
    render_cells.add_argument("--marker-size", type=float, default=1.4)
    render_cells.add_argument("--opacity", type=float, default=0.48)
    render_cells.add_argument("--no-contours", action="store_true")
    render_cells.add_argument("--title", default="HistoSeg 3D cell cloud")
    render_cells.add_argument("--performance-warning-threshold", type=int, default=500000)

    gland_qc = subparsers.add_parser(
        "render-gland-qc-atlas",
        help="Render a per-gland 3D QC atlas from an aligned contour stack.",
    )
    gland_qc.add_argument("--stack-root", required=True, help="HistoSeg 3D reconstruction output directory.")
    gland_qc.add_argument("--out-dir", required=True, help="Output directory for the gland QC atlas.")
    gland_qc.add_argument(
        "--aligned-cells-parquet",
        default=None,
        help="Optional aligned 3D cell Parquet for local cell context.",
    )
    gland_qc.add_argument(
        "--structures",
        nargs="*",
        default=(),
        help="Optional structure labels to include. Comma-separated chunks are accepted.",
    )
    gland_qc.add_argument("--group-property", default="structure")
    gland_qc.add_argument("--pixel-size-um", type=float, default=0.2125)
    gland_qc.add_argument("--padding-um", type=float, default=250.0)
    gland_qc.add_argument("--z-visual-scale", type=float, default=8.0)
    gland_qc.add_argument("--max-cells-per-gland", type=int, default=50000)
    gland_qc.add_argument(
        "--max-gland-pages",
        type=int,
        default=0,
        help="Render only the top N priority gland pages. 0 renders every gland page.",
    )
    gland_qc.add_argument("--label-column", default="leiden_1_0")
    gland_qc.add_argument("--random-state", type=int, default=0)
    gland_qc.add_argument("--min-overlap-fraction", type=float, default=0.05)
    gland_qc.add_argument("--good-overlap-fraction", type=float, default=0.20)
    gland_qc.add_argument("--centroid-fallback-um", type=float, default=160.0)
    gland_qc.add_argument("--max-area-ratio-for-fallback", type=float, default=4.0)
    gland_qc.add_argument("--centroid-jump-review-um", type=float, default=350.0)
    gland_qc.add_argument("--area-cv-review-threshold", type=float, default=1.0)

    detect_glands = subparsers.add_parser(
        "detect-gland-instances",
        help="Segment and track individual gland/crypt instances across a 3D stack.",
    )
    detect_glands.add_argument("--stack-root", required=True, help="HistoSeg 3D reconstruction output directory.")
    detect_glands.add_argument("--out-dir", required=True, help="Output directory for gland instance artifacts.")
    detect_glands.add_argument(
        "--aligned-cells-parquet",
        default=None,
        help="Aligned 3D cell Parquet, usually aligned_leiden_3d_cells.parquet.",
    )
    detect_glands.add_argument(
        "--epithelial-structures",
        nargs="*",
        default=("Structure 3", "Structure 4"),
        help="Semantic epithelial structure labels. Comma-separated chunks are accepted.",
    )
    detect_glands.add_argument(
        "--markers",
        nargs="*",
        default=("EPCAM", "MUC2", "LGR5", "OLFM4", "MKI67"),
        help="Epithelial marker genes. Comma-separated chunks are accepted.",
    )
    detect_glands.add_argument("--pixel-size-um", type=float, default=0.2125)
    detect_glands.add_argument("--raster-pixel-size-um", type=float, default=5.0)
    detect_glands.add_argument("--lumen-density-threshold", type=float, default=0.05)
    detect_glands.add_argument("--lumen-min-area-um2", type=float, default=200.0)
    detect_glands.add_argument("--min-gland-area-um2", type=float, default=500.0)
    detect_glands.add_argument("--min-ring-support-score", type=float, default=0.3)
    detect_glands.add_argument("--group-property", default="structure")
    detect_glands.add_argument("--cell-density-sigma-um", type=float, default=12.0)
    detect_glands.add_argument("--epithelial-density-threshold", type=float, default=0.10)
    detect_glands.add_argument("--support-closing-um", type=float, default=12.0)
    detect_glands.add_argument("--max-slices", type=int, default=0)
    detect_glands.add_argument("--max-centroid-distance-um", type=float, default=300.0)
    detect_glands.add_argument("--min-overlap-ratio", type=float, default=0.15)
    detect_glands.add_argument("--min-area-ratio", type=float, default=0.25)
    detect_glands.add_argument("--max-area-ratio", type=float, default=4.0)
    detect_glands.add_argument("--lumen-center-weight", type=float, default=0.4)
    detect_glands.add_argument("--marker-profile-weight", type=float, default=0.3)
    detect_glands.add_argument(
        "--allow-many-to-many",
        action="store_true",
        help="Use mutual-best links instead of one-to-one Hungarian assignment.",
    )
    detect_glands.add_argument(
        "--max-gland-pages",
        type=int,
        default=250,
        help="Render only the top N gland detail pages. 0 renders every gland page.",
    )
    detect_glands.add_argument("--z-visual-scale", type=float, default=8.0)

    gland_position = subparsers.add_parser(
        "render-gland-position-atlas",
        help="Render an interactive 3D atlas of tracked gland positions and fission-family links.",
    )
    gland_position.add_argument(
        "--gland-instance-dir",
        required=True,
        help="Directory containing gland_instance_tracks.csv and gland_instance_qc_index.csv.",
    )
    gland_position.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for the position atlas. Defaults to gland_instance_dir/gland_position_atlas.",
    )
    gland_position.add_argument(
        "--biology-dir",
        default=None,
        help="Optional biology mining directory with gland_biology_feature_matrix.csv.",
    )
    gland_position.add_argument(
        "--candidate-score-threshold",
        type=float,
        default=0.55,
        help="Minimum branch/merge candidate score included in the derived family graph.",
    )
    gland_position.add_argument("--z-visual-scale", type=float, default=8.0)
    gland_position.add_argument(
        "--color-by",
        default="gland_family_id",
        choices=[
            "gland_family_id",
            "semantic_structure",
            "qc_flags",
            "slice_count",
            "state_entropy",
            "fission_candidate_score",
            "boundary_wnt_niche_score",
            "lumen_collapse_score",
        ],
        help="Initial color mode for the interactive atlas.",
    )
    gland_position.add_argument(
        "--max-fission-pages",
        type=int,
        default=50,
        help="Render local detail pages for the top N fission/budding candidate edges.",
    )

    gland_surface = subparsers.add_parser(
        "render-gland-surface-atlas",
        help="Render SpaceMap-style solid 3D gland/lumen surface meshes.",
    )
    gland_surface.add_argument(
        "--gland-instance-dir",
        required=True,
        help="Directory containing gland_instance_tracks.csv and slice_gland_instances.geojson.",
    )
    gland_surface.add_argument("--out-dir", required=True, help="Output directory for solid surface artifacts.")
    gland_surface.add_argument(
        "--aligned-cells-parquet",
        default=None,
        help="Optional aligned 3D cell Parquet for local cell overlays.",
    )
    gland_surface.add_argument("--biology-dir", default=None, help="Optional biology/family output directory.")
    gland_surface.add_argument("--gland-ids", nargs="*", default=(), help="Explicit gland IDs to render.")
    gland_surface.add_argument("--family-ids", nargs="*", default=(), help="Family IDs to expand into glands.")
    gland_surface.add_argument("--max-glands", type=int, default=30)
    gland_surface.add_argument("--padding-um", type=float, default=120.0)
    gland_surface.add_argument("--voxel-size-xy-um", type=float, default=6.0)
    gland_surface.add_argument("--voxel-size-z-um", type=float, default=None)
    gland_surface.add_argument("--z-interpolation-factor", type=int, default=3)
    gland_surface.add_argument("--z-visual-scale", type=float, default=8.0)
    gland_surface.add_argument("--surface-smoothing-iterations", type=int, default=10)
    gland_surface.add_argument("--surface-smoothing-lambda", type=float, default=0.35)
    gland_surface.add_argument("--max-faces-per-mesh", type=int, default=50000)
    gland_surface.add_argument("--max-vertices-per-mesh", type=int, default=30000)
    gland_surface.add_argument("--max-cells-per-view", type=int, default=25000)
    gland_surface.add_argument("--cell-color-by", default="leiden_1_0")
    gland_surface.add_argument("--preset", choices=["publication", "qc"], default="publication")
    gland_surface.add_argument("--export-meshes", action="store_true")
    gland_surface.add_argument("--transparent-shell", action="store_true")
    gland_surface.add_argument("--random-state", type=int, default=0)

    gland_biology = subparsers.add_parser(
        "analyze-gland-biology",
        help="Mine gland-level biology from tracked 3D gland instances.",
    )
    gland_biology.add_argument(
        "--gland-instance-dir",
        required=True,
        help="Directory containing gland_instance_tracks.csv and slice_gland_instances.geojson.",
    )
    gland_biology.add_argument("--stack-root", required=True, help="HistoSeg 3D reconstruction output directory.")
    gland_biology.add_argument("--out-dir", default=None, help="Output directory for biology mining tables.")
    gland_biology.add_argument(
        "--aligned-cells-parquet",
        default=None,
        help="Aligned 3D cell Parquet. Defaults to stack_root/aligned_leiden_3d_cells.parquet.",
    )
    gland_biology.add_argument("--outer-ring-inner-um", type=float, default=10.0)
    gland_biology.add_argument("--outer-ring-outer-um", type=float, default=30.0)
    gland_biology.add_argument("--max-slices", type=int, default=0)
    gland_biology.add_argument("--high-confidence-min-slice-count", type=int, default=3)
    gland_biology.add_argument("--high-confidence-max-branch-candidates", type=int, default=0)

    discover = subparsers.add_parser(
        "discover-spatial-modules",
        help="Batch-map genes into 3D enrichment fields, surfaces, SDF metrics, and spatial module matrices.",
    )
    discover.add_argument("--h5ad", required=True, help="Merged AnnData .h5ad with gene expression.")
    discover.add_argument(
        "--aligned-cells-parquet",
        required=True,
        help="Aligned 3D cell table, usually aligned_leiden_3d_cells.parquet.",
    )
    discover.add_argument("--stack-root", required=True, help="HistoSeg 3D reconstruction output directory.")
    discover.add_argument("--out-dir", default=None, help="Batch output directory.")
    discover.add_argument("--genes", nargs="*", default=(), help="Gene names or comma-separated gene chunks.")
    discover.add_argument("--gene-file", default=None, help="Optional newline-delimited gene list.")
    discover.add_argument("--sample-column", default="sample_id")
    discover.add_argument("--barcode-column", default="barcode")
    discover.add_argument("--skip-order-check", action="store_true")
    discover.add_argument("--template-density-summary", default=None)
    discover.add_argument("--xy-voxel-size-um", type=float, default=80.0)
    discover.add_argument("--z-voxel-size-um", type=float, default=5.0)
    discover.add_argument("--smoothing-sigma-xy-um", type=float, default=120.0)
    discover.add_argument("--smoothing-sigma-z-um", type=float, default=10.0)
    discover.add_argument("--surface-smoothing-sigma-voxels", default="1.0,0.9,0.9")
    discover.add_argument("--valid-min-cell-count", type=float, default=8.0)
    discover.add_argument("--min-positive-cells", type=int, default=200)
    discover.add_argument("--min-expression-sum", type=float, default=0.0)
    discover.add_argument("--min-surface-voxels", type=int, default=20)
    discover.add_argument("--mesh-export-formats", default="ply,obj")
    discover.add_argument("--structures", nargs="*", default=None)
    discover.add_argument("--group-property", default="structure")
    discover.add_argument("--pixel-size-um", type=float, default=0.2125)
    discover.add_argument("--force-rebuild-masks", action="store_true")

    quantify = subparsers.add_parser(
        "quantify-gene-structure",
        help="Quantify an existing gene density output against 3D structure masks.",
    )
    quantify.add_argument("--stack-root", required=True)
    quantify.add_argument("--gene-density-dir", required=True)
    quantify.add_argument("--gene", required=True)
    quantify.add_argument("--structures", nargs="*", default=None)
    quantify.add_argument("--group-property", default="structure")
    quantify.add_argument("--pixel-size-um", type=float, default=0.2125)
    quantify.add_argument("--force-rebuild-masks", action="store_true")

    plot = subparsers.add_parser(
        "plot-spatial-modules",
        help="Plot clustered gene-by-structure heatmaps from spatial module matrices.",
    )
    plot.add_argument("--batch-dir", required=True)
    plot.add_argument("--hotspot", default="top05", choices=["top15", "top10", "top05"])
    plot.add_argument(
        "--matrix",
        default="fraction_inside",
        choices=["fraction_inside", "overlap_fraction", "signed_distance"],
    )
    plot.add_argument("--out-png", default=None)
    plot.add_argument("--raw-values", action="store_true", help="Plot raw values instead of row z-scores.")
    plot.add_argument("--cmap", default=None)

    args = parser.parse_args(argv)
    if args.command == "reconstruct":
        rbf_smoothing_reconstruct = (
            args.rbf_smoothing if args.rbf_smoothing == "auto" else float(args.rbf_smoothing)
        )
        result = run_3d_contour_reconstruction(
            ThreeDContourReconstructionConfig(
                fixed_geojson=args.fixed_geojson,
                moving_hard_aligned_geojson=args.moving_hard_aligned_geojson,
                out_dir=args.out_dir,
                group_property=args.group_property,
                sampling_distance_um=args.sampling_distance_um,
                max_landmark_distance_um=args.max_landmark_distance_um,
                landmarks_per_structure=args.landmarks_per_structure or None,
                diagnostic_structure_landmarks=args.diagnostic_structure_landmarks or None,
                landmark_candidate_count=args.landmark_candidate_count,
                landmark_candidate_spacing_um=args.landmark_candidate_spacing_um,
                landmark_normal_weight_um=args.landmark_normal_weight_um,
                landmark_normal_step_um=args.landmark_normal_step_um,
                rbf_kernel=args.rbf_kernel,
                rbf_neighbors=args.rbf_neighbors or None,
                rbf_smoothing=rbf_smoothing_reconstruct,
                topology_grid_size=args.topology_grid_size,
                topology_min_area_ratio=args.topology_min_area_ratio,
                topology_max_area_ratio=args.topology_max_area_ratio,
                diagnostic_structure=args.diagnostic_structure or None,
                dpi=args.dpi,
                save_preview_png=not args.no_preview,
                curvature_landmark_weight=args.curvature_landmark_weight,
                mutual_nn_check=not args.no_mutual_nn_check,
                icp_iterations=args.icp_iterations,
                zero_anchor_count=args.zero_anchor_count,
                landmark_outlier_mad_threshold=(
                    args.landmark_outlier_mad_threshold
                    if args.landmark_outlier_mad_threshold > 0
                    else None
                ),
                per_structure_soft_acceptance=not args.no_per_structure_soft_acceptance,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "align-contours-label-free":
        result = align_contours_label_free(
            LabelFreeContourAlignmentConfig(
                fixed_geojson=args.fixed_geojson,
                moving_geojson=args.moving_geojson,
                out_dir=args.out_dir,
                maxiter=args.maxiter,
                multistart=not args.no_multistart,
                affine_fallback_iou_threshold=args.affine_fallback_iou_threshold,
                run_soft_tps=not args.no_soft_tps,
                sampling_distance_um=args.sampling_distance_um,
                max_landmark_distance_um=args.max_landmark_distance_um,
                landmark_candidate_count=args.landmark_candidate_count,
                landmark_candidate_spacing_um=args.landmark_candidate_spacing_um,
                landmark_normal_weight_um=args.landmark_normal_weight_um,
                landmark_normal_step_um=args.landmark_normal_step_um,
                rbf_kernel=args.rbf_kernel,
                rbf_neighbors=args.rbf_neighbors or None,
                rbf_smoothing=args.rbf_smoothing,
                topology_grid_size=args.topology_grid_size,
                topology_min_area_ratio=args.topology_min_area_ratio,
                topology_max_area_ratio=args.topology_max_area_ratio,
                min_component_area_um2=args.min_component_area_um2,
                max_component_weight=args.max_component_weight,
                boundary_sample_count=args.boundary_sample_count,
                component_sample_count=args.component_sample_count,
                partial_correspondence=args.partial_correspondence,
                diagnostic_only=args.diagnostic_only,
                search_window=args.search_window,
                knn_neighbors=args.knn_neighbors,
                min_anchor_score=args.min_anchor_score,
                min_review_score=args.min_review_score,
                min_anchor_count=args.min_anchor_count,
                overlap_ransac=args.overlap_ransac,
                overlap_candidate_count=args.overlap_candidate_count,
                overlap_ransac_trials=args.overlap_ransac_trials,
                overlap_min_descriptor_score=args.overlap_min_descriptor_score,
                overlap_allow_scale=args.overlap_allow_scale,
                group_correspondence=args.group_correspondence,
                group_candidate_count=args.group_candidate_count,
                group_ransac_trials=args.group_ransac_trials,
                group_min_descriptor_score=args.group_min_descriptor_score,
                group_residual_limit_um=args.group_residual_limit_um,
                group_min_component_area_um2=args.group_min_component_area_um2,
                save_preview_png=not args.no_preview,
                overwrite=args.overwrite,
                dpi=args.dpi,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "close-contour-tear":
        result = run_contour_tear_closure(
            ContourTearClosureConfig(
                fixed_geojson=args.fixed_geojson,
                moving_aligned_geojson=args.moving_aligned_geojson,
                moving_cells_csv=args.moving_cells_csv,
                fixed_cells_csv=args.fixed_cells_csv,
                out_dir=args.out_dir,
                group_property=args.group_property,
                moving_x_column=args.moving_x_column,
                moving_y_column=args.moving_y_column,
                fixed_x_column=args.fixed_x_column,
                fixed_y_column=args.fixed_y_column,
                cluster_column=args.cluster_column,
                output_coordinate_prefix=args.output_coordinate_prefix,
                centroid_search_radius_um=args.centroid_search_radius_um,
                area_ratio_min=args.area_ratio_min,
                area_ratio_max=args.area_ratio_max,
                min_pair_score=args.min_pair_score,
                min_motion_um=args.min_motion_um,
                tear_motion_threshold_um=args.tear_motion_threshold_um,
                influence_radius_um=args.influence_radius_um,
                max_neighbors=args.max_neighbors,
                max_displacement_um=args.max_displacement_um,
                save_preview_png=not args.no_preview,
                overwrite=args.overwrite,
                dpi=args.dpi,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "export-spatialdata-3d":
        result = export_stack_to_spatialdata_3d(
            SpatialData3DExportConfig(
                stack_root=args.stack_root,
                out_zarr=args.out_zarr,
                aligned_cells_parquet=args.aligned_cells_parquet,
                group_property=args.group_property,
                xenium_pixel_size_um=args.xenium_pixel_size_um,
                include_contour_points=not args.no_contour_points,
                include_cells=args.include_cells,
                max_cells=args.max_cells,
                cell_x_column=args.cell_x_column,
                cell_y_column=args.cell_y_column,
                cell_z_column=args.cell_z_column,
                overwrite=args.overwrite,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "project-cells":
        result = run_cell_cloud_projection(
            CellCloudProjectionConfig(
                h5ad=args.h5ad,
                stack_root=args.stack_root,
                out_parquet=args.out_parquet,
                sample_column=args.sample_column,
                barcode_column=args.barcode_column,
                x_column=args.x_column,
                y_column=args.y_column,
                label_columns=tuple(_parse_cli_chunks(args.label_columns)),
                pixel_size_um=args.pixel_size_um,
                chunk_size=args.chunk_size,
                ignore_cache=args.ignore_cache,
                fail_on_stale_cache=args.fail_on_stale_cache,
                write_cache=args.write_cache,
                cache_h5ad=args.cache_h5ad,
                write_scanpy_spatial=args.write_scanpy_spatial,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "infer-local-z-orientation":
        result = run_local_z_orientation_correction(
            LocalZOrientationConfig(
                xenium_root=args.xenium_root,
                stack_root=args.stack_root,
                out_dir=args.out_dir,
                sample_glob=args.sample_glob,
                transcript_relpath=args.transcript_relpath,
                gene_column=args.gene_column,
                x_column=args.x_column,
                y_column=args.y_column,
                z_column=args.z_column,
                transcript_id_column=args.transcript_id_column,
                structure_column=args.structure_column,
                pixel_size_um=args.pixel_size_um,
                z_band_fraction=args.z_band_fraction,
                min_band_transcripts=args.min_band_transcripts,
                max_signature_genes=args.max_signature_genes,
                low_confidence_margin=args.low_confidence_margin,
                orientation_spatial_unit=args.orientation_spatial_unit,
                contour_group_property=args.contour_group_property,
                contour_min_transcripts=args.contour_min_transcripts,
                contour_match_min_iou=args.contour_match_min_iou,
                contour_match_max_distance_um=args.contour_match_max_distance_um,
                orientation_bootstrap_iterations=args.orientation_bootstrap_iterations,
                orientation_bootstrap_seed=args.orientation_bootstrap_seed,
                vertical_qc_backend=args.vertical_qc_backend,
                apply_local_z_flip=args.apply_local_z_flip,
                ovrlpy_kde_bandwidth=args.ovrlpy_kde_bandwidth,
                ovrlpy_n_components=args.ovrlpy_n_components,
                ovrlpy_n_workers=args.ovrlpy_n_workers,
                ovrlpy_fit_umap=args.ovrlpy_fit_umap,
                ovrlpy_min_transcripts=args.ovrlpy_min_transcripts,
                doublet_min_signal=args.doublet_min_signal,
                doublet_integrity_sigma=args.doublet_integrity_sigma,
                doublet_exclusion_radius_um=args.doublet_exclusion_radius_um,
                chunk_size=args.chunk_size,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "render-cell-cloud":
        aligned_cells_parquet = args.aligned_cells_parquet
        if aligned_cells_parquet:
            if args.h5ad or args.out_parquet:
                parser.error("--aligned-cells-parquet cannot be combined with --h5ad or --out-parquet.")
        else:
            if not args.h5ad or not args.out_parquet:
                parser.error("render-cell-cloud requires either --aligned-cells-parquet or both --h5ad and --out-parquet.")
            label_columns = tuple(dict.fromkeys([args.label_column, *_parse_cli_chunks(args.label_columns)]))
            projection = run_cell_cloud_projection(
                CellCloudProjectionConfig(
                    h5ad=args.h5ad,
                    stack_root=args.stack_root,
                    out_parquet=args.out_parquet,
                    sample_column=args.sample_column,
                    barcode_column=args.barcode_column,
                    x_column=args.x_column,
                    y_column=args.y_column,
                    label_columns=label_columns,
                    pixel_size_um=args.pixel_size_um,
                    chunk_size=args.chunk_size,
                    ignore_cache=args.ignore_cache,
                    fail_on_stale_cache=args.fail_on_stale_cache,
                    write_cache=args.write_cache,
                    cache_h5ad=args.cache_h5ad,
                    write_scanpy_spatial=args.write_scanpy_spatial,
                )
            )
            aligned_cells_parquet = projection.out_parquet

        result = render_cell_cloud_html(
            CellCloudRenderConfig(
                aligned_cells_parquet=aligned_cells_parquet,
                stack_root=args.stack_root,
                out_html=args.out_html,
                label_column=args.label_column,
                max_points=args.max_points,
                random_state=args.random_state,
                z_visual_scale=args.z_visual_scale,
                marker_size=args.marker_size,
                opacity=args.opacity,
                include_contours=not args.no_contours,
                title=args.title,
                performance_warning_threshold=args.performance_warning_threshold,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "render-gland-qc-atlas":
        result = render_gland_qc_atlas(
            GlandQCAtlasConfig(
                stack_root=args.stack_root,
                out_dir=args.out_dir,
                aligned_cells_parquet=args.aligned_cells_parquet,
                structures=tuple(_parse_cli_chunks(args.structures)),
                group_property=args.group_property,
                pixel_size_um=args.pixel_size_um,
                padding_um=args.padding_um,
                z_visual_scale=args.z_visual_scale,
                max_cells_per_gland=args.max_cells_per_gland,
                max_gland_pages=args.max_gland_pages or None,
                label_column=args.label_column,
                random_state=args.random_state,
                min_overlap_fraction=args.min_overlap_fraction,
                good_overlap_fraction=args.good_overlap_fraction,
                centroid_fallback_um=args.centroid_fallback_um,
                max_area_ratio_for_fallback=args.max_area_ratio_for_fallback,
                centroid_jump_review_um=args.centroid_jump_review_um,
                area_cv_review_threshold=args.area_cv_review_threshold,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "detect-gland-instances":
        aligned_cells_parquet = args.aligned_cells_parquet
        if aligned_cells_parquet is None:
            default_cells = Path(args.stack_root) / "aligned_leiden_3d_cells.parquet"
            if not default_cells.exists():
                parser.error(
                    "detect-gland-instances requires --aligned-cells-parquet unless "
                    "stack_root/aligned_leiden_3d_cells.parquet exists."
                )
            aligned_cells_parquet = str(default_cells)
        result = run_gland_instance_detection(
            segmentation_config=GlandInstanceSegmentationConfig(
                stack_root=args.stack_root,
                aligned_cells_parquet=aligned_cells_parquet,
                out_dir=args.out_dir,
                structures=tuple(_parse_cli_chunks(args.epithelial_structures)),
                epithelial_markers=tuple(_parse_cli_chunks(args.markers)),
                lumen_min_area_um2=args.lumen_min_area_um2,
                lumen_cell_density_threshold=args.lumen_density_threshold,
                pixel_size_um=args.pixel_size_um,
                raster_pixel_size_um=args.raster_pixel_size_um,
                min_ring_support_score=args.min_ring_support_score,
                min_gland_area_um2=args.min_gland_area_um2,
                group_property=args.group_property,
                cell_density_sigma_um=args.cell_density_sigma_um,
                epithelial_density_threshold=args.epithelial_density_threshold,
                support_closing_um=args.support_closing_um,
                max_slices=args.max_slices or None,
            ),
            tracking_config=GlandInstanceTrackingConfig(
                segmentation_result_dir=args.out_dir,
                out_dir=args.out_dir,
                max_centroid_distance_um=args.max_centroid_distance_um,
                min_overlap_ratio=args.min_overlap_ratio,
                min_area_ratio=args.min_area_ratio,
                max_area_ratio=args.max_area_ratio,
                lumen_center_weight=args.lumen_center_weight,
                marker_profile_weight=args.marker_profile_weight,
                use_one_to_one=not args.allow_many_to_many,
            ),
            max_gland_pages=args.max_gland_pages or None,
            z_visual_scale=args.z_visual_scale,
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "render-gland-position-atlas":
        result = render_gland_position_atlas(
            GlandPositionAtlasConfig(
                gland_instance_dir=args.gland_instance_dir,
                out_dir=args.out_dir,
                biology_dir=args.biology_dir,
                candidate_score_threshold=args.candidate_score_threshold,
                z_visual_scale=args.z_visual_scale,
                color_by=args.color_by,
                max_fission_pages=args.max_fission_pages,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "render-gland-surface-atlas":
        result = render_gland_surface_atlas(
            GlandSurfaceAtlasConfig(
                gland_instance_dir=args.gland_instance_dir,
                out_dir=args.out_dir,
                aligned_cells_parquet=args.aligned_cells_parquet,
                biology_dir=args.biology_dir,
                gland_ids=tuple(_parse_cli_chunks(args.gland_ids)),
                family_ids=tuple(_parse_cli_chunks(args.family_ids)),
                max_glands=args.max_glands,
                padding_um=args.padding_um,
                voxel_size_xy_um=args.voxel_size_xy_um,
                voxel_size_z_um=args.voxel_size_z_um,
                z_interpolation_factor=args.z_interpolation_factor,
                z_visual_scale=args.z_visual_scale,
                surface_smoothing_iterations=args.surface_smoothing_iterations,
                surface_smoothing_lambda=args.surface_smoothing_lambda,
                max_faces_per_mesh=args.max_faces_per_mesh,
                max_vertices_per_mesh=args.max_vertices_per_mesh,
                max_cells_per_view=args.max_cells_per_view,
                cell_color_by=args.cell_color_by,
                preset=args.preset,
                export_meshes=args.export_meshes,
                transparent_shell=args.transparent_shell,
                random_state=args.random_state,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "analyze-gland-biology":
        result = run_gland_biology_mining(
            GlandBiologyMiningConfig(
                gland_instance_dir=args.gland_instance_dir,
                stack_root=args.stack_root,
                out_dir=args.out_dir,
                aligned_cells_parquet=args.aligned_cells_parquet,
                outer_ring_inner_um=args.outer_ring_inner_um,
                outer_ring_outer_um=args.outer_ring_outer_um,
                max_slices=args.max_slices or None,
                high_confidence_min_slice_count=args.high_confidence_min_slice_count,
                high_confidence_max_branch_candidates=args.high_confidence_max_branch_candidates,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "discover-spatial-modules":
        surface_sigma = tuple(
            float(part.strip())
            for part in args.surface_smoothing_sigma_voxels.split(",")
            if part.strip()
        )
        mesh_export_formats = tuple(
            fmt.strip().lower()
            for fmt in args.mesh_export_formats.split(",")
            if fmt.strip()
        )
        structures = tuple(args.structures) if args.structures else (
            "Structure 1",
            "Structure 2",
            "Structure 3",
            "Structure 4",
            "Structure 5",
        )
        result = run_spatial_module_discovery(
            SpatialModuleDiscoveryConfig(
                h5ad=args.h5ad,
                aligned_cells_parquet=args.aligned_cells_parquet,
                stack_root=args.stack_root,
                out_dir=args.out_dir,
                genes=tuple(args.genes),
                gene_file=args.gene_file,
                sample_column=args.sample_column,
                barcode_column=args.barcode_column,
                skip_order_check=args.skip_order_check,
                template_density_summary=args.template_density_summary,
                xy_voxel_size_um=args.xy_voxel_size_um,
                z_voxel_size_um=args.z_voxel_size_um,
                smoothing_sigma_xy_um=args.smoothing_sigma_xy_um,
                smoothing_sigma_z_um=args.smoothing_sigma_z_um,
                surface_smoothing_sigma_voxels_zyx=surface_sigma,
                valid_min_cell_count=args.valid_min_cell_count,
                min_positive_cells=args.min_positive_cells,
                min_expression_sum=args.min_expression_sum,
                min_surface_voxels=args.min_surface_voxels,
                mesh_export_formats=mesh_export_formats,
                structures=structures,
                group_property=args.group_property,
                pixel_size_um=args.pixel_size_um,
                force_rebuild_masks=args.force_rebuild_masks,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "quantify-gene-structure":
        structures = tuple(args.structures) if args.structures else (
            "Structure 1",
            "Structure 2",
            "Structure 3",
            "Structure 4",
            "Structure 5",
        )
        result = quantify_gene_structure_relationships(
            GeneStructureQuantificationConfig(
                stack_root=args.stack_root,
                gene_density_dir=args.gene_density_dir,
                gene=args.gene,
                structures=structures,
                group_property=args.group_property,
                pixel_size_um=args.pixel_size_um,
                force_rebuild_masks=args.force_rebuild_masks,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "plot-spatial-modules":
        result = plot_spatial_module_clustermap(
            SpatialModulePlotConfig(
                batch_dir=args.batch_dir,
                hotspot=args.hotspot,
                matrix=args.matrix,
                out_png=args.out_png,
                row_zscore=not args.raw_values,
                cmap=args.cmap,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    elif args.command == "reconstruct-stack":
        mesh_smoothing_sigma_um = args.mesh_smoothing_sigma_um
        if mesh_smoothing_sigma_um < 0:
            parser.error("--mesh-smoothing-sigma-um must be >= 0. Use 0 to disable smoothing.")
        if mesh_smoothing_sigma_um == 0:
            mesh_smoothing_sigma_um = None
        mesh_export_formats = tuple(
            fmt.strip().lower()
            for fmt in args.mesh_export_formats.split(",")
            if fmt.strip()
        )
        rbf_smoothing_stack = (
            args.rbf_smoothing if args.rbf_smoothing == "auto" else float(args.rbf_smoothing)
        )
        result = run_3d_stack_reconstruction(
            ThreeDStackReconstructionConfig(
                xenium_root=args.xenium_root,
                out_dir=args.out_dir,
                contour_manifest=args.contour_manifest,
                segmentation_strategy=args.segmentation_strategy,
                sample_glob=args.sample_glob,
                z_spacing_um=args.z_spacing_um,
                xenium_pixel_size_um=args.xenium_pixel_size_um,
                group_property=args.group_property,
                cluster_column_name=args.cluster_column_name,
                clusters_relpath=args.clusters_relpath,
                merged_h5ad=args.merged_h5ad,
                merged_cluster_column=args.merged_cluster_column,
                merged_sample_column=args.merged_sample_column,
                merged_barcode_column=args.merged_barcode_column,
                bins_x=args.bins_x,
                bins_y=args.bins_y,
                gaussian_sigma=args.gaussian_sigma,
                min_cells=args.min_cells,
                min_component_pixels=args.min_component_pixels,
                save_slice_preview_png=args.save_slice_preview,
                registration_backend=args.registration_backend,
                hard_alignment_maxiter=args.hard_alignment_maxiter,
                hard_alignment_multistart=not args.no_multistart,
                affine_fallback_iou_threshold=args.affine_fallback_iou_threshold,
                coda_raster_size=args.coda_raster_size,
                coda_angle_step=args.coda_angle_step,
                coda_phase_upsample_factor=args.coda_phase_upsample_factor,
                coda_mask_padding_fraction=args.coda_mask_padding_fraction,
                label_free_search_window=args.label_free_search_window,
                label_free_knn_neighbors=args.label_free_knn_neighbors,
                label_free_min_anchor_count=args.label_free_min_anchor_count,
                label_free_group_candidate_count=args.label_free_group_candidate_count,
                label_free_group_ransac_trials=args.label_free_group_ransac_trials,
                label_free_group_min_descriptor_score=args.label_free_group_min_descriptor_score,
                label_free_group_residual_limit_um=args.label_free_group_residual_limit_um,
                label_free_group_min_component_area_um2=args.label_free_group_min_component_area_um2,
                label_free_guarded_alignment=not args.no_label_free_guarded_alignment,
                label_free_min_similarity_anchor_count=(
                    args.label_free_min_similarity_anchor_count
                ),
                label_free_max_rotation_degrees=args.label_free_max_rotation_degrees,
                label_free_high_rotation_min_anchor_count=(
                    args.label_free_high_rotation_min_anchor_count
                ),
                label_free_high_rotation_residual_limit_um=(
                    args.label_free_high_rotation_residual_limit_um
                ),
                label_free_min_anchor_coverage=args.label_free_min_anchor_coverage,
                label_free_min_anchor_quadrants=args.label_free_min_anchor_quadrants,
                label_free_near_180_rotation_degrees=(
                    args.label_free_near_180_rotation_degrees
                ),
                global_drift_correction=args.global_drift_correction,
                run_soft_alignment=not args.no_soft,
                soft_alignment_mode=(
                    "none" if args.no_soft else args.soft_alignment_mode
                ),
                anchor_only_bbox_padding_fraction=args.anchor_only_bbox_padding_fraction,
                anchor_only_identity_padding_count=args.anchor_only_identity_padding_count,
                anchor_only_rbf_smoothing=args.anchor_only_rbf_smoothing,
                anchor_only_jacobian_grid_size=args.anchor_only_jacobian_grid_size,
                anchor_only_max_negative_jacobian_ratio=(
                    args.anchor_only_max_negative_jacobian_ratio
                ),
                iterative_contour_refinement=(
                    not args.no_iterative_contour_refinement
                ),
                iterative_contour_refinement_rounds=(
                    args.iterative_contour_refinement_rounds
                ),
                iterative_contour_min_pair_count=args.iterative_contour_min_pair_count,
                iterative_contour_min_anchor_count=args.iterative_contour_min_anchor_count,
                iterative_contour_min_pair_score=args.iterative_contour_min_pair_score,
                iterative_contour_min_overlap=args.iterative_contour_min_overlap,
                iterative_contour_centroid_search_radius_um=(
                    args.iterative_contour_centroid_search_radius_um
                ),
                iterative_contour_accepted_centroid_radius_um=(
                    args.iterative_contour_accepted_centroid_radius_um
                ),
                iterative_contour_max_anchor_distance_um=(
                    args.iterative_contour_max_anchor_distance_um
                ),
                iterative_contour_residual_limit_um=args.iterative_contour_residual_limit_um,
                iterative_contour_max_negative_jacobian_ratio=(
                    args.iterative_contour_max_negative_jacobian_ratio
                ),
                sampling_distance_um=args.sampling_distance_um,
                max_landmark_distance_um=args.max_landmark_distance_um,
                landmarks_per_structure=args.landmarks_per_structure or None,
                diagnostic_structure_landmarks=args.diagnostic_structure_landmarks or None,
                landmark_candidate_count=args.landmark_candidate_count,
                landmark_candidate_spacing_um=args.landmark_candidate_spacing_um,
                landmark_normal_weight_um=args.landmark_normal_weight_um,
                landmark_normal_step_um=args.landmark_normal_step_um,
                rbf_neighbors=args.rbf_neighbors or None,
                rbf_smoothing=rbf_smoothing_stack,
                topology_grid_size=args.topology_grid_size,
                topology_min_area_ratio=args.topology_min_area_ratio,
                topology_max_area_ratio=args.topology_max_area_ratio,
                diagnostic_structure=args.diagnostic_structure or None,
                save_alignment_preview_png=not args.no_alignment_preview,
                curvature_landmark_weight=args.curvature_landmark_weight,
                mutual_nn_check=not args.no_mutual_nn_check,
                icp_iterations=args.icp_iterations,
                zero_anchor_count=args.zero_anchor_count,
                landmark_outlier_mad_threshold=(
                    args.landmark_outlier_mad_threshold
                    if args.landmark_outlier_mad_threshold > 0
                    else None
                ),
                per_structure_soft_acceptance=not args.no_per_structure_soft_acceptance,
                point_sample_distance_um=args.point_sample_distance_um,
                voxel_size_um=args.voxel_size_um,
                mesh_method=args.mesh_method,
                mesh_smoothing_sigma_um=mesh_smoothing_sigma_um,
                mesh_level=args.mesh_level,
                mesh_export_formats=mesh_export_formats,
                mesh_cleanup=not args.no_mesh_cleanup,
                min_mesh_component_volume_um3=args.min_mesh_component_volume_um3,
                mesh_max_faces_for_html=args.mesh_max_faces_for_html,
                local_z_orientation=args.local_z_orientation,
                vertical_qc_backend=args.vertical_qc_backend,
                apply_local_z_flip=args.apply_local_z_flip,
                transcript_relpath=args.transcript_relpath,
                orientation_spatial_unit=args.orientation_spatial_unit,
                contour_group_property=args.contour_group_property,
                contour_min_transcripts=args.contour_min_transcripts,
                contour_match_min_iou=args.contour_match_min_iou,
                contour_match_max_distance_um=args.contour_match_max_distance_um,
                orientation_bootstrap_iterations=args.orientation_bootstrap_iterations,
                orientation_bootstrap_seed=args.orientation_bootstrap_seed,
                ovrlpy_n_components=args.ovrlpy_n_components,
                ovrlpy_n_workers=args.ovrlpy_n_workers,
                ovrlpy_fit_umap=args.ovrlpy_fit_umap,
                ovrlpy_min_transcripts=args.ovrlpy_min_transcripts,
                overwrite=args.overwrite,
                dpi=args.dpi,
            )
        )
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))


def _parse_cli_chunks(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in str(value).split(",") if part.strip())
    return result


if __name__ == "__main__":
    main()
