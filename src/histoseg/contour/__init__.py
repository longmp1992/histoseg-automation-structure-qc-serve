"""Contour Analysis public API."""

from histoseg._cellgps import require_cellgps_attrs

_cellgps = require_cellgps_attrs(
    (
        "compute_cophenetic_distances_from_df",
        "compute_cophenetic_from_distance_matrix",
        "compute_searcher_findee_distance_matrix_from_df",
        "plot_cophenetic_heatmap",
    )
)
compute_cophenetic_distances_from_df = _cellgps.compute_cophenetic_distances_from_df
compute_cophenetic_from_distance_matrix = _cellgps.compute_cophenetic_from_distance_matrix
compute_searcher_findee_distance_matrix_from_df = (
    _cellgps.compute_searcher_findee_distance_matrix_from_df
)
plot_cophenetic_heatmap = _cellgps.plot_cophenetic_heatmap

from .auto_structure import (
    AutoStructureDiscoveryConfig,
    AutoStructureDiscoveryResult,
    discover_auto_structures,
    resolve_xenium_output_folder,
)
from .boundary_network import (
    BoundaryNetworkConfig,
    BoundaryNetworkResult,
    build_group_boundary_graph,
    draw_group_boundary_network,
    normalize_group_boundary_overlap,
    run_group_boundary_network,
)
from .contour_adjacency import (
    ContourAdjacencyConfig,
    ContourAdjacencyResult,
    build_contour_adjacency_edges,
    build_contour_adjacency_graph,
    build_contour_adjacency_matrix,
    draw_contour_adjacency_heatmap,
    draw_contour_adjacency_network,
    load_contours_csv,
    run_contour_adjacency,
)
from .gene_isoline import (
    GeneTranscriptIsolineConfig,
    GeneTranscriptIsolineResult,
    run_gene_transcript_isoline,
)
from .multi_structure import (
    MultiStructureContourConfig,
    MultiStructureContourResult,
    MultiStructureSpec,
    run_multi_structure_contours,
)
from .pattern1_isoline import (
    Pattern1IsolineConfig,
    Pattern1IsolineResult,
    SegmentationConfidenceResult,
    compute_segmentation_confidence_score,
    compute_segmentation_confidence_score_from_merged,
    run_pattern1_isoline,
    run_pattern1_isoline_from_hf,
)
from .topology import (
    ContourTopologyConfig,
    ContourTopologyResult,
    summarize_contour_topology,
)

__all__ = [
    "BoundaryNetworkConfig",
    "BoundaryNetworkResult",
    "AutoStructureDiscoveryConfig",
    "AutoStructureDiscoveryResult",
    "ContourAdjacencyConfig",
    "ContourAdjacencyResult",
    "ContourTopologyConfig",
    "ContourTopologyResult",
    "GeneTranscriptIsolineConfig",
    "GeneTranscriptIsolineResult",
    "MultiStructureContourConfig",
    "MultiStructureContourResult",
    "MultiStructureSpec",
    "Pattern1IsolineConfig",
    "Pattern1IsolineResult",
    "SegmentationConfidenceResult",
    "build_contour_adjacency_edges",
    "build_contour_adjacency_graph",
    "build_contour_adjacency_matrix",
    "build_group_boundary_graph",
    "discover_auto_structures",
    "compute_cophenetic_distances_from_df",
    "compute_cophenetic_from_distance_matrix",
    "compute_searcher_findee_distance_matrix_from_df",
    "compute_segmentation_confidence_score",
    "compute_segmentation_confidence_score_from_merged",
    "draw_contour_adjacency_heatmap",
    "draw_contour_adjacency_network",
    "draw_group_boundary_network",
    "load_contours_csv",
    "normalize_group_boundary_overlap",
    "plot_cophenetic_heatmap",
    "resolve_xenium_output_folder",
    "run_contour_adjacency",
    "run_gene_transcript_isoline",
    "run_group_boundary_network",
    "run_multi_structure_contours",
    "run_pattern1_isoline",
    "run_pattern1_isoline_from_hf",
    "summarize_contour_topology",
]
