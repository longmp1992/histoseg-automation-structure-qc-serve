<div align="center">

# HistoSeg Structure QC Serve

This repository deploys an end-to-end SciLifeLab Serve app. It starts from the
two Xenium-derived input tables rather than from a precomputed HistoSeg ZIP:

1. upload `cells.parquet` and `clusters.csv`;
2. build the Cell-GPS StructureMap and candidate dendrogram branches;
3. select one or more candidate structures;
4. generate non-overlapping HistoSeg structure contours;
5. automatically run fixed-window CSR and random-labelling QC;
6. download PASS/PARTIAL/FAIL calls, assigned-cluster HOTSPOT calls, and tree-consistent
   `cluster_ids` split suggestions with all ordinary HistoSeg outputs.

The complete CSR method and thresholds are documented in
[`STRUCTURE_QC_ALGORITHM.md`](STRUCTURE_QC_ALGORITHM.md). The container image is
published as `ghcr.io/longmp1992/histoseg-automation-structure-qc-serve:latest`.

## Local run

```bash
docker build -t histoseg-automation-structure-qc-serve .
docker run --rm -p 7860:7860 histoseg-automation-structure-qc-serve
```

The app is upload-only and writes run artifacts under `APP_DATA_DIR`.

---

# Upstream HistoSeg

<p align="center">
  <a href="https://pypi.org/project/histoseg/"><img alt="PyPI" src="https://img.shields.io/pypi/v/histoseg.svg"></a>
  <a href="https://anaconda.org/conda-forge/histoseg"><img alt="Conda" src="https://img.shields.io/conda/vn/conda-forge/histoseg.svg"></a>
  <a href="https://histoseg.readthedocs.io/en/latest/"><img alt="Docs" src="https://readthedocs.org/projects/histoseg/badge/?version=latest"></a>
  <a href="https://github.com/hutaobo/HistoSeg/actions/workflows/publish.yml"><img alt="Publish to PyPI" src="https://github.com/hutaobo/HistoSeg/actions/workflows/publish.yml/badge.svg?event=release"></a>
  <a href="https://github.com/hutaobo/HistoSeg/actions/workflows/docker-image-ghcr.yml"><img alt="Serve Docker image" src="https://github.com/hutaobo/HistoSeg/actions/workflows/docker-image-ghcr.yml/badge.svg?branch=main"></a>
  <a href="https://polyformproject.org/licenses/noncommercial/1.0.0/"><img alt="License: PolyForm Noncommercial 1.0.0" src="https://img.shields.io/badge/License-PolyForm--Noncommercial%201.0.0-blue.svg"></a>
</p>

**HistoSeg: StructureMap-guided semantic contours and robust 3D tissue reconstruction via Signed Distance Fields (SDFs).**

</div>

HistoSeg is centered on same-sample, multi-slice **3D Xenium contour
reconstruction**. It converts selected or curated tissue structure groups,
including groups defined or audited with Cell-GPS Search-and-Find/StructureMap
relationships, into continuous semantic contours. Those named 2D contours then
become aligned contour stacks, sampled 3D points, smoothed PLY/OBJ meshes, QC
metrics, interactive HTML views, and SDF-based gene-structure measurements. The
package also includes the H&E and 2D contour workflows needed to prepare and
review the structures that feed the 3D reconstruction pipeline.

HistoSeg is organized around its 3D reconstruction surface, with two supporting
analysis groups:

- **3D Reconstruction** (`histoseg.threed`) for same-sample, multi-slice Xenium contour alignment, 3D contour stacks, mesh export, and QC visualization.
- **2D Contour Analysis** (`histoseg.contour`) for StructureMap-guided semantic contour extraction from spatial/cell-coordinate data, including Pattern1 isolines and multi-structure Xenium exports.
- **H&E Analysis** (`histoseg.he`) for image-based H&E tissue segmentation, neutral tissue partitioning, and aligned-image change detection.
- **Spatial Pathologist Serve app** (`histoseg.spatial_pathologist.serve_app`) for the browser-based SciLifeLab Serve workflow around dendrogram-guided multi-structure contour selection.

Full documentation: [histoseg.readthedocs.io](https://histoseg.readthedocs.io)

## When To Use Each Feature Group

Use **3D Reconstruction** when you are preparing for multi-slice Xenium contour reconstruction from the same sample. It can soft-align a hard-aligned moving contour GeoJSON to a fixed reference slice, or build a pyXenium-backed multi-slice contour stack with 3D points, smoothed PLY/OBJ surface meshes, and an interactive HTML view.

Use **Contour Analysis** when your input is spatial cell-coordinate data such as Xenium `cells.parquet` plus cluster assignments, and you want geometry extracted from cell neighborhoods or selected cluster groups. The selected groups can be curated directly or informed by Cell-GPS Search-and-Find / cophenetic StructureMap relationships.

Use **HE Analysis** when your input is an H&E image such as PNG, JPG, TIFF, or GeoTIFF and you want masks, overlays, heatmaps, GeoJSON polygons, or region tables.

## Installation

```bash
pip install -U histoseg
```

Or install from conda-forge:

```bash
conda install -c conda-forge histoseg
```

For flagship 3D Xenium stack reconstruction, install the pyXenium-backed 3D extra:

```bash
pip install -U "histoseg[threed]"
```

For reproducible static 3D figure rendering and local documentation builds:

```bash
pip install -U "histoseg[threed,viz,docs]"
```

For the browser-based Spatial Pathologist Serve app:

```bash
pip install -U "histoseg[serve]"
histoseg-ai-driven-spatial-pathologist-serve
```

For development:

```bash
git clone https://github.com/hutaobo/HistoSeg.git
cd HistoSeg
pip install -U pip
pip install -e ".[threed,he]"
```

Conda environments and Docker targets are provided for reproducible runs:

```bash
conda env create -f environment.yml
conda env create -f environment-viz.yml

docker build --target core -t histoseg:core .
docker build --target viz -t histoseg:viz .
docker build --target serve -t histoseg-ai-driven-spatial-pathologist-serve .
```

The `core` Docker target is for CPU/headless 3D analysis. The `viz` target adds
Mesa/Xvfb and PyVista for static documentation or paper figure rendering. The
`serve` target runs the Gradio web app used by the SciLifeLab Serve deployment.

For local Hugging Face MedSAM-backed HE segmentation only:

```bash
pip install -U "histoseg[he]"
```

## Flagship 3D Reconstruction Quickstart

```python
from histoseg.threed import (
    ThreeDContourReconstructionConfig,
    run_3d_contour_reconstruction,
)

cfg = ThreeDContourReconstructionConfig(
    fixed_geojson="slice_01.geojson",
    moving_hard_aligned_geojson="slice_02_hard_aligned_to_01.geojson",
    out_dir="outputs/3d_soft_alignment",
    group_property="structure",
    diagnostic_structure="Structure 5",
)

result = run_3d_contour_reconstruction(cfg)
print(result.soft_aligned_geojson)
print(result.diagnostic_report_png)
```

```bash
histoseg-3d reconstruct \
  --fixed-geojson slice_01.geojson \
  --moving-hard-aligned-geojson slice_02_hard_aligned_to_01.geojson \
  --out-dir outputs/3d_soft_alignment \
  --group-property structure \
  --diagnostic-structure "Structure 5"
```

For a complete same-sample Xenium stack:

```bash
histoseg-3d reconstruct-stack \
  --xenium-root polyp \
  --segmentation-strategy "polyp/contour for alignment/segmentationstrategy.txt" \
  --merged-h5ad "polyp/pdc_merge_leiden/polyp_32samples_processed_leiden.h5ad" \
  --merged-cluster-column leiden_1_0 \
  --out-dir outputs/polyp_3d_reconstruction \
  --z-spacing-um 5 \
  --mesh-smoothing-sigma-um 40 \
  --mesh-export-formats ply,obj
```

3D Reconstruction writes aligned per-slice GeoJSON files, pairwise alignment
metrics, sampled 3D contour points, per-structure PLY/OBJ meshes, mesh QC
metrics, and an interactive Plotly HTML visualization. Use
`--mesh-smoothing-sigma-um 0` to disable 3D smoothing for direct Marching Cubes
output. The stack CLI defaults to `--registration-backend auto`, which compares
the standard semantic contour seed with a label-free cross-group seed and keeps
labels unchanged when cross-named contour groups provide the best geometric
anchor.

When each slice has already been converted into slice-local semantic contours,
use a contour manifest instead of a global segmentation strategy. This mode is
intended for independently clustered slices whose structure names are not
globally comparable:

```bash
histoseg-3d reconstruct-stack \
  --contour-manifest slice_local_contour_manifest.csv \
  --out-dir outputs/polyp_slice_local_3d \
  --registration-backend auto \
  --z-spacing-um 5
```

The manifest must contain `order`, `sample_id`, `z_um`, and `geojson`. HistoSeg
copies each GeoJSON into the stack output, preserves its labels, and aligns
adjacent slices with the same label-free group machinery used by the automatic
backend. A merged Leiden/AnnData object can still be used later for cell-cloud
coloring or downstream biology, but it is not required to define contours in
this mode.

Render an aligned 3D cell cloud as a browser-shareable Plotly HTML:

```bash
histoseg-3d render-cell-cloud \
  --stack-root outputs/polyp_3d_reconstruction \
  --aligned-cells-parquet outputs/polyp_3d_reconstruction/aligned_leiden_3d_cells.parquet \
  --out-html outputs/polyp_3d_reconstruction/leiden_3d_cells.html \
  --label-column leiden_1_0 \
  --max-points 300000
```

If the aligned cell table does not exist yet, `render-cell-cloud` can project a
merged AnnData first by replacing `--aligned-cells-parquet` with `--h5ad` and
`--out-parquet`.

For local inspection of many small gland-like contour components, render a
per-gland QC atlas from an existing aligned stack:

```bash
histoseg-3d render-gland-qc-atlas \
  --stack-root outputs/polyp_3d_reconstruction \
  --aligned-cells-parquet outputs/polyp_3d_reconstruction/aligned_leiden_3d_cells.parquet \
  --out-dir outputs/polyp_3d_reconstruction/gland_qc \
  --structures "Structure 3" "Structure 4" \
  --max-gland-pages 250
```

The atlas assigns cross-slice `gland_id` values from aligned contour components
and writes `gland_qc_atlas.html`, per-gland local 3D zoom pages, and CSV QC
tables for slice continuity review. The CSV index is always full; use
`--max-gland-pages` to render only the highest-priority local HTML pages first.

For lumen-seeded gland/crypt instance segmentation and cross-slice tracking:

```bash
histoseg-3d detect-gland-instances \
  --stack-root outputs/polyp_3d_reconstruction \
  --aligned-cells-parquet outputs/polyp_3d_reconstruction/aligned_leiden_3d_cells.parquet \
  --out-dir outputs/polyp_3d_reconstruction/gland_instances \
  --epithelial-structures "Structure 3" "Structure 4" \
  --markers EPCAM MUC2 LGR5 OLFM4 MKI67
```

The tracker uses one-to-one Hungarian assignment by default. Add
`--allow-many-to-many` only for exploratory branch/merge review; candidate
two-to-one links are reported in CSV/HTML QC outputs.

## HE Analysis Quickstart

```python
from histoseg.he import HESegmentationConfig, run_he_segmentation

result = run_he_segmentation(
    HESegmentationConfig(
        image="/path/to/he.png",
        out_dir="outputs/he_all_elements",
        task="all_elements",
        backend="heuristic",
        n_components=6,
    )
)

print(result.overlay_png)
print(result.geojson)
```

```bash
histoseg-he all-elements \
  --image /path/to/he.png \
  --out-dir outputs/he_all_elements \
  --backend heuristic
```

HE Analysis currently supports:

- `single`: tissue foreground extraction, or user-prompted region extraction from boxes/points
- `all_elements`: neutral tissue component partitioning (`component_1`, `component_2`, ...)
- `change`: aligned before/after H&E change detection

## Contour Analysis Quickstart

```python
from histoseg.contour import Pattern1IsolineConfig, run_pattern1_isoline

cfg = Pattern1IsolineConfig(
    clusters_csv="/path/to/clusters.csv",
    cells_parquet="/path/to/cells.parquet",
    tissue_boundary_csv="/path/to/tissue_boundary.csv",
    out_dir="outputs/pattern1_isoline0p5",
    pattern1_clusters=(10, 23, 19, 27, 14, 20, 25, 26),
)

result = run_pattern1_isoline(cfg)
print(result.preview_png)
print(len(result.contours))
```

For Xenium transcript-defined niches such as GREM1-positive regions:

```python
from histoseg.contour import GeneTranscriptIsolineConfig, run_gene_transcript_isoline

result = run_gene_transcript_isoline(
    GeneTranscriptIsolineConfig(
        xenium_root="/path/to/polyp",
        out_dir="outputs/gene_isolines",
        genes=("GREM1",),
        sample_glob="A079-C-008_*",
    )
)
print(result.run_log_csv)
```

```bash
histoseg-contour pattern1 \
  --clusters-csv clusters.csv \
  --cells-parquet cells.parquet \
  --out-dir outputs/pattern1 \
  --pattern1-clusters 10,23,19

histoseg-contour gene-isoline \
  --xenium-root polyp \
  --sample-glob "A079-C-008_*" \
  --genes GREM1,COL1A1 \
  --out-dir outputs/gene_isolines
```

Contour Analysis currently supports:

- StructureMap-guided semantic contour synthesis from selected or curated structure groups
- Pattern1 isoline contour generation from clustered cell coordinates
- gene/transcript isoline contour generation from Xenium transcript tables
- multi-structure contour partitioning
- Xenium Explorer annotation exports
- Hugging Face dataset helper workflows for Xenium-style inputs

## Outputs

HistoSeg workflows write reviewable artifacts such as:

- aligned 3D contour stacks and sampled 3D point clouds
- PLY/OBJ surface meshes and mesh QC summaries
- interactive Plotly 3D HTML views
- PNG previews and overlays
- label maps and heatmaps
- GeoJSON polygons
- CSV/Parquet region or contour tables
- `params.json` and `metrics.json` provenance files

## Documentation

- [3D Reconstruction](https://histoseg.readthedocs.io/en/latest/3d_analysis.html)
- [Polyp 24-gene 3D spatial modules tutorial](https://histoseg.readthedocs.io/en/latest/tutorials/polyp_24_gene_3d_spatial_modules.html)
- [3D contour stack reconstruction tutorial](https://histoseg.readthedocs.io/en/latest/tutorials/3d_stack_reconstruction.html)
- [3D contour soft alignment tutorial](https://histoseg.readthedocs.io/en/latest/tutorials/3d_soft_alignment.html)
- [Contour Analysis](https://histoseg.readthedocs.io/en/latest/contour_analysis.html)
- [HE Analysis](https://histoseg.readthedocs.io/en/latest/he_analysis.html)
- [API Reference](https://histoseg.readthedocs.io/en/latest/api/index.html)

## Scientific Foundation & Reproducibility

HistoSeg's methods are documented as an implementation-faithful manuscript
draft in [Online Methods: Semantic Contours, SDF Quantification And Topology-Aware Alignment](docs/manuscripts/histoseg_online_methods_sdf_alignment.md).
The draft describes the full pipeline from Cell-GPS Search-and-Find/StructureMap
relationships to HistoSeg semantic isoline contours, topology-aware stack
alignment, and the exact anisotropic SDF contract used by the package:
`scipy.ndimage.distance_transform_edt(..., sampling=(z_um, y_um, x_um))`,
negative distances inside structure masks, and positive distances outside.

The manuscript figure roadmap is maintained in
[docs/manuscripts/figure_plan.md](docs/manuscripts/figure_plan.md). It defines
the proof target for workflow infrastructure, SDF robustness, and the 32-slice
polyp biological discovery case. Users can recreate the visualization-oriented
outputs locally with:

```bash
pip install -U "histoseg[threed,viz]"
```

or with the reproducible viz environment:

```bash
conda env create -f environment-viz.yml
docker build --target viz -t histoseg:viz .
```

See [reproducibility/README.md](reproducibility/README.md) for the paper
environment, tutorial artifact map, and alignment-hash provenance checks.
The local wrapper `python reproducibility/run_paper_pipeline.py` regenerates
the paper-facing cell-cloud HTML and spatial-module clustermaps from the
validated 32-slice polyp paths and writes `reproducibility/results_manifest.json`.

Citation placeholder for the HistoSeg methods paper:

```bibtex
@article{histoseg_method_paper,
  title   = {HistoSeg: robust 3D reconstruction of tissue architecture from multi-slice spatial transcriptomics},
  author  = {Hu, Taobo and HistoSeg contributors},
  journal = {Manuscript in preparation},
  year    = {2026},
  note    = {Zenodo DOI pending}
}
```

## License

This project is distributed under the **PolyForm Noncommercial 1.0.0** license.
Academic and other noncommercial use is permitted.
Any commercial use requires a separate commercial license from **SPATHO AB**.
See [LICENSE](LICENSE) for details.
