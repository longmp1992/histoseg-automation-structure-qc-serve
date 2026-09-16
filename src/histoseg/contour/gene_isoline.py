"""Gene/transcript isoline contours for Xenium transcript tables.

This workflow adapts Xenium ``transcripts.parquet`` points into the existing
Pattern1 isoline engine by treating selected gene transcripts as the target
class and cell centroids as spatial background.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import tempfile
from typing import Mapping, Optional, Sequence, Union
from uuid import uuid4

import numpy as np
import pandas as pd

from histoseg.geometry import generate_tissue_boundary

from .pattern1_isoline import Pattern1IsolineConfig, Pattern1IsolineResult, run_pattern1_isoline


PathLike = Union[str, Path]


GENE_ISOLINE_PALETTE = [
    "#6EF0D4",
    "#78B9FF",
    "#FFB870",
    "#C8A2FF",
    "#FF8DA1",
    "#90F184",
    "#FFD76C",
    "#80E1FF",
    "#F4A6FF",
    "#FFA07A",
]


@dataclass(frozen=True)
class GeneTranscriptIsolineConfig:
    xenium_root: PathLike
    out_dir: PathLike
    genes: Sequence[str]
    sample_glob: str = "*"
    xenium_output_glob: str = "output-*"
    qv_min: float = 20
    min_transcripts: int = 10
    grid_n: int = 1200
    knn_k: int = 30
    smooth_sigma: float = 5.0
    min_cells_inside: int = 10
    use_synth_bg: bool = True
    alpha: float = 0.05
    xenium_pixel_size_um: float = 0.2125
    compute_confidence_score: bool = False
    keep_prepared_inputs: bool = False
    fail_fast: bool = False

    transcript_id_col: str = "transcript_id"
    transcript_feature_col: str = "feature_name"
    transcript_x_col: str = "x_location"
    transcript_y_col: str = "y_location"
    transcript_qv_col: str = "qv"
    cell_id_col: str = "cell_id"
    cell_x_col: str = "x_centroid"
    cell_y_col: str = "y_centroid"


@dataclass
class GeneTranscriptIsolineResult:
    out_dir: Path
    run_log_csv: Path
    records: list[Mapping[str, object]]


@dataclass(frozen=True)
class _GeneSampleInput:
    order: int
    sample_id: str
    sample_dir: Path
    xenium_dir: Path


def run_gene_transcript_isoline(
    cfg: GeneTranscriptIsolineConfig,
) -> GeneTranscriptIsolineResult:
    """Run gene/transcript isolines for all discovered Xenium samples."""

    genes = _normalize_gene_list(cfg.genes)
    if not genes:
        raise ValueError("genes must contain at least one non-empty gene name.")
    if cfg.min_transcripts < 1:
        raise ValueError("min_transcripts must be greater than 0.")
    if cfg.xenium_pixel_size_um <= 0:
        raise ValueError("xenium_pixel_size_um must be greater than 0.")

    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = _discover_xenium_transcript_samples(
        cfg.xenium_root,
        sample_glob=cfg.sample_glob,
        xenium_output_glob=cfg.xenium_output_glob,
    )
    if not samples:
        raise ValueError(f"No Xenium transcript samples were found under {cfg.xenium_root!s}.")

    gene_path_names = _gene_path_names(genes)
    gene_ids = {gene: index for index, gene in enumerate(genes, start=1)}
    sample_features: dict[str, list[dict[str, object]]] = {item.sample_id: [] for item in samples}
    sample_csv_rows: dict[str, list[dict[str, object]]] = {item.sample_id: [] for item in samples}
    sample_summary_rows: dict[str, list[dict[str, object]]] = {item.sample_id: [] for item in samples}
    records: list[dict[str, object]] = []

    for sample in samples:
        try:
            transcripts, cells = _load_xenium_tables(sample.xenium_dir, cfg)
        except Exception as exc:
            if cfg.fail_fast:
                raise
            for gene in genes:
                records.append(
                    _record(
                        sample=sample,
                        gene=gene,
                        status=f"error_load_tables: {exc}",
                    )
                )
            continue

        for gene in genes:
            gene_dir = out_dir / sample.sample_id / gene_path_names[gene]
            pattern_out_dir = gene_dir / "pattern1_isoline"
            try:
                gene_tx = _filter_gene_transcripts(transcripts, gene, cfg)
                n_transcripts = int(len(gene_tx))
                n_cells = int(len(cells))
                if n_transcripts < cfg.min_transcripts:
                    records.append(
                        _record(
                            sample=sample,
                            gene=gene,
                            status="skip_too_few_transcripts",
                            n_transcripts=n_transcripts,
                            n_cells=n_cells,
                            pattern1_out_dir=pattern_out_dir,
                        )
                    )
                    continue

                if cfg.keep_prepared_inputs:
                    prepared_dir = gene_dir / "prepared_inputs"
                    prepared_dir.mkdir(parents=True, exist_ok=True)
                    result = _run_one_gene(
                        cfg,
                        gene=gene,
                        gene_id=gene_ids[gene],
                        gene_transcripts=gene_tx,
                        cells=cells,
                        prepared_dir=prepared_dir,
                        pattern_out_dir=pattern_out_dir,
                    )
                else:
                    with tempfile.TemporaryDirectory(prefix="histoseg_gene_isoline_") as tmpdir:
                        result = _run_one_gene(
                            cfg,
                            gene=gene,
                            gene_id=gene_ids[gene],
                            gene_transcripts=gene_tx,
                            cells=cells,
                            prepared_dir=Path(tmpdir),
                            pattern_out_dir=pattern_out_dir,
                        )

                export_payload = _contours_to_xenium_export_rows(
                    contours=result.contours,
                    sample_id=sample.sample_id,
                    gene=gene,
                    gene_id=gene_ids[gene],
                    color=GENE_ISOLINE_PALETTE[(gene_ids[gene] - 1) % len(GENE_ISOLINE_PALETTE)],
                    transcript_count=n_transcripts,
                    xenium_pixel_size_um=cfg.xenium_pixel_size_um,
                )
                sample_features[sample.sample_id].extend(export_payload["features"])
                sample_csv_rows[sample.sample_id].extend(export_payload["csv_rows"])
                sample_summary_rows[sample.sample_id].extend(export_payload["summary_rows"])

                records.append(
                    _record(
                        sample=sample,
                        gene=gene,
                        status="ok",
                        n_transcripts=n_transcripts,
                        n_cells=n_cells,
                        n_contours=len(result.contours),
                        confidence=result.segmentation_confidence_score,
                        pattern1_out_dir=pattern_out_dir,
                        preview_png=result.preview_png,
                        sample_geojson=out_dir / sample.sample_id / "xenium_explorer_annotations.geojson",
                    )
                )
            except Exception as exc:
                if cfg.fail_fast:
                    raise
                records.append(
                    _record(
                        sample=sample,
                        gene=gene,
                        status=f"error: {exc}",
                        n_cells=int(len(cells)),
                        pattern1_out_dir=pattern_out_dir,
                    )
                )

    for sample in samples:
        if sample_features[sample.sample_id]:
            _write_sample_xenium_exports(
                out_dir / sample.sample_id,
                features=sample_features[sample.sample_id],
                csv_rows=sample_csv_rows[sample.sample_id],
                summary_rows=sample_summary_rows[sample.sample_id],
            )

    run_log_path = out_dir / "gene_isoline_run_log.csv"
    pd.DataFrame(
        records,
        columns=[
            "sample",
            "sample_order",
            "gene",
            "status",
            "n_transcripts",
            "n_cells",
            "n_contours",
            "confidence",
            "xenium_dir",
            "pattern1_out_dir",
            "preview_png",
            "sample_geojson",
        ],
    ).to_csv(run_log_path, index=False)

    return GeneTranscriptIsolineResult(
        out_dir=out_dir,
        run_log_csv=run_log_path,
        records=records,
    )


def _discover_xenium_transcript_samples(
    xenium_root: PathLike,
    *,
    sample_glob: str = "*",
    xenium_output_glob: str = "output-*",
) -> list[_GeneSampleInput]:
    root = Path(xenium_root).expanduser().resolve()
    if _looks_like_xenium_transcript_dir(root):
        return [
            _GeneSampleInput(
                order=1,
                sample_id=root.name,
                sample_dir=root,
                xenium_dir=root,
            )
        ]
    root_output = _find_xenium_transcript_output_dir(
        root,
        xenium_output_glob=xenium_output_glob,
    )
    if root_output is not None:
        return [
            _GeneSampleInput(
                order=1,
                sample_id=root.name,
                sample_dir=root,
                xenium_dir=root_output,
            )
        ]

    samples: list[_GeneSampleInput] = []
    candidates = [path for path in root.glob(sample_glob) if path.is_dir()]
    for candidate in candidates:
        xenium_dir = _find_xenium_transcript_output_dir(
            candidate,
            xenium_output_glob=xenium_output_glob,
        )
        if xenium_dir is None:
            continue
        samples.append(
            _GeneSampleInput(
                order=0,
                sample_id=candidate.name,
                sample_dir=candidate,
                xenium_dir=xenium_dir,
            )
        )

    ordered = sorted(samples, key=lambda item: _sample_sort_key(item.sample_id))
    return [
        _GeneSampleInput(
            order=index,
            sample_id=item.sample_id,
            sample_dir=item.sample_dir,
            xenium_dir=item.xenium_dir,
        )
        for index, item in enumerate(ordered, start=1)
    ]


def _looks_like_xenium_transcript_dir(path: Path) -> bool:
    return (path / "cells.parquet").exists() and (path / "transcripts.parquet").exists()


def _find_xenium_transcript_output_dir(
    sample_dir: Path,
    *,
    xenium_output_glob: str,
) -> Optional[Path]:
    if _looks_like_xenium_transcript_dir(sample_dir):
        return sample_dir
    for child in sorted(path for path in sample_dir.glob(xenium_output_glob) if path.is_dir()):
        if _looks_like_xenium_transcript_dir(child):
            return child
    return None


def _sample_sort_key(sample_id: str) -> tuple[int, str]:
    matches = re.findall(r"\d+", sample_id)
    if matches:
        return int(matches[-1]), sample_id
    return 10**9, sample_id


def _load_xenium_tables(
    xenium_dir: Path,
    cfg: GeneTranscriptIsolineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transcript_columns = _unique_columns(
        [
            cfg.transcript_id_col,
            cfg.transcript_feature_col,
            cfg.transcript_x_col,
            cfg.transcript_y_col,
            cfg.transcript_qv_col,
        ]
    )
    cell_columns = _unique_columns([cfg.cell_id_col, cfg.cell_x_col, cfg.cell_y_col])
    transcripts = pd.read_parquet(xenium_dir / "transcripts.parquet", columns=transcript_columns)
    cells = pd.read_parquet(xenium_dir / "cells.parquet", columns=cell_columns)
    return transcripts, cells


def _unique_columns(columns: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for column in columns:
        if column not in seen:
            result.append(column)
            seen.add(column)
    return result


def _filter_gene_transcripts(
    transcripts: pd.DataFrame,
    gene: str,
    cfg: GeneTranscriptIsolineConfig,
) -> pd.DataFrame:
    qv = pd.to_numeric(transcripts[cfg.transcript_qv_col], errors="coerce")
    mask = (transcripts[cfg.transcript_feature_col].astype(str) == str(gene)) & (qv >= cfg.qv_min)
    return transcripts.loc[mask].copy()


def _run_one_gene(
    cfg: GeneTranscriptIsolineConfig,
    *,
    gene: str,
    gene_id: int,
    gene_transcripts: pd.DataFrame,
    cells: pd.DataFrame,
    prepared_dir: Path,
    pattern_out_dir: Path,
) -> Pattern1IsolineResult:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    pattern_out_dir.mkdir(parents=True, exist_ok=True)

    inputs = _write_pattern1_inputs(
        cfg,
        gene=gene,
        gene_id=gene_id,
        gene_transcripts=gene_transcripts,
        cells=cells,
        out_dir=prepared_dir,
    )

    return run_pattern1_isoline(
        Pattern1IsolineConfig(
            clusters_csv=inputs["clusters_csv"],
            cells_parquet=inputs["cells_parquet"],
            tissue_boundary_csv=inputs["tissue_boundary_csv"],
            out_dir=pattern_out_dir,
            pattern1_clusters=["1"],
            grid_n=cfg.grid_n,
            knn_k=cfg.knn_k,
            smooth_sigma=cfg.smooth_sigma,
            min_cells_inside=cfg.min_cells_inside,
            use_synth_bg=cfg.use_synth_bg,
            label_scheme="p1_is_one",
            save_params_json=False,
            compute_confidence_score=cfg.compute_confidence_score,
        )
    )


def _write_pattern1_inputs(
    cfg: GeneTranscriptIsolineConfig,
    *,
    gene: str,
    gene_id: int,
    gene_transcripts: pd.DataFrame,
    cells: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    gene_prefix = f"gene{gene_id}_{_safe_path_part(gene)}"
    target = pd.DataFrame(
        {
            "cell_id": (
                gene_prefix
                + "_tx_"
                + gene_transcripts[cfg.transcript_id_col].astype(str).to_numpy()
            ),
            "x_centroid": pd.to_numeric(
                gene_transcripts[cfg.transcript_x_col],
                errors="coerce",
            ).to_numpy(dtype=float),
            "y_centroid": pd.to_numeric(
                gene_transcripts[cfg.transcript_y_col],
                errors="coerce",
            ).to_numpy(dtype=float),
            "Cluster": "1",
        }
    ).dropna(subset=["x_centroid", "y_centroid"])

    background = pd.DataFrame(
        {
            "cell_id": "cell_" + cells[cfg.cell_id_col].astype(str),
            "x_centroid": pd.to_numeric(cells[cfg.cell_x_col], errors="coerce"),
            "y_centroid": pd.to_numeric(cells[cfg.cell_y_col], errors="coerce"),
            "Cluster": "0",
        }
    ).dropna(subset=["x_centroid", "y_centroid"])

    all_points = pd.concat(
        [
            target[["cell_id", "x_centroid", "y_centroid"]],
            background[["cell_id", "x_centroid", "y_centroid"]],
        ],
        ignore_index=True,
    )
    clusters = pd.concat(
        [
            target[["cell_id", "Cluster"]],
            background[["cell_id", "Cluster"]],
        ],
        ignore_index=True,
    ).rename(columns={"cell_id": "Barcode"})

    cells_path = out_dir / "cells.parquet"
    clusters_path = out_dir / "clusters.csv"
    tissue_boundary_path = out_dir / "tissue_boundary.csv"
    all_points.to_parquet(cells_path, index=False)
    clusters.to_csv(clusters_path, index=False)
    generate_tissue_boundary(
        background[["x_centroid", "y_centroid"]],
        x_col="x_centroid",
        y_col="y_centroid",
        method="alpha_shape",
        alpha=cfg.alpha,
        output_csv=tissue_boundary_path,
    )

    return {
        "cells_parquet": cells_path,
        "clusters_csv": clusters_path,
        "tissue_boundary_csv": tissue_boundary_path,
    }


def _contours_to_xenium_export_rows(
    *,
    contours: Sequence[np.ndarray],
    sample_id: str,
    gene: str,
    gene_id: int,
    color: str,
    transcript_count: int,
    xenium_pixel_size_um: float,
) -> dict[str, list[dict[str, object]]]:
    rgb_color = _hex_to_rgb_triplet(color)
    features: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    base_name = f"G{int(gene_id)} {gene}".strip()

    valid_polygons = [_prepare_polygon(contour, xenium_pixel_size_um) for contour in contours]
    valid_polygons = [polygon for polygon in valid_polygons if polygon is not None]

    for polygon_index, polygon in enumerate(valid_polygons, start=1):
        selection_name = base_name if len(valid_polygons) == 1 else f"{base_name} #{polygon_index}"
        feature = {
            "type": "Feature",
            "id": str(uuid4()),
            "geometry": {
                "type": "Polygon",
                "coordinates": [[list(map(float, point)) for point in polygon]],
            },
            "properties": {
                "objectType": "annotation",
                "name": selection_name,
                "classification": {
                    "name": selection_name,
                    "color": rgb_color,
                },
                "structure": gene,
                "structure_id": int(gene_id),
                "assigned_structure": gene,
                "component_index": 1,
                "polygon_index": int(polygon_index),
                "gene": gene,
                "sample_id": sample_id,
                "source": "gene_transcript_isoline",
                "transcript_count": int(transcript_count),
            },
        }
        features.append(feature)

        for x_value, y_value in polygon:
            csv_rows.append(
                {
                    "Selection": selection_name,
                    "X": float(x_value),
                    "Y": float(y_value),
                }
            )

        summary_rows.append(
            {
                "Selection": selection_name,
                "StructureID": int(gene_id),
                "AssignedStructure": gene,
                "Gene": gene,
                "SampleID": sample_id,
                "Source": "gene_transcript_isoline",
                "ComponentIndex": 1,
                "PolygonIndex": int(polygon_index),
                "VertexCount": int(polygon.shape[0]),
                "TranscriptCount": int(transcript_count),
            }
        )

    return {"features": features, "csv_rows": csv_rows, "summary_rows": summary_rows}


def _prepare_polygon(
    contour: np.ndarray,
    xenium_pixel_size_um: float,
) -> Optional[np.ndarray]:
    polygon = np.asarray(contour, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 4:
        return None
    polygon = polygon / float(xenium_pixel_size_um)
    polygon = polygon[np.isfinite(polygon).all(axis=1)]
    if polygon.shape[0] < 4:
        return None
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0]])
    return polygon


def _write_sample_xenium_exports(
    output_dir: Path,
    *,
    features: Sequence[dict[str, object]],
    csv_rows: Sequence[dict[str, object]],
    summary_rows: Sequence[dict[str, object]],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = output_dir / "xenium_explorer_annotations.geojson"
    csv_path = output_dir / "xenium_explorer_annotations.csv"
    summary_path = output_dir / "xenium_explorer_annotations_summary.csv"

    geojson_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": list(features),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(csv_rows, columns=["Selection", "X", "Y"]).to_csv(csv_path, index=False)
    pd.DataFrame(
        summary_rows,
        columns=[
            "Selection",
            "StructureID",
            "AssignedStructure",
            "Gene",
            "SampleID",
            "Source",
            "ComponentIndex",
            "PolygonIndex",
            "VertexCount",
            "TranscriptCount",
        ],
    ).to_csv(summary_path, index=False)
    return {"geojson": geojson_path, "csv": csv_path, "summary": summary_path}


def _hex_to_rgb_triplet(hex_color: str) -> list[int]:
    value = str(hex_color).strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {hex_color!r}")
    return [int(value[index : index + 2], 16) for index in (0, 2, 4)]


def _normalize_gene_list(genes: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        value = str(gene).strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _gene_path_names(genes: Sequence[str]) -> dict[str, str]:
    counts: dict[str, int] = {}
    result: dict[str, str] = {}
    for gene in genes:
        base = _safe_path_part(gene)
        count = counts.get(base, 0) + 1
        counts[base] = count
        result[gene] = base if count == 1 else f"{base}_{count}"
    return result


def _safe_path_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    safe = safe.strip("._-")
    return safe or "gene"


def _record(
    *,
    sample: _GeneSampleInput,
    gene: str,
    status: str,
    n_transcripts: int = 0,
    n_cells: int = 0,
    n_contours: int = 0,
    confidence: Optional[float] = None,
    pattern1_out_dir: Optional[Path] = None,
    preview_png: Optional[Path] = None,
    sample_geojson: Optional[Path] = None,
) -> dict[str, object]:
    return {
        "sample": sample.sample_id,
        "sample_order": int(sample.order),
        "gene": gene,
        "status": status,
        "n_transcripts": int(n_transcripts),
        "n_cells": int(n_cells),
        "n_contours": int(n_contours),
        "confidence": confidence,
        "xenium_dir": str(sample.xenium_dir),
        "pattern1_out_dir": str(pattern1_out_dir) if pattern1_out_dir is not None else None,
        "preview_png": str(preview_png) if preview_png is not None else None,
        "sample_geojson": str(sample_geojson) if sample_geojson is not None else None,
    }


__all__ = [
    "GeneTranscriptIsolineConfig",
    "GeneTranscriptIsolineResult",
    "run_gene_transcript_isoline",
]
