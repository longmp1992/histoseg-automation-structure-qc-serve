from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import pandas as pd

from histoseg.openai_json_client import OpenAIJsonClient, OpenAISettings

from .evidence_pack import build_cluster_evidence_pack
from .heuristic_labeler import annotate_cluster_heuristically
from .openai_annotator import annotate_cluster_with_openai, review_case_annotations_with_openai
from .report import write_annotation_report


@dataclass(frozen=True)
class ClusterAnnotationPipelineConfig:
    case_name: str
    study_context: str
    cluster_csv: Path
    differential_expression_csv: Path
    projection_csv: Path
    output_dir: Path
    annotation_taxonomy: str = "lung"

    openai_enabled: bool = True
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_model: str = "gpt-5.4"
    openai_reasoning_effort: str = "medium"
    openai_store: bool = False
    force_recompute: bool = False

    top_positive_markers: int = 15
    top_negative_markers: int = 6
    min_log2fc: float = 0.5
    max_adjusted_p_value: float = 0.05
    top_neighbors: int = 5


def _heuristic_case_review(cluster_annotations: list[dict[str, Any]]) -> dict[str, Any]:
    high_priority = [
        int(item["cluster_id"])
        for item in cluster_annotations
        if str(item["review_priority"]) == "high"
    ]
    discovery_candidates = [
        int(item["cluster_id"])
        for item in cluster_annotations
        if item["label_id"] in {
            "tumor_high_heterogeneity",
            "proliferating_cells",
            "proliferating_tumor_or_immune",
            "proliferating_tumor_cells",
            "basal_like_tumor",
        }
    ]
    return {
        "headline": f"Automated annotation completed for {len(cluster_annotations)} clusters.",
        "overall_impression": (
            f"{len(high_priority)} clusters need closer review because of ambiguous or proliferative programs."
        ),
        "high_priority_cluster_ids": high_priority,
        "consistency_notes": [
            "Review proliferative clusters against morphology and CNV when available.",
            "Tumor-vs-normal adjudication is marker-driven in this run because no CNV artifact was supplied.",
        ],
        "discovery_candidates": discovery_candidates,
    }


def run_cluster_annotation_pipeline(cfg: ClusterAnnotationPipelineConfig) -> dict[str, str]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_pack = build_cluster_evidence_pack(
        cluster_csv=cfg.cluster_csv,
        differential_expression_csv=cfg.differential_expression_csv,
        projection_csv=cfg.projection_csv,
        case_name=cfg.case_name,
        study_context=cfg.study_context,
        top_positive_markers=cfg.top_positive_markers,
        top_negative_markers=cfg.top_negative_markers,
        min_log2fc=cfg.min_log2fc,
        max_adjusted_p_value=cfg.max_adjusted_p_value,
        top_neighbors=cfg.top_neighbors,
    )
    (output_dir / "cluster_evidence.json").write_text(
        json.dumps(evidence_pack, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    settings = OpenAISettings(
        enabled=cfg.openai_enabled,
        api_key_env=cfg.openai_api_key_env,
        model=cfg.openai_model,
        reasoning_effort=cfg.openai_reasoning_effort,
        store=cfg.openai_store,
    )
    llm_client = OpenAIJsonClient(settings) if settings.available else None

    cluster_annotations: list[dict[str, Any]] = []
    for cluster_evidence in evidence_pack["clusters"]:
        heuristic_annotation = annotate_cluster_heuristically(
            cluster_evidence,
            taxonomy_name=cfg.annotation_taxonomy,
        )
        if llm_client is not None:
            try:
                cluster_annotations.append(
                    annotate_cluster_with_openai(
                        llm_client,
                        case_name=cfg.case_name,
                        study_context=cfg.study_context,
                        cluster_evidence=cluster_evidence,
                        heuristic_annotation=heuristic_annotation,
                        taxonomy_name=cfg.annotation_taxonomy,
                    )
                )
                continue
            except Exception:
                pass
        cluster_annotations.append(heuristic_annotation)

    case_review = _heuristic_case_review(cluster_annotations)
    if llm_client is not None:
        try:
            case_review = review_case_annotations_with_openai(
                llm_client,
                case_name=cfg.case_name,
                study_context=cfg.study_context,
                cluster_annotations=cluster_annotations,
                taxonomy_name=cfg.annotation_taxonomy,
            )
        except Exception:
            pass

    full_table = pd.DataFrame(cluster_annotations).sort_values("cluster_id")
    compatibility = full_table[["cluster_id", "detailed_label"]].copy()
    compatibility.columns = ["cluster", "celltype"]

    full_table.to_csv(output_dir / "cluster_annotations_openai.csv", index=False)
    compatibility.to_csv(output_dir / "cluster_celltype_annotation.csv", index=False)
    (output_dir / "cluster_annotations_openai.json").write_text(
        json.dumps(cluster_annotations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "annotation_case_review.json").write_text(
        json.dumps(case_review, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "annotation_run_metadata.json").write_text(
        json.dumps(
            {
                "openai_requested": cfg.openai_enabled,
                "openai_api_key_env": cfg.openai_api_key_env,
                "openai_api_key_present": settings.api_key is not None,
                "openai_dependency_available": settings.available,
                "annotation_taxonomy": cfg.annotation_taxonomy,
                "openai_model": cfg.openai_model,
                "openai_reasoning_effort": cfg.openai_reasoning_effort,
                "openai_store": cfg.openai_store,
                "cluster_count": len(cluster_annotations),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report_path = write_annotation_report(
        output_dir=output_dir,
        evidence_pack=evidence_pack,
        cluster_annotations=cluster_annotations,
        case_review=case_review,
    )
    return {
        "output_dir": str(output_dir),
        "cluster_evidence_json": str(output_dir / "cluster_evidence.json"),
        "cluster_annotations_json": str(output_dir / "cluster_annotations_openai.json"),
        "cluster_annotations_csv": str(output_dir / "cluster_annotations_openai.csv"),
        "compatibility_csv": str(output_dir / "cluster_celltype_annotation.csv"),
        "case_review_json": str(output_dir / "annotation_case_review.json"),
        "report_html": str(report_path),
    }
