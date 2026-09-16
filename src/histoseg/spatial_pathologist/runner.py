from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from .artifact_loader import build_case_bundle
from .config import SpatialPathologistConfig
from .heuristics import (
    build_case_summary,
    build_cluster_review,
    build_structure_review,
)
from .openai_client import OpenAIJsonClient, OpenAISettings
from .pathology_ai_api import PathologyAIClient, PathologyAISettings
from .report import write_html_report


def _cluster_review_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "standardized_label": {"type": "string"},
            "cell_family": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "review_priority": {"type": "string", "enum": ["low", "medium", "high"]},
            "summary": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "concerns": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        },
        "required": [
            "standardized_label",
            "cell_family",
            "confidence",
            "review_priority",
            "summary",
            "evidence",
            "concerns",
        ],
    }


def _structure_review_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "assigned_label": {"type": "string"},
            "behavior": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "review_priority": {"type": "string", "enum": ["low", "medium", "high"]},
            "discovery_flag": {"type": "boolean"},
            "summary": {"type": "string"},
            "top_celltype_summary": {"type": "string"},
            "key_evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
            "differential": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "recommended_checks": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        },
        "required": [
            "title",
            "assigned_label",
            "behavior",
            "confidence",
            "review_priority",
            "discovery_flag",
            "summary",
            "top_celltype_summary",
            "key_evidence",
            "differential",
            "recommended_checks",
        ],
    }


def _case_summary_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "headline": {"type": "string"},
            "overall_impression": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "review_priorities": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "discovery_candidates": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        },
        "required": [
            "headline",
            "overall_impression",
            "key_findings",
            "review_priorities",
            "discovery_candidates",
        ],
    }


def _cluster_prompt(cluster: dict, study_context: str) -> tuple[str, str]:
    system = (
        "You are an expert spatial transcriptomics reviewer. "
        "Return JSON only with keys: standardized_label, cell_family, confidence, "
        "review_priority, summary, evidence, concerns."
    )
    user = json.dumps(
        {"study_context": study_context, "cluster": cluster},
        ensure_ascii=False,
        indent=2,
    )
    return system, user


def _structure_prompt(structure: dict, study_context: str) -> tuple[str, str]:
    system = (
        "You are an expert pathologist reviewing structure-level spatial transcriptomics evidence. "
        "Return JSON only with keys: title, assigned_label, behavior, confidence, review_priority, "
        "discovery_flag, summary, top_celltype_summary, key_evidence, differential, recommended_checks."
    )
    user = json.dumps(
        {"study_context": study_context, "structure": structure},
        ensure_ascii=False,
        indent=2,
    )
    return system, user


def _case_prompt(case_bundle: dict, structure_reviews: list[dict]) -> tuple[str, str]:
    system = (
        "You are an expert spatial pathology summarizer. "
        "Return JSON only with keys: headline, overall_impression, key_findings, "
        "review_priorities, discovery_candidates."
    )
    user = json.dumps(
        {
            "case_name": case_bundle["case_name"],
            "study_context": case_bundle["study_context"],
            "structures": structure_reviews,
        },
        ensure_ascii=False,
        indent=2,
    )
    return system, user


def _merge_review(default: dict, llm_result: dict, *, engine_name: str) -> dict:
    merged = dict(default)
    for key, value in llm_result.items():
        if value is None:
            continue
        merged[key] = value
    merged["engine"] = engine_name
    return merged


def _format_citation(hit: dict) -> str:
    title = str(hit.get("title", "Unknown source"))
    page_start = hit.get("page_start", "?")
    page_end = hit.get("page_end", page_start)
    retrieval_source = hit.get("retrieval_source", "rag")
    preview = " ".join(str(hit.get("text_preview", "")).split())
    if preview:
        preview = f": {preview[:180]}"
    return f"{title} pp.{page_start}-{page_end} [{retrieval_source}]{preview}"


def _structure_rag_question(structure: dict, heuristic_review: dict, study_context: str) -> str:
    top_clusters = ", ".join(
        f"C{int(cluster['cluster_id'])} {cluster['label']} ({100.0 * float(cluster['fraction_of_structure']):.1f}%)"
        for cluster in structure.get("top_clusters", [])[:4]
    ) or "No dominant clusters"
    alternatives = ", ".join(
        str(candidate.get("label", ""))
        for candidate in structure.get("top_candidates", [])[1:4]
        if candidate.get("label")
    ) or "No strong alternative labels"
    return (
        f"Study context: {study_context}\n"
        f"Spatial structure: S{int(structure['structure_id'])}\n"
        f"Current assigned label: {structure['assigned_label']}\n"
        f"Behavior: {structure['behavior']}\n"
        f"Cells in structure: {int(structure['case_cell_count'])}\n"
        f"Fraction of case: {100.0 * float(structure['case_cell_fraction']):.1f}%\n"
        f"Dominant harmonized cell types: {heuristic_review['top_celltype_summary']}\n"
        f"Dominant contributing clusters: {top_clusters}\n"
        f"Rule-based score total: {float(structure['score_total']):.3f}\n"
        f"Winner margin to second leaf: {float(structure['winner_margin_to_second_leaf']):.3f}\n"
        f"Top alternative labels: {alternatives}\n\n"
        "Using pathology textbook evidence, write a concise interpretation of whether this structure label is "
        "pathologically plausible, note the most relevant differentials, and mention the most important manual review checks."
    )


def _case_rag_question(case_bundle: dict, structure_reviews: list[dict]) -> str:
    structure_lines = []
    for review in structure_reviews[:6]:
        structure_lines.append(
            f"S{int(review['structure_id'])}: {review['assigned_label']} | priority={review['review_priority']} | "
            f"confidence={float(review['confidence']):.3f} | top_celltypes={review.get('top_celltype_summary', '')[:180]}"
        )
    return (
        f"Study context: {case_bundle['study_context']}\n"
        f"Case name: {case_bundle['case_name']}\n"
        f"Total cells: {int(case_bundle['total_cells'])}\n"
        f"Number of spatial structures: {len(structure_reviews)}\n\n"
        "Structure-level interpretations:\n"
        + "\n".join(structure_lines)
        + "\n\nUsing pathology textbook evidence, provide a brief overall pathology impression for this case, "
          "highlight key differentials, and mention the most important structures that should be manually confirmed."
    )


def _merge_pathology_ai_structure_review(
    default: dict,
    rag_response: dict,
) -> dict:
    merged = dict(default)
    answer = str(rag_response.get("answer", "")).strip()
    citations = [_format_citation(hit) for hit in rag_response.get("citations", [])]
    if answer:
        merged["summary"] = answer
    merged["key_evidence"] = (list(default.get("key_evidence", [])) + citations[:3])[:10]
    merged["pathology_ai_citations"] = citations[:5]
    merged["engine"] = f"pathology_ai_rag:{rag_response.get('used_model', 'unknown')}"
    if not rag_response.get("used_rag", False):
        merged["recommended_checks"] = (
            ["No strong pathology textbook support was retrieved; weigh structure evidence and manual review more heavily."]
            + list(default.get("recommended_checks", []))
        )[:8]
    return merged


def _merge_pathology_ai_case_summary(
    default: dict,
    rag_response: dict,
) -> dict:
    merged = dict(default)
    answer = str(rag_response.get("answer", "")).strip()
    citations = [_format_citation(hit) for hit in rag_response.get("citations", [])]
    if answer:
        merged["overall_impression"] = answer
    merged["pathology_ai_citations"] = citations[:5]
    merged["engine"] = f"pathology_ai_rag:{rag_response.get('used_model', 'unknown')}"
    return merged


def run_spatial_pathologist(cfg: SpatialPathologistConfig) -> dict[str, str]:
    case_bundle = build_case_bundle(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_client = None
    pathology_ai_client = None
    settings = OpenAISettings(
        enabled=cfg.openai_enabled,
        api_key_env=cfg.openai_api_key_env,
        model=cfg.openai_model,
        reasoning_effort=cfg.openai_reasoning_effort,
        store=cfg.openai_store,
    )
    if cfg.pathology_review_backend == "openai" and settings.available:
        llm_client = OpenAIJsonClient(settings)
    if cfg.pathology_review_backend == "pathology_ai_api":
        pathology_ai_settings = PathologyAISettings(
            base_url=cfg.pathology_ai_api_base_url,
            top_k=cfg.pathology_ai_top_k,
            answer_language=cfg.pathology_ai_answer_language,
            document_ids=cfg.pathology_ai_document_ids,
        )
        if pathology_ai_settings.available:
            pathology_ai_client = PathologyAIClient(pathology_ai_settings)

    cluster_reviews = []
    for cluster in case_bundle["clusters"]:
        heuristic_review = build_cluster_review(cluster)
        if cfg.pathology_review_backend == "openai" and llm_client is not None:
            try:
                system, user = _cluster_prompt(cluster, case_bundle["study_context"])
                llm_review = llm_client.generate_json(
                    system_prompt=system,
                    user_prompt=user,
                    schema_name="cluster_review",
                    schema=_cluster_review_schema(),
                )
                cluster_reviews.append(
                    _merge_review(heuristic_review, llm_review, engine_name=f"openai:{cfg.openai_model}")
                )
                continue
            except Exception:
                pass
        cluster_reviews.append(heuristic_review)

    structure_reviews = []
    for structure in case_bundle["structures"]:
        heuristic_review = build_structure_review(
            structure,
            low_confidence_threshold=cfg.low_confidence_threshold,
            ambiguity_margin_threshold=cfg.ambiguity_margin_threshold,
        )
        if cfg.pathology_review_backend == "openai" and llm_client is not None:
            try:
                system, user = _structure_prompt(structure, case_bundle["study_context"])
                llm_review = llm_client.generate_json(
                    system_prompt=system,
                    user_prompt=user,
                    schema_name="structure_review",
                    schema=_structure_review_schema(),
                )
                structure_reviews.append(
                    _merge_review(heuristic_review, llm_review, engine_name=f"openai:{cfg.openai_model}")
                )
                continue
            except Exception:
                pass
        if cfg.pathology_review_backend == "pathology_ai_api" and pathology_ai_client is not None:
            try:
                rag_review = pathology_ai_client.ask(
                    _structure_rag_question(structure, heuristic_review, case_bundle["study_context"])
                )
                structure_reviews.append(_merge_pathology_ai_structure_review(heuristic_review, rag_review))
                continue
            except Exception:
                pass
        structure_reviews.append(heuristic_review)

    case_summary = build_case_summary(case_bundle, structure_reviews)
    if cfg.pathology_review_backend == "openai" and llm_client is not None:
        try:
            system, user = _case_prompt(case_bundle, structure_reviews)
            llm_summary = llm_client.generate_json(
                system_prompt=system,
                user_prompt=user,
                schema_name="case_summary",
                schema=_case_summary_schema(),
            )
            case_summary = _merge_review(case_summary, llm_summary, engine_name=f"openai:{cfg.openai_model}")
        except Exception:
            pass
    elif cfg.pathology_review_backend == "pathology_ai_api" and pathology_ai_client is not None:
        try:
            rag_summary = pathology_ai_client.ask(_case_rag_question(case_bundle, structure_reviews))
            case_summary = _merge_pathology_ai_case_summary(case_summary, rag_summary)
        except Exception:
            pass

    (output_dir / "cluster_reviews.json").write_text(
        json.dumps(cluster_reviews, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(cluster_reviews).to_csv(
        output_dir / "cluster_reviews.csv",
        index=False,
    )
    (output_dir / "structure_reviews.json").write_text(
        json.dumps(structure_reviews, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(structure_reviews).to_csv(
        output_dir / "structure_reviews.csv",
        index=False,
    )
    (output_dir / "case_summary.json").write_text(
        json.dumps(case_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "pathology_review_backend": cfg.pathology_review_backend,
                "pathology_ai_api_base_url": cfg.pathology_ai_api_base_url,
                "pathology_ai_top_k": cfg.pathology_ai_top_k,
                "pathology_ai_answer_language": cfg.pathology_ai_answer_language,
                "pathology_ai_document_ids": list(cfg.pathology_ai_document_ids),
                "openai_requested": cfg.openai_enabled,
                "openai_api_key_env": cfg.openai_api_key_env,
                "openai_api_key_present": settings.api_key is not None,
                "openai_dependency_available": settings.available,
                "openai_model": cfg.openai_model,
                "openai_reasoning_effort": cfg.openai_reasoning_effort,
                "openai_store": cfg.openai_store,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_path = write_html_report(
        output_dir=output_dir,
        case_bundle=case_bundle,
        cluster_reviews=cluster_reviews,
        structure_reviews=structure_reviews,
        case_summary=case_summary,
    )

    return {
        "output_dir": str(output_dir),
        "report_html": str(report_path),
        "cluster_reviews_json": str(output_dir / "cluster_reviews.json"),
        "structure_reviews_json": str(output_dir / "structure_reviews.json"),
        "case_summary_json": str(output_dir / "case_summary.json"),
    }
