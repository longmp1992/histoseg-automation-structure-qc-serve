from __future__ import annotations

from typing import Any
import json

from histoseg.openai_json_client import OpenAIJsonClient

from .taxonomy import get_label_by_id, label_taxonomy_payload, normalize_taxonomy_name


def _cluster_annotation_schema(label_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label_id": {
                "type": "string",
                "enum": label_ids,
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "review_priority": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
            "supporting_markers": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "conflicting_markers": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "alternative_label_ids": {
                "type": "array",
                "items": {"type": "string", "enum": label_ids},
                "maxItems": 4,
            },
            "reasoning_summary": {"type": "string"},
            "tumor_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "recommended_follow_up": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
            },
        },
        "required": [
            "label_id",
            "confidence",
            "review_priority",
            "supporting_markers",
            "conflicting_markers",
            "alternative_label_ids",
            "reasoning_summary",
            "tumor_evidence",
            "recommended_follow_up",
        ],
    }


def _case_review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "headline": {"type": "string"},
            "overall_impression": {"type": "string"},
            "high_priority_cluster_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 8,
            },
            "consistency_notes": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "discovery_candidates": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 8,
            },
        },
        "required": [
            "headline",
            "overall_impression",
            "high_priority_cluster_ids",
            "consistency_notes",
            "discovery_candidates",
        ],
    }


def annotate_cluster_with_openai(
    client: OpenAIJsonClient,
    *,
    case_name: str,
    study_context: str,
    cluster_evidence: dict[str, Any],
    heuristic_annotation: dict[str, Any],
    taxonomy_name: str = "lung",
) -> dict[str, Any]:
    taxonomy_name = normalize_taxonomy_name(taxonomy_name)
    label_by_id = get_label_by_id(taxonomy_name)
    label_ids = list(label_by_id)
    system_prompt = (
        "You are an expert spatial transcriptomics annotator. "
        "Choose the single best label_id from the provided controlled vocabulary. "
        "Use only the supplied evidence. If evidence is mixed or uncertain, keep confidence lower and raise review_priority. "
        "supporting_markers and conflicting_markers must come from the provided marker lists."
    )
    user_prompt = json.dumps(
        {
            "case_name": case_name,
            "study_context": study_context,
            "annotation_taxonomy": taxonomy_name,
            "controlled_vocabulary": label_taxonomy_payload(taxonomy_name),
            "heuristic_starting_point": {
                "label_id": heuristic_annotation["label_id"],
                "label": heuristic_annotation["detailed_label"],
                "confidence": heuristic_annotation["confidence"],
                "alternatives": heuristic_annotation["alternative_labels"],
            },
            "cluster_evidence": cluster_evidence,
        },
        ensure_ascii=False,
        indent=2,
    )
    result = client.generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="cluster_annotation",
        schema=_cluster_annotation_schema(label_ids),
    )
    spec = label_by_id[str(result["label_id"])]
    alternative_ids = [label_id for label_id in result.get("alternative_label_ids", []) if label_id != spec.id]
    return {
        "cluster_id": int(cluster_evidence["cluster_id"]),
        "label_id": spec.id,
        "detailed_label": spec.label,
        "broad_family": spec.broad_family,
        "malignancy_state": spec.malignancy_state,
        "confidence": round(float(result["confidence"]), 3),
        "supporting_markers": [str(value) for value in result.get("supporting_markers", [])],
        "conflicting_markers": [str(value) for value in result.get("conflicting_markers", [])],
        "alternative_label_ids": alternative_ids,
        "alternative_labels": [label_by_id[label_id].label for label_id in alternative_ids],
        "reasoning_summary": str(result.get("reasoning_summary", "")).strip(),
        "review_priority": str(result["review_priority"]),
        "tumor_evidence": [str(value) for value in result.get("tumor_evidence", [])],
        "recommended_follow_up": [str(value) for value in result.get("recommended_follow_up", [])],
        "downstream_cell_type": spec.label,
        "engine": f"openai:{client.settings.model}",
        "prompt_version": f"openai-cluster-{taxonomy_name}-v1",
    }


def review_case_annotations_with_openai(
    client: OpenAIJsonClient,
    *,
    case_name: str,
    study_context: str,
    cluster_annotations: list[dict[str, Any]],
    taxonomy_name: str = "lung",
) -> dict[str, Any]:
    taxonomy_name = normalize_taxonomy_name(taxonomy_name)
    system_prompt = (
        "You are an expert spatial transcriptomics reviewer performing a case-level consistency pass. "
        "Summarize the overall case, list high-priority clusters for manual review, and flag discovery candidates."
    )
    user_prompt = json.dumps(
        {
            "case_name": case_name,
            "study_context": study_context,
            "annotation_taxonomy": taxonomy_name,
            "cluster_annotations": cluster_annotations,
        },
        ensure_ascii=False,
        indent=2,
    )
    return client.generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="annotation_case_review",
        schema=_case_review_schema(),
    )
