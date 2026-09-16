from __future__ import annotations

from math import exp
from typing import Any
import re


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _summary_brief(text: str, *, max_chars: int = 180) -> str:
    normalized = str(text).replace("\r", "\n")
    normalized = re.sub(r"^\s*#{1,4}\s*", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"\*\*(.+?)\*\*", r"\1", normalized)
    normalized = re.sub(r"\*(.+?)\*", r"\1", normalized)
    normalized = re.sub(r"`(.+?)`", r"\1", normalized)
    normalized = " ".join(normalized.split())
    normalized = re.sub(r"^\*+|\*+$", "", normalized).strip()
    if not normalized:
        return "No summary available."
    if len(normalized) <= max_chars:
        return normalized
    cutoff = normalized[: max_chars + 1].rsplit(" ", 1)[0].strip()
    return cutoff + "..."


def _label_family(label: str) -> str:
    text = str(label).lower()
    if any(token in text for token in ("tumor", "adenocarcinoma", "carcinoma", "neuroendocrine")):
        return "tumor"
    if any(token in text for token in ("fibro", "endothelial", "pericyte", "stroma", "smooth muscle", "mural")):
        return "stromal"
    if any(token in text for token in ("t cell", "b cell", "plasma", "macrophage", "monocyte", "mast", "dendritic", "nk")):
        return "immune"
    if any(token in text for token in ("ciliated", "club", "alveolar", "epithelial", "bronchio")):
        return "epithelial"
    return "other"


def structure_confidence(structure: dict[str, Any]) -> float:
    score_total = float(structure["score_total"])
    margin = min(float(structure["winner_margin_to_second_leaf"]), 0.20)
    return round(max(0.01, min(0.99, _sigmoid((2.2 * score_total) + (3.0 * margin) - 0.15))), 3)


def structure_review_priority(
    structure: dict[str, Any],
    *,
    low_confidence_threshold: float,
    ambiguity_margin_threshold: float,
) -> str:
    confidence = structure_confidence(structure)
    margin = float(structure["winner_margin_to_second_leaf"])
    if confidence < low_confidence_threshold or margin < ambiguity_margin_threshold or structure["fallback_applied"]:
        return "high"
    if confidence < (low_confidence_threshold + 0.12):
        return "medium"
    return "low"


def structure_discovery_flag(
    structure: dict[str, Any],
    *,
    low_confidence_threshold: float,
    ambiguity_margin_threshold: float,
) -> bool:
    return (
        structure_confidence(structure) < low_confidence_threshold
        or float(structure["winner_margin_to_second_leaf"]) < ambiguity_margin_threshold
        or bool(structure["fallback_applied"])
    )


def build_cluster_review(cluster: dict[str, Any]) -> dict[str, Any]:
    family = _label_family(cluster["label"])
    structure_name = str(cluster["structure_name"])

    review_priority = "low"
    if family == "tumor" and "adenocarcinoma" not in structure_name.lower():
        review_priority = "high"
    elif family == "epithelial" and "stroma" in structure_name.lower():
        review_priority = "medium"

    confidence = 0.92 if review_priority == "low" else (0.72 if review_priority == "medium" else 0.58)
    summary = (
        f"Cluster C{cluster['cluster_id']} is currently interpreted as {cluster['label']} and sits within "
        f"{structure_name}. It accounts for {_format_percent(cluster['fraction_of_structure'])} of its parent "
        f"structure and {_format_percent(cluster['fraction_of_case'])} of the full case."
    )
    return {
        "cluster_id": int(cluster["cluster_id"]),
        "standardized_label": str(cluster["label"]),
        "cell_family": family,
        "confidence": round(confidence, 3),
        "review_priority": review_priority,
        "summary": summary,
        "evidence": [
            f"Parent structure: {structure_name}",
            f"Cluster cell count: {int(cluster['cluster_cell_count'])}",
            f"Fraction of structure: {_format_percent(cluster['fraction_of_structure'])}",
        ],
        "concerns": (
            ["Label-to-structure mismatch worth manual review."]
            if review_priority != "low"
            else []
        ),
        "engine": "heuristic",
    }


def build_structure_review(
    structure: dict[str, Any],
    *,
    low_confidence_threshold: float,
    ambiguity_margin_threshold: float,
) -> dict[str, Any]:
    confidence = structure_confidence(structure)
    review_priority = structure_review_priority(
        structure,
        low_confidence_threshold=low_confidence_threshold,
        ambiguity_margin_threshold=ambiguity_margin_threshold,
    )
    discovery_flag = structure_discovery_flag(
        structure,
        low_confidence_threshold=low_confidence_threshold,
        ambiguity_margin_threshold=ambiguity_margin_threshold,
    )

    top_clusters = structure["top_clusters"][:3]
    top_cluster_labels = ", ".join(cluster["label"] for cluster in top_clusters) if top_clusters else "No dominant clusters"
    top_celltypes = sorted(
        structure["harmonized_composition"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    top_celltype_summary = ", ".join(f"{label} ({_format_percent(weight)})" for label, weight in top_celltypes)

    summary = (
        f"S{structure['structure_id']} is most consistent with {structure['assigned_label']} "
        f"({structure['behavior']}). The region contains {int(structure['case_cell_count'])} cells "
        f"({_format_percent(structure['case_cell_fraction'])} of the case), with dominant support from "
        f"{top_cluster_labels}."
    )

    recommended_checks = []
    if discovery_flag:
        recommended_checks.append("Review the differential list and overlay before accepting the top label.")
    if structure["fallback_applied"]:
        recommended_checks.append("Fallback to a parent label was applied in the rule-based scorer.")
    if "bronchiole" in str(structure["assigned_label"]).lower():
        recommended_checks.append("Confirm bronchiolar localization on the H&E overlay.")
    if "adenocarcinoma" in str(structure["assigned_label"]).lower():
        recommended_checks.append("Inspect tumor-stroma interfaces and mixed immune pockets.")

    return {
        "structure_id": int(structure["structure_id"]),
        "title": f"S{structure['structure_id']} {structure['assigned_label']}",
        "assigned_label": str(structure["assigned_label"]),
        "behavior": str(structure["behavior"]),
        "confidence": confidence,
        "review_priority": review_priority,
        "discovery_flag": discovery_flag,
        "summary": summary,
        "top_celltype_summary": top_celltype_summary,
        "top_clusters": top_clusters,
        "differential": [
            candidate["label"]
            for candidate in structure["top_candidates"][1:4]
        ],
        "key_evidence": [
            f"Rule-based score total: {float(structure['score_total']):.3f}",
            f"Winner margin: {float(structure['winner_margin_to_second_leaf']):.3f}",
            f"Top harmonized cell types: {top_celltype_summary}",
            f"Polygon count: {int(structure['polygon_count'])}",
        ],
        "recommended_checks": recommended_checks,
        "engine": "heuristic",
    }


def build_case_summary(
    case_bundle: dict[str, Any],
    structure_reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    sorted_reviews = sorted(
        structure_reviews,
        key=lambda item: (item["review_priority"] != "high", -item["confidence"]),
    )
    high_priority = [item for item in sorted_reviews if item["review_priority"] == "high"]
    discovery_candidates = [item for item in sorted_reviews if item["discovery_flag"]]
    dominant_structures = sorted(
        structure_reviews,
        key=lambda item: next(
            (
                structure["case_cell_fraction"]
                for structure in case_bundle["structures"]
                if int(structure["structure_id"]) == int(item["structure_id"])
            ),
            0.0,
        ),
        reverse=True,
    )[:3]

    headline = (
        f"{case_bundle['case_name']} contains {len(structure_reviews)} spatial pathology structures "
        f"across {int(case_bundle['total_cells'])} cells."
    )
    dominant_labels = ", ".join(review["assigned_label"] for review in dominant_structures)
    impression = (
        f"The dominant structure pattern is {dominant_labels}. "
        f"Automated review suggests {len(high_priority)} high-priority structures for manual confirmation."
    )

    return {
        "headline": headline,
        "overall_impression": impression,
        "key_findings": [
            f"S{review['structure_id']}: {_summary_brief(review['summary'])}" for review in dominant_structures
        ],
        "review_priorities": [
            f"S{review['structure_id']}: {review['assigned_label']}"
            for review in high_priority
        ],
        "discovery_candidates": [
            f"S{review['structure_id']}: {review['assigned_label']}"
            for review in discovery_candidates
        ],
        "engine": "heuristic",
    }
