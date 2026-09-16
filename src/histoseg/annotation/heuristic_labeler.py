from __future__ import annotations

from typing import Any

from .taxonomy import ClusterLabelSpec, get_label_by_id, get_label_specs, normalize_taxonomy_name


_PROLIFERATION_GENES = {"MKI67", "TOP2A", "CCNB1", "BIRC5", "PLK1", "CDC20", "AURKB", "CCNA2", "ASPM"}
_TUMOR_GENES = {"FOLR1", "LAPTM4B", "HOXB13", "NQO1", "HHLA2", "SSX2", "SALL4", "CRNDE", "ABCC3", "PRAME"}
_MESOTHELIAL_GENES = {"MSLN", "UPK3B"}
_ARTERIAL_PERICYTE_GENES = {"KCNT1", "GJC1", "ADRA1D", "SFRP5"}
_TIP_ENDOTHELIAL_GENES = {"ANGPT2", "ESM1", "KDR"}
_IMMUNE_LABEL_IDS = {
    "cd4_t_cells",
    "classical_monocytes_macrophages",
    "neutrophils",
    "nk_nkt_cells",
    "b_cells",
    "dendritic_cells",
    "plasma_cells",
    "tam",
    "treg",
    "mast_cells",
    "cd8_exhausted_t_cells",
}
_TUMOR_LABEL_IDS = {
    "tumor_adenocarcinoma",
    "tumor_high_heterogeneity",
    "at2_like_tumor",
    "neuroendocrine_tumor_cells",
}
_BREAST_TUMOR_LABEL_IDS = {
    "secretory_luminal_tumor",
    "er_positive_luminal_tumor",
    "basal_like_tumor",
    "proliferating_tumor_cells",
}
_BREAST_PROLIFERATION_GENES = {"MKI67", "TOP2A", "CDC20", "CCNB1", "BIRC5", "AURKA", "UBE2C", "CENPF", "NDC80"}
_BREAST_LUMINAL_GENES = {"ESR1", "GATA3", "FOXA1", "PIP", "SCUBE2", "ANKRD30A", "ANKRD30B", "AGR3", "AR"}
_BREAST_SECRETORY_GENES = {"SCGB2A1", "SCGB2A2", "OPRPN", "CYP4Z1", "NPY1R", "EPCAM", "KRT15", "VEGFA"}
_BREAST_BASAL_TUMOR_GENES = {"WFDC2", "KRT6B", "KRT23", "FOXC1", "CLDN4", "LAMC2", "CEACAM6", "SERPINA3", "S100A8", "S100A9"}
_BREAST_MYO_GENES = {"COL17A1", "TP63", "KRT5", "KRT14", "LAMB3", "KRT17", "ACTA2", "CNN1"}
_BREAST_ENDOTHELIAL_GENES = {"VWF", "PECAM1", "PLVAP", "CLEC14A", "FLT1", "AQP1", "HSPG2", "LEPR"}
_BREAST_PERICYTE_GENES = {"RGS5", "PDGFRB", "MYH11", "ACTA2", "TAGLN", "MYLK", "COL4A1", "HEYL"}
_BREAST_MACROPHAGE_GENES = {"CD163", "ITGAX", "LYVE1", "CD68", "LYZ", "HMOX1", "APOC1", "MMP12"}
_BREAST_LYMPHOCYTE_GENES = {"TRAC", "CD3E", "CCL5", "CD8A", "IL7R", "MS4A1", "CD27", "CTLA4"}
_BREAST_FIBROBLAST_GENES = {"SFRP4", "CCDC80", "LAMA2", "CXCL12", "CXCL14", "FBLN1", "LOXL1", "C1R"}


def _positive_marker_map(cluster_evidence: dict[str, Any]) -> dict[str, float]:
    marker_map: dict[str, float] = {}
    for marker in cluster_evidence.get("top_positive_markers", []):
        gene = str(marker.get("gene", "")).upper()
        if not gene:
            continue
        marker_map[gene] = float(marker.get("log2fc", 0.0))
    return marker_map


def _support_hits(spec: ClusterLabelSpec, marker_map: dict[str, float]) -> list[str]:
    hits = [gene for gene in spec.marker_genes if gene in marker_map]
    hits.sort(key=lambda gene: marker_map.get(gene, 0.0), reverse=True)
    return hits


def _score_spec(spec: ClusterLabelSpec, marker_map: dict[str, float]) -> float:
    hits = _support_hits(spec, marker_map)
    score = sum(marker_map.get(gene, 0.0) for gene in hits) + 0.35 * len(hits)
    for gene in spec.negative_markers:
        if gene in marker_map:
            score -= 0.65 * marker_map[gene]
    return float(score)


def _apply_lung_rule_adjustments(scores: dict[str, float], marker_map: dict[str, float]) -> None:
    proliferation_score = sum(marker_map.get(gene, 0.0) for gene in _PROLIFERATION_GENES if gene in marker_map)
    tumor_score = sum(marker_map.get(gene, 0.0) for gene in _TUMOR_GENES if gene in marker_map)
    mesothelial_score = sum(marker_map.get(gene, 0.0) for gene in _MESOTHELIAL_GENES if gene in marker_map)
    arterial_pericyte_score = sum(marker_map.get(gene, 0.0) for gene in _ARTERIAL_PERICYTE_GENES if gene in marker_map)
    tip_endothelial_score = sum(marker_map.get(gene, 0.0) for gene in _TIP_ENDOTHELIAL_GENES if gene in marker_map)

    if proliferation_score >= 3.0:
        scores["proliferating_cells"] += 1.2
        if tumor_score >= 1.5:
            scores["proliferating_tumor_or_immune"] += 1.3
        immune_peak = max(scores[label_id] for label_id in _IMMUNE_LABEL_IDS)
        if immune_peak >= 2.2:
            scores["proliferating_tumor_or_immune"] += 0.7

    if tumor_score >= 2.0:
        scores["tumor_adenocarcinoma"] += 0.8
        scores["tumor_high_heterogeneity"] += 0.6

    if {"SSX2", "SALL4", "PRAME", "CRNDE"} & marker_map.keys():
        scores["tumor_high_heterogeneity"] += 1.0

    if mesothelial_score >= 1.0:
        scores["at1_mesothelial"] += 0.9

    if arterial_pericyte_score >= 1.0:
        scores["pericytes_arterial"] += 0.8

    if tip_endothelial_score >= 1.0:
        scores["lymphatic_tip_endothelial"] += 0.8

    if "FOXP3" in marker_map:
        scores["treg"] += 1.2
    if "PDCD1" in marker_map or "CXCL13" in marker_map:
        scores["cd8_exhausted_t_cells"] += 0.9
    if "NCR1" in marker_map or "KLRD1" in marker_map:
        scores["nk_nkt_cells"] += 0.7
    if "MARCO" in marker_map or "TREM2" in marker_map:
        scores["tam"] += 1.0
    if "CD207" in marker_map or "FCER1A" in marker_map:
        scores["dendritic_cells"] += 0.9
    if "CALCA" in marker_map or "ASCL1" in marker_map:
        scores["neuroendocrine_tumor_cells"] += 1.5
    if "DNAH9" in marker_map and ("TEKT1" in marker_map or "ZMYND10" in marker_map):
        scores["ciliated_epithelial_subtype"] += 0.7
    if "LAMP3" in marker_map or "ABCA3" in marker_map:
        scores["at2_like_tumor"] += 0.9


def _apply_breast_rule_adjustments(scores: dict[str, float], marker_map: dict[str, float]) -> None:
    proliferation_score = sum(marker_map.get(gene, 0.0) for gene in _BREAST_PROLIFERATION_GENES if gene in marker_map)
    luminal_score = sum(marker_map.get(gene, 0.0) for gene in _BREAST_LUMINAL_GENES if gene in marker_map)
    secretory_score = sum(marker_map.get(gene, 0.0) for gene in _BREAST_SECRETORY_GENES if gene in marker_map)
    basal_tumor_score = sum(marker_map.get(gene, 0.0) for gene in _BREAST_BASAL_TUMOR_GENES if gene in marker_map)
    myo_score = sum(marker_map.get(gene, 0.0) for gene in _BREAST_MYO_GENES if gene in marker_map)
    endothelial_score = sum(marker_map.get(gene, 0.0) for gene in _BREAST_ENDOTHELIAL_GENES if gene in marker_map)
    pericyte_score = sum(marker_map.get(gene, 0.0) for gene in _BREAST_PERICYTE_GENES if gene in marker_map)
    macrophage_score = sum(marker_map.get(gene, 0.0) for gene in _BREAST_MACROPHAGE_GENES if gene in marker_map)
    lymphocyte_score = sum(marker_map.get(gene, 0.0) for gene in _BREAST_LYMPHOCYTE_GENES if gene in marker_map)
    fibroblast_score = sum(marker_map.get(gene, 0.0) for gene in _BREAST_FIBROBLAST_GENES if gene in marker_map)

    if proliferation_score >= 3.0:
        scores["proliferating_tumor_cells"] += 1.4
    if luminal_score >= 1.6:
        scores["er_positive_luminal_tumor"] += 1.0
    if secretory_score >= 1.8:
        scores["secretory_luminal_tumor"] += 1.1
    if basal_tumor_score >= 1.8:
        scores["basal_like_tumor"] += 1.2
    if myo_score >= 1.6:
        scores["basal_myoepithelial_cells"] += 1.1
    if endothelial_score >= 1.4:
        scores["endothelial_cells"] += 1.2
    if pericyte_score >= 1.4:
        scores["pericytes_myofibroblasts"] += 1.1
    if macrophage_score >= 1.4:
        scores["macrophages_myeloid"] += 1.2
    if lymphocyte_score >= 1.3:
        scores["lymphocytes_mixed"] += 1.0
    if fibroblast_score >= 1.4:
        scores["fibroblasts_stromal"] += 1.2

    if "FOXC1" in marker_map and "WFDC2" in marker_map:
        scores["basal_like_tumor"] += 0.8
    if "ESR1" in marker_map and "GATA3" in marker_map:
        scores["er_positive_luminal_tumor"] += 0.8
    if "SCGB2A2" in marker_map and "OPRPN" in marker_map:
        scores["secretory_luminal_tumor"] += 0.8


def _apply_rule_adjustments(scores: dict[str, float], marker_map: dict[str, float], *, taxonomy_name: str) -> None:
    if taxonomy_name == "breast":
        _apply_breast_rule_adjustments(scores, marker_map)
        return
    _apply_lung_rule_adjustments(scores, marker_map)


def _confidence(best_score: float, second_score: float, hit_count: int) -> float:
    margin = max(best_score - second_score, 0.0)
    raw = 0.35 + 0.08 * min(best_score, 6.0) + 0.05 * min(hit_count, 6) + 0.06 * min(margin, 4.0)
    return max(0.2, min(0.98, raw))


def _review_priority(label_id: str, confidence: float, margin: float) -> str:
    if confidence < 0.6 or margin < 0.35:
        return "high"
    if label_id in {"tumor_high_heterogeneity", "proliferating_cells", "proliferating_tumor_or_immune"}:
        return "high"
    if confidence < 0.72 or margin < 0.75:
        return "medium"
    return "low"


def annotate_cluster_heuristically(cluster_evidence: dict[str, Any], *, taxonomy_name: str = "lung") -> dict[str, Any]:
    taxonomy_name = normalize_taxonomy_name(taxonomy_name)
    label_specs = get_label_specs(taxonomy_name)
    label_by_id = get_label_by_id(taxonomy_name)
    marker_map = _positive_marker_map(cluster_evidence)
    scores = {spec.id: _score_spec(spec, marker_map) for spec in label_specs}
    _apply_rule_adjustments(scores, marker_map, taxonomy_name=taxonomy_name)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_id, best_score = ranked[0]
    second_id, second_score = ranked[1]
    best_spec = label_by_id[best_id]
    best_hits = _support_hits(best_spec, marker_map)
    confidence = _confidence(best_score, second_score, len(best_hits))
    review_priority = _review_priority(best_id, confidence, best_score - second_score)

    alternative_ids = [label_id for label_id, score in ranked[1:4] if score > 0]
    conflicting_markers: list[str] = []
    for alternative_id in alternative_ids:
        alternative_hits = _support_hits(label_by_id[alternative_id], marker_map)
        for gene in alternative_hits:
            if gene not in best_hits and gene not in conflicting_markers:
                conflicting_markers.append(gene)
    if not conflicting_markers:
        conflicting_markers = [str(marker.get("gene")) for marker in cluster_evidence.get("top_negative_markers", [])[:4]]

    tumor_evidence = [
        gene
        for gene in best_hits
        if gene in _TUMOR_GENES or gene in {"ASCL1", "CALCA", "LAMP3", "ABCA3"}
    ]
    if taxonomy_name == "breast":
        tumor_evidence = [
            gene
            for gene in best_hits
            if gene in (_BREAST_LUMINAL_GENES | _BREAST_SECRETORY_GENES | _BREAST_BASAL_TUMOR_GENES | _BREAST_PROLIFERATION_GENES)
        ]
        tumor_label_ids = _BREAST_TUMOR_LABEL_IDS
    else:
        tumor_label_ids = _TUMOR_LABEL_IDS

    if best_id in tumor_label_ids and not tumor_evidence:
        tumor_evidence = best_hits[:3]

    reasoning_summary = (
        f"Cluster {cluster_evidence['cluster_id']} matches {best_spec.label} based on "
        f"{', '.join(best_hits[:4]) or 'limited canonical markers'}."
    )
    if alternative_ids:
        reasoning_summary += f" Main alternative: {label_by_id[alternative_ids[0]].label}."

    return {
        "cluster_id": int(cluster_evidence["cluster_id"]),
        "label_id": best_spec.id,
        "detailed_label": best_spec.label,
        "broad_family": best_spec.broad_family,
        "malignancy_state": best_spec.malignancy_state,
        "confidence": round(confidence, 3),
        "supporting_markers": best_hits[:6] or [str(marker.get("gene")) for marker in cluster_evidence.get("top_positive_markers", [])[:6]],
        "conflicting_markers": conflicting_markers[:6],
        "alternative_label_ids": alternative_ids,
        "alternative_labels": [label_by_id[label_id].label for label_id in alternative_ids],
        "reasoning_summary": reasoning_summary,
        "review_priority": review_priority,
        "tumor_evidence": tumor_evidence[:6],
        "recommended_follow_up": (
            ["Manual review recommended because the marker program is ambiguous."]
            if review_priority == "high"
            else []
        ),
        "downstream_cell_type": best_spec.label,
        "engine": "heuristic",
        "prompt_version": f"heuristic-{taxonomy_name}-v1",
    }
