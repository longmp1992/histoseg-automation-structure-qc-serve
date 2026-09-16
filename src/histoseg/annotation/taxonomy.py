from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClusterLabelSpec:
    id: str
    label: str
    broad_family: str
    malignancy_state: str
    description: str
    marker_genes: tuple[str, ...]
    negative_markers: tuple[str, ...] = ()


LUNG_LABEL_SPECS: tuple[ClusterLabelSpec, ...] = (
    ClusterLabelSpec(
        id="cd4_t_cells",
        label="CD4+ T cells",
        broad_family="immune",
        malignancy_state="non_tumor",
        description="Helper T-cell program with T-cell receptor signaling and activation markers.",
        marker_genes=("ZAP70", "CCR4", "CD40LG", "CD2", "ITK", "IL7R", "LTB"),
        negative_markers=("GZMB", "NCR1", "MKI67"),
    ),
    ClusterLabelSpec(
        id="classical_monocytes_macrophages",
        label="Classical Monocytes / Macrophages",
        broad_family="immune",
        malignancy_state="non_tumor",
        description="Inflammatory monocyte/macrophage compartment.",
        marker_genes=("FCN1", "CD300E", "LILRB2", "CLEC10A", "IL1B", "F13A1", "CTSB"),
    ),
    ClusterLabelSpec(
        id="capillary_endothelial_cells",
        label="Capillary Endothelial Cells",
        broad_family="vascular",
        malignancy_state="non_tumor",
        description="Normal capillary endothelial identity.",
        marker_genes=("CA4", "CLDN5", "TMEM100", "ROBO4", "NOS3", "EMCN", "RGCC"),
    ),
    ClusterLabelSpec(
        id="caf",
        label="Cancer-Associated Fibroblasts (CAF)",
        broad_family="stromal",
        malignancy_state="microenvironment",
        description="Activated fibroblast program with matrix remodeling markers.",
        marker_genes=("FAP", "LRRC15", "COL11A1", "COL10A1", "MMP11", "COMP", "POSTN"),
    ),
    ClusterLabelSpec(
        id="smooth_muscle",
        label="Cardiomyocytes / Smooth Muscle",
        broad_family="stromal",
        malignancy_state="non_tumor",
        description="Contractile mural or smooth-muscle program.",
        marker_genes=("CASQ2", "PLN", "FLNC", "LDB3", "JPH2", "MYH11", "ACTA2", "TAGLN"),
    ),
    ClusterLabelSpec(
        id="neutrophils",
        label="Neutrophils",
        broad_family="immune",
        malignancy_state="non_tumor",
        description="Granulocytic neutrophil program.",
        marker_genes=("CXCR1", "CXCR2", "DEFA1", "PADI4", "CEACAM3", "FCGR3B", "S100A8"),
    ),
    ClusterLabelSpec(
        id="nk_nkt_cells",
        label="NK / NKT Cells",
        broad_family="immune",
        malignancy_state="non_tumor",
        description="Cytotoxic NK or NKT-like lymphocytes.",
        marker_genes=("PRF1", "KLRD1", "KLRF1", "GZMH", "KIR2DL1", "NCR1", "NKG7"),
        negative_markers=("FOXP3", "CCR4"),
    ),
    ClusterLabelSpec(
        id="tumor_adenocarcinoma",
        label="Tumor Cells (Adenocarcinoma)",
        broad_family="epithelial",
        malignancy_state="tumor",
        description="Pulmonary adenocarcinoma-like epithelial tumor program.",
        marker_genes=("FOLR1", "LAPTM4B", "HOXB13", "NQO1", "HHLA2", "KRT19", "MUC1"),
    ),
    ClusterLabelSpec(
        id="tumor_high_heterogeneity",
        label="Tumor Cells (high heterogeneity)",
        broad_family="epithelial",
        malignancy_state="tumor",
        description="Heterogeneous tumor program with oncofetal or atypical markers.",
        marker_genes=("SSX2", "SALL4", "CRNDE", "CYP3A5", "ABCC3", "PRAME", "ACE2"),
    ),
    ClusterLabelSpec(
        id="adventitial_fibroblasts",
        label="Adventitial Fibroblasts",
        broad_family="stromal",
        malignancy_state="microenvironment",
        description="Adventitial or matrix-producing fibroblast program.",
        marker_genes=("VEGFD", "SLIT2", "SVEP1", "FGFR4", "CCBE1", "COL15A1", "DCN"),
    ),
    ClusterLabelSpec(
        id="b_cells",
        label="B Cells (naive/mature)",
        broad_family="immune",
        malignancy_state="non_tumor",
        description="B-cell lineage markers.",
        marker_genes=("MS4A1", "CD19", "PAX5", "CD22", "BLK", "BANK1", "CD79A"),
    ),
    ClusterLabelSpec(
        id="lymphatic_endothelial",
        label="Lymphatic Endothelial Cells",
        broad_family="vascular",
        malignancy_state="non_tumor",
        description="Lymphatic endothelial identity.",
        marker_genes=("PLVAP", "FLT4", "SELE", "SELP", "ANGPT2", "SOX18", "PROX1"),
    ),
    ClusterLabelSpec(
        id="dendritic_cells",
        label="Dendritic Cells (cDC2 / Langerhans)",
        broad_family="immune",
        malignancy_state="non_tumor",
        description="Antigen-presenting dendritic cell program.",
        marker_genes=("CD207", "CD1A", "CD1C", "FCER1A", "IDO1", "CXCL10", "HLA-DRA"),
    ),
    ClusterLabelSpec(
        id="plasma_cells",
        label="Plasma Cells",
        broad_family="immune",
        malignancy_state="non_tumor",
        description="Antibody-secreting plasma cell program.",
        marker_genes=("MZB1", "TNFRSF17", "CD38", "CD79A", "IGHE", "SLAMF7", "JCHAIN"),
    ),
    ClusterLabelSpec(
        id="tam",
        label="Tumor-Associated Macrophages (TAM)",
        broad_family="immune",
        malignancy_state="microenvironment",
        description="Macrophage state associated with tumor microenvironment remodeling.",
        marker_genes=("MARCO", "TREM2", "MMP12", "MSR1", "CCL18", "CHIT1", "SPP1"),
    ),
    ClusterLabelSpec(
        id="pericytes_mural",
        label="Pericytes / Mural Cells",
        broad_family="stromal",
        malignancy_state="non_tumor",
        description="Pericyte or mural-cell identity.",
        marker_genes=("SOX10", "NOTCH3", "SFRP1", "ANGPTL7", "L1CAM", "RGS5", "MCAM"),
    ),
    ClusterLabelSpec(
        id="at1_mesothelial",
        label="AT1 / Mesothelial Cells",
        broad_family="epithelial",
        malignancy_state="non_tumor",
        description="Alveolar type I program with mesothelial features.",
        marker_genes=("AGER", "CLDN18", "UPK3B", "MSLN", "AQP4", "RTKN2", "CAV1"),
    ),
    ClusterLabelSpec(
        id="basal_squamous",
        label="Basal / Squamous Epithelial Cells",
        broad_family="epithelial",
        malignancy_state="uncertain",
        description="Basal or squamous epithelial program that may reflect reactive or tumor-associated cells.",
        marker_genes=("TP63", "SERPINB4", "SERPINB7", "MUC5B", "COL7A1", "KRT5", "KRT14"),
    ),
    ClusterLabelSpec(
        id="at2_like_tumor",
        label="AT2-like Tumor Cells",
        broad_family="epithelial",
        malignancy_state="tumor",
        description="Alveolar type II-like tumor epithelial program.",
        marker_genes=("LAMP3", "LRP2", "PCSK9", "BMP2", "ALPL", "ABCA3", "NAPSA"),
    ),
    ClusterLabelSpec(
        id="treg",
        label="Regulatory T Cells (Treg)",
        broad_family="immune",
        malignancy_state="non_tumor",
        description="Regulatory T-cell program.",
        marker_genes=("FOXP3", "CCL19", "CD2", "BCL11B", "KLRB1", "CCR4", "IL2RA"),
        negative_markers=("GZMB", "NCR1"),
    ),
    ClusterLabelSpec(
        id="venous_endothelial",
        label="Venous Endothelial Cells",
        broad_family="vascular",
        malignancy_state="non_tumor",
        description="Venous endothelial identity.",
        marker_genes=("CA4", "NOTCH4", "SOX17", "TEK", "IL6", "SLC6A4", "VWF"),
    ),
    ClusterLabelSpec(
        id="at1_cells",
        label="AT1 Cells (Alveolar Type 1)",
        broad_family="epithelial",
        malignancy_state="non_tumor",
        description="Alveolar type I epithelial program.",
        marker_genes=("AGER", "CLDN18", "AQP4", "MSLN", "ELSPBP1", "UPK3B", "CAV1"),
    ),
    ClusterLabelSpec(
        id="pericytes_arterial",
        label="Pericytes (arterial)",
        broad_family="stromal",
        malignancy_state="non_tumor",
        description="Arterial pericyte or mural-cell subtype.",
        marker_genes=("SOX10", "NOTCH3", "KCNT1", "GJC1", "SFRP5", "ADRA1D", "MCAM"),
    ),
    ClusterLabelSpec(
        id="lymphatic_tip_endothelial",
        label="Lymphatic / Tip Endothelial Cells",
        broad_family="vascular",
        malignancy_state="non_tumor",
        description="Tip-like or lymphatic endothelial program.",
        marker_genes=("ANGPT2", "FLT4", "ESM1", "CLDN5", "PROX1", "KDR", "PLVAP"),
    ),
    ClusterLabelSpec(
        id="ciliated_epithelial",
        label="Ciliated Epithelial Cells",
        broad_family="epithelial",
        malignancy_state="non_tumor",
        description="Ciliated bronchiolar epithelial program.",
        marker_genes=("DNAI1", "DNAH2", "DNAH9", "DNAH12", "CFAP46", "CFAP47", "FOXJ1"),
    ),
    ClusterLabelSpec(
        id="mast_cells",
        label="Mast Cells",
        broad_family="immune",
        malignancy_state="non_tumor",
        description="Mast-cell lineage program.",
        marker_genes=("HDC", "MS4A2", "CMA1", "CTSG", "KIT", "GATA2", "HPGDS"),
    ),
    ClusterLabelSpec(
        id="ciliated_epithelial_subtype",
        label="Ciliated Epithelial Cells (subtype)",
        broad_family="epithelial",
        malignancy_state="non_tumor",
        description="Subtype of ciliated epithelial cells with strong motile cilia genes.",
        marker_genes=("DNAH9", "SPAG6", "DRC1", "TPPP3", "TEKT1", "ZMYND10", "FOXJ1"),
    ),
    ClusterLabelSpec(
        id="proliferating_cells",
        label="Proliferating Cells (G2/M phase)",
        broad_family="mixed",
        malignancy_state="uncertain",
        description="Cell-cycle dominant cluster without enough lineage certainty.",
        marker_genes=("MKI67", "TOP2A", "CCNB1", "BIRC5", "PLK1", "CDC20", "AURKB"),
    ),
    ClusterLabelSpec(
        id="cd8_exhausted_t_cells",
        label="CD8+ Cytotoxic / Exhausted T Cells",
        broad_family="immune",
        malignancy_state="non_tumor",
        description="Activated cytotoxic or exhausted CD8 T-cell program.",
        marker_genes=("CD8A", "GZMB", "GZMA", "IFNG", "PDCD1", "CXCL13", "FASLG"),
        negative_markers=("FOXP3",),
    ),
    ClusterLabelSpec(
        id="neuroendocrine_tumor_cells",
        label="Neuroendocrine Tumor Cells",
        broad_family="epithelial",
        malignancy_state="tumor",
        description="Pulmonary neuroendocrine or small-cell carcinoma-like tumor program.",
        marker_genes=("ASCL1", "CALCA", "SCG2", "DDC", "TPH1", "DLK1", "SCGN"),
    ),
    ClusterLabelSpec(
        id="proliferating_tumor_or_immune",
        label="Proliferating Tumor / Immune Cells",
        broad_family="mixed",
        malignancy_state="uncertain",
        description="Cell-cycle dominant cluster with mixed tumor and immune evidence.",
        marker_genes=("MKI67", "AURKB", "GZMB", "CCNA2", "TOP2A", "ASPM", "BIRC5"),
    ),
)


BREAST_LABEL_SPECS: tuple[ClusterLabelSpec, ...] = (
    ClusterLabelSpec(
        id="lymphocytes_mixed",
        label="Lymphocytes (T/B mixed)",
        broad_family="immune",
        malignancy_state="non_tumor",
        description="Mixed adaptive lymphocyte program with both T-cell and B-cell markers.",
        marker_genes=("TRAC", "CD3E", "CCL5", "CD8A", "IL7R", "MS4A1", "CD27", "CTLA4"),
        negative_markers=("EPCAM", "COL1A1", "VWF"),
    ),
    ClusterLabelSpec(
        id="fibroblasts_stromal",
        label="Fibroblasts / Stromal Cells",
        broad_family="stromal",
        malignancy_state="microenvironment",
        description="Matrix-rich fibroblastic stroma with CXCL12/SFRP4/LAMA2-like programs.",
        marker_genes=("SFRP4", "CCDC80", "LAMA2", "CXCL12", "CXCL14", "FBLN1", "LOXL1", "C1R"),
        negative_markers=("EPCAM", "VWF", "PTPRC"),
    ),
    ClusterLabelSpec(
        id="macrophages_myeloid",
        label="Macrophages / Myeloid Cells",
        broad_family="immune",
        malignancy_state="microenvironment",
        description="Macrophage-rich myeloid compartment including LYVE1/CD163/APOC1-like states.",
        marker_genes=("CD163", "ITGAX", "LYVE1", "CD68", "LYZ", "HMOX1", "APOC1", "MMP12"),
        negative_markers=("EPCAM", "KRT19"),
    ),
    ClusterLabelSpec(
        id="pericytes_myofibroblasts",
        label="Pericytes / Myofibroblasts",
        broad_family="stromal",
        malignancy_state="microenvironment",
        description="Perivascular mural or myofibroblastic program.",
        marker_genes=("RGS5", "PDGFRB", "MYH11", "ACTA2", "TAGLN", "MYLK", "COL4A1", "HEYL"),
        negative_markers=("EPCAM", "MS4A1"),
    ),
    ClusterLabelSpec(
        id="endothelial_cells",
        label="Endothelial Cells",
        broad_family="vascular",
        malignancy_state="non_tumor",
        description="Vascular endothelial program.",
        marker_genes=("VWF", "CLEC14A", "PECAM1", "PLVAP", "FLT1", "AQP1", "HSPG2", "LEPR"),
        negative_markers=("EPCAM", "KRT19"),
    ),
    ClusterLabelSpec(
        id="basal_myoepithelial_cells",
        label="Basal / Myoepithelial Cells",
        broad_family="epithelial",
        malignancy_state="non_tumor",
        description="Basal and myoepithelial program lining normal ducts or DCIS boundaries.",
        marker_genes=("COL17A1", "TP63", "KRT5", "KRT14", "LAMB3", "KRT17", "ACTA2", "CNN1"),
        negative_markers=("ESR1", "PIP"),
    ),
    ClusterLabelSpec(
        id="secretory_luminal_tumor",
        label="Secretory Luminal Tumor Cells",
        broad_family="epithelial",
        malignancy_state="tumor",
        description="Secretory luminal breast epithelial tumor program with SCGB2A2/OPRPN-like markers.",
        marker_genes=("SCGB2A2", "SCGB2A1", "OPRPN", "CYP4Z1", "EPCAM", "NPY1R", "KRT15", "VEGFA"),
        negative_markers=("COL1A1", "VWF"),
    ),
    ClusterLabelSpec(
        id="er_positive_luminal_tumor",
        label="ER-positive Luminal Tumor Cells",
        broad_family="epithelial",
        malignancy_state="tumor",
        description="Hormone receptor-positive luminal tumor program.",
        marker_genes=("ESR1", "GATA3", "FOXA1", "PIP", "SCUBE2", "ANKRD30A", "AGR3", "AR"),
        negative_markers=("VWF", "COL1A1", "PTPRC"),
    ),
    ClusterLabelSpec(
        id="basal_like_tumor",
        label="Basal-like Tumor Cells",
        broad_family="epithelial",
        malignancy_state="tumor",
        description="Basal-like or aggressive epithelial tumor program enriched for FOXC1/WFDC2/CLDN4-like markers.",
        marker_genes=("WFDC2", "KRT6B", "KRT23", "FOXC1", "CLDN4", "LAMC2", "CEACAM6", "SERPINA3"),
        negative_markers=("ESR1", "SCUBE2"),
    ),
    ClusterLabelSpec(
        id="proliferating_tumor_cells",
        label="Proliferating Tumor Cells",
        broad_family="epithelial",
        malignancy_state="tumor",
        description="Cell-cycle dominant epithelial/tumor compartment.",
        marker_genes=("MKI67", "TOP2A", "CDC20", "CCNB1", "BIRC5", "AURKA", "UBE2C", "CENPF"),
        negative_markers=("COL1A1", "VWF"),
    ),
)


TAXONOMY_REGISTRY: dict[str, tuple[ClusterLabelSpec, ...]] = {
    "lung": LUNG_LABEL_SPECS,
    "breast": BREAST_LABEL_SPECS,
}


def normalize_taxonomy_name(taxonomy_name: str | None) -> str:
    normalized = str(taxonomy_name or "lung").strip().lower().replace("-", "_")
    if normalized not in TAXONOMY_REGISTRY:
        raise ValueError(
            f"Unsupported annotation taxonomy: {taxonomy_name}. "
            f"Expected one of: {', '.join(sorted(TAXONOMY_REGISTRY))}"
        )
    return normalized


def get_label_specs(taxonomy_name: str | None = None) -> tuple[ClusterLabelSpec, ...]:
    return TAXONOMY_REGISTRY[normalize_taxonomy_name(taxonomy_name)]


def get_label_by_id(taxonomy_name: str | None = None) -> dict[str, ClusterLabelSpec]:
    return {spec.id: spec for spec in get_label_specs(taxonomy_name)}


LABEL_SPECS = LUNG_LABEL_SPECS
LABEL_BY_ID = get_label_by_id("lung")


def label_taxonomy_payload(taxonomy_name: str | None = None) -> list[dict[str, object]]:
    return [
        {
            "label_id": spec.id,
            "label": spec.label,
            "broad_family": spec.broad_family,
            "malignancy_state": spec.malignancy_state,
            "description": spec.description,
            "marker_genes": list(spec.marker_genes),
        }
        for spec in get_label_specs(taxonomy_name)
    ]
