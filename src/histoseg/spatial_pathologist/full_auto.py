from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from histoseg.annotation import ClusterAnnotationPipelineConfig, run_cluster_annotation_pipeline
from histoseg.annotation.evidence_pack import (
    infer_differential_expression_csv,
    infer_projection_csv,
)

from .artifact_loader import ensure_base_pipeline_outputs, load_base_pipeline_config
from .config import SpatialPathologistConfig
from .runner import run_spatial_pathologist


@dataclass(frozen=True)
class FullAutoSpatialPathologistConfig:
    case_name: str
    study_context: str
    base_pipeline_config: Path
    output_root: Path
    annotation_taxonomy: str = "lung"

    pathology_review_backend: str = "openai"
    pathology_ai_api_base_url: str = "http://127.0.0.1:8000"
    pathology_ai_top_k: int = 6
    pathology_ai_answer_language: str = "en"
    pathology_ai_document_ids: tuple[str, ...] = ()

    differential_expression_csv: Path | None = None
    projection_csv: Path | None = None

    openai_enabled: bool = True
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_model: str = "gpt-5.4"
    openai_reasoning_effort: str = "medium"
    openai_store: bool = False

    force_recompute_annotation: bool = False
    force_recompute_pipeline: bool = False

    top_positive_markers: int = 15
    top_negative_markers: int = 6
    min_log2fc: float = 0.5
    max_adjusted_p_value: float = 0.05
    top_neighbors: int = 5

    low_confidence_threshold: float = 0.65
    ambiguity_margin_threshold: float = 0.08
    top_clusters_per_structure: int = 8


def _resolve_path(value: str | Path | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_full_auto_spatial_pathologist_config(path: str | Path) -> FullAutoSpatialPathologistConfig:
    config_path = Path(path).resolve()
    payload: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    return FullAutoSpatialPathologistConfig(
        case_name=str(payload["case_name"]),
        study_context=str(payload["study_context"]),
        base_pipeline_config=_resolve_path(payload["base_pipeline_config"], config_path.parent),
        output_root=_resolve_path(payload["output_root"], config_path.parent),
        annotation_taxonomy=str(payload.get("annotation_taxonomy", "lung")),
        pathology_review_backend=str(payload.get("pathology_review_backend", "openai")),
        pathology_ai_api_base_url=str(payload.get("pathology_ai_api_base_url", "http://127.0.0.1:8000")),
        pathology_ai_top_k=int(payload.get("pathology_ai_top_k", 6)),
        pathology_ai_answer_language=str(payload.get("pathology_ai_answer_language", "en")),
        pathology_ai_document_ids=tuple(str(item) for item in payload.get("pathology_ai_document_ids", [])),
        differential_expression_csv=_resolve_path(payload.get("differential_expression_csv"), config_path.parent),
        projection_csv=_resolve_path(payload.get("projection_csv"), config_path.parent),
        openai_enabled=bool(payload.get("openai_enabled", True)),
        openai_api_key_env=str(payload.get("openai_api_key_env", "OPENAI_API_KEY")),
        openai_model=str(payload.get("openai_model", "gpt-5.4")),
        openai_reasoning_effort=str(payload.get("openai_reasoning_effort", "medium")),
        openai_store=bool(payload.get("openai_store", False)),
        force_recompute_annotation=bool(payload.get("force_recompute_annotation", False)),
        force_recompute_pipeline=bool(payload.get("force_recompute_pipeline", False)),
        top_positive_markers=int(payload.get("top_positive_markers", 15)),
        top_negative_markers=int(payload.get("top_negative_markers", 6)),
        min_log2fc=float(payload.get("min_log2fc", 0.5)),
        max_adjusted_p_value=float(payload.get("max_adjusted_p_value", 0.05)),
        top_neighbors=int(payload.get("top_neighbors", 5)),
        low_confidence_threshold=float(payload.get("low_confidence_threshold", 0.65)),
        ambiguity_margin_threshold=float(payload.get("ambiguity_margin_threshold", 0.08)),
        top_clusters_per_structure=int(payload.get("top_clusters_per_structure", 8)),
    )


def _build_runtime_base_config(
    base_cfg: dict[str, Any],
    *,
    annotation_csv: Path,
    output_root: Path,
) -> dict[str, Any]:
    runtime_cfg = dict(base_cfg)
    runtime_cfg["cluster_annotation_csv"] = annotation_csv.resolve()
    runtime_cfg["analysis_output_dir"] = (output_root / "pipeline" / "structure_assignment").resolve()
    runtime_cfg["validation_output_dir"] = (output_root / "pipeline" / "validation").resolve()
    runtime_cfg["config_path"] = (output_root / "runtime_configs" / "generated_runtime_config.json").resolve()
    return runtime_cfg


def _write_runtime_config(runtime_cfg: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for key, value in runtime_cfg.items():
        if key in {"project_root"}:
            continue
        payload[key] = str(value) if isinstance(value, Path) else value
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _should_recompute_pipeline(
    *,
    annotation_csv: Path,
    output_root: Path,
    force_recompute_pipeline: bool,
) -> bool:
    if force_recompute_pipeline:
        return True

    structure_assignments = output_root / "pipeline" / "structure_assignment" / "structure_assignments.csv"
    validation_summary = output_root / "pipeline" / "validation" / "xenium_explorer_annotations_summary.csv"
    if not structure_assignments.exists() or not validation_summary.exists():
        return True

    annotation_mtime = annotation_csv.stat().st_mtime
    structure_mtime = structure_assignments.stat().st_mtime
    validation_mtime = validation_summary.stat().st_mtime
    return annotation_mtime > min(structure_mtime, validation_mtime)


def run_full_auto_spatial_pathologist(cfg: FullAutoSpatialPathologistConfig) -> dict[str, str]:
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_cfg = load_base_pipeline_config(cfg.base_pipeline_config)
    differential_expression_csv = Path(
        cfg.differential_expression_csv or infer_differential_expression_csv(base_cfg["cluster_csv"])
    )
    projection_csv = Path(
        cfg.projection_csv or infer_projection_csv(base_cfg["dataset_root"])
    )

    annotation_outputs = run_cluster_annotation_pipeline(
        ClusterAnnotationPipelineConfig(
            case_name=cfg.case_name,
            study_context=cfg.study_context,
            cluster_csv=Path(base_cfg["cluster_csv"]),
            differential_expression_csv=differential_expression_csv,
            projection_csv=projection_csv,
            output_dir=output_root / "annotation",
            annotation_taxonomy=cfg.annotation_taxonomy,
            openai_enabled=cfg.openai_enabled,
            openai_api_key_env=cfg.openai_api_key_env,
            openai_model=cfg.openai_model,
            openai_reasoning_effort=cfg.openai_reasoning_effort,
            openai_store=cfg.openai_store,
            force_recompute=cfg.force_recompute_annotation,
            top_positive_markers=cfg.top_positive_markers,
            top_negative_markers=cfg.top_negative_markers,
            min_log2fc=cfg.min_log2fc,
            max_adjusted_p_value=cfg.max_adjusted_p_value,
            top_neighbors=cfg.top_neighbors,
        )
    )

    runtime_cfg = _build_runtime_base_config(
        base_cfg,
        annotation_csv=Path(annotation_outputs["compatibility_csv"]),
        output_root=output_root,
    )
    runtime_cfg_path = _write_runtime_config(
        runtime_cfg,
        output_root / "runtime_configs" / "generated_runtime_config.json",
    )

    recompute_pipeline = _should_recompute_pipeline(
        annotation_csv=Path(annotation_outputs["compatibility_csv"]),
        output_root=output_root,
        force_recompute_pipeline=cfg.force_recompute_pipeline,
    )

    ensure_base_pipeline_outputs(
        SpatialPathologistConfig(
            case_name=cfg.case_name,
            study_context=cfg.study_context,
            base_pipeline_config=runtime_cfg_path,
            output_dir=output_root / "pathology_review",
            pathology_review_backend=cfg.pathology_review_backend,
            pathology_ai_api_base_url=cfg.pathology_ai_api_base_url,
            pathology_ai_top_k=cfg.pathology_ai_top_k,
            pathology_ai_answer_language=cfg.pathology_ai_answer_language,
            pathology_ai_document_ids=cfg.pathology_ai_document_ids,
            openai_enabled=cfg.openai_enabled,
            openai_api_key_env=cfg.openai_api_key_env,
            openai_model=cfg.openai_model,
            openai_reasoning_effort=cfg.openai_reasoning_effort,
            openai_store=cfg.openai_store,
            force_recompute_pipeline=recompute_pipeline,
            low_confidence_threshold=cfg.low_confidence_threshold,
            ambiguity_margin_threshold=cfg.ambiguity_margin_threshold,
            top_clusters_per_structure=cfg.top_clusters_per_structure,
        )
    )

    pathology_outputs = run_spatial_pathologist(
        SpatialPathologistConfig(
            case_name=cfg.case_name,
            study_context=cfg.study_context,
            base_pipeline_config=runtime_cfg_path,
            output_dir=output_root / "pathology_review",
            pathology_review_backend=cfg.pathology_review_backend,
            pathology_ai_api_base_url=cfg.pathology_ai_api_base_url,
            pathology_ai_top_k=cfg.pathology_ai_top_k,
            pathology_ai_answer_language=cfg.pathology_ai_answer_language,
            pathology_ai_document_ids=cfg.pathology_ai_document_ids,
            openai_enabled=cfg.openai_enabled,
            openai_api_key_env=cfg.openai_api_key_env,
            openai_model=cfg.openai_model,
            openai_reasoning_effort=cfg.openai_reasoning_effort,
            openai_store=cfg.openai_store,
            force_recompute_pipeline=False,
            low_confidence_threshold=cfg.low_confidence_threshold,
            ambiguity_margin_threshold=cfg.ambiguity_margin_threshold,
            top_clusters_per_structure=cfg.top_clusters_per_structure,
        )
    )

    workflow_summary = {
        "case_name": cfg.case_name,
        "study_context": cfg.study_context,
        "output_root": str(output_root),
        "annotation_outputs": annotation_outputs,
        "runtime_base_pipeline_config": str(runtime_cfg_path),
        "pathology_outputs": pathology_outputs,
        "responses_api_ready": True,
        "annotation_taxonomy": cfg.annotation_taxonomy,
        "pathology_review_backend": cfg.pathology_review_backend,
        "pathology_ai_api_base_url": cfg.pathology_ai_api_base_url,
        "pathology_ai_top_k": cfg.pathology_ai_top_k,
        "pathology_ai_answer_language": cfg.pathology_ai_answer_language,
        "pathology_ai_document_ids": list(cfg.pathology_ai_document_ids),
        "pipeline_recomputed": recompute_pipeline,
        "openai_api_key_env": cfg.openai_api_key_env,
        "openai_model": cfg.openai_model,
        "openai_reasoning_effort": cfg.openai_reasoning_effort,
        "openai_store": cfg.openai_store,
    }
    (output_root / "workflow_summary.json").write_text(
        json.dumps(workflow_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "output_root": str(output_root),
        "annotation_report_html": annotation_outputs["report_html"],
        "annotation_csv": annotation_outputs["compatibility_csv"],
        "runtime_base_pipeline_config": str(runtime_cfg_path),
        "pathology_report_html": pathology_outputs["report_html"],
        "workflow_summary_json": str(output_root / "workflow_summary.json"),
    }
