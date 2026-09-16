from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class SpatialPathologistConfig:
    case_name: str
    study_context: str
    base_pipeline_config: Path
    output_dir: Path

    pathology_review_backend: str = "openai"
    pathology_ai_api_base_url: str = "http://127.0.0.1:8000"
    pathology_ai_top_k: int = 6
    pathology_ai_answer_language: str = "en"
    pathology_ai_document_ids: tuple[str, ...] = ()

    openai_enabled: bool = True
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_model: str = "gpt-5.4"
    openai_reasoning_effort: str = "medium"
    openai_store: bool = False

    force_recompute_pipeline: bool = False
    low_confidence_threshold: float = 0.65
    ambiguity_margin_threshold: float = 0.08
    top_clusters_per_structure: int = 8


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_spatial_pathologist_config(path: str | Path) -> SpatialPathologistConfig:
    config_path = Path(path).resolve()
    payload: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))

    return SpatialPathologistConfig(
        case_name=str(payload["case_name"]),
        study_context=str(payload["study_context"]),
        base_pipeline_config=_resolve_path(payload["base_pipeline_config"], config_path.parent),
        output_dir=_resolve_path(payload["output_dir"], config_path.parent),
        pathology_review_backend=str(payload.get("pathology_review_backend", "openai")),
        pathology_ai_api_base_url=str(payload.get("pathology_ai_api_base_url", "http://127.0.0.1:8000")),
        pathology_ai_top_k=int(payload.get("pathology_ai_top_k", 6)),
        pathology_ai_answer_language=str(payload.get("pathology_ai_answer_language", "en")),
        pathology_ai_document_ids=tuple(str(item) for item in payload.get("pathology_ai_document_ids", [])),
        openai_enabled=bool(payload.get("openai_enabled", True)),
        openai_api_key_env=str(payload.get("openai_api_key_env", "OPENAI_API_KEY")),
        openai_model=str(payload.get("openai_model", "gpt-5.4")),
        openai_reasoning_effort=str(payload.get("openai_reasoning_effort", "medium")),
        openai_store=bool(payload.get("openai_store", False)),
        force_recompute_pipeline=bool(payload.get("force_recompute_pipeline", False)),
        low_confidence_threshold=float(payload.get("low_confidence_threshold", 0.65)),
        ambiguity_margin_threshold=float(payload.get("ambiguity_margin_threshold", 0.08)),
        top_clusters_per_structure=int(payload.get("top_clusters_per_structure", 8)),
    )
