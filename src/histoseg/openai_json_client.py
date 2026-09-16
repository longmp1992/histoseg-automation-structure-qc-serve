from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import os
import re


try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


@dataclass(frozen=True)
class OpenAISettings:
    enabled: bool
    api_key_env: str
    model: str
    reasoning_effort: str
    store: bool = False

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.api_key and OpenAI is not None)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(stripped[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Could not parse JSON object from model output: {text[:300]!r}")


class OpenAIJsonClient:
    def __init__(self, settings: OpenAISettings) -> None:
        if not settings.available:
            raise RuntimeError(
                "OpenAI client is unavailable. Install the `openai` package and set the API key env var."
            )
        self.settings = settings
        self.client = OpenAI(api_key=settings.api_key)

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "store": self.settings.store,
        }
        if self.settings.reasoning_effort:
            request_kwargs["reasoning"] = {"effort": self.settings.reasoning_effort}
        if schema is not None:
            request_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name or "structured_response",
                    "schema": schema,
                    "strict": True,
                },
                "verbosity": "low",
            }
        else:
            request_kwargs["text"] = {
                "format": {"type": "json_object"},
                "verbosity": "low",
            }

        response = self.client.responses.create(**request_kwargs)
        output_text = getattr(response, "output_text", "") or ""
        if not output_text:
            raise ValueError("Model returned no output_text.")
        return extract_json_object(output_text)
