from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
from urllib import error, request


@dataclass(frozen=True)
class PathologyAISettings:
    base_url: str = "http://127.0.0.1:8000"
    top_k: int = 6
    answer_language: str = "en"
    document_ids: tuple[str, ...] = ()
    timeout_seconds: float = 60.0

    @property
    def normalized_base_url(self) -> str:
        return str(self.base_url).rstrip("/")

    @property
    def available(self) -> bool:
        return bool(self.normalized_base_url)


class PathologyAIClient:
    def __init__(self, settings: PathologyAISettings) -> None:
        self.settings = settings

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        answer_language: str | None = None,
        document_ids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "top_k": int(top_k or self.settings.top_k),
            "answer_language": str(answer_language or self.settings.answer_language),
            "document_ids": [str(item) for item in (document_ids or self.settings.document_ids)],
            "attachments": [],
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url=f"{self.settings.normalized_base_url}/ask",
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.settings.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"pathology-ai API returned HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"pathology-ai API request failed: {exc.reason}") from exc
