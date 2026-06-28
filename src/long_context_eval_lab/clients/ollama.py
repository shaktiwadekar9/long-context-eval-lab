from __future__ import annotations

import json
import time
from typing import Any

import requests

from long_context_eval_lab.clients.base import ModelClient
from long_context_eval_lab.schemas import ModelResponse


class OllamaClient(ModelClient):
    """Ollama implementation using the local REST API.

    Uses `/api/chat` with `stream=false`. We request JSON output using
    `format="json"`, but the parser still has a fallback because local models
    can sometimes return extra text around JSON.
    """

    provider_name = "ollama"

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 300,
        force_json: bool = True,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.force_json = force_json

    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.force_json:
            payload["format"] = "json"

        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Ollama request failed. Check that Ollama is running and the model is pulled. "
                f"Base URL: {self.base_url}, model: {self.model_name}. Original error: {exc}"
            ) from exc

        latency = time.perf_counter() - started
        data = response.json()
        raw_text = data.get("message", {}).get("content", "")
        parsed = self._parse_model_json(raw_text)

        return ModelResponse(
            answer=str(parsed.get("answer", raw_text)).strip(),
            evidence=self._normalize_evidence(parsed.get("evidence", [])),
            confidence=self._normalize_confidence(parsed.get("confidence")),
            raw_text=raw_text,
            latency_seconds=latency,
            provider_metadata={
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
                "done_reason": data.get("done_reason"),
            },
        )

    @staticmethod
    def _parse_model_json(raw_text: str) -> dict[str, Any]:
        try:
            value = json.loads(raw_text)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

        # Fallback for models that wrap JSON in prose.
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                value = json.loads(raw_text[start : end + 1])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass

        return {"answer": raw_text, "evidence": [], "confidence": None}

    @staticmethod
    def _normalize_evidence(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _normalize_confidence(value: Any) -> float | None:
        try:
            if value is None:
                return None
            number = float(value)
            return max(0.0, min(1.0, number))
        except (TypeError, ValueError):
            return None
