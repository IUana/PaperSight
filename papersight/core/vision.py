import base64
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from papersight.core.prompts import FIGURE_PROMPT_TEMPLATE


@dataclass(frozen=True)
class FigureAnalysis:
    figure_type: str
    is_pipeline: bool
    confidence: float
    steps: list[str]
    components: list[str]
    relations: list[str]
    summary: str


class OllamaFigureAnalyzer:
    def __init__(self, model: str, api_url: str = "http://127.0.0.1:11434/api/chat") -> None:
        self.model = model
        self.api_url = api_url

    def analyze(self, image_bytes: bytes, mime_type: str, caption: str | None = None) -> FigureAnalysis:
        if not image_bytes:
            return self._fallback_analysis(caption)

        prompt = FIGURE_PROMPT_TEMPLATE
        if caption:
            prompt = f"{prompt}\n\n参考图注：{caption}"

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64],
                }
            ],
        }

        try:
            response_payload = self._post_json(payload)
            content = (
                response_payload.get("message", {}).get("content")
                if isinstance(response_payload, dict)
                else ""
            )
            if not isinstance(content, str):
                return self._fallback_analysis(caption)
            parsed = self._parse_content(content)
            if parsed is None:
                return self._fallback_analysis(caption)
            return parsed
        except (HTTPError, URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError):
            return self._fallback_analysis(caption)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self.api_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            response_text = response.read().decode("utf-8")
        parsed = json.loads(response_text)
        if not isinstance(parsed, dict):
            raise ValueError("Invalid Ollama response format")
        return parsed

    def _parse_content(self, content: str) -> FigureAnalysis | None:
        json_blob = self._extract_json_blob(content)
        if not json_blob:
            return None

        payload = json.loads(json_blob)
        if not isinstance(payload, dict):
            return None

        figure_type = str(payload.get("figure_type", "other"))
        is_pipeline = bool(payload.get("is_pipeline", False))
        confidence = self._normalize_confidence(payload.get("confidence", 0.0))
        steps = self._normalize_string_list(payload.get("steps"))
        components = self._normalize_string_list(payload.get("components"))
        relations = self._normalize_string_list(payload.get("relations"))
        summary = str(payload.get("summary", "")).strip()

        if not summary:
            summary = "图像解析未返回足够文字信息。"

        return FigureAnalysis(
            figure_type=figure_type,
            is_pipeline=is_pipeline,
            confidence=confidence,
            steps=steps,
            components=components,
            relations=relations,
            summary=summary,
        )

    def _extract_json_blob(self, content: str) -> str | None:
        code_fence = re.search(r"```json\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
        if code_fence:
            return code_fence.group(1)

        generic_fence = re.search(r"```\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
        if generic_fence:
            return generic_fence.group(1)

        plain_json = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if plain_json:
            return plain_json.group(0)
        return None

    def _normalize_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))

    def _normalize_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized[:24]

    def _fallback_analysis(self, caption: str | None) -> FigureAnalysis:
        lowered = (caption or "").lower()
        hint_pipeline = any(token in lowered for token in ["pipeline", "workflow", "framework", "流程", "步骤"])
        summary = caption.strip() if caption else "图像解析失败，仅保留占位信息。"

        return FigureAnalysis(
            figure_type="pipeline" if hint_pipeline else "other",
            is_pipeline=hint_pipeline,
            confidence=0.0,
            steps=[],
            components=[],
            relations=[],
            summary=summary,
        )
