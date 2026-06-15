from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class VisionWord:
    english: str
    chinese: str
    german: str
    confidence: float = 0.60


class OllamaVisionRecognizer:
    """Local vision-language recognizer through Ollama's HTTP API.

    This backend is designed for broad object vocabulary extraction, not precise bounding boxes.
    It requires a local Ollama server and a vision-capable model.
    """

    def __init__(self, model_name: str = "gemma3:4b", host: str = "http://localhost:11434") -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")

    @staticmethod
    def _image_to_base64(image: Image.Image, max_side: int = 1280) -> str:
        image = image.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / longest
            image = image.resize((int(width * scale), int(height * scale)))

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _extract_json_array(text: str) -> list[dict[str, Any]]:
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
                return [item for item in parsed["items"] if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return []

        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            return []

        return []

    @staticmethod
    def _clean_text(value: object) -> str:
        return str(value or "").strip()

    def recognize(self, image: Image.Image, max_items: int = 20) -> tuple[list[VisionWord], str | None]:
        prompt = f"""
You are a careful multilingual vocabulary extractor for language learning.
Look at the image and list the main visible physical objects.
Return JSON only. No markdown. No explanation.

Rules:
- Return at most {max_items} items.
- Prefer concrete nouns, not vague scene descriptions.
- Avoid duplicates.
- If unsure, omit the item.
- English should be lowercase singular noun phrase.
- German nouns must include article: der/die/das.

JSON format:
[
  {{"english": "cup", "chinese": "杯子", "german": "die Tasse"}}
]
""".strip()

        image_b64 = self._image_to_base64(image)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
        }

        request = urllib.request.Request(
            url=f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            return [], f"无法连接 Ollama：{exc}. 请确认 Ollama 正在运行，且模型 {self.model_name} 已安装。"
        except TimeoutError:
            return [], "Ollama 视觉模型响应超时。可以换更小的模型，或减少图片尺寸。"

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return [], "Ollama 返回了非 JSON 响应。"

        content = ""
        message = data.get("message")
        if isinstance(message, dict):
            content = str(message.get("content", ""))
        elif "response" in data:
            content = str(data.get("response", ""))

        items = self._extract_json_array(content)
        words: list[VisionWord] = []
        seen: set[str] = set()

        for item in items:
            english = self._clean_text(item.get("english")).lower()
            chinese = self._clean_text(item.get("chinese")) or "待确认"
            german = self._clean_text(item.get("german")) or "待确认"

            if not english or english in seen:
                continue
            seen.add(english)
            words.append(VisionWord(english=english, chinese=chinese, german=german))

        if not words:
            return [], "Ollama 没有返回可解析的物品 JSON。可以换模型，或重试一张更清晰的图片。"

        return words[:max_items], None
