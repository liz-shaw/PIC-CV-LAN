from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    english: str
    confidence: float


class ObjectDetector:
    """YOLO object detector wrapper."""

    def __init__(self, model_name: str = "yolo11n.pt", confidence_threshold: float = 0.35) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_name)

    def detect(self, image: Image.Image | str | Path) -> tuple[list[Detection], np.ndarray | None]:
        """Return unique labels with max confidence and an annotated image array."""

        results = self.model(image)
        names: dict[int, str] = self.model.names
        label_to_confidence: dict[str, float] = {}
        annotated: np.ndarray | None = None

        for result in results:
            try:
                annotated = result.plot()
            except Exception:
                annotated = None

            boxes: Any = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue

                label = names[cls_id]
                label_to_confidence[label] = max(label_to_confidence.get(label, 0.0), confidence)

        detections = [
            Detection(english=label, confidence=confidence)
            for label, confidence in sorted(label_to_confidence.items(), key=lambda item: -item[1])
        ]
        return detections, annotated
