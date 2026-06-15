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
    count: int = 1


class ObjectDetector:
    """YOLO object detector wrapper.

    The default device is CPU so the app works on servers without NVIDIA GPUs.
    Change device to "cuda:0" only when CUDA is actually available.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
        image_size: int = 640,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.image_size = image_size
        self.model = YOLO(model_name)

    def detect(
        self,
        image: Image.Image | str | Path,
        confidence_threshold: float | None = None,
        image_size: int | None = None,
        unique_labels: bool = True,
        max_detections: int = 100,
    ) -> tuple[list[Detection], np.ndarray | None]:
        """Detect objects from an image.

        Args:
            image: PIL image or image path.
            confidence_threshold: Runtime confidence threshold. Lower values show more boxes but may add false positives.
            image_size: Runtime inference size. Larger values may detect small objects better but run slower on CPU.
            unique_labels: If True, merge duplicate labels and keep a count. If False, return every detected instance.
            max_detections: Maximum number of raw detections returned by YOLO.
        """

        effective_conf = self.confidence_threshold if confidence_threshold is None else confidence_threshold
        effective_imgsz = self.image_size if image_size is None else image_size

        results = self.model.predict(
            source=image,
            conf=effective_conf,
            device=self.device,
            imgsz=effective_imgsz,
            max_det=max_detections,
            verbose=False,
        )
        names: dict[int, str] = self.model.names
        raw_detections: list[Detection] = []
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
                if confidence < effective_conf:
                    continue

                label = names[cls_id]
                raw_detections.append(Detection(english=label, confidence=confidence, count=1))

        raw_detections.sort(key=lambda item: -item.confidence)

        if not unique_labels:
            return raw_detections[:max_detections], annotated

        label_to_stats: dict[str, tuple[float, int]] = {}
        for detection in raw_detections:
            best_confidence, count = label_to_stats.get(detection.english, (0.0, 0))
            label_to_stats[detection.english] = (max(best_confidence, detection.confidence), count + 1)

        merged_detections = [
            Detection(english=label, confidence=confidence, count=count)
            for label, (confidence, count) in label_to_stats.items()
        ]
        merged_detections.sort(key=lambda item: (-item.confidence, item.english))
        return merged_detections, annotated
