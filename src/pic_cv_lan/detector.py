from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    english: str
    confidence: float
    count: int = 1
    bbox: Box | None = None


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

    @staticmethod
    def _iou(box_a: Box, box_b: Box) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @classmethod
    def _deduplicate_instances(
        cls,
        detections: list[Detection],
        duplicate_mode: str = "same_label",
        iou_threshold: float = 0.70,
    ) -> list[Detection]:
        """Suppress duplicate boxes after YOLO prediction.

        duplicate_mode:
        - none: keep all detections.
        - same_label: suppress only highly overlapping boxes with the same label.
        - any_label: suppress highly overlapping boxes even when labels are different.
        """

        if duplicate_mode == "none":
            return detections

        kept: list[Detection] = []
        for detection in sorted(detections, key=lambda item: -item.confidence):
            if detection.bbox is None:
                kept.append(detection)
                continue

            should_drop = False
            for existing in kept:
                if existing.bbox is None:
                    continue
                if duplicate_mode == "same_label" and detection.english != existing.english:
                    continue
                if cls._iou(detection.bbox, existing.bbox) >= iou_threshold:
                    should_drop = True
                    break

            if not should_drop:
                kept.append(detection)

        return kept

    @staticmethod
    def _to_pil_image(image: Image.Image | str | Path) -> Image.Image | None:
        try:
            if isinstance(image, Image.Image):
                return image.convert("RGB")
            return Image.open(image).convert("RGB")
        except Exception:
            return None

    @staticmethod
    def _draw_annotations(image: Image.Image | str | Path, detections: list[Detection]) -> np.ndarray | None:
        pil_image = ObjectDetector._to_pil_image(image)
        if pil_image is None:
            return None

        draw = ImageDraw.Draw(pil_image)

        for detection in detections:
            if detection.bbox is None:
                continue
            x1, y1, x2, y2 = detection.bbox
            label = f"{detection.english} {detection.confidence:.2f}"
            draw.rectangle((x1, y1, x2, y2), outline=(37, 99, 235), width=3)
            text_box = draw.textbbox((x1, y1), label)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            label_y1 = max(0, y1 - text_h - 8)
            draw.rectangle((x1, label_y1, x1 + text_w + 8, label_y1 + text_h + 6), fill=(37, 99, 235))
            draw.text((x1 + 4, label_y1 + 3), label, fill=(255, 255, 255))

        return np.array(pil_image)

    def detect(
        self,
        image: Image.Image | str | Path,
        confidence_threshold: float | None = None,
        image_size: int | None = None,
        unique_labels: bool = True,
        max_detections: int = 100,
        duplicate_mode: str = "same_label",
        iou_threshold: float = 0.70,
    ) -> tuple[list[Detection], np.ndarray | None]:
        """Detect objects from an image.

        Args:
            image: PIL image or image path.
            confidence_threshold: Runtime confidence threshold. Lower values show more boxes but may add false positives.
            image_size: Runtime inference size. Larger values may detect small objects better but run slower on CPU.
            unique_labels: If True, merge duplicate labels and keep a count. If False, return every detected instance.
            max_detections: Maximum number of raw detections returned by YOLO.
            duplicate_mode: none, same_label, or any_label.
            iou_threshold: IoU threshold for duplicate suppression. Lower means more aggressive deduplication.
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

        for result in results:
            boxes: Any = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if confidence < effective_conf:
                    continue

                label = names[cls_id]
                bbox_values = tuple(float(value) for value in box.xyxy[0].tolist())
                raw_detections.append(
                    Detection(
                        english=label,
                        confidence=confidence,
                        count=1,
                        bbox=bbox_values,  # type: ignore[arg-type]
                    )
                )

        filtered_instances = self._deduplicate_instances(
            raw_detections,
            duplicate_mode=duplicate_mode,
            iou_threshold=iou_threshold,
        )
        filtered_instances.sort(key=lambda item: -item.confidence)
        annotated = self._draw_annotations(image, filtered_instances)

        if not unique_labels:
            return filtered_instances[:max_detections], annotated

        label_to_stats: dict[str, tuple[float, int, Box | None]] = {}
        for detection in filtered_instances:
            best_confidence, count, best_bbox = label_to_stats.get(detection.english, (0.0, 0, None))
            if detection.confidence >= best_confidence:
                best_bbox = detection.bbox
            label_to_stats[detection.english] = (max(best_confidence, detection.confidence), count + 1, best_bbox)

        merged_detections = [
            Detection(english=label, confidence=confidence, count=count, bbox=bbox)
            for label, (confidence, count, bbox) in label_to_stats.items()
        ]
        merged_detections.sort(key=lambda item: (-item.confidence, item.english))
        return merged_detections, annotated
