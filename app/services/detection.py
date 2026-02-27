from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image


@dataclass
class DetectionResult:
    bbox: tuple[int, int, int, int]
    yolo_conf: float
    class_id: int
    class_name: str
    crop: Image.Image


class DetectionService:
    def __init__(self, weights_path: str, conf_threshold: float, iou_threshold: float):
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect_and_crop(self, image: Image.Image) -> List[DetectionResult]:
        rgb_image = image.convert("RGB")
        np_image = np.array(rgb_image)
        results = self.model.predict(np_image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)

        detections: List[DetectionResult] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        width, height = rgb_image.size

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(width, int(x2)), min(height, int(y2))

            if x2i <= x1i or y2i <= y1i:
                continue

            class_id = int(box.cls[0].item()) if box.cls is not None else -1
            class_name = result.names.get(class_id, "unknown") if hasattr(result, "names") else "unknown"
            yolo_conf = float(box.conf[0].item()) if box.conf is not None else 0.0

            crop = rgb_image.crop((x1i, y1i, x2i, y2i))
            detections.append(
                DetectionResult(
                    bbox=(x1i, y1i, x2i, y2i),
                    yolo_conf=yolo_conf,
                    class_id=class_id,
                    class_name=class_name,
                    crop=crop,
                )
            )

        return detections
