"""Person detectors.

The scheduler only depends on the tiny :class:`PersonDetector` protocol, so the
neural network can be replaced by any callable that maps a BGR frame to a list
of :class:`~motiongate.types.Detection` objects -- another YOLO version, an
exported ONNX/TensorRT model, or a fake detector in unit tests.

Ultralytics is imported lazily so that the gate, the scheduler and the tests do
not require PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .datatypes import Detection

COCO_PERSON_CLASS_ID = 0


class PersonDetector(Protocol):
    def __call__(self, frame: np.ndarray) -> list[Detection]:
        """Return the person detections of one BGR frame."""


class UltralyticsPersonDetector:
    """Wrap an Ultralytics YOLO model and keep only person detections.

    Parameters
    ----------
    weights:
        Path or name of an Ultralytics checkpoint, e.g. ``"yolov8m.pt"``.  Named
        official checkpoints are downloaded automatically on first use.
    confidence:
        Minimum confidence ``s_min``.
    image_size:
        Inference size; Ultralytics letterboxes the frame (aspect ratio is kept)
        and maps boxes back to the original frame coordinates.
    device:
        ``"cpu"``, ``"cuda:0"``, ``"mps"``, ...
    person_class_id:
        Class index of *person* in the model's label map (0 for COCO models).

    Class filtering is applied *after* the library's standard post-processing,
    so the non-maximum suppression of the model is left unchanged.
    """

    def __init__(
        self,
        weights: str | Path = "yolov8m.pt",
        confidence: float = 0.25,
        image_size: int = 640,
        device: str = "cpu",
        person_class_id: int = COCO_PERSON_CLASS_ID,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "UltralyticsPersonDetector needs the 'ultralytics' package: pip install ultralytics"
            ) from exc
        self.weights = str(weights)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.person_class_id = person_class_id
        self.model = YOLO(self.weights)

    def warmup(self, height: int = 640, width: int = 640) -> None:
        """Run one inference on a black frame so that timing excludes lazy initialisation."""
        self(np.zeros((height, width, 3), dtype=np.uint8))

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        return [
            Detection(box=tuple(float(v) for v in box), score=float(score), class_id=int(cls))
            for box, score, cls in zip(boxes, scores, classes)
            if cls == self.person_class_id and score >= self.confidence
        ]
