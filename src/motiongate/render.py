"""Visualisation helpers.  Drawing always happens on a copy of the frame."""

from __future__ import annotations

import cv2
import numpy as np

from .datatypes import FrameResult

GREEN = (0, 170, 0)      # motion regions
RED = (0, 0, 220)        # fresh person detections
ORANGE = (0, 140, 255)   # retained person detections


def draw_overlay(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    canvas = frame.copy()
    for x1, y1, x2, y2 in result.gate.boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GREEN, 1)
    colour = RED if result.fresh else ORANGE
    for det in result.detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        if result.fresh:
            suffix = ""
        elif result.tracked:
            suffix = f" (tracked, age {result.detection_age})"
        else:
            suffix = f" (age {result.detection_age})"
        label = f"person {det.score:.2f}" + suffix
        cv2.putText(canvas, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    status = f"frame {result.index} | {result.reason} | {len(result.detections)} person(s)"
    cv2.putText(canvas, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas
