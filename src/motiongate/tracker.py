"""Lightweight box propagation with sparse pyramidal Lucas--Kanade flow.

The tracker is deliberately detector-agnostic. A detector refresh seeds feature
points inside each person box; subsequent frames propagate each box by the
median displacement of forward-backward-consistent feature tracks. Tracking
quality is explicit so the scheduler can request re-detection when propagation
becomes unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .config import TrackingConfig
from .datatypes import Detection, TrackingResult


class BoxTracker(Protocol):
    """Interface used by the motion-gated scheduler."""

    def initialize(self, frame: np.ndarray, detections: tuple[Detection, ...]) -> None:
        """Seed tracking state from fresh detector output."""

    def update(self, frame: np.ndarray) -> TrackingResult:
        """Propagate boxes to the next frame."""

    def reset(self) -> None:
        """Forget all temporal state."""


@dataclass
class _TrackState:
    detection: Detection
    points: np.ndarray


def _gray(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be an HxWx3 numpy array (BGR)")
    if frame.dtype != np.uint8:
        raise TypeError(f"frame must have dtype uint8, got {frame.dtype}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _clip_box(
    detection: Detection,
    dx: float,
    dy: float,
    width: int,
    height: int,
) -> Detection | None:
    x1, y1, x2, y2 = detection.box
    x1 = float(np.clip(x1 + dx, 0.0, float(width)))
    x2 = float(np.clip(x2 + dx, 0.0, float(width)))
    y1 = float(np.clip(y1 + dy, 0.0, float(height)))
    y2 = float(np.clip(y2 + dy, 0.0, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return Detection((x1, y1, x2, y2), detection.score, detection.class_id)


class SparseOpticalFlowTracker:
    """Propagate axis-aligned boxes with sparse forward-backward optical flow."""

    name = "flow"

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._previous_gray: np.ndarray | None = None
        self._tracks: list[_TrackState] = []

    def reset(self) -> None:
        self._previous_gray = None
        self._tracks = []

    def _features(self, gray: np.ndarray, detection: Detection) -> np.ndarray:
        h, w = gray.shape
        x1, y1, x2, y2 = detection.box
        left = max(0, min(w, int(np.floor(x1))))
        top = max(0, min(h, int(np.floor(y1))))
        right = max(0, min(w, int(np.ceil(x2))))
        bottom = max(0, min(h, int(np.ceil(y2))))
        if right <= left or bottom <= top:
            return np.empty((0, 1, 2), dtype=np.float32)

        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[top:bottom, left:right] = 255
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_corners_per_box,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            mask=mask,
            blockSize=self.config.block_size,
        )
        if points is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return points.astype(np.float32, copy=False)

    def initialize(self, frame: np.ndarray, detections: tuple[Detection, ...]) -> None:
        gray = _gray(frame)
        self._previous_gray = gray
        self._tracks = [_TrackState(det, self._features(gray, det)) for det in detections]

    def update(self, frame: np.ndarray) -> TrackingResult:
        current = _gray(frame)
        if self._previous_gray is None:
            self._previous_gray = current
            return TrackingResult((), 0.0, False, 0)
        if not self._tracks:
            self._previous_gray = current
            return TrackingResult((), 1.0, True, 0)

        previous = self._previous_gray
        h, w = current.shape
        next_tracks: list[_TrackState] = []
        detections: list[Detection] = []
        qualities: list[float] = []
        tracked_points = 0
        all_reliable = True

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        lk_args = {
            "winSize": (self.config.win_size, self.config.win_size),
            "maxLevel": self.config.max_level,
            "criteria": criteria,
        }

        for state in self._tracks:
            p0 = state.points
            if len(p0) == 0:
                qualities.append(0.0)
                all_reliable = False
                next_tracks.append(state)
                detections.append(state.detection)
                continue

            p1, status_forward, _ = cv2.calcOpticalFlowPyrLK(previous, current, p0, None, **lk_args)
            if p1 is None or status_forward is None:
                qualities.append(0.0)
                all_reliable = False
                next_tracks.append(_TrackState(state.detection, np.empty((0, 1, 2), np.float32)))
                detections.append(state.detection)
                continue

            p0_back, status_backward, _ = cv2.calcOpticalFlowPyrLK(current, previous, p1, None, **lk_args)
            if p0_back is None or status_backward is None:
                qualities.append(0.0)
                all_reliable = False
                next_tracks.append(_TrackState(state.detection, np.empty((0, 1, 2), np.float32)))
                detections.append(state.detection)
                continue

            p0_flat = p0.reshape(-1, 2)
            p1_flat = p1.reshape(-1, 2)
            back_flat = p0_back.reshape(-1, 2)
            valid = (status_forward.reshape(-1) == 1) & (status_backward.reshape(-1) == 1)
            valid &= np.isfinite(p1_flat).all(axis=1) & np.isfinite(back_flat).all(axis=1)
            fb_error = np.linalg.norm(back_flat - p0_flat, axis=1)
            valid &= fb_error <= self.config.fb_threshold

            good0 = p0_flat[valid]
            good1 = p1_flat[valid]
            count = len(good1)
            tracked_points += count
            quality = count / len(p0_flat)
            qualities.append(float(quality))

            moved = state.detection
            if count:
                dx, dy = np.median(good1 - good0, axis=0)
                shifted = _clip_box(state.detection, float(dx), float(dy), w, h)
                if shifted is not None:
                    moved = shifted
                else:
                    all_reliable = False

            reliable = count >= self.config.min_points and quality >= self.config.min_quality
            all_reliable = all_reliable and reliable
            next_points = good1.reshape(-1, 1, 2).astype(np.float32, copy=False)
            next_tracks.append(_TrackState(moved, next_points))
            detections.append(moved)

        self._previous_gray = current
        self._tracks = next_tracks
        quality = min(qualities) if qualities else 1.0
        return TrackingResult(tuple(detections), quality, all_reliable, tracked_points)
