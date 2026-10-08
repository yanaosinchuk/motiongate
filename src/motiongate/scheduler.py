"""Motion-gated inference scheduler.

Suppressed motion remains pending until the minimum interval permits a detector
call. Retained detections are at most K - 1 frames old.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .config import SchedulerConfig
from .datatypes import Detection, FrameResult
from .detector import PersonDetector
from .gates import MotionGate

REASON_INITIAL = "initial"
REASON_MOTION = "motion"
REASON_REFRESH = "refresh"
REASON_SKIPPED = "skipped"


@dataclass
class SchedulerStats:
    frames: int = 0
    invocations: int = 0
    reasons: Counter = field(default_factory=Counter)
    gate_ms: float = 0.0
    detector_ms: float = 0.0

    @property
    def invocation_ratio(self) -> float:
        return self.invocations / self.frames if self.frames else float("nan")

    @property
    def total_ms(self) -> float:
        return self.gate_ms + self.detector_ms


class MotionGatedDetector:
    """Run detector only on frames selected by the gate or periodic refresh."""

    def __init__(
        self,
        detector: PersonDetector,
        gate: MotionGate,
        config: SchedulerConfig | None = None,
    ) -> None:
        self.detector = detector
        self.gate = gate
        self.config = config or SchedulerConfig()
        self.stats = SchedulerStats()
        self._index = -1
        self._last_inference = -1
        self._last_detections: tuple[Detection, ...] = ()
        self._shape: tuple[int, ...] | None = None
        self._pending_motion = False

    def _reset_temporal_state(self) -> None:
        """Reset video-dependent state while preserving counters and frame index."""
        self.gate.reset()
        self._last_inference = -1
        self._last_detections = ()
        self._shape = None
        self._pending_motion = False

    def reset(self) -> None:
        """Start a completely new stream and clear accumulated statistics."""
        self._index = -1
        self._reset_temporal_state()
        self.reset_stats()

    def reset_stats(self) -> None:
        """Clear performance counters without changing the current stream state."""
        self.stats = SchedulerStats()

    def process(self, frame: np.ndarray) -> FrameResult:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 numpy array (BGR)")
        if frame.dtype != np.uint8:
            raise TypeError(f"frame must have dtype uint8, got {frame.dtype}")

        self._index += 1
        if self._shape is not None and frame.shape != self._shape:
            # A resolution change invalidates temporal state and retained boxes,
            # but remains part of the same input stream for statistics.
            self._reset_temporal_state()
        first = self._shape is None
        self._shape = frame.shape

        start = perf_counter()
        gate_result = self.gate.update(frame)
        gate_ms = 1000.0 * (perf_counter() - start)

        gap = self._index - self._last_inference
        self._pending_motion = self._pending_motion or gate_result.active
        if first:
            reason = REASON_INITIAL
        elif self._pending_motion and gap >= self.config.min_interval:
            reason = REASON_MOTION
        elif gap >= self.config.refresh_interval:
            reason = REASON_REFRESH
        else:
            reason = REASON_SKIPPED

        detector_ms = 0.0
        fresh = reason != REASON_SKIPPED
        if fresh:
            start = perf_counter()
            detections = tuple(self.detector(frame))
            detector_ms = 1000.0 * (perf_counter() - start)
            self._last_detections = detections
            self._last_inference = self._index
            self._pending_motion = False

        self.stats.frames += 1
        self.stats.invocations += int(fresh)
        self.stats.reasons[reason] += 1
        self.stats.gate_ms += gate_ms
        self.stats.detector_ms += detector_ms

        return FrameResult(
            index=self._index,
            detections=self._last_detections,
            fresh=fresh,
            reason=reason,
            detection_age=self._index - self._last_inference,
            gate=gate_result,
            gate_ms=gate_ms,
            detector_ms=detector_ms,
        )
