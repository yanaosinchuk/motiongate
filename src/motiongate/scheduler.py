"""Motion-gated inference scheduler.

Motion and tracker-failure events remain pending until the minimum interval
permits a detector call. Periodic refresh still bounds detector age by K - 1
frames. When an optional box tracker is present, reliable tracked boxes replace
frozen retained boxes between detector calls without changing detector-age
semantics.
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
from .tracker import BoxTracker

REASON_INITIAL = "initial"
REASON_MOTION = "motion"
REASON_TRACKING = "tracking"
REASON_REFRESH = "refresh"
REASON_SKIPPED = "skipped"


@dataclass
class SchedulerStats:
    frames: int = 0
    invocations: int = 0
    reasons: Counter = field(default_factory=Counter)
    gate_ms: float = 0.0
    tracker_ms: float = 0.0
    detector_ms: float = 0.0

    @property
    def invocation_ratio(self) -> float:
        return self.invocations / self.frames if self.frames else float("nan")

    @property
    def total_ms(self) -> float:
        return self.gate_ms + self.tracker_ms + self.detector_ms


class MotionGatedDetector:
    """Run a detector only when the scheduler decides fresh inference is needed.

    Tracking is optional. Without a tracker the behaviour matches the paper's
    scheduler: skipped frames reuse the most recent detector output. With a
    tracker, reliable propagation updates those boxes, and unreliable tracking
    requests a detector refresh subject to the same minimum interval M.
    """

    def __init__(
        self,
        detector: PersonDetector,
        gate: MotionGate,
        config: SchedulerConfig | None = None,
        tracker: BoxTracker | None = None,
    ) -> None:
        self.detector = detector
        self.gate = gate
        self.config = config or SchedulerConfig()
        self.tracker = tracker
        self.stats = SchedulerStats()
        self._index = -1
        self._last_inference = -1
        self._last_detections: tuple[Detection, ...] = ()
        self._shape: tuple[int, ...] | None = None
        self._pending_motion = False
        self._pending_tracking = False

    def _reset_temporal_state(self) -> None:
        """Reset video-dependent state while preserving counters and frame index."""
        self.gate.reset()
        if self.tracker is not None:
            self.tracker.reset()
        self._last_inference = -1
        self._last_detections = ()
        self._shape = None
        self._pending_motion = False
        self._pending_tracking = False

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
            self._reset_temporal_state()
        first = self._shape is None
        self._shape = frame.shape

        start = perf_counter()
        gate_result = self.gate.update(frame)
        gate_ms = 1000.0 * (perf_counter() - start)

        tracking_result = None
        tracker_ms = 0.0
        gap = self._index - self._last_inference
        self._pending_motion = self._pending_motion or gate_result.active

        # Decide all already-known detector triggers before spending time on
        # optical flow. Tracking is useful only when this frame would otherwise
        # be skipped; a fresh detector result will immediately replace it.
        if first:
            reason = REASON_INITIAL
        elif self._pending_tracking and gap >= self.config.min_interval:
            reason = REASON_TRACKING
        elif self._pending_motion and gap >= self.config.min_interval:
            reason = REASON_MOTION
        elif gap >= self.config.refresh_interval:
            reason = REASON_REFRESH
        else:
            if self.tracker is not None and self._last_detections:
                start = perf_counter()
                tracking_result = self.tracker.update(frame)
                tracker_ms += 1000.0 * (perf_counter() - start)
                self._pending_tracking = self._pending_tracking or not tracking_result.reliable
            reason = (
                REASON_TRACKING
                if self._pending_tracking and gap >= self.config.min_interval
                else REASON_SKIPPED
            )

        detector_ms = 0.0
        fresh = reason != REASON_SKIPPED
        tracked = False
        if fresh:
            start = perf_counter()
            detections = tuple(self.detector(frame))
            detector_ms = 1000.0 * (perf_counter() - start)
            self._last_detections = detections
            self._last_inference = self._index
            self._pending_motion = False
            self._pending_tracking = False
            if self.tracker is not None:
                start = perf_counter()
                self.tracker.initialize(frame, detections)
                tracker_ms += 1000.0 * (perf_counter() - start)
        elif tracking_result is not None and tracking_result.reliable:
            self._last_detections = tracking_result.detections
            tracked = True

        self.stats.frames += 1
        self.stats.invocations += int(fresh)
        self.stats.reasons[reason] += 1
        self.stats.gate_ms += gate_ms
        self.stats.tracker_ms += tracker_ms
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
            tracker_ms=tracker_ms,
            tracked=tracked,
            tracking_quality=tracking_result.quality if tracking_result is not None else None,
        )
