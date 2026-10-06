"""Motion-gated inference scheduler.

The scheduler implements the decision rule

    p_t = p_{t-1} or g_t                      (motion pending since t_last)
    z_t = 1[ t = 0  or  (p_t and t - t_last >= M)  or  t - t_last >= K ]

where ``g_t`` is the motion-gate decision, ``t_last`` the index of the last
detector call, ``K`` the refresh interval and ``M`` the minimum interval
(``p`` is cleared whenever the detector runs).  With M = 1 this is the
original rule ``z_t = 1[t = 0 or g_t or t - t_last >= K]``.  If ``z_t = 0``
the most recent detections are *retained*.  Every result states whether its detections are
fresh, why the detector ran, and how old retained detections are, so that a
consumer can never mistake a retained box for a new observation.

Guarantees: retained detections are at most ``K - 1`` frames old; a motion
trigger is served within ``M - 1`` frames; the invocation ratio lies between
1/K (no motion) and 1/M (motion in every frame), up to the initial frame.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .config import SchedulerConfig
from .detector import PersonDetector
from .gates import MotionGate
from .datatypes import Detection, FrameResult

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
    """Run ``detector`` only on frames selected by ``gate`` or by the periodic refresh."""

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

    def reset(self) -> None:
        """Start a new stream: the next frame is treated as the first frame."""
        self.gate.reset()
        self._last_inference = -1
        self._last_detections = ()
        self._shape = None
        self._pending_motion = False

    def process(self, frame: np.ndarray) -> FrameResult:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be an HxWx3 numpy array (BGR)")
        self._index += 1
        if self._shape is not None and frame.shape != self._shape:
            # A resolution change invalidates both the gate reference and the boxes.
            self.reset()
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
            # Replace the retained set even when it is empty: this clears stale
            # boxes after a person has left the scene.
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
