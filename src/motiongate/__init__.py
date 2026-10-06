"""Motion-gated person detection for static-camera video."""

from .config import GateConfig, SchedulerConfig, StabilizationConfig
from .detector import PersonDetector, UltralyticsPersonDetector
from .gates import FrameDifferenceGate, MOG2Gate, StabilizedDifferenceGate, make_gate
from .scheduler import MotionGatedDetector, SchedulerStats
from .datatypes import Detection, FrameResult, GateResult

__version__ = "1.0.0"

__all__ = [
    "Detection", "FrameResult", "GateResult",
    "GateConfig", "SchedulerConfig", "StabilizationConfig",
    "FrameDifferenceGate", "StabilizedDifferenceGate", "MOG2Gate", "make_gate",
    "PersonDetector", "UltralyticsPersonDetector",
    "MotionGatedDetector", "SchedulerStats",
]
