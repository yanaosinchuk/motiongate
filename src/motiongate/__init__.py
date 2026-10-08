"""Motion-gated person detection for static-camera video."""

from .config import GateConfig, SchedulerConfig, StabilizationConfig, TrackingConfig
from .detector import PersonDetector, UltralyticsPersonDetector
from .gates import FrameDifferenceGate, MOG2Gate, StabilizedDifferenceGate, make_gate
from .scheduler import MotionGatedDetector, SchedulerStats
from .tracker import BoxTracker, SparseOpticalFlowTracker, make_tracker
from .datatypes import Detection, FrameResult, GateResult, TrackingResult

__version__ = "1.1.0"

__all__ = [
    "Detection", "FrameResult", "GateResult", "TrackingResult",
    "GateConfig", "SchedulerConfig", "StabilizationConfig", "TrackingConfig",
    "FrameDifferenceGate", "StabilizedDifferenceGate", "MOG2Gate", "make_gate",
    "PersonDetector", "UltralyticsPersonDetector",
    "MotionGatedDetector", "SchedulerStats",
    "BoxTracker", "SparseOpticalFlowTracker", "make_tracker",
]
