import cv2
import numpy as np
import pytest

from motiongate import Detection, SparseOpticalFlowTracker, TrackingConfig, make_tracker


def translated(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        frame,
        matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def test_sparse_flow_propagates_box_translation():
    rng = np.random.default_rng(12)
    first = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    second = translated(first, 4.0, 3.0)
    detection = Detection((30.0, 30.0, 90.0, 90.0), 0.9)

    tracker = SparseOpticalFlowTracker()
    tracker.initialize(first, (detection,))
    result = tracker.update(second)

    assert result.reliable
    assert result.tracked_points >= TrackingConfig().min_points
    assert result.quality >= TrackingConfig().min_quality
    x1, y1, x2, y2 = result.detections[0].box
    assert (x1, y1, x2, y2) == pytest.approx((34.0, 33.0, 94.0, 93.0), abs=1.0)


def test_sparse_flow_reports_low_quality_without_texture():
    frame = np.full((96, 96, 3), 127, np.uint8)
    detection = Detection((20.0, 20.0, 70.0, 70.0), 0.9)

    tracker = SparseOpticalFlowTracker()
    tracker.initialize(frame, (detection,))
    result = tracker.update(frame.copy())

    assert not result.reliable
    assert result.quality == 0.0
    assert result.tracked_points == 0


def test_empty_detection_set_is_a_valid_tracking_state():
    frame = np.zeros((64, 64, 3), np.uint8)
    tracker = SparseOpticalFlowTracker()
    tracker.initialize(frame, ())

    result = tracker.update(frame.copy())

    assert result.reliable
    assert result.quality == 1.0
    assert result.detections == ()


def test_tracker_factory_keeps_tracking_opt_in():
    assert make_tracker("none") is None
    assert isinstance(make_tracker("flow"), SparseOpticalFlowTracker)
    with pytest.raises(ValueError, match="unknown tracker"):
        make_tracker("unknown")
