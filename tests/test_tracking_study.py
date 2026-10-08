import math

from benchmark.tracking_study import Accumulator, OracleDetector
from motiongate import Detection, FrameResult, GateResult


def result(detections, *, fresh=False, reason="skipped", age=1):
    return FrameResult(
        index=1,
        detections=tuple(detections),
        fresh=fresh,
        reason=reason,
        detection_age=age,
        gate=GateResult(active=False, ready=True),
        gate_ms=2.0,
        detector_ms=0.0,
        tracker_ms=1.0,
    )


def test_oracle_detector_returns_current_truth_box():
    detector = OracleDetector()
    detector.box = (1, 2, 11, 12)
    out = detector(None)
    assert out == [Detection((1.0, 2.0, 11.0, 12.0), 1.0)]


def test_accumulator_counts_recall_iou_and_stale_false_positive():
    acc = Accumulator()
    acc.add((0, 0, 10, 10), result([Detection((0, 0, 10, 10), 1.0)], age=0))
    acc.add(None, result([Detection((0, 0, 10, 10), 1.0)], age=1))

    row = acc.row(detector_ms=100.0)
    assert row["recall"] == 1.0
    assert row["mean_iou_positive"] == 1.0
    assert row["stale_fp_frames"] == 1
    assert row["mean_detection_age"] == 0.5
    assert row["gate_ms_per_frame"] == 2.0
    assert row["tracker_ms_per_frame"] == 1.0
    assert math.isfinite(row["model_speedup"])
