import pytest

from benchmark.tracking import _evaluate_frame
from motiongate import Detection


def test_tracking_frame_metric_counts_localised_hit():
    result = _evaluate_frame(
        (10, 10, 30, 30),
        (Detection((11, 10, 31, 30), 0.9),),
    )
    assert result["tp"] == 1
    assert result["fn"] == 0
    assert result["box_fp"] == 0
    assert result["ious"][0] > 0.8


def test_tracking_frame_metric_counts_stale_box_on_empty_frame():
    result = _evaluate_frame(
        None,
        (Detection((10, 10, 30, 30), 0.9),),
    )
    assert result["fp_frame"] == 1
    assert result["box_fp"] == 1
    assert result["tn"] == 0


def test_tracking_frame_metric_counts_miss_below_iou_threshold():
    result = _evaluate_frame(
        (10, 10, 30, 30),
        (Detection((40, 40, 60, 60), 0.9),),
    )
    assert result["tp"] == 0
    assert result["fn"] == 1
    assert result["box_fp"] == 1
    assert result["ious"] == []
