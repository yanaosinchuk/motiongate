import math

import pytest

from motiongate import GateConfig, SchedulerConfig, StabilizationConfig, TrackingConfig


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"threshold": math.inf}, ValueError),
        ({"threshold": 255}, ValueError),
        ({"min_area": 0}, ValueError),
        ({"dilation_iterations": 1.5}, TypeError),
        ({"blur_kernel": 4}, ValueError),
    ],
)
def test_gate_config_rejects_invalid_values(kwargs, exc):
    with pytest.raises(exc):
        GateConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"estimation_scale": 0},
        {"max_shift": 0},
        {"min_response": 1.1},
        {"min_residual_gain": 0},
        {"min_shift": -0.1},
        {"saturation_margin": 128},
        {"registration_tolerance": -0.1},
        {"saturation_offset": -1},
    ],
)
def test_stabilization_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        StabilizationConfig(**kwargs)


def test_scheduler_config_rejects_boolean_intervals():
    with pytest.raises(TypeError):
        SchedulerConfig(True, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_corners_per_box": 0},
        {"quality_level": 0},
        {"quality_level": 1.1},
        {"min_distance": -1},
        {"block_size": 1},
        {"win_size": 2},
        {"max_level": -1},
        {"fb_threshold": 0},
        {"min_points": 0},
        {"min_quality": -0.1},
        {"min_quality": 1.1},
    ],
)
def test_tracking_config_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        TrackingConfig(**kwargs)
