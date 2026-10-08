import math

import pytest

from motiongate import GateConfig, SchedulerConfig, StabilizationConfig


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
