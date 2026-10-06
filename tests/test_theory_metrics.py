import math

import cv2
import numpy as np
import pytest

from benchmark.evaluation import iou, match, wilson
from motiongate import GateConfig, theory
from motiongate.gates import classic_difference_mask


def test_iou_and_matching():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    tp, fp, fn, _ = match([(0, 0, 10, 10), (1, 1, 11, 11)], [(0, 0, 10, 10)])
    assert (tp, fp, fn) == (1, 1, 0)     # one-to-one: a duplicate box is a false positive


def test_wilson_interval():
    lo, hi = wilson(50, 50)
    assert hi == pytest.approx(1.0) and 0.92 < lo < 0.94
    assert all(math.isnan(v) for v in wilson(0, 0))


def test_rectification_bias_matches_opencv_pipeline():
    """The mean of the original gate's smoothed difference on pure noise is 2 sigma / sqrt(pi)."""
    rng = np.random.default_rng(3)
    sigma = 6.0
    a = np.clip(128 + rng.normal(0, sigma, (400, 400, 3)), 0, 255).astype(np.uint8)
    b = np.clip(128 + rng.normal(0, sigma, (400, 400, 3)), 0, 255).astype(np.uint8)
    d = cv2.GaussianBlur(cv2.cvtColor(cv2.absdiff(a, b), cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0)
    mean, sd = theory.classic_noise_moments(sigma)
    assert d.mean() == pytest.approx(mean, rel=0.05)
    assert d.std() == pytest.approx(sd, rel=0.25)   # quantisation and spatial correlation of cvtColor


def test_predicted_noise_onset_orders_the_two_pipelines():
    lo_fd = theory.critical_sigma(20, 1e-3, "classic")
    lo_fds = theory.critical_sigma(20, 1e-3, "robust")
    assert 11 < lo_fd < 14 and lo_fds > 1.5 * lo_fd
    rng = np.random.default_rng(4)
    frame = lambda s: np.clip(128 + rng.normal(0, s, (200, 200, 3)), 0, 255).astype(np.uint8)
    cfg = GateConfig(dilation_iterations=0)
    below = classic_difference_mask(frame(0.7 * lo_fd), frame(0.7 * lo_fd), cfg).mean() / 255
    above = classic_difference_mask(frame(1.3 * lo_fd), frame(1.3 * lo_fd), cfg).mean() / 255
    assert below < 1e-3 < 0.05 < above


def test_cost_model_and_bounds():
    assert theory.speedup(1.0, 100.0, 0.5) == pytest.approx(100 / 51)
    assert theory.break_even_ratio(1.0, 100.0) == pytest.approx(0.99)
    assert theory.invocation_bounds(15, 3) == (1 / 15, 1 / 3)
    assert theory.area_model_critical_speed(113) == pytest.approx(900 / 119 - 6)
