import cv2
import numpy as np
import pytest

from motiongate import FrameDifferenceGate, GateConfig, MOG2Gate, StabilizationConfig, StabilizedDifferenceGate, make_gate
from motiongate.gates import stabilized_difference_mask


def shift(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def with_square(img, x, y, size=60, value=255):
    out = img.copy()
    out[y:y + size, x:x + size] = value
    return out


@pytest.mark.parametrize("gate_cls", [FrameDifferenceGate, StabilizedDifferenceGate, MOG2Gate])
def test_first_frame_is_not_ready(gate_cls, texture):
    result = gate_cls().update(texture)
    assert not result.ready and not result.active


@pytest.mark.parametrize("gate_cls", [FrameDifferenceGate, StabilizedDifferenceGate])
def test_static_scene_is_inactive_and_moving_square_is_active(gate_cls, sharp_texture):
    gate = gate_cls()
    gate.update(with_square(sharp_texture, 50, 80))
    assert not gate.update(with_square(sharp_texture, 50, 80)).active
    moved = gate.update(with_square(sharp_texture, 62, 80))
    assert moved.active
    # A change mask is not an object mask: it covers the trailing and leading edges.
    assert min(b[0] for b in moved.boxes) <= 62 and max(b[2] for b in moved.boxes) >= 110
    assert all(b[2] - b[0] < 100 for b in moved.boxes)      # no frame-wide region
    assert moved.shift == (0.0, 0.0)                         # the object is not mistaken for camera motion


def test_camera_translation_fools_fd_but_not_fds(texture):
    fd, fds = FrameDifferenceGate(), StabilizedDifferenceGate()
    a, b = shift(texture, 0.3, -0.4), shift(texture, 3.3, -2.4)
    fd.update(a), fds.update(a)
    assert fd.update(b).active
    res = fds.update(b)
    assert not res.active
    assert res.shift == pytest.approx((3.0, -2.0), abs=0.8)   # sign convention; low-texture image


def test_global_illumination_step_fools_fd_but_not_fds(texture):
    fd, fds = FrameDifferenceGate(), StabilizedDifferenceGate()
    brighter = np.clip(texture.astype(int) + 40, 0, 255).astype(np.uint8)
    fd.update(texture), fds.update(texture)
    assert fd.update(brighter).active
    res = fds.update(brighter)
    assert not res.active and res.offset == pytest.approx(40, abs=1)


def test_fds_still_sees_object_under_jitter(texture):
    fds = StabilizedDifferenceGate()
    fds.update(with_square(shift(texture, 0, 0), 60, 60))
    assert fds.update(with_square(shift(texture, 2.5, 1.5), 80, 60)).active


def test_noise_below_prediction_is_ignored(texture):
    rng = np.random.default_rng(1)
    def noisy():
        return np.clip(texture + rng.normal(0, 3, texture.shape), 0, 255).astype(np.uint8)
    for gate in (FrameDifferenceGate(), StabilizedDifferenceGate()):
        gate.update(noisy())
        assert not any(gate.update(noisy()).active for _ in range(5))


def test_mog2_detects_new_object(texture):
    gate = MOG2Gate()
    for _ in range(5):
        gate.update(texture)
    assert gate.update(with_square(texture, 100, 100, 70)).active


def test_resolution_change_resets(texture):
    gate = FrameDifferenceGate()
    gate.update(texture)
    assert not gate.update(texture[:128, :128].copy()).ready


def test_invalid_input_and_config():
    with pytest.raises(TypeError):
        FrameDifferenceGate().update(np.zeros((10, 10, 3), np.float32))
    with pytest.raises(ValueError):
        FrameDifferenceGate().update(np.zeros((10, 10, 4), np.uint8))
    with pytest.raises(ValueError):
        GateConfig(blur_kernel=4)
    with pytest.raises(ValueError):
        make_gate("optical-flow")


def test_mask_kept_only_on_request(texture):
    gate = FrameDifferenceGate(GateConfig(keep_mask=True))
    gate.update(texture)
    assert gate.update(texture).mask is not None


def test_stabilized_mask_compensates_global_offset():
    current = np.full((64, 64), 140.0, np.float32)
    reference = np.full((64, 64), 100.0, np.float32)
    config = GateConfig(dilation_iterations=0)
    mask, offset = stabilized_difference_mask(current, reference, config, StabilizationConfig())

    assert offset == pytest.approx(40.0)
    assert not np.any(mask)


def test_stabilized_mask_rejects_shape_mismatch():
    config = GateConfig()
    with pytest.raises(ValueError, match="same shape"):
        stabilized_difference_mask(
            np.zeros((16, 16), np.float32),
            np.zeros((15, 16), np.float32),
            config,
            StabilizationConfig(),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"history": 0},
        {"var_threshold": 0},
        {"learning_rate": -2},
        {"learning_rate": 1.1},
    ],
)
def test_mog2_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        MOG2Gate(**kwargs)
