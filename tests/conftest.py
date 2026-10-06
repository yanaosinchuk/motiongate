import cv2
import numpy as np
import pytest


@pytest.fixture
def texture():
    """A smooth random texture resembling a natural background (256 x 256 x 3, uint8)."""
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.uniform(0, 255, (256, 256, 3)).astype(np.float32), (0, 0), 3)
    base = (base - base.min()) / (base.max() - base.min()) * 200 + 25
    return base.astype(np.uint8)


@pytest.fixture
def sharp_texture():
    """Strong texture everywhere: image structure is spread over the whole frame, as in real scenes."""
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(rng.uniform(0, 255, (256, 256, 3)).astype(np.float32), (0, 0), 1.0)
    base = (base - base.min()) / (base.max() - base.min()) * 200 + 25
    return base.astype(np.uint8)
