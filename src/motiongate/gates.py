"""Motion gates: cheap per-frame decisions whether the expensive detector must run.

Three gates share one interface (``update(frame) -> GateResult`` and ``reset()``):

``FrameDifferenceGate``
    The original prototype: absolute colour difference of adjacent frames,
    grayscale conversion, 5x5 Gaussian blur, fixed threshold, dilation and a
    minimum contour area.  Note the order of operations: the absolute value is
    taken *before* smoothing, which rectifies sensor noise into a positive bias
    of about 1.13 sigma that the blur cannot remove (see ``motiongate.theory``).

``StabilizedDifferenceGate``
    Grayscale conversion and smoothing *before* a signed difference, global
    translation compensation by phase correlation, and removal of the global
    (median) intensity offset.  These three changes target the failure modes of
    the original gate: sensor noise, camera jitter and abrupt illumination steps.

``MOG2Gate``
    OpenCV's adaptive Gaussian-mixture background subtractor (Zivkovic, 2004),
    included as the classical adaptive-background reference.  It answers a
    different question ("does the frame differ from the learned background?")
    and therefore stays active while a newly arrived object is still being
    absorbed into the background model.
"""

from __future__ import annotations

import math
from typing import Protocol

import cv2
import numpy as np

from .config import GateConfig, StabilizationConfig
from .datatypes import GateResult, IntBoxXYXY

__all__ = [
    "MotionGate",
    "FrameDifferenceGate",
    "StabilizedDifferenceGate",
    "MOG2Gate",
    "classic_difference_mask",
    "analyse_mask",
    "make_gate",
    "GATE_NAMES",
]


class MotionGate(Protocol):
    """Interface of every gate."""

    def update(self, frame: np.ndarray) -> GateResult:
        """Consume the next frame and decide whether it contains relevant change."""

    def reset(self) -> None:
        """Forget all temporal state (e.g. after a resolution change or a stream restart)."""


def _check_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame must be a numpy array, got {type(frame).__name__}")
    if frame.dtype != np.uint8:
        raise TypeError(f"frame must have dtype uint8, got {frame.dtype}")
    if frame.ndim not in (2, 3) or (frame.ndim == 3 and frame.shape[2] not in (1, 3)):
        raise ValueError(f"frame must be HxW or HxWx3, got shape {frame.shape}")
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError(f"frame is too small: {frame.shape}")


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _edge_residual(residual: np.ndarray, edges: np.ndarray) -> float:
    """Median absolute residual on edge pixels after removing the global median (offset)."""
    return float(np.median(np.abs(residual[edges] - np.median(residual[::2, ::2]))))


def analyse_mask(mask: np.ndarray, min_area: float) -> tuple[bool, tuple[IntBoxXYXY, ...], float]:
    """Extract external contours and apply the minimum-area rule.

    Returns ``(active, boxes, largest_area)`` where ``boxes`` are the bounding
    rectangles (x1, y1, x2, y2) of contours with area >= ``min_area``.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[IntBoxXYXY] = []
    largest = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        largest = max(largest, area)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, x + w, y + h))
    return bool(boxes), tuple(boxes), largest


def classic_difference_mask(previous: np.ndarray, current: np.ndarray, config: GateConfig) -> np.ndarray:
    """Binary change mask of the original prototype (uint8, values 0/255)."""
    diff = cv2.absdiff(previous, current)
    gray = _to_gray(diff)
    k = config.blur_kernel
    blurred = cv2.GaussianBlur(gray, (k, k), 0) if k > 1 else gray
    _, binary = cv2.threshold(blurred, config.threshold, 255, cv2.THRESH_BINARY)
    if config.dilation_iterations > 0:
        binary = cv2.dilate(binary, None, iterations=config.dilation_iterations)
    return binary


class FrameDifferenceGate:
    """Adjacent-frame differencing exactly as in the original OpenCV prototype."""

    name = "difference"

    def __init__(self, config: GateConfig | None = None) -> None:
        self.config = config or GateConfig()
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def update(self, frame: np.ndarray) -> GateResult:
        _check_frame(frame)
        previous = self._previous
        # Copy: the caller may reuse its frame buffer (e.g. cv2.VideoCapture.read(image=...)).
        self._previous = frame.copy()
        if previous is None or previous.shape != frame.shape:
            return GateResult(active=False, ready=False)
        mask = classic_difference_mask(previous, frame, self.config)
        active, boxes, largest = analyse_mask(mask, self.config.min_area)
        return GateResult(
            active=active,
            boxes=boxes,
            largest_area=largest,
            mask=mask if self.config.keep_mask else None,
        )


class StabilizedDifferenceGate:
    """Smoothed signed differencing with global-motion and illumination compensation.

    Processing of a new frame ``I_t`` (reference ``I_{t-1}``):

    1. ``G_t = K * gray(I_t)`` (grayscale, Gaussian blur, float32);
    2. translation ``s`` from phase correlation of down-scaled ``G_{t-1}``, ``G_t``,
       accepted only if it explains the image edges better than a static
       camera (model selection, see ``_estimate_shift``);
    3. ``R_t = G_t - warp(G_{t-1}, s)`` (a signed difference);
    4. ``beta_t = median(R_t)``; ``D_t = |R_t - beta_t|``;
    5. threshold, ignore the warped border and pixels that are saturated in
       either frame (clipping violates the global-offset model), dilate, apply
       the contour area filter.
    """

    name = "stabilized"

    def __init__(
        self,
        config: GateConfig | None = None,
        stabilization: StabilizationConfig | None = None,
    ) -> None:
        self.config = config or GateConfig()
        self.stabilization = stabilization or StabilizationConfig()
        self._previous: np.ndarray | None = None
        self._previous_small: np.ndarray | None = None
        self._window: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None
        self._previous_small = None
        self._window = None

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        gray = _to_gray(frame).astype(np.float32)
        k = self.config.blur_kernel
        smooth = cv2.GaussianBlur(gray, (k, k), 0) if k > 1 else gray
        small = None
        if self.stabilization.compensate_motion:
            scale = self.stabilization.estimation_scale
            small = smooth if scale == 1 else cv2.resize(
                smooth, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        return smooth, small

    def _estimate_shift(self, previous_small: np.ndarray, current_small: np.ndarray) -> tuple[float, float]:
        h, w = current_small.shape
        if self._window is None or self._window.shape != (h, w):
            self._window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(previous_small, current_small, self._window)
        scale = self.stabilization.estimation_scale
        dx, dy = dx / scale, dy / scale
        if response < self.stabilization.min_response or math.hypot(dx, dy) > self.stabilization.max_shift:
            return 0.0, 0.0  # unreliable estimate: fall back to plain differencing
        if math.hypot(dx, dy) < self.stabilization.min_shift:
            return 0.0, 0.0  # below the accuracy of the estimator: treat the camera as static
        # Model selection between "static camera" and "camera translated by s".
        # Phase correlation assumes that the background dominates the image; a
        # large, high-contrast moving object can capture the correlation peak.
        # Registration can only be verified where the image has structure, so the
        # shift is accepted only if it lowers the median residual on the 10 %
        # strongest-gradient pixels by the configured factor.  On these pixels
        # the (static or jittering) background dominates; a moving object is a
        # minority and cannot pull the decision.  The global median of the
        # residual is removed first, which makes the test illumination invariant.
        m = np.float32([[1.0, 0.0, dx * scale], [0.0, 1.0, dy * scale]])
        warped = cv2.warpAffine(previous_small, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        b = int(math.ceil(max(abs(dx), abs(dy)) * scale)) + 1
        inner = (slice(b, h - b), slice(b, w - b))
        cur, prev, wrp = current_small[inner], previous_small[inner], warped[inner]
        grad = cv2.magnitude(cv2.Sobel(cur, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(cur, cv2.CV_32F, 0, 1, ksize=3))
        edges = grad >= np.quantile(grad[::2, ::2], 0.9)
        if _edge_residual(cur - wrp, edges) > self.stabilization.min_residual_gain * _edge_residual(cur - prev, edges):
            return 0.0, 0.0
        return float(dx), float(dy)

    def update(self, frame: np.ndarray) -> GateResult:
        _check_frame(frame)
        current, current_small = self._prepare(frame)
        previous, previous_small = self._previous, self._previous_small
        self._previous, self._previous_small = current, current_small
        if previous is None or previous.shape != current.shape:
            return GateResult(active=False, ready=False)

        h, w = current.shape
        dx = dy = 0.0
        if self.stabilization.compensate_motion and previous_small is not None and current_small is not None:
            dx, dy = self._estimate_shift(previous_small, current_small)

        if dx != 0.0 or dy != 0.0:
            matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
            # Cubic interpolation: bilinear warping would blur only the reference
            # frame and leave residuals along every strong edge.
            reference = cv2.warpAffine(
                previous, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            margin = int(math.ceil(max(abs(dx), abs(dy)))) + 1
        else:
            reference = previous
            margin = 0

        residual = current - reference
        inside = np.ones(current.shape, dtype=bool)
        if margin:
            inside[:margin, :] = False
            inside[h - margin :, :] = False
            inside[:, :margin] = False
            inside[:, w - margin :] = False
        lo = self.stabilization.saturation_margin
        hi = 255.0 - lo
        unsaturated = inside & (current > lo) & (current < hi) & (reference > lo) & (reference < hi)

        offset = 0.0
        if self.stabilization.compensate_illumination:
            # A strided median is ~20x cheaper than the full median and just as robust.
            sample = residual[::4, ::4][unsaturated[::4, ::4]]
            if sample.size:
                offset = float(np.median(sample))
        # Clipped pixels cannot follow a global offset, so they are ignored -- but
        # only while an offset is actually compensated; otherwise a saturated
        # (e.g. white) moving object would become invisible.
        valid = unsaturated if abs(offset) > self.stabilization.saturation_offset else inside

        threshold: float | np.ndarray = self.config.threshold
        eps = self.stabilization.registration_tolerance
        if margin and eps > 0:
            # First-order error propagation: a registration error e leaves a
            # residual of about e . grad(I).  After compensation, a pixel is
            # therefore called "changed" only if |R| > tau + eps * |grad G_t|.
            gx = cv2.Sobel(current, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(current, cv2.CV_32F, 0, 1, ksize=3)
            threshold = self.config.threshold + (eps / 8.0) * cv2.magnitude(gx, gy)
        mask = ((np.abs(residual - offset) > threshold) & valid).astype(np.uint8) * 255
        if self.config.dilation_iterations > 0:
            mask = cv2.dilate(mask, None, iterations=self.config.dilation_iterations)

        active, boxes, largest = analyse_mask(mask, self.config.min_area)
        return GateResult(
            active=active,
            boxes=boxes,
            largest_area=largest,
            shift=(dx, dy),
            offset=offset,
            mask=mask if self.config.keep_mask else None,
        )


class MOG2Gate:
    """Gate based on OpenCV's adaptive Gaussian-mixture background model (MOG2).

    The first frame initialises the model and is reported as not ready.  Shadow
    pixels (value 127 in OpenCV's output) are ignored; a 3x3 opening removes
    isolated foreground pixels before the shared dilation and area rule.
    """

    name = "mog2"

    def __init__(
        self,
        config: GateConfig | None = None,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        learning_rate: float = -1.0,
    ) -> None:
        self.config = config or GateConfig()
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.learning_rate = learning_rate
        self._shape: tuple[int, ...] | None = None
        self._model = self._new_model()

    def _new_model(self):
        return cv2.createBackgroundSubtractorMOG2(
            history=self.history, varThreshold=self.var_threshold, detectShadows=self.detect_shadows
        )

    def reset(self) -> None:
        self._model = self._new_model()
        self._shape = None

    def update(self, frame: np.ndarray) -> GateResult:
        _check_frame(frame)
        if self._shape != frame.shape:
            self.reset()
            self._shape = frame.shape
            self._model.apply(frame, learningRate=self.learning_rate)
            return GateResult(active=False, ready=False)
        foreground = self._model.apply(frame, learningRate=self.learning_rate)
        mask = np.where(foreground == 255, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if self.config.dilation_iterations > 0:
            mask = cv2.dilate(mask, None, iterations=self.config.dilation_iterations)
        active, boxes, largest = analyse_mask(mask, self.config.min_area)
        return GateResult(
            active=active,
            boxes=boxes,
            largest_area=largest,
            mask=mask if self.config.keep_mask else None,
        )


GATE_NAMES = ("difference", "stabilized", "mog2")


def make_gate(
    name: str,
    config: GateConfig | None = None,
    stabilization: StabilizationConfig | None = None,
) -> MotionGate:
    """Factory used by the CLI and the benchmark."""
    if name == "difference":
        return FrameDifferenceGate(config)
    if name == "stabilized":
        return StabilizedDifferenceGate(config, stabilization)
    if name == "mog2":
        return MOG2Gate(config)
    raise ValueError(f"unknown gate {name!r}; choose one of {GATE_NAMES}")
