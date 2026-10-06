"""Validated configuration objects.

The default values of :class:`GateConfig` reproduce the original OpenCV
prototype (5x5 Gaussian kernel, threshold 20, three dilations, minimum contour
area 900 px).  Validation happens once, at construction time, so that a bad
parameter fails loudly instead of silently producing an always-on or never-on
gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GateConfig:
    """Parameters shared by all difference-based gates."""

    threshold: float = 20.0          # tau, on the 0..255 intensity scale
    min_area: float = 900.0          # A_min, contour area in pixels
    dilation_iterations: int = 3     # d, iterations of a 3x3 square dilation
    blur_kernel: int = 5             # k, odd Gaussian kernel size
    keep_mask: bool = False          # keep the binary mask in GateResult (diagnostics)

    def __post_init__(self) -> None:
        if not 0 <= self.threshold < 255:
            raise ValueError(f"threshold must be in [0, 255), got {self.threshold}")
        if self.min_area < 0:
            raise ValueError(f"min_area must be non-negative, got {self.min_area}")
        if self.dilation_iterations < 0:
            raise ValueError("dilation_iterations must be non-negative")
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError(f"blur_kernel must be a positive odd integer, got {self.blur_kernel}")


@dataclass(frozen=True)
class StabilizationConfig:
    """Extra parameters of the stabilised, illumination-compensated gate."""

    compensate_motion: bool = True        # estimate and remove global translation
    compensate_illumination: bool = True  # remove the global (median) intensity offset
    estimation_scale: float = 0.5         # downscale factor for phase correlation
    max_shift: float = 20.0               # reject implausible shift estimates (pixels)
    min_response: float = 0.05            # reject weak phase-correlation peaks
    min_residual_gain: float = 0.9        # accept a shift only if it lowers the edge residual by >= 10 %
    min_shift: float = 0.1                # shifts below this (px) are below estimation accuracy -> treated as zero
    saturation_margin: float = 6.0        # ignore pixels within this distance of 0 or 255 (clipped sensor values)
    registration_tolerance: float = 0.5   # epsilon: assumed registration error in pixels (threshold tau + eps*|grad|)
    saturation_offset: float = 2.0        # mask saturated pixels only when a larger global offset is compensated

    def __post_init__(self) -> None:
        if not 0 < self.estimation_scale <= 1:
            raise ValueError("estimation_scale must be in (0, 1]")
        if self.max_shift <= 0:
            raise ValueError("max_shift must be positive")
        if not 0 <= self.min_response <= 1:
            raise ValueError("min_response must be in [0, 1]")
        if not 0 < self.min_residual_gain <= 1:
            raise ValueError("min_residual_gain must be in (0, 1]")


@dataclass(frozen=True)
class SchedulerConfig:
    """Parameters of the inference scheduler.

    ``refresh_interval`` (K): the detector is forced to run once K frames have
    passed since its last call, so retained detections are never older than
    K - 1 frames.

    ``min_interval`` (M): motion triggers the detector only if at least M frames
    have passed since its last call.  Motion is never dropped, only delayed: a
    suppressed trigger stays pending until the gap reaches M.  M = 1 reproduces
    the original rule "run on every motion frame".  Together the two knobs bound
    the invocation ratio between 1/K (static scene) and 1/M (continuous motion).
    """

    refresh_interval: int = 30
    min_interval: int = 1

    def __post_init__(self) -> None:
        if self.refresh_interval < 1:
            raise ValueError("refresh_interval must be at least 1 frame")
        if not 1 <= self.min_interval <= self.refresh_interval:
            raise ValueError("min_interval must satisfy 1 <= min_interval <= refresh_interval")

    @classmethod
    def from_max_staleness(cls, seconds: float, fps: float, min_interval: int = 1) -> "SchedulerConfig":
        """Derive K from a maximum tolerated detection age in seconds.

        Retained detections are at most ``K - 1`` frames, i.e. ``(K - 1) / fps``
        seconds, old; the largest K meeting the requirement is returned.
        """
        if seconds < 0 or fps <= 0:
            raise ValueError("seconds must be >= 0 and fps > 0")
        k = max(1, math.floor(seconds * fps) + 1)
        return cls(refresh_interval=k, min_interval=min(min_interval, k))
