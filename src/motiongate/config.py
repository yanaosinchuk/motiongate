"""Validated configuration objects.

The defaults reproduce the original OpenCV prototype: a 5x5 Gaussian kernel,
threshold 20, three dilations and a minimum contour area of 900 pixels.
Validation happens once at construction time so invalid parameters fail loudly
instead of being silently corrected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


@dataclass(frozen=True)
class GateConfig:
    """Parameters shared by all difference-based gates."""

    threshold: float = 20.0
    min_area: float = 900.0
    dilation_iterations: int = 3
    blur_kernel: int = 5
    keep_mask: bool = False

    def __post_init__(self) -> None:
        _require_finite("threshold", self.threshold)
        _require_finite("min_area", self.min_area)
        if not 0 <= self.threshold < 255:
            raise ValueError(f"threshold must be in [0, 255), got {self.threshold}")
        if self.min_area <= 0:
            raise ValueError(f"min_area must be positive, got {self.min_area}")
        if not isinstance(self.dilation_iterations, int) or isinstance(self.dilation_iterations, bool):
            raise TypeError("dilation_iterations must be an integer")
        if self.dilation_iterations < 0:
            raise ValueError("dilation_iterations must be non-negative")
        if not isinstance(self.blur_kernel, int) or isinstance(self.blur_kernel, bool):
            raise TypeError("blur_kernel must be an integer")
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError(f"blur_kernel must be a positive odd integer, got {self.blur_kernel}")


@dataclass(frozen=True)
class StabilizationConfig:
    """Extra parameters of the stabilised, illumination-compensated gate."""

    compensate_motion: bool = True
    compensate_illumination: bool = True
    estimation_scale: float = 0.5
    max_shift: float = 20.0
    min_response: float = 0.05
    min_residual_gain: float = 0.9
    min_shift: float = 0.1
    saturation_margin: float = 6.0
    registration_tolerance: float = 0.5
    saturation_offset: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "estimation_scale",
            "max_shift",
            "min_response",
            "min_residual_gain",
            "min_shift",
            "saturation_margin",
            "registration_tolerance",
            "saturation_offset",
        ):
            _require_finite(name, getattr(self, name))
        if not 0 < self.estimation_scale <= 1:
            raise ValueError("estimation_scale must be in (0, 1]")
        if self.max_shift <= 0:
            raise ValueError("max_shift must be positive")
        if not 0 <= self.min_response <= 1:
            raise ValueError("min_response must be in [0, 1]")
        if not 0 < self.min_residual_gain <= 1:
            raise ValueError("min_residual_gain must be in (0, 1]")
        if self.min_shift < 0:
            raise ValueError("min_shift must be non-negative")
        if not 0 <= self.saturation_margin < 127.5:
            raise ValueError("saturation_margin must be in [0, 127.5)")
        if self.registration_tolerance < 0:
            raise ValueError("registration_tolerance must be non-negative")
        if self.saturation_offset < 0:
            raise ValueError("saturation_offset must be non-negative")


@dataclass(frozen=True)
class TrackingConfig:
    """Parameters of the sparse Lucas--Kanade box tracker."""

    max_corners_per_box: int = 40
    quality_level: float = 0.01
    min_distance: float = 5.0
    block_size: int = 7
    win_size: int = 21
    max_level: int = 3
    fb_threshold: float = 1.5
    min_points: int = 4
    min_quality: float = 0.5

    def __post_init__(self) -> None:
        for name in ("quality_level", "min_distance", "fb_threshold", "min_quality"):
            _require_finite(name, getattr(self, name))
        for name in ("max_corners_per_box", "block_size", "win_size", "max_level", "min_points"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if self.max_corners_per_box < 1:
            raise ValueError("max_corners_per_box must be at least 1")
        if not 0 < self.quality_level <= 1:
            raise ValueError("quality_level must be in (0, 1]")
        if self.min_distance < 0:
            raise ValueError("min_distance must be non-negative")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.win_size < 3:
            raise ValueError("win_size must be at least 3")
        if self.max_level < 0:
            raise ValueError("max_level must be non-negative")
        if self.fb_threshold <= 0:
            raise ValueError("fb_threshold must be positive")
        if self.min_points < 1:
            raise ValueError("min_points must be at least 1")
        if not 0 <= self.min_quality <= 1:
            raise ValueError("min_quality must be in [0, 1]")


@dataclass(frozen=True)
class SchedulerConfig:
    """Parameters of the inference scheduler.

    refresh_interval (K) forces a detector call once K frames have passed since
    the previous call. min_interval (M) rate-limits motion-triggered calls.
    Motion is delayed rather than discarded.
    """

    refresh_interval: int = 30
    min_interval: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.refresh_interval, int) or isinstance(self.refresh_interval, bool):
            raise TypeError("refresh_interval must be an integer")
        if not isinstance(self.min_interval, int) or isinstance(self.min_interval, bool):
            raise TypeError("min_interval must be an integer")
        if self.refresh_interval < 1:
            raise ValueError("refresh_interval must be at least 1 frame")
        if not 1 <= self.min_interval <= self.refresh_interval:
            raise ValueError("min_interval must satisfy 1 <= min_interval <= refresh_interval")

    @classmethod
    def from_max_staleness(cls, seconds: float, fps: float, min_interval: int = 1) -> "SchedulerConfig":
        """Derive K from a maximum tolerated detection age in seconds.

        Retained detections are at most K - 1 frames old. Invalid combinations
        are rejected rather than silently clamped.
        """
        _require_finite("seconds", seconds)
        _require_finite("fps", fps)
        if seconds < 0 or fps <= 0:
            raise ValueError("seconds must be >= 0 and fps > 0")
        k = max(1, math.floor(seconds * fps) + 1)
        return cls(refresh_interval=k, min_interval=min_interval)
