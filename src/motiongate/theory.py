"""Closed-form predictions used to explain (and test) the behaviour of the gates.

All quantities refer to 8-bit intensities.  Noise is modelled as independent,
zero-mean Gaussian noise with standard deviation ``sigma`` per colour channel
and pixel, drawn independently for every frame.

Rectification bias of the original gate
---------------------------------------
The original gate computes ``D = K * sum_c w_c |I_t,c - I_t-1,c|``: the absolute
value is applied *before* the Gaussian smoothing K.  For pure noise the channel
difference is N(0, 2 sigma^2), whose absolute value has mean ``2 sigma/sqrt(pi)``
(about 1.13 sigma).  Smoothing averages away the fluctuation but not this mean,
so the gate fires everywhere once ``1.13 sigma`` approaches the threshold tau.
The stabilised gate smooths a *signed* difference, which has mean zero; only
its (much smaller) standard deviation matters.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# OpenCV's BGR2GRAY weights (ITU-R BT.601) in B, G, R order.
GRAY_WEIGHTS = (0.114, 0.587, 0.299)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def kernel_energy(ksize: int = 5) -> float:
    """Sum of squared weights of OpenCV's 2-D Gaussian kernel (variance reduction factor)."""
    if ksize <= 1:
        return 1.0
    k1 = cv2.getGaussianKernel(ksize, 0)[:, 0]
    return float(np.sum(k1**2) ** 2)


def gray_energy() -> float:
    return float(sum(w * w for w in GRAY_WEIGHTS))


def classic_noise_moments(sigma: float, ksize: int = 5) -> tuple[float, float]:
    """Mean and standard deviation of the smoothed rectified difference for pure noise."""
    mean = 2.0 * sigma / math.sqrt(math.pi)
    var = kernel_energy(ksize) * gray_energy() * 2.0 * sigma**2 * (1.0 - 2.0 / math.pi)
    return mean, math.sqrt(var)


def robust_noise_sd(sigma: float, ksize: int = 5) -> float:
    """Standard deviation of the smoothed signed grayscale difference for pure noise."""
    return sigma * math.sqrt(2.0 * gray_energy() * kernel_energy(ksize))


def pixel_exceedance(sigma: float, tau: float, pipeline: str = "classic", ksize: int = 5) -> float:
    """Per-pixel probability that pure noise exceeds the threshold (normal approximation)."""
    if sigma <= 0:
        return 0.0
    if pipeline == "classic":
        mean, sd = classic_noise_moments(sigma, ksize)
        return 1.0 - _phi((tau - mean) / sd)
    if pipeline == "robust":
        sd = robust_noise_sd(sigma, ksize)
        return 2.0 * (1.0 - _phi(tau / sd))
    raise ValueError("pipeline must be 'classic' or 'robust'")


def critical_sigma(tau: float, pixel_probability: float, pipeline: str = "classic", ksize: int = 5) -> float:
    """Noise level at which the per-pixel exceedance probability reaches ``pixel_probability``."""
    lo, hi = 1e-6, 10.0 * tau + 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if pixel_exceedance(mid, tau, pipeline, ksize) < pixel_probability:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def percolation_probability(dilation: int = 3) -> float:
    """Per-pixel exceedance probability at which dilated exceedances start to connect.

    ``d`` dilations with a 3x3 element turn every exceeding pixel into a
    (2d+1) x (2d+1) square.  Randomly placed aligned squares of side L form
    spanning clusters once their density reaches eta_c / L^2 with
    eta_c ~= 1.0988 (continuum percolation of aligned squares, Mertens & Moore,
    2012).  Near this density clusters larger than A_min appear in every frame,
    so the gate switches from "never" to "always" active.
    """
    side = 2 * dilation + 1
    return 1.0988 / side**2


def critical_illumination_step(tau: float) -> float:
    """A uniform intensity step Delta changes every unsaturated pixel by |Delta|.

    Without photometric compensation the gate therefore fires once |Delta|
    exceeds tau (strictly: ``> tau`` because ``cv2.threshold`` keeps values
    greater than the threshold).
    """
    return float(tau)


def area_model_critical_speed(height: float, min_area: float = 900.0, dilation: int = 3) -> float:
    """Smallest displacement per frame that triggers the gate for a rigid silhouette.

    A silhouette of height h translating by v pixels changes two strips of about
    v x h pixels (leading and trailing edge).  d iterations of a 3x3 dilation
    grow each strip to (v + 2d) x (h + 2d).  If the strips stay separate the gate
    fires iff (v + 2d)(h + 2d) >= A_min.  Interior texture only adds changed
    pixels, so for textured objects this is a conservative estimate.
    """
    return min_area / (height + 2.0 * dilation) - 2.0 * dilation


def speedup(gate_cost: float, detector_cost: float, invocation_ratio: float) -> float:
    """S = C_y / (C_m + r C_y)."""
    return detector_cost / (gate_cost + invocation_ratio * detector_cost)


def break_even_ratio(gate_cost: float, detector_cost: float) -> float:
    """Gating pays off iff r < 1 - C_m / C_y."""
    return 1.0 - gate_cost / detector_cost


def invocation_bounds(refresh_interval: int, min_interval: int = 1) -> tuple[float, float]:
    """Asymptotic bounds 1/K <= r <= 1/M on the invocation ratio of the scheduler."""
    return 1.0 / refresh_interval, 1.0 / min_interval
