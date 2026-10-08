"""Shared benchmark constants.

Keeping experiment settings in one place prevents plotting, policy replay and
the orchestration script from silently drifting apart.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GATES = ("difference", "stabilized", "mog2")
LABEL = {
    "difference": "FD (original)",
    "stabilized": "FD-S (stabilised)",
    "mog2": "MOG2",
}
SHORT = {"difference": "FD", "stabilized": "FDS", "mog2": "MOG"}
COLOR = {
    "difference": "#D55E00",
    "stabilized": "#0072B2",
    "mog2": "#009E73",
    "fixed": "#7F7F7F",
    "every": "#000000",
}

REFRESH = 15
K_VALUES = (5, 10, 15, 30, 45, 90)
M_VALUES = (1, 2, 3, 4, 6, 8)
FIXED_N = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 45)

VIDEO_REFRESH = 10
VIDEO_M = (1, 2, 3, 4, 6)
VIDEO_FIXED_N = (1, 2, 3, 4, 5, 6, 8, 10)

IOU_THRESHOLD = 0.5
CONFIDENCE = 0.25
IMAGE_SIZE = 640

SWEEP_SIGMA = (0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 28, 32, 36, 40, 48)
SWEEP_DELTA = (0, 5, 10, 15, 18, 19, 20, 21, 22, 25, 30, 40, 60, 80)
SWEEP_JITTER = (0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8)
SWEEP_SPEED = (1, 2, 3, 4, 5, 6, 8, 10)
SWEEP_SCALES = (0.15, 0.25, 0.7)

E2E_SCENARIOS = ("stop_and_go", "enter_stop_exit", "empty", "jitter")
