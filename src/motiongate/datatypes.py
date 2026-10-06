"""Plain data types exchanged between the gate, the detector and the scheduler.

All boxes use the ``(x1, y1, x2, y2)`` convention in pixel coordinates of the
frame that was passed to the component.  The types are immutable so that a
retained detection can never be modified accidentally between frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BoxXYXY = tuple[float, float, float, float]
IntBoxXYXY = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Detection:
    """One detector output: an axis-aligned box, a confidence score and a class id."""

    box: BoxXYXY
    score: float
    class_id: int = 0


@dataclass(frozen=True, slots=True)
class GateResult:
    """Decision of a motion gate for one frame.

    Attributes
    ----------
    active:
        ``True`` if at least one change region passed the area filter.
    ready:
        ``False`` while the gate has no reference yet (first frame after a reset).
    boxes:
        Bounding rectangles of the qualifying change regions.
    largest_area:
        Area (in pixels) of the largest change contour, qualifying or not.
    shift:
        Estimated global translation ``(dx, dy)`` between the previous and the
        current frame.  Always ``(0, 0)`` for gates without motion compensation.
    offset:
        Estimated global intensity offset removed before thresholding.
    mask:
        Optional binary change mask (only kept when ``keep_mask=True``).
    """

    active: bool
    ready: bool = True
    boxes: tuple[IntBoxXYXY, ...] = ()
    largest_area: float = 0.0
    shift: tuple[float, float] = (0.0, 0.0)
    offset: float = 0.0
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Everything the scheduler knows about one processed frame.

    ``fresh`` distinguishes detections computed on *this* frame from detections
    retained from an earlier frame; ``detection_age`` is the number of frames
    since the retained detections were produced (0 when fresh).
    """

    index: int
    detections: tuple[Detection, ...]
    fresh: bool
    reason: str
    detection_age: int
    gate: GateResult
    gate_ms: float
    detector_ms: float

    @property
    def total_ms(self) -> float:
        return self.gate_ms + self.detector_ms
