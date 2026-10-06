from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

FRAME_SIZE = 640
CROP_ORIGIN = (85, 250)   # (x0, y0) of the 640x640 crop, as in the original study
GROUND_Y = 567            # bottom row of the pasted person (feet), crop coordinates
CANVAS_PAD = 16           # real scene content around the crop, revealed by camera jitter
NOISE_SIGMA = 1.5         # sensor noise per channel (8-bit scale)
NEAR_SCALE = 0.70         # person height ~317 px
SMALL_SCALE = 0.25        # person height ~113 px ("distant pedestrian")
N_FRAMES = 90             # 3 s at 30 FPS


@dataclass
class Scene:
    canvas: np.ndarray       # crop plus CANVAS_PAD pixels of real content on every side
    person: np.ndarray
    alpha: np.ndarray
    source_box: tuple[int, int, int, int]
    _sprites: dict = field(default_factory=dict, repr=False)

    @property
    def background(self) -> np.ndarray:
        p = CANVAS_PAD
        return self.canvas[p : p + FRAME_SIZE, p : p + FRAME_SIZE]

    def sprite(self, scale: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(scale, 4)
        if key not in self._sprites:
            h, w = self.person.shape[:2]
            size = (max(1, int(w * scale)), max(1, int(h * scale)))
            self._sprites[key] = (
                cv2.resize(self.person, size, interpolation=cv2.INTER_AREA),
                cv2.resize(self.alpha, size, interpolation=cv2.INTER_NEAREST),
            )
        return self._sprites[key]


def default_image_path() -> Path:
    from ultralytics.utils import ASSETS  # bus.jpg ships with the ultralytics package
    return Path(ASSETS) / "bus.jpg"


def prepare_scene(seg_weights: str | Path = "yolov8n-seg.pt", image_path: str | Path | None = None) -> Scene:
    from ultralytics import YOLO

    path = Path(image_path) if image_path else default_image_path()
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"cannot read {path}")
    result = YOLO(str(seg_weights))(image, device="cpu", verbose=False, retina_masks=True)[0]
    if result.masks is None:
        raise RuntimeError("the segmentation model returned no masks")
    height, width = image.shape[:2]
    persons = []
    for i, box in enumerate(result.boxes):
        if int(box.cls.item()) != 0:
            continue
        mask = (result.masks.data[i].cpu().numpy() > 0.5).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        touches = x1 == 0 or y1 == 0 or x2 == width or y2 == height
        persons.append({"mask": mask, "box": (x1, y1, x2, y2), "touches": touches, "ratio": (y2 - y1) / (x2 - x1)})
    candidates = [p for p in persons if not p["touches"]]
    if not candidates:
        raise RuntimeError("no fully visible person found in the source image")

    union = np.zeros((height, width), np.uint8)
    for p in persons:
        union |= p["mask"]
    union = cv2.dilate(union * 255, np.ones((11, 11), np.uint8), iterations=2)
    clean = cv2.inpaint(image, union, 7, cv2.INPAINT_TELEA)
    x0, y0, p = CROP_ORIGIN[0], CROP_ORIGIN[1], CANVAS_PAD
    canvas = clean[y0 - p : y0 + FRAME_SIZE + p, x0 - p : x0 + FRAME_SIZE + p].copy()

    # Cut-out: the fully visible, most upright pedestrian (largest height/width ratio).
    target = max(candidates, key=lambda p: p["ratio"])
    x1, y1, x2, y2 = target["box"]
    return Scene(canvas, image[y1:y2, x1:x2].copy(), (target["mask"][y1:y2, x1:x2] * 255).astype(np.uint8), target["box"])


def composite(canvas: np.ndarray, patch: np.ndarray, alpha: np.ndarray, x: int, y: int):
    """Alpha-composite ``patch`` at (x, y); return the frame and the visible tight box (or None)."""
    h, w = patch.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 >= x2 or y1 >= y2:
        return canvas, None
    px1, py1 = x1 - x, y1 - y
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)
    a = (alpha[py1:py2, px1:px2].astype(np.float32) / 255.0)[..., None]
    roi = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = np.clip(a * patch[py1:py2, px1:px2] + (1.0 - a) * roi, 0, 255).astype(np.uint8)
    yy, xx = np.nonzero(alpha[py1:py2, px1:px2])
    if xx.size == 0:
        return canvas, None
    return canvas, (x1 + int(xx.min()), y1 + int(yy.min()), x1 + int(xx.max()) + 1, y1 + int(yy.max()) + 1)


def pixel_position(x: float | None) -> int | None:
    return None if x is None else int(math.floor(x + 0.5))


def render_frame(scene, rng, *, x=None, scale=NEAR_SCALE, offset=0.0, shift=(0.0, 0.0), sigma=NOISE_SIGMA):
    """Render one frame: composite -> global intensity offset -> camera translation -> sensor noise.

    The person is composited on the padded canvas; the 640x640 frame is then
    extracted at a (sub-pixel) offset, so camera jitter reveals real scene
    content at the border instead of padding.  The target box is the tight box
    of the visible alpha mask in frame coordinates.
    """
    p, size = CANVAS_PAD, FRAME_SIZE
    frame = scene.canvas.copy()
    mask = None
    if x is not None:
        patch, alpha = scene.sprite(scale)
        px, py = pixel_position(x) + p, GROUND_Y - patch.shape[0] + p
        frame, _ = composite(frame, patch, alpha, px, py)
        mask = np.zeros(frame.shape[:2], np.uint8)
        composite(mask[..., None], np.full(patch.shape[:2] + (1,), 255, np.uint8), alpha, px, py)
    if offset:
        frame = np.clip(frame.astype(np.int16) + int(round(offset)), 0, 255).astype(np.uint8)
    dx, dy = shift
    if dx or dy:
        m = np.float32([[1, 0, dx - p], [0, 1, dy - p]])
        # Lanczos interpolation approximates a band-limited (physical) sub-pixel
        # camera translation; bilinear warping would blur each frame differently.
        frame = cv2.warpAffine(frame, m, (size, size), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        if mask is not None:
            mask = cv2.warpAffine(mask, m, (size, size), flags=cv2.INTER_LINEAR)
    else:
        frame = frame[p : p + size, p : p + size].copy()
        mask = None if mask is None else mask[p : p + size, p : p + size]
    box = None
    if mask is not None:
        ys, xs = np.nonzero(mask > 127)
        if xs.size:
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    if sigma > 0:
        frame = np.clip(frame.astype(np.float32) + rng.normal(0.0, sigma, frame.shape), 0, 255).astype(np.uint8)
    return frame, box


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    xs: tuple
    scale: float = NEAR_SCALE
    offsets: tuple | None = None
    jitter: float = 0.0
    has_person: bool = True


def build_scenarios(n: int = N_FRAMES) -> list[Scenario]:
    """Ten scenarios of ``n`` frames (defaults assume n = 90)."""
    near_w = int(121 * NEAR_SCALE)  # width of the near cut-out (84 px)
    third, two_third = n // 3, 2 * n // 3
    stop_go = [40 + 6 * min(t, third - 1) if t < two_third else 40 + 6 * (third - 1) + 6 * (t - two_third + 1)
               for t in range(n)]
    ese = []
    for t in range(n):
        if t < 10:
            ese.append(None)
        elif t < 20:
            ese.append(-near_w + 12 * (t - 9))
        elif t < 60:
            ese.append(-near_w + 120)
        else:
            ese.append(-near_w + 120 + 24 * (t - 59))
    illum = tuple(0 if t < third else (32 if t < two_third else -18) for t in range(n))
    none = (None,) * n
    return [
        Scenario("walk", "Walk", tuple(20 + 6 * t for t in range(n))),
        Scenario("walk_slow", "Slow walk", tuple(150 + 1.5 * t for t in range(n))),
        Scenario("stop_and_go", "Stop and go", tuple(stop_go)),
        Scenario("enter_stop_exit", "Enter, stop, exit", tuple(ese)),
        Scenario("stationary", "Stationary person", (280,) * n),
        Scenario("small_walk", "Distant walker", tuple(100 + 2 * t for t in range(n)), scale=SMALL_SCALE),
        Scenario("empty", "Empty scene", none, has_person=False),
        Scenario("illumination", "Illumination steps", none, offsets=illum, has_person=False),
        Scenario("jitter", "Camera jitter", none, jitter=3.0, has_person=False),
        Scenario("jitter_walk", "Jitter + walk", tuple(20 + 6 * t for t in range(n)), jitter=3.0),
    ]


@dataclass
class FrameRecord:
    index: int
    frame: np.ndarray
    box: tuple | None
    motion: bool


def scenario_rng(seed: int, scenario_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, scenario_index]))


def render_scenario(scene: Scene, scenario: Scenario, rng: np.random.Generator,
                    sigma: float = NOISE_SIGMA) -> Iterator[FrameRecord]:
    """Yield frames one by one (a 90-frame scenario would otherwise occupy ~110 MB)."""
    prev_pos, prev_visible = None, False
    for t, x in enumerate(scenario.xs):
        shift = tuple(rng.uniform(-scenario.jitter, scenario.jitter, 2)) if scenario.jitter else (0.0, 0.0)
        offset = scenario.offsets[t] if scenario.offsets else 0.0
        frame, box = render_frame(scene, rng, x=x, scale=scenario.scale, offset=offset, shift=shift, sigma=sigma)
        pos, visible = pixel_position(x), box is not None
        motion = t > 0 and (visible or prev_visible) and pos != prev_pos
        yield FrameRecord(t, frame, box, motion)
        prev_pos, prev_visible = pos, visible


# ---------------------------------------------------------------------------
# Operating-envelope sequences (gate only): one nuisance factor at a time.
# ---------------------------------------------------------------------------

def noise_sequence(scene, rng, sigma, n):
    return [render_frame(scene, rng, sigma=sigma)[0] for _ in range(n + 1)]


def illumination_sequence(scene, rng, delta, n):
    return [render_frame(scene, rng, offset=(delta if t % 2 else 0.0))[0] for t in range(n + 1)]


def jitter_sequence(scene, rng, amplitude, n):
    """Consecutive frames differ by a translation of exactly ``amplitude`` pixels in a random direction."""
    p = rng.uniform(-1.0, 1.0, 2)
    frames = []
    for _ in range(n + 1):
        frames.append(render_frame(scene, rng, shift=(float(p[0]), float(p[1])))[0])
        centre = math.atan2(-p[1], -p[0]) if np.hypot(*p) > 3.0 else rng.uniform(-math.pi, math.pi)
        phi = centre + (rng.uniform(-math.pi / 2, math.pi / 2) if np.hypot(*p) > 3.0 else 0.0)
        p = p + amplitude * np.array([math.cos(phi), math.sin(phi)])
    return frames


def walk_sequence(scene, rng, speed, scale, n):
    return [render_frame(scene, rng, x=20 + speed * t, scale=scale)[0] for t in range(n + 1)]
