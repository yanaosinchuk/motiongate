import numpy as np
import pytest

from benchmark.evaluation import schedule
from motiongate import Detection, GateResult, MotionGatedDetector, SchedulerConfig, TrackingResult


class ScriptedGate:
    def __init__(self, decisions):
        self.decisions, self.t = list(decisions), -1

    def update(self, frame):
        self.t += 1
        return GateResult(active=bool(self.decisions[self.t]), ready=self.t > 0)

    def reset(self):
        self.t = -1


class ScriptedTracker:
    def __init__(self, results):
        self.results = list(results)
        self.t = -1
        self.initializations = 0

    def initialize(self, frame, detections):
        self.initializations += 1

    def update(self, frame):
        self.t += 1
        return self.results[self.t]

    def reset(self):
        self.t = -1


class CountingDetector:
    def __init__(self, outputs=None):
        self.calls, self.outputs = [], outputs

    def __call__(self, frame):
        self.calls.append(int(frame[0, 0, 0]))
        if self.outputs is not None:
            return self.outputs[int(frame[0, 0, 0])]
        return [Detection((0, 0, 10, 10), 0.9)]


def frames(n, shape=(8, 8, 3)):
    for t in range(n):
        f = np.zeros(shape, np.uint8)
        f[0, 0, 0] = t
        yield f


def run(decisions, k, m=1, outputs=None):
    det = CountingDetector(outputs)
    system = MotionGatedDetector(det, ScriptedGate(decisions), SchedulerConfig(k, m))
    return [system.process(f) for f in frames(len(decisions))], det, system


def test_initial_refresh_and_motion():
    results, det, _ = run([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], k=4)
    assert det.calls == [0, 3, 7]
    assert [r.reason for r in results][:4] == ["initial", "skipped", "skipped", "motion"]
    assert results[7].reason == "refresh"
    assert max(r.detection_age for r in results) == 3          # <= K - 1


def test_retained_detections_and_clearing():
    outputs = {0: [Detection((0, 0, 5, 5), 0.8)], 2: []}
    results, _, _ = run([0, 0, 1, 0], k=10, outputs=outputs)
    assert results[1].detections == results[0].detections and not results[1].fresh
    assert results[1].detection_age == 1
    assert results[2].detections == () and results[3].detections == ()   # exit clears stale boxes


def test_min_interval_rate_limits_and_keeps_pending_motion():
    _, det, system = run([0] + [1] * 11, k=30, m=3)
    assert det.calls == [0, 3, 6, 9]
    assert system.stats.invocation_ratio == pytest.approx(4 / 12)
    _, det, _ = run([0, 1, 0, 0, 0, 0], k=30, m=3)
    assert det.calls == [0, 3]      # motion at t=1 is delayed to t=3, not dropped


def test_resolution_change_restarts_stream():
    det = CountingDetector()
    system = MotionGatedDetector(det, ScriptedGate([0, 0, 0]), SchedulerConfig(10))
    system.process(np.zeros((8, 8, 3), np.uint8))
    system.process(np.zeros((8, 8, 3), np.uint8))
    assert system.process(np.zeros((16, 16, 3), np.uint8)).reason == "initial"


@pytest.mark.parametrize("k,m", [(1, 1), (5, 1), (15, 1), (15, 4), (8, 8)])
def test_offline_replay_matches_online_scheduler(k, m):
    rng = np.random.default_rng(k * 10 + m)
    decisions = [False] + list(rng.random(200) < 0.3)
    results, _, _ = run(decisions, k=k, m=m)
    invoked, source = schedule(len(decisions), gate=decisions, refresh=k, min_interval=m)
    assert [r.fresh for r in results] == list(invoked)
    assert [r.index - r.detection_age for r in results] == list(source)


def test_config_validation_and_staleness_conversion():
    with pytest.raises(ValueError):
        SchedulerConfig(0)
    with pytest.raises(ValueError):
        SchedulerConfig(5, 6)
    cfg = SchedulerConfig.from_max_staleness(1.0, 30)
    assert cfg.refresh_interval == 31        # ages <= 30 frames = 1 s


def test_reset_starts_a_new_stream_and_clears_stats():
    results, _, system = run([0, 0, 0], k=10)
    assert results[-1].index == 2
    assert system.stats.frames == 3

    system.reset()
    result = system.process(next(frames(1)))

    assert result.index == 0
    assert result.reason == "initial"
    assert result.detection_age == 0
    assert system.stats.frames == 1
    assert system.stats.invocations == 1


def test_reset_stats_preserves_stream_state():
    det = CountingDetector()
    system = MotionGatedDetector(det, ScriptedGate([0, 0, 0]), SchedulerConfig(10))
    stream = frames(3)
    system.process(next(stream))
    second = system.process(next(stream))
    assert second.detection_age == 1

    system.reset_stats()
    next_result = system.process(next(stream))

    assert next_result.index == 2
    assert next_result.reason == "skipped"
    assert next_result.detection_age == 2
    assert system.stats.frames == 1


def test_resolution_change_preserves_global_stats_and_index():
    det = CountingDetector()
    system = MotionGatedDetector(det, ScriptedGate([0, 0, 0]), SchedulerConfig(10))
    system.process(np.zeros((8, 8, 3), np.uint8))
    system.process(np.zeros((8, 8, 3), np.uint8))
    result = system.process(np.zeros((16, 16, 3), np.uint8))

    assert result.reason == "initial"
    assert result.index == 2
    assert system.stats.frames == 3


def test_from_max_staleness_rejects_incompatible_min_interval():
    with pytest.raises(ValueError, match="min_interval"):
        SchedulerConfig.from_max_staleness(0.1, 10, min_interval=3)


def test_reliable_tracker_updates_boxes_on_skipped_frame():
    tracked_detection = Detection((2, 1, 12, 11), 0.9)
    tracker = ScriptedTracker([TrackingResult((tracked_detection,), 0.9, True, 8)])
    detector = CountingDetector()
    system = MotionGatedDetector(
        detector,
        ScriptedGate([0, 0]),
        SchedulerConfig(10),
        tracker=tracker,
    )
    stream = frames(2)
    first = system.process(next(stream))
    second = system.process(next(stream))

    assert first.fresh
    assert not second.fresh
    assert second.tracked
    assert second.tracking_quality == pytest.approx(0.9)
    assert second.detections == (tracked_detection,)
    assert second.detection_age == 1


def test_tracker_failure_requests_refresh_without_breaking_rate_limit():
    failed = TrackingResult((Detection((0, 0, 10, 10), 0.9),), 0.1, False, 1)
    tracker = ScriptedTracker([failed] * 9)
    detector = CountingDetector()
    system = MotionGatedDetector(
        detector,
        ScriptedGate([0] * 10),
        SchedulerConfig(30, 3),
        tracker=tracker,
    )

    results = [system.process(frame) for frame in frames(10)]

    assert detector.calls == [0, 3, 6, 9]
    assert results[1].reason == "skipped"
    assert results[2].reason == "skipped"
    assert results[3].reason == "tracking"
    assert all(
        b - a >= 3
        for a, b in zip(detector.calls, detector.calls[1:])
    )


def test_tracker_is_not_run_when_motion_already_requires_detection():
    tracked_detection = Detection((2, 1, 12, 11), 0.9)
    tracker = ScriptedTracker([TrackingResult((tracked_detection,), 0.9, True, 8)])
    detector = CountingDetector()
    system = MotionGatedDetector(
        detector,
        ScriptedGate([0, 1]),
        SchedulerConfig(10, 1),
        tracker=tracker,
    )
    stream = frames(2)
    system.process(next(stream))
    second = system.process(next(stream))

    assert second.reason == "motion"
    assert second.fresh
    assert second.tracking_quality is None
    assert tracker.t == -1
