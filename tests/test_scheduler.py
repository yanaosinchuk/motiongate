import numpy as np
import pytest

from benchmark.evaluation import schedule
from motiongate import Detection, GateResult, MotionGatedDetector, SchedulerConfig


class ScriptedGate:
    def __init__(self, decisions):
        self.decisions, self.t = list(decisions), -1

    def update(self, frame):
        self.t += 1
        return GateResult(active=bool(self.decisions[self.t]), ready=self.t > 0)

    def reset(self):
        pass


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
