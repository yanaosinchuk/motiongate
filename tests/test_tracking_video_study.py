from benchmark.tracking_video_study import ReplayDetector
from motiongate import Detection


def test_replay_detector_returns_reference_frame():
    reference = [
        (Detection((0, 0, 10, 10), 0.9),),
        (Detection((1, 1, 11, 11), 0.8),),
    ]
    detector = ReplayDetector(reference)
    detector.index = 1
    assert detector(None) == reference[1]
