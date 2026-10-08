import argparse

import pytest

from motiongate.cli import _scheduler_config, build_parser


def parse(*args: str) -> argparse.Namespace:
    return build_parser().parse_args(["input.mp4", *args])


def test_scheduler_config_uses_explicit_refresh_without_clamping():
    cfg = _scheduler_config(parse("--refresh", "15", "--min-interval", "4"), fps=30.0)
    assert cfg.refresh_interval == 15
    assert cfg.min_interval == 4


def test_scheduler_config_rejects_invalid_interval_pair():
    with pytest.raises(ValueError, match="min_interval"):
        _scheduler_config(parse("--refresh", "5", "--min-interval", "6"), fps=30.0)


def test_scheduler_config_from_seconds_rejects_incompatible_interval():
    with pytest.raises(ValueError, match="min_interval"):
        _scheduler_config(parse("--max-staleness", "0.1", "--min-interval", "5"), fps=10.0)


def test_refresh_and_max_staleness_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse("--refresh", "15", "--max-staleness", "0.5")


def test_max_frames_must_be_non_negative():
    with pytest.raises(SystemExit):
        parse("--max-frames", "-1")
