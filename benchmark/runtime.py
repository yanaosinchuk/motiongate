"""Runtime helpers shared by benchmark stages."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """Raised before an expensive stage once the configured time budget is used."""


@dataclass
class TimeBudget:
    seconds: float | None = None
    start: float = field(default_factory=time.perf_counter)

    def configure(self, seconds: float | None) -> None:
        self.seconds = seconds if seconds and seconds > 0 else None
        self.start = time.perf_counter()

    def check(self, stage: str) -> None:
        if self.seconds is not None and time.perf_counter() - self.start > self.seconds:
            raise BudgetExceeded(stage)


BUDGET = TimeBudget()


def configure_budget(seconds: float | None) -> None:
    BUDGET.configure(seconds)


def check_budget(stage: str) -> None:
    BUDGET.check(stage)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
