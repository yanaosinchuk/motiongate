# Changelog

## Unreleased

- Clarify exact finite-horizon scheduler call bounds versus long-run invocation ratios.
- Add cache-independent `--fresh` benchmark runs and explicit one-thread CPU timing control.
- Make the manual tracking benchmark use the pinned reproducibility environment.
- Harden CI across Python 3.10 and 3.12 and add compile checks.
- Add citation metadata and tighten benchmark/reproduction documentation.

## 1.1.0

- Add opt-in sparse Lucas--Kanade box propagation between detector calls.
- Use forward--backward consistency and explicit tracking quality for reliability.
- Request re-detection when tracking degrades while preserving the scheduler's minimum-interval cost bound.
- Report whether boxes were tracked and account for tracker runtime in frame and aggregate timings.
- Keep tracking disabled by default so the published paper remains exactly reproducible.

## 1.0.0

- Stabilised motion gate with noise, illumination and camera-translation compensation.
- Scheduler with bounded staleness and detector-call rate.
- Reproducible controlled benchmark, real-video check, tests and accompanying paper.
