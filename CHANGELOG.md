# Changelog

## 1.1.0

- Add opt-in sparse Lucas--Kanade box propagation between detector calls.
- Use forward--backward consistency and explicit tracking quality for reliability.
- Request re-detection when tracking degrades while preserving the scheduler's minimum-interval cost bound.
- Report whether boxes were tracked and account for tracker runtime in frame and aggregate timings.
- Keep tracking disabled by default so the published paper remains exactly reproducible.
- Add a controlled oracle-detector benchmark showing that at K=15, M=8 optical flow improves frame recall from 0.782 to 0.987 and mean IoU from 0.748 to 0.987 at the same detector-call ratio.

## 1.0.0

- Stabilised motion gate with noise, illumination and camera-translation compensation.
- Scheduler with bounded staleness and detector-call rate.
- Reproducible controlled benchmark, real-video check, tests and accompanying paper.
