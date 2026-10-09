# Optical-flow tracking benchmark

This note documents the post-paper experiment that compares the original
**retain-last-box** policy with MotionGate v1.1's optional sparse optical-flow
tracker. It is deliberately separate from the seminar paper: the paper's
algorithm, tables and reported numbers are not retroactively changed.

## Question

When the detector is not invoked on a frame, is it better to keep the most
recent detector box fixed, or to propagate it with lightweight optical flow?

The comparison is designed so that the detector itself is not a confounder.
YOLOv8m is run once on every controlled frame and its outputs and per-frame
runtime are cached. Both policies then use exactly those same detector outputs
at the frames selected by the scheduler.

## Protocol

- 10 deterministic controlled scenarios, 90 frames each (900 frames total)
- stabilised motion gate (FD-S)
- refresh interval `K = 15`
- minimum detector intervals `M in {1, 2, 4}`
- YOLOv8m person detector
- IoU hit threshold: 0.5
- seed: `20260922`
- tracker: Shi--Tomasi features + pyramidal Lucas--Kanade flow
- reliability: forward--backward consistency, minimum point count and minimum
  retained-point quality
- tracker cost is included in the runtime estimate

The reported run used one controlled rendering seed. This is an engineering
comparison of the two box-propagation policies, not a claim of general
real-world tracking accuracy.

## Aggregate results

| Policy | M | Detector ratio | Recall | Mean IoU | Estimated speed-up |
|---|---:|---:|---:|---:|---:|
| Retain | 1 | 0.542 | 0.9967 | 0.9810 | 1.82x |
| Flow | 1 | 0.542 | 0.9967 | 0.9810 | 1.80x |
| Retain | 2 | 0.291 | 0.9951 | 0.9394 | 3.35x |
| Flow | 2 | 0.291 | 0.9951 | 0.9795 | 3.30x |
| Retain | 4 | 0.168 | 0.9707 | 0.8794 | 5.70x |
| Flow | 4 | 0.168 | 0.9919 | 0.9805 | 5.62x |

At `M=1`, there is almost no room for tracking to help because the detector is
already invoked on most frames where a person is moving.

At `M=2`, optical flow preserves the same recall and detector-call ratio while
raising mean IoU from **0.9394 to 0.9795**. The additional tracker cost reduces
the estimated speed-up only slightly, from **3.35x to 3.30x**.

At `M=4`, the difference becomes much larger. Optical flow raises recall from
**0.9707 to 0.9919** and mean IoU from **0.8794 to 0.9805**, with the detector
still invoked on only **16.8%** of frames. Estimated speed-up changes from
**5.70x to 5.62x**.

A particularly useful comparison is the tracked `M=4` mode against the
paper-style retain `M=1` operating point. The tracked mode uses **69% fewer
detector calls** (151 instead of 488 across the 900-frame benchmark), while
recall falls by only **0.49 percentage points** and mean IoU is essentially
unchanged (0.9805 vs. 0.9810).

## Where the gain comes from

The largest localisation improvements at `M=4` occur in the moving-person
scenarios:

| Scenario | Retain mean IoU | Flow mean IoU |
|---|---:|---:|
| Walk | 0.815 | 0.981 |
| Slow walk | 0.943 | 0.982 |
| Stop and go | 0.872 | 0.983 |
| Distant walker | 0.820 | 0.970 |
| Jitter + walk | 0.804 | 0.984 |

For the enter-stop-exit sequence, recall also rises from **0.760 to 0.933** at
`M=4`. This is the expected failure mode of frozen boxes: when inference is
rate-limited, the retained box can lag behind a moving target or remain visible
after the target has moved substantially. Optical flow reduces that
localisation staleness between detector calls.

## Interpretation

The experiment supports a narrow but useful conclusion:

> Lightweight optical-flow propagation can preserve detector-level
> localisation at substantially lower detector invocation rates than retaining
> the last box unchanged, on the controlled MotionGate benchmark.

It does **not** establish that sparse optical flow is a universally strong
multi-object tracker. The benchmark contains a single composited pedestrian
and controlled nuisance factors. Occlusion, identity switches, non-rigid scale
changes, long-term disappearance and crowded scenes need a separate
real-world tracking benchmark.

## Reproduction

```bash
pip install -e ".[benchmark]"
python -m benchmark.tracking_run
```

The command writes the aggregate results to
`results/tracking_controlled.csv`, per-scenario results to
`results/tracking_scenarios.csv`, metadata to
`results/tracking_metadata.json`, and the comparison figure to
`figures/fig_tracking_comparison.{png,pdf}`.
