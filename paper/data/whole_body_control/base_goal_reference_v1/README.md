# Continuous base endpoint references

See the [report](../../../../docs/experiments/base_goal_reference_20261116.md).
This is a reference-construction and control-formulation/calibration experiment,
with unchanged learned parameter tensors and no new PPO steps. Mars supplies
native rollout evidence; both Mars and Stretch have compiler/actor unit coverage.
It is not a learned-policy improvement estimate or hardware qualification.

`results.json` separates every failed/successful development producer, the
actual exported candidate's existing cohorts, raw static goals, compiled
rest-to-rest development endpoints, and the independently frozen endpoint audit.
Variants reuse deterministic cases and are not pooled as independent trials.
Governed runs report final endpoint error separately from active-reference lag.

The exact probe/native producers are preserved as `.py.txt` files. Shared
evidence aggregation and offline W&B logging producers are preserved alongside
them. Case-per-line JSON retains the complete compact records without thousands
of incidental pretty-print lines. Shared
collection/evaluation helpers are preserved in the adjacent
`base_direction_bootstrap_v1/train.py.txt` and
`base_navigation_feedback_v1/readout-runner.py.txt`; `helper-hashes.json` binds
the raw helper files used here. `lateral-goal.svg` shows the same development
lateral goal, including its original failure. This directory contains compact
summaries, protocols and hashes; raw traces, weights, XML and source archives
remain ignored under logs. No compressed evidence blobs are committed.

The compiler has a distinct continuous-reference record format. Its display
samples are not a wire-compatible linear task-pose schedule. Initial rest,
geometry, reference derivative bounds and dynamic execution are separate checks.
Moving-start replans, continuous UI dragging, loaded tasks, disturbance/jerk
qualification, actor promotion and Herman hardware motion are outside this
readout. No service default is changed.
