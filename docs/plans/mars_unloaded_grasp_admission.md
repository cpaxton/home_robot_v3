# Unloaded grasp-pose boundary and teacher admission

This is the completed experiment's predeclared protocol. The boundary is now
implemented; its [readout](../experiments/mars_unloaded_grasp_admission_20261116.md)
has closed admission. Continue from the unloaded-prefix planning milestone in
[Active work](../rl/ACTIVE_WORK.md), rather than implementing this boundary
again or automatically rerunning the same failed population.

Continue the [WBC plan](mars_wbc_next_steps.md) after
[the task-stage diagnosis](../experiments/mars_task_arrival_20261115.md).
The primitive environment originally stopped at pregrasp. The extension adds an
explicit `primitive="grasp_pose"` option that executes the real public-tool
approach, pregrasp and grasp pose, unwinding the executor **before closure**.
Default `pregrasp` behavior and all original tolerances remain unchanged.

## Fixed experiment, declared before native admission

Run exactly twelve teacher episodes: both boundaries on the original four
development and two audit cases in
`configs/benchmarks/mars_whole_body_v2_development.json`, in manifest order.
Use current-COM wheel feedback from fixture construction, no added initial
settling, original nominal reset, original stage tolerances and `s_curve`
references. No new mass/seed/reset selection, task-region enlargement, time
extension, physics change, loaded execution or PPO update.

Each terminal boundary still requires two seconds of measured quiet hold and a
confirmed stop. Record pose residual and open-jaw state after that hold/stop;
the new grasp-pose completion and both stage readouts require those to remain
inside the original contract. Historical pregrasp success semantics stay fixed.
Keep individual tick rewards and put the terminal bonus on the
final physical tick, preserving physical-time PPO discounting. Publish stage,
boundary-reached flag, actual initial-state/model identities, safety flags,
command identities and all failed outcomes. A failed earlier approach remains
a failure, rather than a successful grasp-pose observation.

The stage-specific readout requires **all six** cases to succeed, without
borrowing the old pregrasp challenge exception. It rejects stage mixing,
missing/duplicated cases, unknown stops, safety faults, learned-controller
inference, closure and commands beyond the chosen boundary. Even successful
stage admission alone is not a training token: full source-bound regressions,
general-task admission and a separately registered mixed-training protocol
remain required. Loaded retention/support admission stays independent.

Run on one CPU14 native worker; reserve CPU15 for focused/full checks. Threads
one, CUDA hidden, ordinary rendering disabled, through managed jobs. Freeze
source before running. The canonical full suite may use its existing EGL
exception for renderer tests; no new render or GPU training is introduced.

```bash
env PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=disable \
  taskset -c 14 .venv/bin/python scripts/validate_mars_unloaded_primitives.py \
  --output logs/eval/mars-unloaded-grasp-admission-v1
```

Use `emet jobs run` around that command after coordinating the lane; output
must not exist. Retain raw traces under ignored logs and publish compact
results with source/row hashes. Repeating two stages at the same case/seed
does not create twelve independent reset trials.

## Follow-through

If any teacher case fails, keep it in the denominator and diagnose its stage;
do not silently discard it or reuse the historical exception. No automatic
training or retry follows. The fixed-world-hand/base trajectory from the
previous diagnosis remains a separate general-controller coverage task. This
change does not implement its curriculum, rewrite PR #240's audit, or change
the outer general/TAMP training balance.
