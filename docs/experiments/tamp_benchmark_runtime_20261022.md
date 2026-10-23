# MuJoCo benchmark runtime foundation validation — October 22, 2026

Branch: `feat/tamp-benchmark-runtime`, based on #213 head `b7bf7641`.
Implementation scope and remaining gates:
[shared-agent benchmark plan](../plans/2026-10-22_mujoco_tamp_benchmarks.md).

## Results

Managed CPU-only regression job `20261022_083434_cb46b3`: **258 passed,
0 failures, 0 errors, 0 skips**, 67.87 s. JUnit:
`/tmp/emet-tamp-benchmark-regression.xml`; job log:
`/tmp/emet-tamp-benchmark-regression-job/job.log`.

This includes the prior 175-test #213 validation set, plus the new runtime and
search tests, renderer-free robosuite smoke, production CHAT prompt/tool checks,
and expanded failed-batch dispatch checks. These overlapping test batches are
not independent experiments or task success denominators.

Standalone native smoke job `20261022_083223_f20808`: **1 passed**, 2.93 s.
It constructs the full robosuite Panda Lift task with seed 7, cameras/renderers
disabled, applies three zero native controller actions, and checks that the
upstream timestep and runtime counter both equal 3. The environment terminates
at the configured horizon and reports **unsuccessful Lift**, as expected. No
further command is accepted. This is native control/termination evidence;
it is not a successful manipulation, certified fixed benchmark reset, or agent
trial. MuJoCo 3.5.0, robosuite 1.5.2.

Other checks: Ruff clean on all changed Python modules; `git diff --check`
clean; `./paper/build.sh` succeeds. Paper text makes no new household performance
claim. Existing stationary-table 2/6 task, 6/12 goal, 6/6 attempted transfer
results are preserved unchanged.

## Covered failure boundaries

- Every gripper/settling/skill-internal native action consumes the horizon.
- Terminated, wall-budget-exhausted and failed-step runtimes reject commands.
- Reset/step exceptions close the upstream environment.
- Upstream success remains independent of final-goal and physical-integrity scores.
- Changed boot, reset or revision rejects a consumed task plan before motion.
- A scene change during feasibility preflight prevents execution.
- Measured missing effects stop a plan even if the command reports success.
- Protected predicates cannot be sacrificed to complete another goal.
- A three-transfer causal swap is found through temporary storage by beam and
  existing UCT, without live mutation during search.
- Planning deadlines and expansion limits fail explicitly; unavailable strategy
  names never select another algorithm.
- Unsupported, missing and infrastructure outcomes remain in fixed-population
  accounting. Unknown physical evidence does not pass strict acceptance.
- New CHAT tools use the existing JSON outcome and batch-stop machinery. They
  are omitted from normal production sessions without a bound task provider.

## Not established

No suite-specific household provider is qualified. LLM sequence/ranking and
PDDLStream bindings, full shared-budget wiring, certified reset manifests,
observed-input execution, upstream/adapted-robot parity and benchmark agent
success remain outstanding. No Caliban VLM run was needed for these CPU contract
and runtime checks. A native runtime failure control must not be counted as an
agent task attempt or as successful Lift.
