# Panda MuJoCo native-reference pilot — November 1

The first native-physics pilot passes **4/4 scripted references**: Panda Lift and
Stack, each at seeds 7 and 13. All **four hold controls fail the task**, as intended.
Stack includes actual grasp, transport, placement, release and retreat, followed
by 30 settled support samples. No object attachment or pose/joint teleport is
performed by the runner after reset. These are privileged controller references,
not TAMP planning, production agent, observed-input or hardware successes.

The [staged plan](../plans/2026-11-01_tamp_sim_realism.md) and
[JSON protocol](../../configs/eval/tamp_sim_realism.json) include Stretch in MuJoCo;
only real Stretch hardware trials are paused. Stretch's existing simulated
adapter can support agent integration while Panda supplies upstream references.
Sourccey remains a separate adapted robot track requiring native task-adapter
qualification. Existing corrected
Sourccey asset/controller checks do not establish native pick/place.

The initial protocol had excluded Stretch entirely. The user clarified that the
exclusion concerns hardware only; this corrects future scope without changing
the Panda pilot or rescoring its results. Its original source/config hashes are
pinned to commit `51616f08` in the JSON record.

## Reported final condition

Managed job `20261101_093532_e0e09a`, `reference-v5`:

| Task | Seed | Reference: upstream / retained-and-settled checks | Native actions | Hold-control task success |
| --- | --- | --- | --- | --- |
| Lift | 7 | pass / pass | 110 | false, 400 actions |
| Lift | 13 | pass / pass | 114 | false, 400 actions |
| Stack | 7 | pass / pass | 215 | false, 400 actions |
| Stack | 13 | pass / pass | 258 | false, 400 actions |

Each reference/hold pair has identical initial simulator state, actuator controls,
model XML and controller configuration. Archived reference repeats also match the
initial states exactly. These are four development task/reset configurations,
repeated during implementation; the repeats are not independent benchmark tasks.

The environment is the full installed robosuite 1.5.2 task, using MuJoCo 3.5.0 and
native Panda OSC/gripper actions at 20 Hz through `RobosuiteAdapter` and
`EpisodeRuntime`. The OSC frame is explicitly world-relative; the archived
controller JSON records all settings. The upstream horizon remains 1000; this
diagnostic has a separately declared 400-action cap. Every gripper wait and settle
action consumes that cap. Native simulated time advances by 0.05 s per recorded
action. No renderer, local GPU, VLM or training is involved.

Upstream success remains unchanged. A stricter diagnostic requires 30 sampled
retained-lift states with bilateral contact, at least 8 cm lift and less than
1 cm relative-position spread. Stack additionally requires 30 post-retreat
released/support-contact states, 10 cm EE clearance, stable object position and
expected center/height over the support. Contact/poses are privileged scorer and
reference-controller inputs. Global collision integrity, sensor availability and
benchmark reset certification are explicitly unqualified.

## Failure boundary and retained attempts

Job `20261101_093535_586f8c` gives the Stack reference a one-action budget. It
executes exactly one native action, then reports
`diagnostic_step_budget_exhausted` with the pregrasp stage retained in
`stage_detail`. Both task and strict reference success remain false. Exit 1 is
expected for that failed reference, not an infrastructure error. Its episode
identity differs from the otherwise corresponding 400-action condition.

Earlier Lift-only, Stack, provenance and receipt iterations remain under
`.tamp-runs/sim-realism-20261101/`. The [compact JSON manifest](tamp_sim_realism_20261101.json)
records every run's job ID, summary hash and scope, including earlier one-action
failure controls. Final runs archive the runner and original task source,
compiled models, model XML, initial state/controls, controller JSON, original
scores and per-action traces. No dense trace arrays are copied into the repo.

## Validation and next gate

- Initial non-Stretch qualification: **36 passed**, no failures/errors/skips
  (`20261101_090449_f58eac`), covering the full native Panda environment/horizon
  seam and Sourccey model/FK/controller-profile checks.
- Shared benchmark runtime/task/search contracts: **42 passed**, no failures,
  errors or skips.
- Independent trace audit checks action counts/order/range, native simulated
  timing, all paired states/profiles, expected task controls, source snapshots,
  budget exhaustion and distinct condition identities. Ruff/whitespace checks pass.

Verified command, launched through the managed job supervisor with CPU-only
MuJoCo and single OpenMP/BLAS threads:

```bash
python scripts/eval_tamp_panda_reference.py --output RUN_DIR --tasks Lift Stack --seeds 7 13
```

CLI help also exposes `--policies` and `--max-steps`; the failure control uses
`--tasks Stack --seeds 7 --policies scripted_reference --max-steps 1`.

Panda next needs a joint-path-compatible native controller and full arm/payload
planner bound to the shared public TAMP tools, then reference requalification.
Simulated Stretch can use its existing privileged native adapter for agent tests
while that upstream binding is built. Its observed collision/contact source gate
still applies.
OSC endpoint success does not certify a redundant joint-space path. Panda agent
runs and clutter/buffer dependencies require that robot's joint-path gate first.
The existing simulated Stretch adapter can already exercise privileged agent
composition; assisted RGBD, observed grounding and observed disturbances still
require the corresponding source and execution qualification.
Both protocol files retain `executable_registry=false`; neither is a qualified
episode registry. No episode or suite provider is silently promoted to qualified status.
