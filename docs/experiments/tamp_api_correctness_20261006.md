# TAMP API correctness repairs (2026-10-06)

This change addresses the three API defects reproduced in review. It does not
change motion tolerances, add object-specific behavior, or claim simulator
readiness. The preceding live gate failed: CHAT placement failed, only one of
three fixtures was admitted, and that fixture passed two of three repeats.

## Repairs

- Immutable planning snapshots capture boot, scene, capabilities and copied
  placements before semantic/grasp/approach grounding. Every planner layer uses
  the supplied placements. Post-plan, storage and execution checks compare with
  the original snapshot; storage cannot legitimize a plan by capturing newer poses.
- Random namespaces prevent task/plan handle reuse across sessions/contexts.
  Discovery stays stable within a session. The scripted runner uses returned
  handles through `$task_ref` / `$plan_ref`, stops dependent calls after failure,
  and uses the resolved task for receptacle scoring.
- Each task operator starts a new evidence scope. The kinematic executor clears
  EE and lift checks, and the collector copies only the matching operation's
  evidence. Placement failure before arm motion no longer reports the preceding
  grasp's successful EE measurement as placement evidence.

The [canonical API](../apis/tamp.md) describes the unchanged version-1 envelope
and updated lifecycle rules. Shared planners accept an optional placements input;
legacy callers retain live-read defaults. Agent plan builds must carry their
original snapshot before a handle can be stored.

## Validation

Regressions cover movement during planning and before storage, boot/capability
changes, finite poses, rotation and quaternion sign equivalence, copied snapshot
independence, shared grounding inputs, handle uniqueness, runner substitutions,
failed-plan dependency stopping, and early placement failure after successful
measured grasp. Detailed live source/job provenance follows after submission.

Only the actual CHAT smoke is rerun for this API repair. The nine-trial matrix
remains deferred until motion changes justify repeating it. A failed motion smoke
with correct JSON and evidence remains a failed readiness gate.

## Frozen source and smoke submission

- Repair commit: `290a8147`; immutable checkout `/tmp/emet-tamp-api-fixes-20261006`.
- Full regression suite: **218 passed**. Ruff, shell syntax, and whitespace checks pass.
- Managed CHAT smoke: `20261006_160739_730b9a`.
- Artifacts: `~/runs/emet/tamp-api-correctness-20261006/`.
- Terminal result inspected October 8: **failed (exit 1)** during placement.

```bash
EMET_UV_RUN=1 .venv/bin/emet jobs status 20261006_160739_730b9a
EMET_UV_RUN=1 .venv/bin/emet jobs logs 20261006_160739_730b9a --tail 60
cat ~/runs/emet/tamp-api-correctness-20261006/results/gate_summary.txt
```

## Terminal result and diagnostic follow-up (2026-10-08)

The selected task was a bowl relocation to the discovered dishwasher. Pickup
completed; the three measured arm arrivals were accepted with errors 34.21,
33.83, and 34.38 mm against the unchanged 35 mm bound. Lift verification measured
4.82 mm target error and 160.43 mm upward displacement. These passes are close to
the arm tolerance and do not establish broad motion robustness.

Placement failed immediately after commanding base pose
`[-1.3003053344777857, -0.8677246613042688, 2.8715410047128485]`.
The captured exception was only `place_execution_error:RuntimeError`; its reason
was discarded. The final object-to-selected-receptacle error was 2.2018 m.
The live log alone cannot distinguish collision rejection from another command
failure.

Returned opaque handles worked, and JSON reported `partial`, completed approach
and grasp, and failure at place. Measurements contained grasp evidence only,
confirming that this early placement failure did not inherit grasp evidence as
placement evidence. This is a failed readiness gate despite correct API behavior
on the exercised path.

Private exception logging now retains the command failure reason without adding
simulator details to the public JSON. The focused TAMP and MCTS regression suite
passes **46 tests**, including a check that the private exception detail stays
out of the public result. A diagnostic rerun is needed to attribute the live
failure before changing placement approach selection.

An initial static reconstruction (`20261008_174338_e34ba5`, artifacts under
`~/runs/emet/tamp-place-diagnosis-20261008/`) used default model joint positions
and base height zero. Its collision results **cannot attribute the live failure**:
the supported live base height and joint state were not reconstructed. Keep this
artifact as preliminary evidence, not a collision or reachability verdict.

Diagnostic replay submitted October 8 as `20261008_174720_1cbb35`, frozen at
`4f16f27a` in `/tmp/emet-tamp-place-diagnosis-20261008`. It acquired the shared GPU
lock; results are pending. Artifacts:
`~/runs/emet/tamp-place-diagnosis-20261008/live-results/` and `live-job/job.log`.
This repeats the same actual CHAT smoke without changing task selection or bounds.
