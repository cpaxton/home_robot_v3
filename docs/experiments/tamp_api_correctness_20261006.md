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
- At handoff, the smoke is **waiting for the shared GPU lock**, behind another
  agent's navigation evaluation. It has not produced a live result yet.

```bash
EMET_UV_RUN=1 .venv/bin/emet jobs status 20261006_160739_730b9a
EMET_UV_RUN=1 .venv/bin/emet jobs logs 20261006_160739_730b9a --tail 60
cat ~/runs/emet/tamp-api-correctness-20261006/results/gate_summary.txt
```

When terminal, inspect the JSON response and final-scene score separately. For an
early placement-approach failure, completed grasp evidence must remain, and no
placement EE record may reuse it. No new motion-success claim is made at handoff.
