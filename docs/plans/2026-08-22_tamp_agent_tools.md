# TAMP agent tools

**Date:** 2026-08-22
**Status:** Implemented offline path; managed simulator acceptance remains.

## Scope

CHAT manipulation has a plan-first path for simulated scenes:

```text
scene_tasks
    │ semantic object/receptacle queries + opaque task_ref
    ▼
plan_pick_place
    │ capability selection + live-scene grounding + IK/TAMP plan
    ▼
execute_pick_place_plan
    │ pose/capability revalidation immediately before execution
    ▼
guarded approach → grasp → place
```

The existing `pick_place` convenience tool uses the same plan-then-execute path
when a live simulator supports it. On hardware, or when no live simulator
grounding is available, it retains the configured manipulation controller.

## Semantic/oracle boundary

Agent-facing schemas contain semantic names and opaque handles only:

- `scene_tasks(object_filter, robot)` reports pickable categories, receptacles,
  reachability summaries, and `task_ref` values.
- `plan_pick_place(task_ref)` or
  `plan_pick_place(object_name, receptacle_name)` returns a `plan_ref`, selected
  capability path, grasp index, and operation names.
- `execute_pick_place_plan(plan_ref)` consumes a one-shot handle.

`object_gt_body` and `receptacle_gt_body` are simulator-adapter details. They
are held inside the session task registry and TAMP plan, where they are needed
to resolve MuJoCo placements and deterministic teleport/kinematic execution;
they are not included in CHAT tool descriptions or user-facing result text.
Metadata extraction may still emit the benchmark `object_gt_body` field for
offline episode generation.

## Guardrails and failure semantics

- A live simulation must advertise `is_simulation` and the required
  `kinematic_manip` or `sim_set_body_pose` capability.
- Semantic queries that match multiple objects or receptacles fail with an
  ambiguity reason and recommend `scene_tasks`.
- A task reference whose grounded body disappears is rejected rather than
  remapped to another instance.
- Before executing a stored plan, the adapter rechecks capability flags,
  object/receptacle existence, and both saved poses. A pose change greater than
  0.20 m returns `scene_changed_replan`.
- Stored plans are one-shot, including failed or stale execution attempts.
- The selected receptacle remains attached to the plan and is used for both
  execution and benchmark scoring. Invalid explicit scoring groundings fail
  closed instead of falling back to another category match.
- Floor controls set the configured `floor_z_m` before mapping in every
  `floor_object` episode. Setup failures abort the episode; `manip_mode: skip`
  remains find-only.

## Simulator matrix

| Path | Scene/capability | Execution |
|------|------------------|-----------|
| CHAT `plan_pick_place` | MolmoSpaces iTHOR + rby1/Galaxea, `kinematic_manip` | IK-ranked grasp and guarded kinematic approach/grasp/place |
| CHAT `pick_place` fallback | Real robot or no live sim | Existing configured controller |
| OVMM `manip_mode: mcts` | RoboCasa + rby1/Galaxea | MCTS assignment, kinematic arm execution, GT placement scoring |
| OVMM `manip_mode: sim` | RoboCasa/Stretch/Sourccey or MolmoSpaces | `sim_set_body_pose` teleport reference |
| OVMM `manip_mode: skip` | Any configured find episode | Find phases only; no manipulation phase |
| OVMM `manip_mode: oracle` | Benchmark control | Copies find success for an upper-bound control |

The OVMM `--manip-mode` override applies to every full-benchmark episode.
Without it, `emet ovmm full` uses the episode YAML value and falls back to
`oracle` only when the episode has no value. `emet ovmm find` always resolves
to `skip`.

## Acceptance criteria

### Offline gate

- CHAT and EQA tool packs remain disjoint.
- Schemas expose `plan_pick_place` and
  `execute_pick_place_plan` without raw simulator body IDs.
- Semantic task references resolve metadata mesh-child names to live freejoint
  placement keys without exposing those keys.
- Stale plans, missing capabilities, invalid receptacles, floor setup errors,
  and partial operation failures produce structured failure results.
- Dense SigLIP close-look crops use the mask head path, retain all spatial
  patches, send at most three unique crops, and report the actual count.

### Managed simulator gate

Run GPU/simulator checks only after `emet eval status`/`diagnose` and through
`emet jobs`:

1. MolmoSpaces iTHOR + rby1 kinematic CHAT plan/execute smoke.
2. Stretch teleport control.
3. RoboCasa floor suite with per-episode `mcts`, `sim`, and `skip` modes.
4. Compare `find_object_success`, `pick_success`, `place_success`, and
   `ovmm_full_partial`; report attach/release flakes separately from planning
   failures.

## Related entry points

- CHAT contract: [`docs/AGENT_RUN.md`](../AGENT_RUN.md)
- Full OVMM controls: [`docs/ovmm_full_benchmark.md`](../ovmm_full_benchmark.md)
- Semantic metadata extraction:
  [`src/emet/eval/scene_task_extractor.py`](../../src/emet/eval/scene_task_extractor.py)
- Agent adapter:
  [`src/emet/controller/task/tamp/agent_bridge.py`](../../src/emet/controller/task/tamp/agent_bridge.py)
