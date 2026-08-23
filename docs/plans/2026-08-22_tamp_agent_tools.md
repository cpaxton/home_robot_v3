# TAMP agent tools

**Date:** 2026-08-22
**Status:** Offline gate green; managed simulator gate partially green (see below).

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
`emet jobs`. Orchestrator: [`scripts/run_tamp_agent_tools_gate.sh`](../../scripts/run_tamp_agent_tools_gate.sh).

```bash
uv run emet eval recover --need-mib 12000
# Routine agent-contract smoke (~2–5 min; rby1 + simulated kinematic base snap):
uv run emet jobs run --name tamp-agent-tools-gate --need-mib 12000 --gpu-exclusive -- \
  ./scripts/run_tamp_agent_tools_gate.sh
# Overnight / paper matrix (1–2 h; full RoboCasa floor suite incl. Stretch):
PROFILE=full uv run emet jobs run --name tamp-agent-tools-gate-full --need-mib 12000 --gpu-exclusive -- \
  ./scripts/run_tamp_agent_tools_gate.sh
# or queue with status log + native scheduling:
./scripts/schedule_tamp_agent_tools_gate.sh
# PROFILE=full DELAY_MIN=120 ./scripts/schedule_tamp_agent_tools_gate.sh
```

**Optional OVMM check** (single rby1 MCTS floor episode, ~5–8 min; avoids Stretch
head sweeps that take 15–45 s each in agentic find):

```bash
uv run python scripts/eval_tamp_floor.py --smoke
```

Queue when another GPU job is live (`--delay-minutes` / `--at` sleep inside the
supervisor; `--gpu-exclusive` waits for unmanaged MuJoCo/VLM servers too). Dry-run
quoting smoke (no sim): `DRY_RUN=1 ./scripts/run_tamp_agent_tools_gate.sh`.

| Profile | Items | Typical wall time |
|---------|-------|-------------------|
| `smoke` (default) | **kinematic**: rby1 CHAT `scene_tasks` → plan → execute | ~2–5 min |
| `full` | chat, kinematic, stretch, **floor** (4 RoboCasa floor episodes) | 1–2 h |

The routine test drives the actual CHAT tools, semantic task/plan handles,
guarded live-scene validation, rby1 IK/grasp/place, and a measured GT object
displacement. `EMET_SIM_NAV_TELEPORT=1` makes its simulated base approaches
instantaneous; it deliberately does **not** measure real navigation latency or
Stretch camera coverage. Those belong in the optional OVMM test or full profile.

| # | Item | Status (2026-08-23) |
|---|------|---------------------|
| 1 | MolmoSpaces iTHOR + rby1 **CHAT** `scene_tasks` → plan → execute (teleport) | **Green** (job `20260823_003208_1379d8`, displacement 2.21 m) |
| 2 | MolmoSpaces iTHOR + rby1 **kinematic** CHAT `scene_tasks` → plan → execute (semantic handles + IK) | **Green** (job `20260823_152854_4c7766`; 81 s sim run; displacement 2.2144 m; placement error 0.0200 m) |
| 3 | Stretch **teleport** control (`plan_pick_place` → `execute_pick_place_plan`) | **Green** (job `20260823_013540_64c74f`, displacement 0.10 m) |
| 4 | RoboCasa **floor-smoke** (`eval_tamp_floor.py --smoke`, rby1 MCTS) | Optional targeted OVMM check |
| 4b | RoboCasa **floor** suite (4 episodes; Stretch + find-only explore) | `PROFILE=full` only |
| 5 | Compare `find_object_success`, `pick_success`, `place_success`, `ovmm_full_partial` | After item 4 |

**Note:** The first scheduled gate (`20260823_025938_6a1aa2`) failed items 1–2
immediately because `run_item` used `bash -c "$*"` (word-split JSON and object
names). Fixed: `run_item` now executes `"$@"` and supports `DRY_RUN=1`.

## Related entry points

- Managed gate: [`scripts/run_tamp_agent_tools_gate.sh`](../../scripts/run_tamp_agent_tools_gate.sh),
  scheduler: [`scripts/schedule_tamp_agent_tools_gate.sh`](../../scripts/schedule_tamp_agent_tools_gate.sh)
- CHAT contract: [`docs/AGENT_RUN.md`](../AGENT_RUN.md)
- Full OVMM controls: [`docs/ovmm_full_benchmark.md`](../ovmm_full_benchmark.md)
- Semantic metadata extraction:
  [`src/emet/eval/scene_task_extractor.py`](../../src/emet/eval/scene_task_extractor.py)
- Agent adapter:
  [`src/emet/controller/task/tamp/agent_bridge.py`](../../src/emet/controller/task/tamp/agent_bridge.py)
