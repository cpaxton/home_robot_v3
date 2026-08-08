# Embodied agent planning: world model + tool calling + motion

**Branch:** `feature/agent-world-model`
**Date:** 2026-08-08
**Status:** Phase 1 design — nothing implemented yet.

## Goal

An embodied agent that explores with the voxel map + scene graph, builds a durable
world representation **including action outcomes** (failed navigation, failed
manipulation, failed "closer look"), and exposes that representation to
tool-calling for planning — without regressing HM-EQA or OVMM.

Today the memory answers "what is where?". After this work it also answers
"what did we try, and how did it go?" — so planners (VLM router, scripted
policies, future TAMP) stop repeating failed attempts and can reason about
reachability and manipulability, not just presence.

## Where we start (status quo, surveyed 2026-08-08)

### Failure / outcome recording is fragmented

| Mechanism | Where | Scope | Consumed by |
|-----------|-------|-------|-------------|
| `nav_attempts` / `nav_failures` / `last_nav_note` on `GraphNode` | `graph_memory.py::record_nav_attempt` | graph lifetime | frontier utility decay, `accessible_from` edge gating, prompt suffix `"; unreachable (N nav failures)"` |
| `_retracted_nav_claims` after close-look ABSENT | `graph_memory.py::retract_phrase_claim_at_obs` | **cleared per question** | label stripping only |
| `expected_absence_count` / `expected_object_missing` change events | `graph_memory.py::observe_visible_labels` | graph lifetime | belief confidence decay |
| `_tried`, `_not_present_streak`, `_nav_loop_flags`, `_place_inspect` | `agentic_eqa.py` executor | **loop-local, per episode** | state-message cards, redirects |
| `_habitat_blocked_goals` | controller | controller lifetime | `pick_uncovered_explore_target` |
| CHAT tool failures (`"Tool X failed: …"`) | `agent/loop.py::_dispatch_tool_calls` | **transcript only** | next LLM turn |
| Manipulation outcomes | — | **not recorded**; Stretch `_pickup`/`_place` always return True | nothing |
| Trace jsonl (`ok:false`, status codes) | `agentic_eqa.py::_append_trace` | on disk, offline | trace audits / tuner only |

### Tool packs and planners

- Two disjoint tool packs (`src/emet/agent/skills/specs.py`): CHAT (~26 tools)
  and EQA_EPISODE (7–8). Membership pinned by
  `src/test/agent/test_skill_packs.py`; EQA tool **names frozen** (traces).
- Base nav: A*/RRT under `src/emet/motion/algo/`; agent-tool → planner boundary
  buried in `controller_dynamem.py::navigate_to_target_pose`.
- Arm: `pinocchio_ik_solver` (Stretch), `mujoco_arm_ik` (position-only),
  `arm_rrt` (joint-space RRT-Connect), voxel/AABB collision checkers
  (default off). `aim_arm_at` chat tool is a stub.
- Evals: HM-EQA (classic default; agentic tool loop opt-in), OVMM find/full
  (scripted), object search tracks 3–4, dynamic-env tracks 6–7. Mobile manip =
  OVMM full with teleport/kinematic modes.

## Phase 1 — Unified action-outcome ledger in graph memory

**Objective:** one structured, queryable record of attempts and outcomes,
living next to the scene graph, replacing today's scattered counters.

- Add an `AttemptRecord` store to `GraphEQAMemory`
  (`src/emet/memory/graph_eqa/graph_memory.py`):

  ```python
  @dataclass(frozen=True)
  class AttemptRecord:
      action_kind: str   # navigate | investigate | verify | closer_look | pick | place
      target_node_id: int | None
      obs_id: int | None
      xyz: np.ndarray | None
      outcome: str       # ok | failed | aborted | absent | unreachable
      status_code: str   # e.g. no_path, timeout, rejected_low_clearance, pregrasp_ik_failed
      note: str
      step: int
  ```

- Generalizes: per-node `nav_attempts`/`nav_failures` become derived views over
  the ledger (keep the fields for compatibility; write both during migration).
- Persist verify-ABSENT evidence beyond the per-question
  `_retracted_nav_claims` reset (config-gated, default off for EQA paper arms).
  Preserve the semantics: ABSENT is a per-view true negative, **not** proof the
  object is gone from the scene.
- Record manipulation outcomes: propagate real success from Stretch/AnyGrasp
  `_pickup`/`_place` and the kinematic IK+RRT path into the ledger (extends the
  existing TODO item "Stretch / AnyGrasp `_pickup` / `_place` always return
  True").
- Surface the ledger to planners:
  - agentic state-message place cards — enrich the existing `[tried: …]` bit
    with outcome + status;
  - CONFIRMED_MEMORY block (attempt summary per phrase/place);
  - CHAT `query_memory` / `navigation_diagnostics` returns.
- Guardrails: additive fields with defaults; `configs/benchmarks/dynagraph.yaml`
  pins unchanged; new config keys under `agent.attempt_ledger.*` (documented in
  `docs/environment_variables.md` if env-toggled).
- Tests: unit tests for ledger write/query/serialization; smoke via
  `uv run emet test src/test/memory/test_memory_backends_smoke.py`.

## Phase 2 — Tool-calling contract cleanup (shared outcome schema)

**Objective:** CHAT and EQA loops report tool outcomes the same way, and both
feed the Phase-1 ledger.

- One `ToolOutcome` shape (`ok`, `status`, `note`, structured payload) shared by
  CHAT `_dispatch_tool_calls` (`src/emet/agent/loop.py`) and EQA
  `handle_tool` (`src/emet/memory/graph_eqa/agentic_eqa.py`). EQA already
  returns dicts with `ok`/`status`; CHAT returns free-form strings — converge on
  the dict shape, render to text at the prompt boundary.
- Both orchestrators write nav / verify / manip / closer-look outcomes to the
  ledger through one recording helper.
- Keep packs disjoint and EQA tool **names** frozen
  (`src/test/agent/test_skill_packs.py` stays green).
- Fold in the router-prompt-hygiene TODO: single source of truth for the two
  `_EQA_FORMAT_BLOCK_*` variants in `agentic_tools.py`, plus a unit test
  asserting `build_graph_eqa_system_prompt` byte-stability (prefix-KV cache).

## Phase 3 — Motion-planning integration behind a narrow interface

**Objective:** tools call planners through a small interface that returns
structured results; plan failures become ledger entries with reasons.

- Extract the agent-tool → planner boundary from
  `controller_dynamem.py::navigate_to_target_pose` into a narrow interface
  returning structured plan results (`no_path`, `timeout`,
  `rejected_low_clearance`, `aborted_waypoint_timeout`, …) that feed the ledger
  directly. Existing outcome strings in `tools.py::format_last_nav_plan_summary`
  become renderings of the structured result.
- Implement `aim_arm_at` / EE "closer look" using existing IK
  (`pinocchio_ik_solver`, `mujoco_arm_ik`) + `arm_rrt`; a failed closer-look is
  a first-class ledger entry (kind `closer_look`, status e.g. `ik_unreachable`).
  Then `take_ee_picture` only after successful aim (existing TODO).
- Mobile-manip readiness: reuse the OVMM-full pick/place path
  (`src/emet/eval/ovmm_full.py`) with ledger recording for pick/place attempts.
  Orientation IK and collision-checker defaults remain separate TODO follow-ups.

## Phase 4 — Evaluation without regressions

**Objective:** prove the ledger helps and nothing else moved.

- Regression gates each phase: `uv run emet test agent-regression`, skill-pack
  tests, memory-backend smoke. GPU tracks (HM-EQA smoke, OVMM find) launched
  via `uv run emet jobs run …` per the GPU workflow rules — never inline.
- New measurement: does failure memory reduce repeated failed attempts?
  Metrics: repeat-nav-failure count per episode, wasted rounds
  (post-failure re-attempts on the same target), episode steps. Report as
  deltas against pinned baselines on the HM-EQA agentic arm and OVMM find —
  **never** by changing pinned configs.
- Sketch (future): mobile manipulation eval = OVMM full + ledger metrics
  (pick attempts per success, re-grasp count) once orientation IK lands.

## Non-goals / guardrails

- No changes to pinned paper configs (`configs/benchmarks/dynagraph.yaml`),
  the `eqa.agentic_verify` default, or EQA tool names.
- No new memory backend — the ledger lives inside `GraphEQAMemory` and is
  consumed via existing surfaces (state message, CONFIRMED_MEMORY, chat tools).
- HM-EQA and OVMM paper progress continue on their own branches; this branch
  merges `main` forward, not the reverse.

## How to extend this plan

- Add a new phase section (`## Phase N — …`) with an **Objective** line, file
  paths, and guardrails; keep phases independently landable as one PR each.
- Track work items in `TODO.md` → "Embodied agent planning" section, one
  checkbox per landable change, linking back here for context.
- Record eval evidence in a status table in this doc (run dir, metric deltas),
  following `2026-07-23_molmospaces_rby1_manip_review.md` conventions.

## Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1 — attempt ledger | not started | design above |
| 2 — tool outcome schema | not started | depends on 1 |
| 3 — motion interface + closer look | not started | can start interface extraction in parallel |
| 4 — eval | not started | gates each phase |
