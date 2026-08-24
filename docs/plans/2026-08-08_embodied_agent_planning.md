# Embodied agent planning: world model + tool calling + motion

**Branch:** `feature/agent-world-model`
**Date:** 2026-08-08
**Status:** Phases 1–3 landed; Phase 4 helpers done; Phase 5 typed semantic
history + static-world progress gate implemented with defaults off. GPU deltas
remain pending.

**Shipped operator/developer reference:** [../attempt_ledger.md](../attempt_ledger.md)
(enable knobs, schema, writers/readers, tests). Keep this plan as design history +
phase checklist; put how-to and API details in that page.

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

Split into landable PRs. Do **not** bundle Stretch manip success or prompt
enrichment into the first ledger PR.

### Phase 1a — nav ledger store (this slice)

- `AttemptRecord` in `src/emet/memory/graph_eqa/attempt_ledger.py`, store on
  `GraphEQAMemory`:

  ```python
  @dataclass(frozen=True)
  class AttemptRecord:
      action_kind: AttemptActionKind  # navigate | investigate | verify | closer_look | pick | place
      outcome: AttemptOutcome         # ok | failed | aborted | absent | unreachable
      status_code: str                # controlled nav codes + short free-form fallback
      note: str
      step: int
      target_node_id: int | None
      obs_id: int | None
      xyz: tuple[float, float, float] | None  # JSON-friendly; not ndarray
      source: AttemptSource           # chat | eqa | unknown
      question_id: str | None
      schema_version: int
  ```

- Config: `eqa.attempt_ledger` (bool or `{enabled: bool}`), default **off**.
  Env: `EMET_EQA_ATTEMPT_LEDGER`. Documented in `docs/emet_config.md` /
  `docs/environment_variables.md`. Pinned `configs/benchmarks/dynagraph.yaml`
  unchanged (implicit off).
- `record_nav_attempt` dual-writes: always updates `GraphNode` counters; when
  ledger on, also appends a `navigate` row. `derive_nav_counters_from_ledger`
  can recompute counters from rows (migration helper; node fields remain
  authoritative until a later dual-write exit).
- Serialize: `export_attempt_ledger` / `import_attempt_ledger`.
- Tests: `src/test/memory/test_attempt_ledger.py`.

**Dual-write exit (later):** once planners read the ledger,
`_tried` / `_place_inspect` / `_nav_loop_flags` become views or are deleted —
do not leave permanent dual writers without an exit criterion.

### Phase 1b — ABSENT persistence (separate PR)

- Config-gated persistence of verify-ABSENT evidence past the per-question
  `_retracted_nav_claims` reset (default off for EQA paper arms).
- Preserve semantics: ABSENT is a per-view true negative, **not** proof the
  object is gone from the scene.

### Phase 1c — manip outcomes (separate PR; blocked)

- Depends on fixing "Stretch / AnyGrasp `_pickup` / `_place` always return True".
- Propagate real success into ledger rows (`pick` / `place`).

### Phase 1d — surface to planners (separate PR)

- Enrich agentic `[tried: …]` place cards with outcome + status.
- CONFIRMED_MEMORY attempt summary (respect `eqa_vl.eqa_prompt_max_tokens`:
  top-K failures per place; truncate with HISTORY).
- CHAT `query_memory` / `navigation_diagnostics` returns.

## Phase 2 — Tool-calling contract cleanup (shared outcome schema)

**Objective:** CHAT and EQA loops report tool outcomes the same way, and both
feed the Phase-1 ledger.

Can land **before** or after 1b–1d: start by converging CHAT string results onto
a dict shape (`ok`, `status`, `note`, payload) at the prompt boundary; wire
ledger writers once 1a exists.

- Normalize EQA `handle_tool` keys (`error` vs `reason` vs `status`) onto the
  shared shape.
- Both orchestrators write nav / verify / manip / closer-look outcomes through
  `GraphEQAMemory.record_attempt`.
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
  Then `take_ee_picture` only after successful aim (existing TODO). Large
  enough for its own PR after the nav interface extraction.
- Mobile-manip readiness: reuse the OVMM-full pick/place path
  (`src/emet/eval/ovmm_full.py`) with ledger recording for pick/place attempts.
  Orientation IK and collision-checker defaults remain separate TODO follow-ups.

## Phase 4 — Evaluation without regressions

**Objective:** prove the ledger helps and nothing else moved.

- Regression gates each phase: `uv run emet test agent-regression`, skill-pack
  tests, memory-backend smoke. GPU tracks (HM-EQA smoke, OVMM find) launched
  via `uv run emet jobs run …` per the GPU workflow rules — never inline.
- **Repeat-failure key (define before GPU runs):** same stable
  `target_kind`/`target_id` first (`view_id` distinguishes verify actions), then
  legacy `target_node_id`, `obs_id`, or `xyz` within 0.25 m planar; same
  `action_kind`; prior row `outcome != ok`. Metrics: repeat-nav-failure count per episode,
  wasted rounds (post-failure re-attempts on that key), episode steps. Report
  as deltas against pinned baselines on the HM-EQA agentic arm and OVMM find —
  **never** by changing pinned configs.
- Sketch (future): mobile manipulation eval = OVMM full + ledger metrics
  (pick attempts per success, re-grasp count) once orientation IK lands.

## Phase 5 — Semantic action history and static-world progress gate

**Objective:** replace opaque round/`obs_id` strings with intent + stable target
semantics, and prevent unchanged duplicate work before the router chooses it.

- `action_history.py` defines deterministic work/equivalence keys, target-local
  progress tokens, typed outcomes, and pure gate decisions.
- Grounded state schema v3 renders intent, stable place/frontier/view identity,
  approach/view variants, outcome, and material progress. Mutable observation
  IDs remain trailing tool adapters only.
- `eqa.action_progress_mode=off|shadow|enforce` is manifest-frozen and
  independent from ledger visibility. `shadow` traces counterfactual decisions
  with byte-for-byte policy behavior; `enforce` omits suppressed IDs from the
  exact rendered allowlist and revalidates dispatch. Run-manifest schema v4
  freezes the axis and completion requires matching summary/trace diagnostics.
- This is not a permanent blacklist. The first policy assumes mostly static
  HM-EQA scenes and suppresses an equivalent terminal/no-progress action only
  while its target-local progress token is unchanged. Alternate approaches,
  new views/target evidence, material frontier/coverage change, and partial
  navigation progress reopen work.
- Dynamic-world follow-up is explicit: target/environment change events, map
  revision, elapsed time/TTL, and evidence staleness must invalidate or decay a
  suppression; a future scheduler may cool down/re-rank instead of filtering.
- Pure key/progress tests and exact q11/q12 trace-derived replay tests are
  complete. The managed `6,11,12,47` shadow/enforce comparison remains a
  separate GPU gate.

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
| 1a — nav ledger store | done (default off) | `attempt_ledger.py` + `record_nav_attempt` dual-write |
| 1b — ABSENT persistence | done | `persist_absent_claims` + verify:absent ledger row on retract |
| 1c — manip outcomes | done | CHAT + Stretch success propagation; OVMM-full `record_manip_attempt` |
| 1d — surface to planners | done | place cards, `navigation_diagnostics`, CONFIRMED_MEMORY `attempts:` tags |
| 2 — tool outcome schema | done | `ToolOutcome` + shared `_EQA_RULE_*` format atoms + byte-stability test |
| 3 — motion interface + closer look | done (v1) | `nav_attempt.sync_*` + `closer_look.aim_wrist_at_phrase` |
| 4 — eval | helpers done | `attempt_metrics.summarize_repeat_failures`; GPU deltas via `emet jobs` still open |
| 5 — semantic action progress | CPU implementation done | typed history + `off|shadow|enforce`; static-world GPU comparison pending |
