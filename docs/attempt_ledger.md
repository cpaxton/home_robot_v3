# Action-outcome ledger (agent world model)

**Plan:** [plans/2026-08-08_embodied_agent_planning.md](plans/2026-08-08_embodied_agent_planning.md)
**Status:** The ledger, typed semantic action history, and static-world progress
gate are implemented with legacy-compatible defaults **off**. HM-EQA variants
freeze history visibility and retry policy independently.

The scene graph already answers “what is where?”. The **attempt ledger** adds “what did we try, and how did it go?” so CHAT tools, agentic EQA, and OVMM manip paths can avoid repeating failed nav / verify / pick / place / closer-look actions.

This page is the **operator and developer reference**. The plan doc is the design history and phase checklist.

---

## Guardrails (paper paths)

| Invariant | Detail |
|-----------|--------|
| **Default off** | No durable `AttemptRecord` rows or dedicated action-history section unless the ledger is enabled. World-evidence provenance may still mirror tool outcomes, but grounded state filters those events unless mode is `agent`. |
| **Retry policy independent and off** | `action_progress_mode` does not follow ledger visibility. `shadow` is observational; `enforce` changes candidate eligibility. |
| **No permanent failure blacklist** | Enforcement suppresses one equivalent action only while its target-local material state is unchanged. The initial contract is explicitly scoped to mostly static HM-EQA scenes. |
| **Pinned configs unchanged** | [`configs/benchmarks/dynagraph.yaml`](../configs/benchmarks/dynagraph.yaml) does not enable the ledger. |
| **EQA tool names frozen** | `investigate`, `explore_frontier`, `verify_siglip`, `submit_answer`, … stay stable for traces. |
| **`eqa.agentic_verify` default** | Still **false**; ledger does not turn on the agentic loop. |
| **Node nav counters remain** | `GraphNode.nav_attempts` / `nav_failures` / `last_nav_note` always update; ledger is an opt-in dual-write. |

---

## Enable

### Grounded Graph Agent visibility modes

`eqa.attempt_ledger_mode` is the paper-facing switch:

| Mode | Collect durable rows | Expose dedicated action history to `grounded_v2` |
|------|----------------------|--------------------------------------------------|
| `off` (default) | no | no |
| `shadow` | yes | no |
| `agent` | yes | yes |

`agent` exposes recent action outcomes, navigation-loop flags, per-place attempt
summaries/failure risk, global attempt rows, and their mirrored provenance events.
`shadow` writes the same ledger and bundle artifact but leaves those channels out
of the router state. Room timeline and live approach-affordance fields remain
controlled by their own axes; this switch does not erase internal safety state.

The 2026-08-23 matched four-question diagnostic validated this manipulation:
all 12 shadow router states were policy-clean, while 10/14 treatment states
contained recent actions, persisted loop flags, and global attempts. Both
scored 3/4; visible history reduced repeat-failure rows 6→4 but increased
attempts 31→37 and mean planning steps 35.25→39.25. The mode remains opt-in. See
[graph_room_evidence.md](experiments/graph_room_evidence.md#focused-action-history-isolation-2026-08-23)
and the frozen
[`action_history_pair_20260823.json`](../paper/data/hmeqa_agentic_h2h/action_history_pair_20260823.json).

### What the memory trace looks like

There are four related representations:

1. `attempt_ledger.json` stores structured durable rows.
2. `agentic_trace.jsonl` records the compact state rendered for each router call.
3. `action_history` trace events store typed signatures, pre/post progress
   tokens, outcome class, and progress reasons for top-level actions.
4. `router_call.action_gate_decisions` records every rendered place/frontier
   candidate; `action_gate_dispatch` records selected explicit verify and motion
   calls. Both are emitted in `shadow`. Automatic post-motion verification is
   summarized by its parent investigate/frontier history entry.

The pre-gate q11 diagnostic exposed the problem as opaque adapters:

```text
Recent action outcomes:
- r0 investigate obs=15 ap=0 verify=ABSENT closest=0.4m
Loop flags:
- obs=15 visits=1 status=STALLED_NAV_LOOP
```

Grounded state schema v3 now leads with semantic intent and stable identity;
the mutable adapter is retained only as a trailing bridge to tool arguments:

```text
Recent action outcomes:
- round=1 action=investigate intent="find the silver trash can"
  target="kitchen island/counter" room=kitchen
  place=place_550b3a0b4a9b approach=0 view=view_00000024@rev8
  closest=0.4m capture=NEW_OBS verify=ABSENT
  result=progress:ABSENT progress=new_view,target_evidence adapter=15
Loop flags:
- action=investigate target='kitchen island/counter' room=kitchen
  stable=place:place_550b3a0b4a9b visits=1 status=STALLED_NAV_LOOP adapter=15
Temporarily suppressed actions:
- action=investigate target='kitchen island/counter' room=kitchen
  stable=place:place_550b3a0b4a9b
  reason='all finite approach variants are temporarily ineligible for the unchanged place state'
  retry=after_target_view_or_geometry_change adapter=15
```

Suppressed rows are explanatory, not actionable: their stable IDs are omitted
from the exact tool allowlist. The same decision remains trace-visible but does
not alter cards or execution in `action_progress_mode=shadow`. Ledger
`attempt_ledger_mode=shadow` independently hides recent/loop/suppression text
from the router.

## Static-world action-progress policy

`eqa.action_progress_mode` is independent from
`eqa.attempt_ledger_mode` and requires `agentic_decision_policy=grounded_v2`:

| Mode | Compute and trace decisions | Remove candidates before routing |
|------|-----------------------------|----------------------------------|
| `off` (default) | no | no |
| `shadow` | yes | no |
| `enforce` | yes | yes |

This is **temporary suppression**, not “failed forever.” The v1 policy is
deliberately benchmark-scoped to HM-EQA's mostly static scenes. It suppresses a
concrete action only when the same equivalence key previously produced a
terminal/no-progress result from the same progress token.

Action identity has two levels:

- the **work key** combines action family, normalized work intent, and stable
  target (`place_id`, `frontier_id`, or question/session identity);
- the **equivalence key** adds the concrete variant: investigate approach slot,
  verify view + phrase/verifier profile, or frontier material geometry + goal
  cell. `investigate` and its `navigate_to_obs` alias share one family.

The rendered intent may retain a useful per-call hint such as
`toward="kitchen"`, but rewording that weak frontier hint does not create new
semantic work or bypass an unchanged no-progress decision.

The action-specific **progress token** includes only facts that can justify a
retry: target view/revision, target-local non-attempt evidence, coverage/local
geometry, material frontier geometry/lineage, and a quantized robot pose cell.
Mirrored ledger events are excluded so a failed action cannot unlock itself by
writing its own failure row. Raw frontier revision increments alone are not
progress. Partial navigation into a new pose cell is a continuation and remains
eligible.

Outcomes distinguish `progress`, `negative_evidence`, `no_progress`,
`terminal_ok`, `transient`, `operator_abort`, and `capability_absent`. Alternate
approaches and new views remain eligible. A known transient gets one fixed
same-token retry; unknown/terminal no-progress actions do not silently loop.

Dynamic-world invalidation is intentionally **not implemented in this slice**.
A future scheduler must reopen or decay suppressions on target/environment
change events, map revision, elapsed-time/TTL, and evidence staleness, and may
cool down/re-rank an action instead of removing it. Do not use `enforce` as a
general lifelong-memory policy until that contract lands.

### Env

```bash
export EMET_EQA_ATTEMPT_LEDGER_MODE=agent  # off | shadow | agent
export EMET_EQA_ACTION_PROGRESS_MODE=shadow  # off | shadow | enforce
# optional:
export EMET_ATTEMPT_LEDGER_MAX=512
export EMET_ATTEMPT_LEDGER_PERSIST_ABSENT=0   # keep ABSENT claim blacklist across questions
```

`EMET_EQA_ATTEMPT_LEDGER=1` remains the legacy boolean write gate. HM-EQA
derives it from the selected mode (`shadow` and `agent` both write).

### YAML

Under `eqa:` (mapping / dynav config) or unified `mapping.eqa` / top-level `eqa:`:

```yaml
eqa:
  attempt_ledger_mode: agent  # off | shadow | agent
  action_progress_mode: shadow  # off | shadow | enforce
```

The lower-level boolean/mapping form remains available for non-HM-EQA writers:

```yaml
eqa:
  attempt_ledger:
    enabled: true
    max_records: 512
    persist_absent_claims: false
```

CLI override example:

```bash
uv run emet run agent --set eqa.attempt_ledger=true
```

For a frozen HM-EQA static-policy comparison, use the checked-in complete
variant files:

```bash
uv run emet hmeqa h2h OUT_SHADOW \
  --variant-config configs/benchmarks/hmeqa_action_progress_shadow.yaml \
  --preset paper-router --arms agentic --ids 6,11,12,47
uv run emet hmeqa h2h OUT_ENFORCE \
  --variant-config configs/benchmarks/hmeqa_action_progress_enforce.yaml \
  --preset paper-router --arms agentic --ids 6,11,12,47
```

Variant schema v2 freezes nine fields, including action progress. The loader
rejects missing or unknown v2 fields; legacy schema-v1 files remain readable
with action progress defaulted to `off`. Explicit CLI variant flags win over the
file; the effective values plus the variant file path and SHA-256 source label
are frozen in schema-v4 `run_manifest.json`. Resume reuses that manifest rather
than re-reading a mutable config. A non-`off` episode cannot publish completion
unless its summary and trace contain parseable, manifest-matching gate
diagnostics.

See [emet_config.md](emet_config.md) and [environment_variables.md](environment_variables.md).

---

## Data model

`AttemptRecord` (`emet.memory.graph_eqa.attempt_ledger`):

| Field | Meaning |
|-------|---------|
| `action_kind` | `navigate` \| `investigate` \| `verify` \| `closer_look` \| `pick` \| `place` \| `explore` |
| `outcome` | `ok` \| `failed` \| `aborted` \| `absent` \| `unreachable` \| `candidate` \| `present` |
| `status_code` | Stable code (`no_path`, `timeout`, `controller_failed`, …) or short free-form |
| `note` | Human/debug string (truncated) |
| `step` | Episode / tool round when known |
| `target_node_id` / `obs_id` / `xyz` | Where the attempt aimed (JSON-friendly xyz tuple) |
| `phrase` | Text query when relevant |
| `source` | `chat` \| `eqa` \| `unknown` |
| `question_id` | Optional episode question id |
| `room` | Canonical room label when known (introduced in schema v2); empty on v1 imports |
| `target_kind` / `target_id` / `view_id` | Stable graph identities (schema v3) |

Stored on `GraphEQAMemory` (`_attempt_records`), capped by `max_records`
(default 512). The current export `schema_version` is **3**.

### Room timeline (graph history)

Separate from classic EQA prompt `HISTORY:` (answer-iteration scrap). `GraphEQAMemory` keeps a capped `_room_events` list (`stamp`, `verify_absent`, `coverage_closed`, …) with `step` + `room`. It survives room-cluster rebuilds, clears on new agentic episode / `clear_eqa_working_memory`, and is **not** gated on the ledger opt-in.

| Method | Role |
|--------|------|
| `record_room_event(...)` | Append when room is a known label (never invents `unknown`) |
| `format_room_history(...)` | Newest-first compact line for agentic `build_state_message` |
| `clear_room_events()` | Drop timeline |

Agentic writers: investigate room-stamp → `stamp`; close ABSENT retract → `verify_absent` (+ ledger `room=` when on); place coverage → `closed` → `coverage_closed`. **Do not** enable investigate stamps on paper-router by default (letter regression). For A/Bs, set `EMET_EQA_ROOM_STAMP_INVESTIGATE=1` / `EMET_EQA_ATTEMPT_LEDGER=1` explicitly — see [experiments/graph_room_evidence.md](experiments/graph_room_evidence.md). This is **agent-visible memory**, not a nav/escape latch.

**Experiment ladder:** [experiments/graph_room_evidence.md](experiments/graph_room_evidence.md) (smoke → rooms probe → wrong-room dwell metrics → optional scale).

### API on `GraphEQAMemory`

| Method | Role |
|--------|------|
| `record_attempt(...)` | Append one row (no-op when ledger off, unless `force=True`); optional `room=` |
| `get_attempt_records()` | Ordered list (oldest first) |
| `export_attempt_ledger()` / `import_attempt_ledger(...)` | Serialize / restore |
| `clear_attempt_ledger()` | Drop rows |
| `attempt_summary_for_obs(obs_id)` | Compact `[attempts: …]` bits for place cards |
| `set_attempt_ledger_question_id(...)` | Tag subsequent rows with a question id |
| `record_nav_attempt(...)` | Always updates node counters; dual-writes a `navigate` row when ledger on |
| `record_room_event` / `format_room_history` | Room-scoped timeline (see above) |

Helpers:

- `infer_nav_status_code` / `infer_nav_outcome` — note → status/outcome
- `summarize_repeat_failures` / `record_manip_attempt` — [`attempt_metrics.py`](../src/emet/memory/graph_eqa/attempt_metrics.py)

---

## Who writes

```mermaid
flowchart LR
  subgraph writers [Writers]
    Nav["_log_nav_attempt / nav_attempt.sync_*"]
    Tool["ToolOutcome + maybe_record_tool_attempt"]
    Retract["retract ABSENT → verify:absent"]
    OVMM["ovmm_full pick/place scoring"]
  end
  GM["GraphEQAMemory.record_attempt"]
  Nav --> GM
  Tool --> GM
  Retract --> GM
  OVMM --> GM
```

| Path | Module | Notes |
|------|--------|-------|
| Base navigation | `emet.controller.nav_attempt` ← `DynamemController._log_nav_attempt` | Structured `NavAttemptResult.status_code`; single write path (agentic avoids double-write) |
| CHAT tools | `emet.agent.loop` → `ToolOutcome` → `maybe_record_tool_attempt` | Maps tool names → `action_kind` |
| EQA_EPISODE tools | `agentic_eqa.handle_tool` → same `ToolOutcome` path | `source="eqa"` |
| Verify ABSENT | `retract_phrase_claim_at_obs` | Ledger row `verify` / `absent`; claim blacklist cleared per question unless `persist_absent_claims` |
| Closer look | CHAT `aim_arm_at` → `closer_look.aim_wrist_at_phrase` | Failures become ledger `closer_look` via tool outcome |
| OVMM full manip | `emet.eval.ovmm_full` | After GT-scored pick/place (`sim` / `attempt`) |
| Stretch manip truth | `DynamemTaskExecutor._pickup` / `_place` | Propagates `agent.manipulate` / `agent.place` / `GraspObjectOperation.was_successful` (no more unconditional `True`) |

Shared tool result shape: [`emet.agent.tool_outcome.ToolOutcome`](../src/emet/agent/tool_outcome.py) (`ok`, `status`, `note`, `payload`).

---

## Who reads (surfaces)

| Surface | When ledger on |
|---------|----------------|
| Agentic place cards | `[attempts: navigate:failed(no_path), …]` via `attempt_summary_for_obs` |
| CHAT `navigation_diagnostics` | Recent attempts in the tool payload |
| CONFIRMED_MEMORY / prompt tags | Compact `attempts:` suffixes (respects prompt token budget) |
| Repeat-failure metrics | `summarize_repeat_failures(records)` for offline / future eval reports |

**Legacy ledger repeat key** (Phase 4 metric): same `action_kind` and stable
`target_kind`/`target_id` first (`view_id` additionally distinguishes verifies),
then legacy `target_node_id`, `obs_id`, or planar `xyz` within **0.25 m**.
This metric is coarser than the live action equivalence/progress gate.

---

## Motion / closer look

- Nav status vocabulary: [`nav_attempt.py`](../src/emet/controller/nav_attempt.py) + [`habitat_nav.NavAttemptResult`](../src/emet/controller/habitat_nav.py).
- Wrist aim: [`closer_look.aim_wrist_at_phrase`](../src/emet/controller/manipulation/closer_look.py) (localize + kinematic EE aim when available). See [motion_planning.md](motion_planning.md).
- **CHAT chain:** successful `aim_arm_at` records a one-shot grant on the agent/context; `take_ee_picture` **refuses** without that grant and **consumes** it on capture. Prefer `face_toward` + `describe_scene` when aim is unavailable.

CHAT vs EQA packs stay disjoint — see [AGENT_RUN.md](AGENT_RUN.md#skill-library-vs-orchestrator-modes). `aim_arm_at` is CHAT-only; EQA uses `investigate` / `look_around`.

---

## Router prompt hygiene

Agentic format blocks in [`agentic_tools.py`](../src/emet/memory/graph_eqa/agentic_tools.py) share `_EQA_RULE_*` atoms so canonical vs `room_policy=llm` variants do not drift. **Byte-stability** of `build_graph_eqa_system_prompt` is required for Qwen3-VL system-prefix KV cache hits — pinned SHA256 + identity checks in `src/test/memory/test_room_policy.py`. Do not casually edit the composed format strings.

---

## Tests

```bash
uv run emet test src/test/memory/test_attempt_ledger.py \
  src/test/memory/test_action_history.py \
  src/test/memory/test_attempt_metrics.py \
  src/test/agent/test_tool_outcome.py \
  src/test/controller/test_nav_attempt.py \
  src/test/controller/test_closer_look.py \
  src/test/controller/test_dynamem_sim_manip_gating.py \
  src/test/agent/test_skill_packs.py \
  src/test/memory/test_room_policy.py -q

uv run emet test agent-regression -q --no-sim
```

GPU A/B (ledger on vs off) must use `uv run emet jobs run …` — never inline Habitat/VLM in an agent turn. See [evaluation.md](evaluation.md) and `.cursor/rules/gpu-eval-workflow.mdc`.

---

## Code map

| Piece | Path |
|-------|------|
| Record + infer helpers | `src/emet/memory/graph_eqa/attempt_ledger.py` |
| Semantic work/equivalence/progress policy | `src/emet/memory/graph_eqa/action_history.py` |
| Candidate filtering + semantic state rendering | `src/emet/memory/graph_eqa/agentic_state.py`, `agentic_eqa.py` |
| Repeat / manip helpers | `src/emet/memory/graph_eqa/attempt_metrics.py` |
| Store + dual-write | `src/emet/memory/graph_eqa/store.py` (ledger list) + `spatial/graph_rooms.py` (`record_nav_attempt`) |
| ToolOutcome | `src/emet/agent/tool_outcome.py` |
| Nav sync | `src/emet/controller/nav_attempt.py` |
| Closer look | `src/emet/controller/manipulation/closer_look.py` |
| OVMM writers | `src/emet/eval/ovmm_full.py` |
| Stretch success | `src/emet/controller/task/dynamem/dynamem_task.py` |
| Living checklist | [TODO.md](../TODO.md) (Embodied agent planning) |

---

## Related docs

- [AGENT_RUN.md](AGENT_RUN.md) — CHAT vs EQA_EPISODE
- [graph_eqa.md](graph_eqa.md) / [graph_memory.md](graph_memory.md) / [dynagraph.md](dynagraph.md) — memory backends
- [ovmm_full_benchmark.md](ovmm_full_benchmark.md) — pick/place scoring + ledger hooks
- [motion_planning.md](motion_planning.md) — base + arm planners
- [emet_config.md](emet_config.md) / [environment_variables.md](environment_variables.md) — knobs
