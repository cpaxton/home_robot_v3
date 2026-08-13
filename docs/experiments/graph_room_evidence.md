# Experiment: graph room evidence (agent-visible timeline)

**Branch:** `feature/graph-room-evidence`  
**Code:** room timeline on `GraphEQAMemory` + `AttemptRecord.room` (schema v2) + state-card `Room history:`  
**Refs:** [attempt_ledger.md](../attempt_ledger.md#room-timeline-graph-history), [agentic_scale.md](agentic_scale.md), [agentic_qwen_context.md](agentic_qwen_context.md#rooms_verify_probe)

## Goal

Test whether **writing room-scoped history into the graph** and **showing it to the paper-router** reduces wrong-room / re-scan budget burns — *without* sticky escape floors.

Hypothesis: when the agent sees ordered facts like `verify_absent` / `coverage_closed` / `stamp` for the current room (plus question target rooms), it chooses `explore_frontier` (or a better hyp) sooner than when those facts stay loop-local or invisible.

Non-goals for this experiment:

- Claiming a letter win on bal-32 / holdout-8 (secondary if cheap)
- Reintroducing room-mismatch `ESCAPE_MIN_TRAVEL_M` latches
- Router-off arms (timeline is for agent-driven routing)

## What “treatment” is

| Knob | Control (baseline) | Treatment |
|------|--------------------|-----------|
| Preset | `--preset paper-router` | same |
| Room stamp | unset / `EMET_EQA_ROOM_STAMP_INVESTIGATE=0` | **on** (paper-router H2H injects `1` on this branch) |
| Attempt ledger | off | **on** (injected with router) |
| Room timeline + state `Room history:` | absent / empty | active writers + `format_room_history` in `build_state_message` |
| Frame-streak escape (3 m after 2 not-present) | **unchanged** (still on `main`) | unchanged |

Fair A/B on one commit: toggle only stamp+ledger+timeline surface via env, or compare this branch’s paper-router vs an older paper-router OUT without room history (document commit SHAs).

## Metrics (primary → secondary)

### Process (primary — must analyze traces)

Per episode / arm, from `bundles/agentic_qN/agentic_trace.jsonl` + graph export if present:

| Metric | How |
|--------|-----|
| `n_room_events` | Count timeline kinds (`stamp`, `verify_absent`, `coverage_closed`, …) |
| `frac_explore_after_history` | Explore rows whose prior state (or same-round user message dump if logged) contained non-empty `Room history` |
| `wrong_room_dwell_rounds` | Rounds where `current_room` (router/graph) ∉ `question_target_rooms` and tool ∈ {investigate, look_around, verify} |
| `path_m_before_first_target_room` | Planar path until first explore/investigate with `frontier_room` or `current_room` in targets (or ∞) |
| `escape_source` histogram | Expect mostly `none` / `frame_streak`; **not** a new latch |
| Letter / steps | Standard H2H scored accuracy + mean steps (secondary) |

Success on process: treatment shows **≥1 room event on ≥70%** of episodes with non-empty `question_target_rooms`, and **median wrong_room_dwell_rounds** drops vs control on the same id set.

### Outcome (secondary)

- Agentic letter accuracy and mean steps vs control on the same ids
- No regression vs pinned paper-router composite story without claiming a new of-record number until bal-32

## Ladder

| Wave | Slice | Arms | Pass / stop |
|------|-------|------|-------------|
| **0** | Unit + CPU | — | `uv run emet test src/test/memory/test_attempt_ledger.py src/test/eval/test_hmeqa_launch.py src/test/memory/test_room_policy.py -q` |
| **1** | Smoke 2 ids | agentic paper-router | Timeline non-empty when rooms known; `Room history` in traces/state; GPU via `emet jobs` |
| **2** | Rooms probe 11 | agentic paper-router | Process metrics vs control; letter not the gate |
| **3** | Wrong-room focus set | agentic paper-router | Dwell / path-to-target-room improve |
| **4** | Holdout-8 or bal-32 | only if wave 2–3 look healthy | Optional; do not overwrite paper figures |

### Id sets

**Wave 1 (smoke):** `2,104`  
- q2: bathroom/bedroom targets (room mismatch opportunity)  
- q104: outdoor spawn / search failure archetype (frame_streak still allowed)

**Wave 2 (rooms_verify_probe):** `6,8,11,12,21,28,39,47,48,80,84`  
(from [agentic_qwen_context.md](agentic_qwen_context.md#rooms_verify_probe); disjoint from holdout-8)

**Wave 3 (wrong-room / leave focus):** start with `2,29,76` plus any wave-2 ids where control `wrong_room_dwell_rounds` is high. Recompute after wave 2 — do not hard-code until traces exist. Prefer ids with non-empty `question_target_rooms(...)`.

**Wave 4:** holdout-8 `15,56,65,68,79,88,104,105` or bal-32 — only after process pass.

## Harness (GPU)

One GPU job at a time. Dogfood CLI; never inline Habitat in a Cursor turn.

```bash
uv run emet eval recover --need-mib 12000
uv run emet habitat safe-start --job-name habitat-egl-probe-room-ev
# wait until: uv run emet jobs status JOB → done; logs show EGL OK

# Wave 1 — treatment (this branch; paper-router injects stamp+ledger)
OUT=~/runs/emet/hmeqa_room_evidence_w1_$(date +%Y%m%d_%H%M%S)
uv run emet hmeqa h2h "$OUT" --preset paper-router --arms agentic \
  --ids 2,104 --job-name room-evidence-w1 \
  -d "Wave1 room timeline smoke q2,q104"

uv run emet jobs status JOB_ID
uv run emet hmeqa inspect "$OUT" --qid 2
```

**Control (same machine, after treatment or separate night):** paper-router agentic with stamp+ledger forced off:

```bash
OUT_CTRL=~/runs/emet/hmeqa_room_evidence_ctrl_w1_$(date +%Y%m%d_%H%M%S)
uv run emet jobs run --name room-evidence-ctrl-w1 --need-mib 12000 -- \
  env EMET_EQA_ROOM_STAMP_INVESTIGATE=0 EMET_EQA_ATTEMPT_LEDGER=0 \
  EMET_ALLOW_SDPA_ATTN=1 EMET_EQA_TRACE=1 ARMS=agentic HOLDOUT_IDS=2,104 \
  EMET_EQA_AGENTIC_ROUTER=1 EMET_EQA_AGENTIC_VERIFIER=none \
  EMET_EQA_AGENTIC_REQUIRE_VERIFIED=0 \
  ./scripts/run_hmeqa_agentic_h2h.sh "$OUT_CTRL"
```

(Adjust to match whatever `hmeqa h2h --preset paper-router` expands to if the script wrapper changes — verify with `uv run emet hmeqa h2h --help` and a dry env dump.)

Wave 2:

```bash
OUT=~/runs/emet/hmeqa_room_evidence_w2_$(date +%Y%m%d_%H%M%S)
uv run emet hmeqa h2h "$OUT" --preset paper-router --arms agentic \
  --ids 6,8,11,12,21,28,39,47,48,80,84 --job-name room-evidence-w2 \
  -d "Wave2 rooms probe + room timeline"
```

Record `git rev-parse HEAD`, job id, and OUT in `emet status` / the run’s `STATUS.log` before launch.

## Analysis recipe

After each wave:

```bash
# Presence of timeline surface in explore / router-adjacent rows
rg -n 'Room history:|verify_absent|coverage_closed|"kind": "stamp"|escape_source' \
  "$OUT"/bundles/agentic_q*/agentic_trace.jsonl | head -80

# Optional: small Python tally (n_room_events, escape_source hist, dwell)
uv run python - <<'PY'
import json, glob, os, collections
out = os.environ["OUT"]
for path in sorted(glob.glob(f"{out}/bundles/agentic_q*/agentic_trace.jsonl")):
    kinds = collections.Counter()
    escape = collections.Counter()
    for line in open(path):
        r = json.loads(line)
        if r.get("event") == "room_stamp_investigate":
            kinds["stamp_event"] += 1
        if r.get("event") == "retract_claim":
            kinds["retract"] += 1
        if r.get("tool") == "explore_frontier":
            escape[r.get("escape_source") or "none"] += 1
            if r.get("escape_min_travel_m"):
                kinds["explore_with_floor"] += 1
    print(path.split("/")[-2], dict(kinds), dict(escape))
PY
```

Qualitative spot-check (2 episodes): did the router explore after `verify_absent` / `coverage_closed` in a non-target room, or keep investigating?

## Decision rules

| Result | Action |
|--------|--------|
| Wave 1: timeline empty because room stays `unknown` | Fix stamp / graph room writers before more GPU |
| Wave 2: history present but dwell unchanged | Improve state-card salience / prompt examples (still no escape latch); optional router few-shot with room-history line |
| Wave 2–3: dwell ↓, letters flat/up | Promote to optional holdout / bal-32 A/B |
| Letters regress hard | Keep timeline for diagnostics only; do not enable stamp+ledger on paper of-record until understood |
| Temptation to add sticky min-travel from room mismatch | **Reject** — out of scope; reopen only as a separate ablation labeled as a latch |

## Status

| Wave | Status | OUT / notes |
|------|--------|-------------|
| 0 unit | **done** | ledger + hmeqa launch + room_policy tests green on `db2ba7eb` |
| 1 smoke | **done (q2 only)** | `~/runs/emet/hmeqa_room_evidence_quick_20260812_122709` job `20260812_122944_49d38f`. Stamp+ledger+router on. 1× stamp blocked (`patio`), 1× stamp ok (`living_room`); explore `room_leave_hint=true` in living_room; `escape_source` all `none`. Letter miss (secondary). q104 not run. |
| 2 rooms probe | **done** | `~/runs/emet/hmeqa_room_evidence_w2_20260812_231348` job `20260812_231618_2e6289`. 11/11 scored, 0 crashes. Process: 11 stamp-ok events (1 blocked), known router room on 10/11 eps, mismatch diag on 3/11 (q6,q11,q48). `escape_source` all `none`. Letters **2/11** (secondary; mean steps ~33). Control A/B not run yet. |
| 3 wrong-room set | next | Focus ids with mismatch: **6,11,48** (+ wave-1 q2). Optional control with stamp/ledger off. |
| 4 scale | blocked | |

## Related

- Abandoned latch PR: [#107](https://github.com/cpaxton/home_robot_v3/pull/107) (escape floor on room mismatch)
- Prior rooms+verify probe: [agentic_qwen_context.md](agentic_qwen_context.md#rooms_verify_probe)
- Scale ladder: [agentic_scale.md](agentic_scale.md)
