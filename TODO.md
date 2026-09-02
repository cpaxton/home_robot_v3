# Living TODOs (small / near-term)

Short checklist for agent/hardware polish that is not worth a full plan doc yet.
Strike through or move to a PR when done.

## Config over env flags

We over-use `EMET_*` for robot/controller policy (head sweep, teleport, TTS, rotate
steps). Embodiment defaults belong in `robots.<id>` / `mapping.*` in
[`configs/emet/default.yaml`](configs/emet/default.yaml) (see Stretch
`look_around_head_sweep`). Env should stay as a rare override (`EMET_FORCE_HEAD_SWEEP`
for paper pans) or for host/GPU process state. Do not add a new `EMET_*` for
something that can be a YAML key + `--set`. `run_ovmm_find_recep_slice.sh` no
longer exports `EMET_SKIP_HEAD_SWEEP` — YAML already defaults pans off; only
`PROFILE=stretch-legacy` sets `EMET_FORCE_HEAD_SWEEP`. Same pass later for other
script-exported knobs that duplicate config.

## Eval orchestrator layers (keep three; do not add a fourth matrix script)

Too many bash wrappers all mean “run a bit of Habitat + OVMM + SQA3D.” Canonical
layers — docs in [`docs/evaluation.md`](docs/evaluation.md) and
[`docs/experiments/README.md`](docs/experiments/README.md):

| Layer | Entry | Job |
|-------|--------|-----|
| Path smoke | `scripts/run_simulation_smoke_battery.sh` | Seven-track merge-gate (often GT/mock) |
| Overnight regression | `scripts/run_overnight_cross_track_smoke.sh` | Tier-0 + pytest + those tracks. **Never** chain VLM after this. |
| Fast OVMM | `scripts/run_ovmm_find_recep_slice.sh` | rby1 / teleport (`PROFILE=smoke`) |
| Habitat OVMM VLM | `scripts/smoke_habitat_ovmm_agentic_find.sh` | One HM3D scene, agentic 4/4 |
| Paper numbers | `scripts/run_paper_matrix.sh` + `emet hmeqa overnight` | Real sizes, one GPU lock |

Deleted 2026-09-02: `scripts/run_overnight_eval_smoke.sh` (tiny real-VLM matrix +
figure pack). Replacements above; figures: `scripts/build_eval_figure_pack.py`.

### Later cleanup

- [x] Fold `run_representative_benchmark_sample.sh` into
      `run_dynagraph_tuned_paper_battery.sh` — **keep both**. Representative =
      static_graph comparison + S0 matrix + tables; paper battery = seven-track
      + holdout/bal32 + tuned dynagraph numbers.
- [x] Make `run_overnight_habitat_eval.sh` a thin wrapper around
      `emet hmeqa overnight`. Extra slices stay as dedicated scripts
      (`run_hmeqa_annotated37_h2h.sh`, `run_hmeqa_paper113_h2h.sh`,
      `run_habitat_iter_subset.sh` for paper-20).
- [ ] Leave `run_hmeqa_*_h2h.sh` until `emet hmeqa h2h` covers those ID sets;
      then delete the one-off H2H scripts.
- [x] `run_habitat_ovmm_joint_gate.sh` vs `run_paper_matrix.sh` — **keep both**.
      Joint gate = count/clock 15-qid + rby1 OVMM `PROFILE=slice`. Paper
      matrix = HM-EQA paper-113 (no semantics) + OVMM S0/S1/S2 numbers.

## Grounded graph room-evidence A/B — no-go for scale; focused history pair authorized (2026-08-23)

Canonical record:
[docs/experiments/graph_room_evidence.md](docs/experiments/graph_room_evidence.md).

### Completed evidence and recovery

- [x] Reconstruct the 2026-08-19 reboot. The last record is an NX-protected
      supervisor instruction-fetch kernel fault; there was no OOM, `libcuda`
      crash, NVRM Xid, or active managed GPU job. This is not evidence of GPU
      starvation; details are preserved in `segfault.md`.
- [x] Archive the old pre-manifest treatment/control OUTs as diagnostic-only.
- [x] Complete the A0 resolver gate at `e85cda20`: **4/6**, including **3/4**
      on regression IDs `6,11,12,47`.
- [x] Complete A1 collection/no-leakage at `fae4b89c`: **4/6**; 621 evidence
      views, 55 ledger rows, 11 room events, and all 40 policy-visible event
      lists empty.
- [x] Complete the strongest clean same-commit comparison at `fae4b89c`:
      A1 shadow **4/6**, 29.2 mean planning steps, 2 budget hits; A2 grounded
      **3/6**, 34.2 steps, 3 budget hits, and room events on only 3/6 episodes.
- [x] Record the A2 decision: **do not promote or scale**. It gained q47, lost
      q12 and q76, did not fix q2, and did not demonstrate lower dwell or fewer
      repeats. This compares the whole grounded treatment, not graph evidence
      alone.
- [x] Complete the pre-fix GT-free q11 infrastructure sequence. Three complete
      runs scored **0/3**; nonnumeric IDs and blocked-navigation redirect worked,
      but trash-can discovery/coverage was still unsolved at that point.
- [x] Fix CPU-audited correctness blockers before another run: strict
      off/shadow/agent visibility, semantic handling of a valid `None` answer,
      placeholder-safe fallback, world-frame room refresh, qualified negative
      evidence, pending-answer visibility, enrich-label ID mapping, explicit
      semantic/enrich oracle axes, and staged compact checkpoints that cannot
      retain stale pixel/voxel state.
- [x] Run one explicitly diagnostic, dirty-tree GT-free q11 A2 canary after an
      EGL safe-start. Job `20260822_175646_219c08`, OUT
      `~/runs/emet/gre_q11_a2_canary_d3ee32e8_20260822_175238`, scored **1/1**
      with a direct present+answerable view at obs 76, semantic answer
      `Next to the refrigerator`, no salvage, and no native failure. This is a
      capability signal, not clean comparison or scale evidence.

### Capability gaps

- [ ] q2: parse one target room and bind gray rug evidence to the shower.
- [ ] q11: diagnose the clean-repeat regression. The dirty diagnostic was 1/1,
      but clean HEAD `81a2c689` reached an initial kitchen view without seeing
      the trash can, then exhausted eight rounds and answered by fallback.
- [ ] q12: validate the new direct-image-over-graph precedence on a clean run;
      the prior grounded treatment trusted a graph undercount.
- [ ] q76: enter and cover a bedroom instead of exhausting living-area frontiers.
- [ ] Add `scripts/summarize_graph_room_evidence.py`; fail on manifest parity
      violations and emit paired dwell/path/repeat/budget/evidence metrics.

### Conservative next experiment

- [x] Freeze and push the reviewed GT-free implementation at `f8ceaa49`
      (`--no-hm3d-semantics`, `--no-enrich-labels`) in PR #115. The focused
      suite, `agent-regression`, and pre-commit pass.
- [x] Finish the consistency audit: apply quantization/tool/nav budgets at
      runtime, reject unfrozen policy environment, hash the small dataset
      inputs, fail closed when requested semantic assets are unavailable, and
      make generic batch resume parity-safe. Focused audit tests pass; keep PR
      #115 draft until the separate CPU baseline gate is resolved.
- [x] Close the simultaneous-launch race: `emet jobs run` now auto-detects
      GPU-like experiment commands and holds a host-wide flock for the full
      job lifetime; zombie supervisor PIDs no longer stall queued jobs.
- [x] Keep benchmark identities explicit: q0–112 is the historical emet
      113-row slice; the upstream GraphEQA HM-EQA semantic-filtered enrich
      sequence has 114 rows. Do not call either mapping authoritative without
      naming which one is used.
- [ ] Close the CPU baseline gate before another GPU run. The full no-sim run
      has three failures after isolating BLAS thread settings: the trajectory
      color and two crop-size assertions in `test_map_grid.py` /
      `test_map_snapshot.py`. Those source files are unchanged from
      `origin/main`; fix or explicitly record that baseline separately, and do
      not describe the full suite as green.
- [x] Repeat the q11 A2 canary from clean PR HEAD `81a2c689` after detached
      EGL probe `20260823_003909_ab5972` passed. Job
      `20260823_004358_462498`, OUT
      `~/runs/emet/gre_q11_a2_clean_81a2c689_20260823_003735`, completed
      without a native failure but scored **0/1**: the first kitchen check and
      every later view marked the trash can absent, then the eight-round budget
      forced `mcq_debias` answer A instead of D at 55 planning steps. This
      fails the direct-evidence gate; stop the scale ladder.
- [x] Fix the first-launch CLI-to-script manifest handoff exposed by the
      canary: the inner script now validates a launcher-prepared manifest
      without switching episode semantics to resume. Managed H2H paths also
      skip in-job stale-process cleanup.
- [x] Finish the PR #115 state-integrity gate in two reviewable commits:
      managed lock/environment/completion integrity first, then grounded
      graph-state contracts and image-free clean/dirty replay regressions.
      **PR #115 CPU integrity gates: complete** — 141 lifecycle/completion,
      182 agent-regression, 108 graph-memory/evidence, and 11 focused
      state/metadata tests pass. The PR stayed draft through the clean q11 and
      action-history GPU diagnostics. Their capability result is negative, but
      the treatment remains opt-in, so that result does not block merging the
      audited state/lifecycle infrastructure.
- [x] After the CPU gate, detached EGL probe `20260823_081023_d24734` passed,
      then clean q11 A2 job `20260823_081229_66d6e3` ran two native-clean
      attempts at `b91059e5`. Both exhausted eight rounds with no direct
      trash-can evidence and answered A instead of D; the capability gate
      remains failed.
- [x] Fix the completion-policy bug exposed by that run: object-crop mosaics
      are best-effort when an episode has no usable instance crops, so absence
      is valid while any diagnostics-declared mosaic remains strictly checked.
      The focused completion suite passes 12/12.
- [x] Add a strict config-file choice for dedicated grounded action history.
      `configs/benchmarks/hmeqa_action_history_{shadow,agent}.yaml` hold
      decision/graph/room/oracle axes fixed; only ledger visibility and variant
      ID differ. Shadow now filters recent outcomes, loop flags, global/place
      attempts, and mirrored attempt events from policy state while retaining
      auditable rows. Focused tests and the paper/docs protocol pass.
- [x] Run the user-authorized, history-only pair on `6,11,12,47`, one managed
      job at a time after successful detached probes. Shadow job
      `20260823_085244_c42458` and agent job `20260823_091138_036fc6` both
      completed native-clean with validated 4-unit `DONE`. Both scored **3/4**
      with identical letters; agent-visible history raised mean planning steps
      **35.25→39.25** and attempts **31→37**, while repeat-failure rows fell
      **6→4**. q11 still failed A vs D; q12 took 16 extra steps. Mechanism and
      no-leakage pass, capability/efficiency do not. Keep it opt-in and do not
      scale; frozen summary is
      `paper/data/hmeqa_agentic_h2h/action_history_pair_20260823.json`.
- [x] Implement typed cross-round action memory and a benchmark-scoped
      pre-router progress gate. Grounded state v3 now leads with semantic intent
      + stable place/frontier/view identity rather than `r0`/`obs=15`, and
      `action_history.py` defines deterministic work/equivalence keys, progress
      tokens, typed outcomes, and one transient retry. The independent
      manifest-frozen axis is `eqa.action_progress_mode=off|shadow|enforce`
      (default off). Shadow leaves cards/execution unchanged; enforce omits
      unchanged terminal/no-progress variants from the exact allowlist and
      covers investigate aliases, explicit verifies, and frontiers; automatic
      post-motion verifies are summarized by their parent action. Alternate
      approaches, new views/evidence/geometry, and partial motion remain
      eligible. Manifest schema v4 freezes the new axis, rejects legacy-policy
      combinations, and validates matching summary/trace diagnostics before
      completion. Exact q11/q12 trace-derived replay fixtures plus the 182-test
      agent, 151-test lifecycle, and 134-test graph-memory CPU gates pass; see
      `docs/attempt_ledger.md#static-world-action-progress-policy`.
- [x] Finish reproducible paper tooling: expose the optional `latexmk` + TeX
      Live bundle through `emet install paper`, `install.sh --paper`, and the
      interactive install menu. The package simulation and 59 focused CLI tests
      pass; documented Docker fallback built the current 38-page appendix/PDF.
- [x] Run the managed static-policy comparison on `6,11,12,47` after CPU gates:
      shadow job `20260823_155106_470fe6` scored **3/4**, mean steps **35.25**,
      attempts **31**; enforce job `20260823_162608_ca49e0` scored **2/4**, mean
      steps **43.25**, attempts **39**. Shadow matched the prior action-history
      shadow baseline; enforce regressed on q47 and added dwell on q12 without
      explicit suppress dispositions. Mechanism passes; do **not** scale enforce.
      Frozen summary:
      `paper/data/hmeqa_agentic_h2h/action_progress_pair_20260823.json`.
      **Merge tooling** with defaults off (`action_progress_mode=off`).
- [ ] Generalize suppression for dynamic worlds before enabling it as lifelong
      memory. Target/environment change events, map revision, elapsed-time/TTL,
      and evidence staleness must invalidate or decay suppression. Consider a
      scheduler that cools down/re-ranks actions instead of removing them.
- [ ] Keep the bundled A0→A1→A2 ladder, `2,76`, A3, rooms-11, holdout, and
      bal-32 blocked until both process and letter gates pass.
- [ ] Keep q104 deferred until scale; it is a known native-crash hot scene.

## HM-EQA count/clock — FIND views, not node-list answers (PR #130, 7/15 = ceiling)

The instance graph **finds** candidate views; it is not the answer (no integer counts
from YoloE node lists). PR #130 (`feat/instance-graph-repair`) hardens graph admission
(`instance_min_confidence: 0.12`, `instance_min_mask_points: 25`), requires label-match
for instance fusion, marks `CONFIRMED_MEMORY` as **LOOK** image ids, pins FIND/Action
RGB as Image 1, attaches detector crops, exposes **VIEW_STATUS** per-Image counters,
and releases the controller stay when a FIND view is spent / Unknown without `read N`.

Frozen 15-qid slice (`12,21,28,32,33,43,47,48,51,60,78,84,86,88,93`,
`scripts/run_hmeqa_countclock_slice.sh`) — **HEAD `fb9fa75d`+VIEW_STATUS: 7/15**, equal
to the no-YoloE ceiling `2f4b4d4f`. q93 (bar stools) recovered via attached FIND RGB;
q47 stays OK; q28/q78 OK via crop-as-Image1. Merged 2026-08-25 (PR #130).

### Remaining gaps (documented in PR, not blockers)

- [ ] **Exploration/coverage**: q86 (table lamps) still finds only 1 of 2 lamps. The
      agent pins the FIND view and loops `read1`/`1` on it (up to 20 planning steps)
      instead of going after the second instance. q32/q48/q51/q60 similarly stop early
      or answer from a partial view. Need coverage-aware targets / second-instance search.
- [ ] **Close-look / legibility**: q33/q43/q84 ("What time is it now?") fail because the
      clock is *visible* but not *legible*. `read N` currently re-attaches the same full
      frame (no detector bbox for clocks ⇒ no crop). A center-zoom / upscale of the
      `read N` target view is the low-hanging next step.
- [ ] Consider a `read N` re-crop path: on `read N`, attach a tighter/zoomed version of
      that image (not just re-send the full frame), so written dials/answers become
      legible without navigating again.

## OVMM agentic find — PR #110 / #111 follow-ups (validated 2026-08-11)

Context: teleport-mode OVMM find on the shared AgenticEQA loop. PR #110 fixes nav
(sample_target_point projection, chunked-nav-as-progress); PR #111 (stacked) adds the
`NavOutcome` enum + question-type-aware verification + camera diagnostic. What's left:

**Branch `feat/tamp-ovmm-perf` (2026-08-23 → 2026-08-28):** OVMM-scoped routing — unified
`_recall_nav_hypotheses`, GT placement seeds, find_recep nearby-investigate bias,
`nav_outcome` in router Recent actions, richer `trace_meta` from find-phase harness.
**2026-08-28 unify mapping (this PR):** `run_mapping_protocol` with `explore_steps>0`
now calls `run_agentic_eqa_result(agent, None, goal="explore and map…", max_nav_steps=n, max_rounds=n+1)`
— same `AgenticEQAExecutor` `mode=explore` as HM-EQA (router only `explore_frontier`/`finish`),
coverage-first frontier picks (no object `toward`), arrival `look_ahead` (tilt 0) facing frontier
then `update` (not `look_front -30°`/pre-sweep), hop-until-arrival for 27-wp kitchens;
`S0` `explore_steps==0` stays rotate-only. Voxel `localize_text` on finished map now beats
camera-pose-at-feet cards (`CAMERA_POSE_PLACE` redirect), SigLIP on arrival RGB is the query.
**Efficiency pivot:** Stretch 3-ep slice took ~3.5 h with 0/3 success — default gate
is now **rby1** (`PROFILE=smoke` / `slice` in `run_ovmm_find_recep_slice.sh`);
`look_around` skips head pans on non-Stretch. Docs:
`docs/experiments/ovmm_agentic_find_teleport.md` + `docs/ovmm_find_phase_benchmark.md`.
**Integration (2026-08-25):** merged `feat/instance-graph-repair` (PR #130) for
clock/count FIND view pinning + `close_look_label`; extend investigate bias to
`find_object` and close-look questions on top.

### 2026-08-28 assessment — does unified mapping fix OVMM failures?

**Historical baseline (pre-unify):** teleport 9-ep sweep ~1/9 FindObj, 0/9 FindRec
(`recep_slice_20260823_172713` 0/3 in 3.5 h). Real-physics 9-ep ~1/3 obj, 1/6 recep.
Mapping was `run_mapping_protocol: spin + N×execute_action("")` (DynaMem multi-goal A*,
`look_around` before drive, `look_front -30°`) → voxel saw floor/sky, not kitchen; find
started a *new* `AgenticEQAExecutor` and scored wall-node cards; `localize_text("jar")`
often empty or floor. Stretch 27-wp kitchen paths truncated at 8 wps (never reached).

**Expected after unify (1–2 turn prompt sanity):**
*Turn 0* `inspect_graph` → hypotheses from `localize_text` on finished voxel map (or none) +
graph nodes/frontiers; prompt = `Rooms: …` + merged `SCENE_GRAPH` (`CONFIRMED_MEMORY` folded,
no dupe), `Recent actions: … nav_outcome=…`, stable allowlists. No object `toward` during
mapping, so frontier picks stay uncovered-first.
*Turn 1* `explore_frontier` → `target_theta` toward frontier, `look_ahead` 0° at arrival,
`capture_and_update` → new observation → SigLIP points on that frontier surface enter voxel.
If a jar/cab was in that frustum, `localize_text` next turn produces `proposal` card
`obs_id<-3M` that **beats** any `CAMERA_POSE_PLACE` view.
*Turn 2* find: `inspect_graph` on finished map now has SigLIP-backed proposal; router
should `investigate(proposal)` before `explore`, `verify_siglip` on that view; one
`explore` only if `ABSENT` (not 150 wall cards). Prompt after 1–2 turns should list
`detections: 1+` with `source=voxel` and `views` separate, `visible_frontier_ids` shrinking.

**Verdict:** unify addresses the load-bearing mapping bug (coverage vs object-biased
`toward`, arrival tilt, hop-until-arrival). Voxel-first find + `CAMERA_POSE_PLACE`
redirect addresses the scoring bug. It **does not** by itself fix the “recep loop
explores away” targeting problem — that is still a search/ranking issue (see below).
Needs a GPU `rby1` smoke + `stretch-kitchen` to confirm `localize_text non-null`
and that `n_explore` now increments `mapping_n_explore` in JSON.

- [x] **Unify OVMM mapping with agentic explore (2026-08-28):** `run_mapping_protocol` for
      `explore_steps>0` now uses `mode=explore` coverage loop; arrival `look_ahead` facing
      frontier; tests for mapping entrypoint/arrival look/voxel-first; docs updated.
- [x] **Object targeting fixes (2026-08-28, live rby1 S0 + kitchen):** three systematic
      find-loop bugs found and fixed on the **shared** AgenticEQAExecutor:
  1. **SigLIP released across find phases** (`agentic/answer.py:_do_submit_answer` →
     `release_siglip_for_vlm` drops `voxel_map.encoder`; nothing re-attached it). FindRec
     (second phase) could never `localize_text`. Fixed: `re_attach_siglip_encoder` wired into
     `warm_siglip_confirmed_memory` (`eval/dynagraph_vram.py`), so each phase localizes on
     the finished map. Rby1 S0 FindRec went from `recep_localize_source: None` → voxel err 0.0.
  2. **Voxel proposals re-chased after close ABSENT** — the VLM router re-emits
     `investigate(obs_id<0)` on the same wall point (`-3000000` visits 2..N). Fixed: proposals
     are one-shot in `_hypothesis_nav_blocked` (`agentic/capture.py`), not just
     `_unused_detection_hypothesis` — S0 obj loop went from 3× chases to 1×.
  3. **Disproven voxel pins were still scored** — a close ABSENT leaves the pin; the harness
     `pinned_xyz_from_phrases` fallback scored the wall point. Fixed: `unpin_localize_xyz`
     (`mapping/voxel_localize.py`) called from `_maybe_retract_claim_after_station` on ABSENT,
     plus clearing the loop-scored voxel record.
  **Measured (rby1 S0 `default_table_rby1_s0_distinct_recep`, 5 runs): FindObj 5/5 (voxel,
  err 0.0); FindRec 3/5 (voxel err 0.0 when found).** FindObj is stable; FindRec is SigLIP
  marginality on the small blue cube (YOLO labels in this scene are garbage:
  box/sign/tv/monitor/divider — graph recall is useless; voxel is the only reliable path).
  Mixed gate still validates EQA: countclock **7/15 = gateAB** (no regression from the
  shared-loop changes).
- [x] **Kitchen explore stall (2026-08-28):** live agentic mapping still creeps (8–20 hops
      cover only ~3 m²; the nearest-uncovered frontier clamps to ~0 and the loop re-picked
      the same frontier). Added a no-progress goal block in `_tool_explore_frontier`
      (`agentic/explore.py`): after a nav that moved < 0.10 m, block the FRONTIER XY in
      `_habitat_recent_goals`/`_habitat_blocked_goals` so the next pick chooses a different
      frontier or falls through to multi-goal explore. Frontier selection now rotates
      (verified live). Kitchen find still 0/2 — coverage volume, not the loop.
- [x] **SigLIP re-attach scoped to OVMM (2026-08-28):** re-attach was briefly in the shared
      `warm_siglip_confirmed_memory`; HM-EQA countclock q47 flipped False, so it is now
      called only from the OVMM harness `run_ovmm_agentic_localize` (before each phase) —
      HM-EQA keeps its released-SigLIP behavior.
- [x] **One-shot proposals refined to close-ABSENT (2026-08-28):** the first one-shot gate
      blocked a proposal after *any* one nav; countclock dropped to 5/15 (q21/q47/q93
      flips). Refined: a proposal is blocked only after a **close ABSENT** recorded on that
      card (`_hypothesis_nav_blocked` + `_unused_detection_hypothesis` now both check
      `_place_inspect.last_verify == ABSENT`), so HM-EQA count/locate targets stay
      re-approachable from a new bearing while OVMM wall-chases stay one-shot. **Countclock
      back to 7/15 = gateAB** (q21/43 recovered; q93 True solo — variance, not systematic).
- [ ] **Recep/object targeting residual — small-object SigLIP marginality.** FindRec (blue cube)
      is 2/4 because the cube's SigLIP cosine hovers at the localize bar (top_sim ≈ 0.10–0.14;
      `localize_text` threshold 0.14). When the map lacks good blue-cube points, no pin forms.
      Levers: (a) lower the voxel localize bar for find (risky false positives; one-shot
      proposals now make ABSENT cheap), (b) a closer start-recep look so YOLO/SigLIP see the
      object, (c) accept variance and report voxel err distribution.
- [ ] **Recep loop explores away from the target, never converges.** "Where is the table?"
      runs all 8 rounds with `nav=0..2 explore=N`; the router keeps picking
      explore_frontier and the assess returns not-present (table not in those views).
      Obj loop now verifies in 2 rounds, so the verify gate + camera are fine — this is a
      **targeting/search problem**. Next steps:
  - [ ] Debug why investigate cards for "table" are absent/ranked low: `hypothesize_nav_targets`
        (graph_memory.py:2689) recall for receptacle labels; dump the card list the router saw
        (`_investigate_hypotheses` at agentic_eqa.py:1015) per recep round.
  - [ ] Check `_prefer_explore` / `_not_present_streak` / `_update_escape_streak` loop: close-look
        ABSENT nudges explore (agentic_eqa.py:2593), which may pull the robot away from the start
        table instead of toward it. Add an investigate bias when the question target is near the
        robot's start receptable.
  - [ ] Consider a distance-gated investigate fallback: if a place card is within `X` m and
        unvisited, prefer `investigate` over `explore_frontier` even when the router leans explore.
- [ ] **Open-question verification semantics are now present-only** (agentic_eqa.py:2441 `open_view_present`).
      Confirm this doesn't over-trust a single glance for location questions where the target is
      visible but mis-localized — verify the verified-obs XYZ error distribution in the next
      full OVMM run (teleport) against GT.
- [ ] **Full teleport OVMM find validation run** (all 9 episodes, `EMET_SIM_NAV_TELEPORT=1`) on
      `feat/nav-outcome-enum` after the recep-targeting fix: want find_object + find_recep rates
      vs the real-physics baseline (1/3 obj, 1/6 recep) and the pre-fix teleport run (1/9 obj, 0/9 recep).
- [ ] **Camera diagnostic** `scripts/debug_ovmm_camera_quality.py`: confirmed no all-black frames
      (mean≈122, std≈93, black_frac=0). Depth stream reads 0 because the diagnostic client is built
      with `allow_missing_depth`; for a true blocked-camera depth check, request depth on the client
      (drop `allow_missing_depth` or set `EMET_ZMQ_FULL_HZ` depth track). Ray trace (`mj_ray`) needs
      model/data on the client — either expose it or run the ray check inside the sim server process.
- [ ] **Slowness**: teleport base nav is ~4s/step (vs ~50s real physics) but each episode is still
      ~30-50 min because (a) every investigate does a full 4-pan head sweep (~40-80s) and (b) two
      agentic loops per episode (obj + recep). Options: `_fast_explore_lookaround` (2-pan) in the
      OVMM find path like run_dynagraph does, and cap recep rounds when the recep is the start
      receptable (already known).
- [x] **NavOutcome in router history**: typed recent actions now render the
      motion enum alongside semantic target/outcome/progress, so the VLM can
      distinguish reached vs progress vs blocked.
- [ ] **NavOutcome in durable nav ledger**: propagate the enum through
      `NavAttemptResult` / `graph_memory.record_nav_attempt` instead of relying
      on status/note reconstruction in offline artifacts.

## Embodied agent planning (world model + tool calling + motion)

Branch `feature/agent-world-model`. Phases 1–3 + Phase 4 helpers are **landed**
(ledger default **off**). Do not touch pinned HM-EQA/OVMM configs.

| Doc | Role |
|-----|------|
| [docs/attempt_ledger.md](docs/attempt_ledger.md) | **Shipped** operator/dev reference (enable, schema, writers, tests) |
| [docs/plans/2026-08-08_embodied_agent_planning.md](docs/plans/2026-08-08_embodied_agent_planning.md) | Design history + phase checklist |

### Phase 1a — nav ledger store
- [x] `AttemptRecord` store on `GraphEQAMemory` (`attempt_ledger.py`); `record_nav_attempt` dual-writes when `eqa.attempt_ledger` / `EMET_EQA_ATTEMPT_LEDGER` on (default **off**); export/import + derive helpers; tests in `src/test/memory/test_attempt_ledger.py`.

### Phase 1b — ABSENT persistence
- [x] Config-gated persistence of verify-ABSENT evidence past the per-question `_retracted_nav_claims` reset (`persist_absent_claims` / `EMET_ATTEMPT_LEDGER_PERSIST_ABSENT`; retract also writes a `verify:absent` ledger row).

### Phase 1c — manip outcomes
- [x] CHAT executor pickup/place failures write to the ledger via `ToolOutcome`.
- [x] Stretch / AnyGrasp `_pickup` / `_place` propagate real success (`agent.manipulate` / `agent.place` / `GraspObjectOperation.was_successful`).

### Phase 1d — surface to planners
- [x] Enriched `[attempts: …]` place cards + CHAT `navigation_diagnostics` recent attempts + CONFIRMED_MEMORY `attempts:` tags.

### Phase 2 — shared tool-outcome schema
- [x] `ToolOutcome` shape shared by CHAT `_dispatch_tool_calls` and EQA `handle_tool` (`emet.agent.tool_outcome`); both write to the ledger.
- [x] Router prompt hygiene: shared `_EQA_RULE_*` atoms compose `_EQA_FORMAT_BLOCK_*` + byte-stability test in `test_room_policy.py`.

### Phase 3 — motion planning behind a narrow interface
- [x] Structured `NavAttemptResult.status_code` + `emet.controller.nav_attempt.sync_nav_attempt_to_ledger` from `_log_nav_attempt` (feeds ledger + `_last_nav_plan`).
- [x] `aim_arm_at` → `closer_look.aim_wrist_at_phrase` (localize + kinematic EE aim when available; structured failure codes otherwise).
- [x] OVMM-full pick/place path records attempts to the ledger (`record_manip_attempt` in `ovmm_full.py`).

### Phase 4 — eval (no regressions)
- [x] Repeat-failure metrics helpers (`attempt_metrics.summarize_repeat_failures`) + unit tests.
- [ ] GPU deltas vs pinned baselines on HM-EQA agentic arm + OVMM find (via `emet jobs`; ledger opt-in). Report repeat-nav-failure / wasted-rounds deltas; never by flipping pinned configs.

### Docs (this branch)
- [x] Canonical [docs/attempt_ledger.md](docs/attempt_ledger.md) + links from README doc map, `AGENT_RUN.md`, `graph_eqa.md`, `dynagraph.md`, `evaluation.md`, `emet_config.md`, `environment_variables.md`, `motion_planning.md`, `ovmm_full_benchmark.md`, plan index.

### Follow-ups (not blocking merge of ledger v1)
- [ ] Dual-write exit: once planners prefer the ledger, retire loop-local `_tried` / `_place_inspect` / `_nav_loop_flags` (or make them views). See plan Phase 1a.
- [x] Closer-look polish: `take_ee_picture` requires a successful `aim_arm_at` grant (consumed on capture); prompt + skill specs updated.
- [ ] Mars/Stretch wrist-frame + collision defaults dogfood on hardware (real EE stream after aim).
- [ ] Mobile-manip eval sketch: OVMM full + ledger metrics (pick attempts per success, re-grasp count) once orientation IK lands.

## Prompt / information flow

- [x] **One fact, one line — de-duplicate EQA memory blocks**: folded CONFIRMED_MEMORY → SCENE_GRAPH is now the **default** (`eqa.merged_memory` on; env can flip). HM-EQA paper row pins `merged_memory: false` in `configs/benchmarks/dynagraph.yaml` to keep published numbers on the standalone block. IMAGE_DESCRIPTIONS (when enabled) omit labels already tagged on SCENE_GRAPH Image-N lines.
- [x] **Room context never reaches the answer VLM**: done — `to_string` emits `Rooms:` + per-node `(room)` tags and `query_answer` passes them to the answer model.
- [x] **EQA answer format → JSON**: HM-EQA/MCQ use `{"reasoning","answer","confidence","action","confidence_reasoning"}` (`eqa.answer_format` / `EMET_EQA_ANSWER_FORMAT`; default json for hmeqa/mcq). `parse_answer` prefers JSON with labeled-field fallback; prefill is `{"reasoning":`.
- [x] **Unified token budget for the EQA prompt**: `eqa_vl.eqa_prompt_max_tokens` default 2500 (`EMET_EQA_PROMPT_MAX_TOKENS`); truncation order HISTORY → CONFIRMED_MEMORY → edges → labels via `build_eqa_prompt_text`.
- [x] **Router prompt hygiene**: shared `_EQA_RULE_*` atoms in `agentic_tools.py`; byte-stability test pins format-block SHA256 + identity across calls.
- [x] **HISTORY loop risk**: HISTORY stores one-line outcomes (`Iter: answer=… conf=… action=… salvage=… | reason`) plus `Nav_result`, not raw model replays.
- [x] **Prompt after 1–2 turns (agentic):** after `inspect_graph` + 1×`explore_frontier(look_ahead→capture)` the state
      message is `Rooms: …` + merged `SCENE_GRAPH` (no dupe `CONFIRMED_MEMORY`), `Recent actions` with
      `nav_outcome, target_theta, room_aligned`, stable allowlists
      `place_ids/place_obs_ids/frontier_ids` via `action_gate`, typed `HISTORY` lines and
      `visible_event_ids`. Verified 2026-08-28 on `AgenticEQAExecutor` `mode=explore` mock: `target_theta` toward frontier,
      `look_ahead` before `capture`, `detections` vs `views` split in `inspect_graph`.
- [ ] **CHAT `_FORMAT_BLOCK` is ~90 lines of routing edge cases** (prompt.py): consider tiered prompt (short default; detailed hints appended only for 4B-class routers) and measure system-prompt chars/tokens with and without hints.
- [ ] **describe_scene grounding**: currently caption + optional graph labels appended ad hoc (`describe_head_camera_scene_text`, controller_dynamem.py:1049). Define one consistent grounding format shared with `query_scene_graph` so the chat VLM sees the same memory vocabulary as EQA.

## Embodied agent / Herman

- [x] **One open-vocab scene graph (not two builders)**: CHAT uses mutually exclusive `agent.memory_backend` (`dynagraph` | `graph_eqa` | `open_vocab` | `dynamem`). Discord presets → Dynagraph memory plug-in only; GraphEQA baseline left frozen for paper. Lifelong save/load writes the active plug-in only.
- [x] **Arm IK “closer look” (v1)**: CHAT `aim_arm_at` → `closer_look.aim_wrist_at_phrase` (localize + kinematic EE aim when available). Follow-ups under Embodied agent planning (EE picture after aim; hardware dogfood).
  - Fallback when aim unavailable: “closer look / inspect X” → **change viewpoint** (rotate / small drive) then `describe_scene` / `send_image` (head), never raw `take_ee_picture`.
- [x] **Chat verify ≈ EQA look**: prompt + tools route “look at / closer look / are you sure” → `face_toward` then `describe_scene` (not blind ±45°); tests in `test_agent_prompt_and_tools` / `test_face_toward`. Full EQA-style `navigate_to_obs` + `verify_siglip` in CHAT still optional later.
- [x] **`emet run` + `--connection`**: wrapper no longer injects default `--robot-ip 127.0.0.1`, so `--connection herman` resolves the profile host.
- [ ] **Interruptible explore**: Discord messages queue until `explore` finishes; drain `unified_input_queue` mid-nav.
- [ ] **VRAM for Discord house chat**: default Mars preset loads tool LLM + EQA 8B VL + SigLIP (+ DA3). Easy OOM on 24 GiB when map update / DA3 overlaps generate. Mitigations: `--onboard-da3`, share one VL (`--llm qwen3-vl-eqa --share-memory-vllm`), or lazy-unload captioner between turns. (Dual OV+GE builders removed; still watch two VLMs.)

## Mapping / safety

- [x] Map-clip `move_forward` (including 0.1 m); refuse when map empty/blank.
- [x] Empty cloud guard in `list_objects_in_an_image` (navigate crash).
- [x] Clearance-aware A* + abort on waypoint timeout (prefer open space; do not raise `dilate_obstacle_size` as the primary fix).

## Memory stack (voxel + graph) — review backlog

- [ ] **Drift correction / relocalization** (long real-world runs): voxel map is robot-start-relative with a fixed `grid_origin`; only `--refine-start` applies an SE(2) fudge at checkpoint load. Plan ICP / pose-graph correction against the voxel map or re-anchor graph nodes during operation.
- [ ] **Pin depth units per source + runtime assert**: `sensor_graph_builder.py:80` assumes meters (`scaling=1.0`) while `core/interfaces.py:196` defaults to `1e-3` (mm). Verify per source (Stretch d405, DA3, Habitat) and assert once at attach.
- [ ] **Robot self-filter lost in DynaMem `add()`**: `voxel_dynamem.py:1109` copies base `voxel.py:403` minus the URDF mesh self-filter — the robot may see itself on real hardware. Reintroduce or document why it is off.
- [ ] **Split `query_answer` + de-dup renumber/rebuild blocks**: `build_eqa_prompt_text` is extracted (#104); `query_answer` is still ~477 lines — still need `run_eqa_prompt(commands)` / `finalize_eqa_answer(parsed)`. The renumber+rebuild block in `maintain`/`_drop_nodes_near`/`absorb_object_node` is copy-pasted 5×.
- [ ] **Rooms as first-class nodes**: runtime `RoomCluster` + prompt `Rooms:` / `(room)` tags exist; still recompute every refresh. Persist room id / name / bounds on nodes and in `graph.json` exports.
- [x] **`room_clustering/` + `partition()` + `proximity`**: naive `near` + XY radius; `room_clusters.py` is the naming/stamp facade.
- [ ] **`occupancy_cc` room backend**: flood-fill free/explored cells on the voxel 2D map; assign instance nodes to occupancy CCs (respects mapped walls).
- [ ] **`portal` room backend**: occupancy CCs cut at narrow passages / doors.
- [ ] **Room clustering backend sweep** once a second geometry backend exists (`eqa.room_clustering.backend` / `EMET_EQA_ROOM_CLUSTERING_BACKEND`). Do not mix with OVMM S0 instance labeling.
- [ ] **Bound voxel memory growth**: `observations` never pruned, `semantic_memory` keeps every subsampled point, pickles dump everything. Trim frames + downsample old semantic points on a schedule.
- [ ] **Prefix-KV timeout leaves a live CUDA worker** (`qwen3_vl_client.py:549-569` raises while the generate thread keeps running): next generate can race. Serialize on a lock or hard-kill the worker.
- [ ] **Silent exception hygiene**: ~23 `except Exception` in graph_memory.py — at least `_logger.debug` with node/obs context. (Bare `except:` in `voxel_dynamem.list_objects_in_an_image` now logs and retries.)
- [x] **Dead code sweep**: removed unused `Qwen3VLClient = Qwen35VLClient` alias from `qwen_client.py` and discarded Candidate 1/2 comment blocks in `voxel_map_dynamem.py`. (`get_change_events` / `alternate_nav_target_for_failed_action` remain in use.)

## Docs / ops

- [x] Document Herman Discord happy path: `innate_mars_hardware.md` Discord section covers `EMET_BASE_ROTATE_ONLY` + `EMET_ALLOW_SDPA_ATTN` / flash-attn with a tethered copy-paste env recipe.
- [x] Action-outcome ledger docs for `feature/agent-world-model`: [docs/attempt_ledger.md](docs/attempt_ledger.md) (see Embodied agent planning § Docs).

## TAMP clutter benchmark + Nori A3 (follow-ups)

Benchmark: [docs/experiments/tamp_clutter.md](docs/experiments/tamp_clutter.md) ·
`scripts/eval_tamp_clutter.py` · `scripts/generate_tamp_clutter_registry.py` (200 episodes,
4 robots: rby1 / stretch / innate_mars / nori). Nori backend: [docs/robots/nori.md](docs/robots/nori.md).
GT+MCTS battery: [docs/experiments/tamp_clutter_testing.md](docs/experiments/tamp_clutter_testing.md).

**Eval code is merge-ready** (blocked-nav scoring, chord-sampled no-snap, reachable
landmarks, sim default manip for mars/nori). **#150** / **#151** already on `main`.
Remaining items are **GPU experiments / product**, not code blockers. Operator table:
[docs/experiments/tamp_clutter.md](docs/experiments/tamp_clutter.md) Remaining experiments
(E1 battery → E5 latch → E3 small YAML → E4 large registry → fill `tab:tamp_clutter`).

- [x] **GT+MCTS battery 24/24** (2026-08-28): pickplace / declutter / navblocked / navclear
      pass for nori, innate_mars, rby1 × iTHOR scenes 0–1, sim-oracle manip, zero AI models.
      That run predates chord-collision + reachable-landmark; re-run below before citing it.
- [ ] **GPU GT+MCTS battery re-run** (after chord-sample / 12 m landmark cap). Furniture on
      the spawn→landmark chord can now fail `navclear`. Queue, do not run inline:

      ```
      NEED_MIB=8000 uv run emet jobs run --name tamp-gt-battery --need-mib 8000 -- \
        uv run python scripts/eval_tamp_clutter.py --test-battery \
        --battery-robots nori --battery-scenes 0,1
      ```

      Then optionally `--battery-robots nori,innate_mars,rby1 --battery-scenes 0,1`.
- [ ] **Fill `tab:tamp_clutter`** from `aggregate_tamp_clutter.csv` (scored denominator;
      exclude `skipped_invalid`). Results section is still placeholders.
- [ ] **Large registry (200 templates)** via `emet jobs`:

      ```
      NEED_MIB=8000 uv run emet jobs run --name tamp-clutter --need-mib 8000 -- \
        uv run python scripts/eval_tamp_clutter.py \
        --episodes configs/ovmm/clutter_episodes_large.yaml
      ```

- [ ] **rby1 latch paper row**: default small-YAML smoke `ithor_cleanup_s1_bin_n3` and
      large-registry rby1 `latch` episodes. Stretch / mars / nori stay `sim` on floor clutter.
- [ ] **Live latch smokes (GPU, via `emet jobs`)**: rby1 default latch
      `--episode-id ithor_cleanup_s1_bin_n3`. Mars/nori floor objects need `--manip-mode latch`
      (expected weak: Nori IK bottoms out ~0.29 m vs z≈0.02 floor). Unit/offline path already
      passes.
- [ ] **Integrate TAMP into the agent** — the real product blocker now that MCTS pick/place
      works. The single-object semantic tools (`scene_tasks` / `plan_pick_place` /
      `execute_pick_place_plan` in `emet.controller.task.tamp.agent_bridge` + `emet/agent/tools.py`)
      exist, but the **multi-object clear chain (`plan_clear_clutter`) is not exposed**. Add a
      `clear_clutter` CHAT skill (resolve scattered objects from scene graph/memory → run the
      MCTS chain to the bin → optional landmark nav) so the LLM agent can parse "clean up the
      room" / "get to the sofa" and drive TAMP end-to-end; reuse `AgentTaskRef`/`AgentPlanBuild`
      handles and the same no-AI test battery for the agent path.
- [x] **innate_mars actuator naming**: the innate_mars MJCF actuators are *unnamed*, and the
      robosuite server applies `{"joint": vec}` via `mj_name2id(mjOBJ_ACTUATOR, aname)` —
      unnamed actuators never receive ctrl from `set_actuator_positions`. Name them (like
      nori_a3.xml) so the kinematic streaming path actually drives the arm.
- [ ] **Nori real-hardware client**: implement an `AbstractRobotClient` adapter over
      `nori-sdk` (WebRTC jog streams → emet motion contract). The SDK is teleop-oriented;
      absolute-joint moves need the action-completion path.
- [ ] **Nori MolmoSpaces spawn metadata**: `emet molmospaces write-spawn-metadata --robot nori`
      → commit `molmospaces_spawn.json`; then `emet serve mujoco --scene ithor --robot nori
      --headless` smoke (see supported_robots.md extension checklist).
- [ ] **Nori RoboCasa** row: strip-replace handling / spawn guards if kitchen scenes are wanted.
- [ ] **Stretch `latch`** via the combined robosuite server — see next section.

## TAMP clutter: kinematic `latch` on Stretch via the combined robosuite server (follow-up)

The capability gate + per-robot arm parser (Phases 1–2) landed **innate_mars** `latch`
(capability gate in `robosuite_server.py`, curated `ArmChain` on the innate_mars spec);
Stretch is deferred. Stretch `latch` needs:

- [ ] Point `get_robot_spec("stretch").mjcf_path` at `src/emet/assets/robot/stretch.xml`
      (or `stretch_mj_3.3.0.xml`) so `ArmManipProfile` can build an offline IK model and
      `KinematicPickPlaceExecutor._ensure_model()` succeeds.
- [ ] Fill the Stretch `ArmChain.actuator_names` (MJCF actuators are unnamed today) once
      the mjcf_path is set.
- [ ] Route Stretch iTHOR / Robocasa scenes through `RobosuiteZmqServer` (the merged-MJCF
      path rby1 / innate_mars use) instead of `MujocoZmqServer` — "combine the Stretch sim
      server". Harness uses `GenericZmqClient` for stretch sim (`EMET_STRETCH_GENERIC_ZMQ=1`
      already exists). Keep **default-table Stretch on `MujocoZmqServer`** so existing tests /
      interactive behavior are untouched; gate the switch on scene kind or env.
- [ ] Enable `ROBOT_DEFAULT_MANIP_MODE["stretch"] = "latch"` after a stretch iTHOR `latch`
      smoke (`plan_clear_clutter`) passes.
- [ ] Risks: stretch telescoping prismatic arm IK + RRT; the curated arm chain already
      lands in Phase 2; verify no regressions on MujocoZmqServer stretch tests
      (`emet test --no-sim` gate).

## Manipulation / MolmoSpaces + rby1 (PR #83 follow-ups)

Offline units + scripted table smokes exist; these are the remaining **real / integration** and product gaps.

### Config / agent path
- [x] **Wire `agent.manip_mode` / `manip_collision` / `manip_planner` into `DynamemTaskExecutor`**: `emet run agent` merges the finalized chat `agent:` section (YAML / `--set agent.manip_*`) into `parameters["agent"]` before executor init; `EMET_MANIP_*` env vars still win. (Manip-only top-level `agent:` blocks are now recognized as chat-agent sections by the loader.)
- [x] **Stretch MuJoCo pick/place default (docs)**: with visual-servo off, any sim advertising `sim_set_body_pose` takes **GT teleport** (`prefer_sim_teleport_manip`; table in `docs/molmospaces.md`; loud callout in `AGENT_RUN.md`). When comparing paper/OVMM Stretch numbers to the old grasp path, pass `-V` / `--visual-servo`.
- [x] **Stretch / AnyGrasp `_pickup` / `_place` always return True**: fixed — Stretch path propagates `agent.manipulate` / `agent.place` / `GraspObjectOperation.was_successful` (and declines confirmation → False).

### Real tests (not yet green on every machine)
- [ ] **MolmoSpaces ithor + rby1** kinematic and teleport smokes when `.venv-molmospaces` + assets are warm (`scripted_sim_pick_place` / `scripted_molmo_grasp_mp` / `scripted_tamp_pick_place` with `--sim configs/sim/molmospaces_ithor_train_0.yaml`).
- [x] **Molmo kinematic approach frame**: fixed `_world_base_xyt` + place detach/`sim_set_body_pose`/verify-before-retract; bowl→microwave kinematic PASS (grasp_err≈0.027, place_err=0; 2026-08-03).
- [x] **OVMM full** episode `molmo_ithor_rby1_s2_bowl_pp` with `manip_mode=sim` (find + teleport pick/place) — reconfirmed 2026-08-03 post base-frame/place fixes.
- [ ] **Robocasa / Stretch** pick-place smoke with and without `--visual-servo` to lock the teleport-vs-servo behavior.
- [ ] **CI / overnight**: mark or gate the above under `RUN_MOLMOSPACES_TESTS` / sim markers so agents use `emet test --no-sim` for the offline pack only.

### Motion / grasp quality
- [ ] **IK orientation**: MuJoCo IK is **position-only**; Molmo grasp quats set approach standoff but are not enforced at the EE. Need orientation (or constrained) IK before claiming grasp fidelity.
- [ ] **Kinematic gripper cmd failures**: `_set_gripper` open/close now logs a warning on ZMQ/actuator errors but still continues (attach/detach is the real hold). Decide whether gripper failure should fail the grasp/place result before attach, or verify finger joints after cmd.
- [ ] **GT body / receptacle disambiguation**: teleport + kinematic still take first category substring match (`receps[0]` / object query). Prefer nearest-to-nav-`point` (or explicit `object_gt_body`) when multiple bowls/microwaves exist.
- [ ] **Voxel / AABB arm collision in live agent**: kinematic path supports `EMET_MANIP_COLLISION=voxel|aabb`; default is `none`. Dogfood on a cluttered Molmo scene and turn voxel on by default only if latency is OK.
- [ ] **Instance-memory `plan_to_frontier`**: still one-sample → one plan (often RRT). Dynamem explore already uses multi-goal A*; port the same top-K → `goals=` pattern if SVM/instance agents stay in use.

### Hardware / product
- [ ] **Real-robot manip** (Stretch / Mars): no GT `sim_set_body_pose` — keep visual-servo / AnyGrasp; add a no-LLM tool-sequence test that asserts we **do not** enter teleport when `is_simulation` is false.
- [x] **`aim_arm_at` → `take_ee_picture` chain**: EE capture gated on successful aim grant (consumed once). Real-robot dogfood after pick failures still open (wrist stream / Mars).
- [ ] **Paper figures**: keep regenerating `manip_figures` / chase-cam MP4s from scripted TAMP on the scene used in the paper; check in paths under `~/runs/emet/` only (not repo blobs).
