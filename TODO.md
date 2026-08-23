# Living TODOs (small / near-term)

Short checklist for agent/hardware polish that is not worth a full plan doc yet.
Strike through or move to a PR when done.

Short checklist for agent/hardware polish that is not worth a full plan doc yet.
Strike through or move to a PR when done.

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
- [ ] Improve cross-round action memory before any new scale experiment. The
      current hard guard catches exhausted `investigate(obs_id)` attempts only
      after the router selects them, emits `NAV_LOOP_BLOCKED`, then redirects
      to generic frontier exploration. Replace this with a pre-router,
      deterministic action-equivalence/progress gate keyed by action + stable
      target + approach/view/evidence revision (and pose cell where needed).
      Filter or re-rank blocked actions before rendering the allowlist; permit a
      retry only after material new evidence or geometry; cover verify and
      frontier actions as well as investigate. Keep the durable ledger in
      `shadow` by default and expose only a compact redirect reason. Add q11/q12
      replay regressions and optimize total attempts/planning steps, not repeat
      count alone. Concrete current traces:
      `docs/attempt_ledger.md#what-the-memory-trace-looks-like`.
- [ ] Keep the bundled A0→A1→A2 ladder, `2,76`, A3, rooms-11, holdout, and
      bal-32 blocked until both process and letter gates pass.
- [ ] Keep q104 deferred until scale; it is a known native-crash hot scene.

## HM-EQA close-look / weak-class validation (PR #120 + follow-ups)

GPUs, not paper rewrites. The 95/113 partial says count 23% / clock 20% are the
collapses the SigLIP-evidence + close-look work targets. All landed via PR #120
(SigLIP evidence line, `dense_siglip_argmax_crop` single crop, opt-in multi-view
close-look consensus). **PR #120 is OPEN on this branch** (mergeable, no reviews
yet) — its two validation jobs died with the box 2026-08-19 and each still needs a
clean GPU-exclusive run before merge. What's left:

- [ ] **Count/clock validation run** (`scripts/run_hmeqa_countclock_slice.sh`, 15 ids)
      single-view vs multiview vs pre-close-look baseline (count 23% / clock 20%).
      Status: first launch (Aug 15) was a silent 0/15 on missing flash-attn (fixed
      `EMET_ALLOW_SDPA_ATTN=1` in-runner). Aug 19 relaunch (jobs `20260819_022652_ece872`
      single-view, `20260819_022702_c3c900` multiview) **died with the box** ~22:35 —
      `pid exited without DONE`. Singleview had scored qids 12,21,28,32,33 of 15;
      multiview died on qid 12. Runner does **not** resume (`run-batch` overwrites the
      jsonl), so relaunch both arms fresh, GPU-exclusive (`NEED_MIB=12000`), then confirm
      the trace logs `close_look_views=N` on the multiview arm.
- [ ] **Temporal close-look consensus** (DeWorldSG idea #1 — implemented, opt-in
      `eqa.agentic_close_look_multiview` / `EMET_EQA_AGENTIC_CLOSE_LOOK_MULTIVIEW`,
      default **off**): aggregate up to 3 close-look crops across views per target
      phrase. Flip the default on only if the multiview arm beats single-view on
      the count/clock slice. (Cross-view retraction variant is PR #124
      `feat/hmeqa-graph-tuning`, open — strip ABSENT across all obs ids.)
- [ ] **Gaussian-similarity node merge** (DeWorldSG idea #2 — not started): replace
      the point-anchored `dynagraph_merge_xy_m` / staleness heuristics with a
      probabilistic per-object Gaussian (depth-aware var). Merge when two objects'
      posteriors overlap; track object-entity persistence across graph refreshes.
      Measurable on location/state classes where disambiguation fails (rug
      q-location, ACZZiU 0/5).
- [ ] **Depth-aware crop** (DeWorldSG idea #3 — not started): `dense_siglip_argmax_crop`
      is pixel-argmax in 2D. Weight the crop by depth continuity / object extent
      (drop false patches on walls/ceilings; prefer small floor objects). Floor
      pick suite (`ovmm full --manip-mode mcts`, `floor_object` eps) is the target.
      Floor-suite baseline (PR #120): ground_truth rby1 mcts pick_success=True /
      ovmm_full_partial=0.75 (place = known rby1 attach/release flake); teleport refs
      1.0; pre-fix dynagraph find_obj=False / partial 0.0–0.5 (cube-vs-brick).
      **SigLIP-validate rerun still owed**: `tamp-floor-siglip-validate` (2026-08-14)
      was INVALID — VL worker OOM / connection refused (co-ran with the paper eval).
      Rerun `scripts/eval_tamp_floor.py --manip-mode sim --backend dynagraph`
      GPU-exclusive and compare find/partial vs the pre-fix table before merging #120.

## OVMM agentic find — PR #110 / #111 follow-ups (validated 2026-08-11)

Context: teleport-mode OVMM find on the shared AgenticEQA loop. PR #110 fixes nav
(sample_target_point projection, chunked-nav-as-progress); PR #111 (stacked) adds the
`NavOutcome` enum + question-type-aware verification + camera diagnostic. What's left:

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
- [ ] **NavOutcome propagation**: `nav_outcome` is recorded in the agentic trace but not yet in
      `NavAttemptResult` consumers (graph_memory `record_nav_attempt`, router state message). Expose
      the enum string in the router "Recent actions" so the VLM can see reached-vs-progress-vs-blocked.

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
- [ ] **Bound voxel memory growth**: `observations` never pruned, `semantic_memory` keeps every subsampled point, pickles dump everything. Trim frames + downsample old semantic points on a schedule.
- [ ] **Prefix-KV timeout leaves a live CUDA worker** (`qwen3_vl_client.py:549-569` raises while the generate thread keeps running): next generate can race. Serialize on a lock or hard-kill the worker.
- [ ] **Silent exception hygiene**: ~23 `except Exception` in graph_memory.py — at least `_logger.debug` with node/obs context. (Bare `except:` in `voxel_dynamem.list_objects_in_an_image` now logs and retries.)
- [x] **Dead code sweep**: removed unused `Qwen3VLClient = Qwen35VLClient` alias from `qwen_client.py` and discarded Candidate 1/2 comment blocks in `voxel_map_dynamem.py`. (`get_change_events` / `alternate_nav_target_for_failed_action` remain in use.)

## Docs / ops

- [x] Document Herman Discord happy path: `innate_mars_hardware.md` Discord section covers `EMET_BASE_ROTATE_ONLY` + `EMET_ALLOW_SDPA_ATTN` / flash-attn with a tethered copy-paste env recipe.
- [x] Action-outcome ledger docs for `feature/agent-world-model`: [docs/attempt_ledger.md](docs/attempt_ledger.md) (see Embodied agent planning § Docs).

## TAMP agent tools

Design and acceptance criteria: [docs/plans/2026-08-22_tamp_agent_tools.md](docs/plans/2026-08-22_tamp_agent_tools.md).

**Merge posture (PR #120):** safe to land — CHAT/TAMP infrastructure is additive;
paper HM-EQA / OVMM-find defaults unchanged. Routine acceptance is the fast
**`PROFILE=smoke`** gate (~1–2 min). OVMM floor perf and Stretch agentic sweeps
are explicit follow-up (`PROFILE=full`, `eval_tamp_floor.py`).

- [x] Keep semantic CHAT task references separate from private simulator
      `*_gt_body` grounding; select and retain the actual receptacle.
- [x] Expose plan-first `plan_pick_place` and
      `execute_pick_place_plan`; route `pick_place` through guarded planning in
      a live simulator and preserve the hardware fallback.
- [x] Revalidate live poses/capabilities before one-shot execution and fail
      closed on stale plans or invalid benchmark receptacles.
- [x] Make floor setup and per-episode `mcts` / `sim` / `skip` modes explicit,
      including configured `floor_z_m` and find-only controls.
- [x] Add offline regressions for semantic grounding, CHAT schemas/dispatch,
      selected-receptacle scoring, floor setup, and close-look crop limits.
- [x] **Stretch teleport CHAT control** (manual): job `20260823_013540_64c74f`,
      displacement 0.10 m (`plan_pick_place` → `execute_pick_place_plan`).
- [x] **MolmoSpaces CHAT `scene_tasks` smoke** (manual, teleport): job
      `20260823_003208_1379d8`, `object_filter=bowl` → `task:1` pick bowl,
      displacement 2.21 m.
- [x] **Managed smoke gate green** (`PROFILE=smoke`, default): rby1 CHAT
      `scene_tasks` → plan → execute through live tools + kinematic base snap.
      Job `20260823_152854_4c7766`: 81 s sim, displacement 2.21 m, placement
      error 0.02 m. Gate quoting bug fixed 2026-08-23 (`run_item` uses `"$@"`).

### Next (OVMM perf / coverage — not blocking merge)

- [ ] **`PROFILE=full` gate once** (`chat kinematic stretch floor`, ~1–2 h):
      `./scripts/run_tamp_agent_tools_gate.sh` with `PROFILE=full` via `emet jobs`.
      Stretch agentic head sweeps and the 4-episode RoboCasa floor matrix belong
      here, not in routine smoke.
- [ ] **OVMM floor smoke** (`eval_tamp_floor.py --smoke`): single rby1 MCTS
      dynagraph episode (~5–8 min). Use when changing agentic find / SigLIP /
      floor scoring — not the default PR gate.
- [ ] **Dynagraph floor find quality**: Stretch head sweeps dominate wall time
      (15–45 s/sweep vs ~1–2 s on rby1). For routine agent tests prefer rby1 +
      `EMET_SIM_NAV_TELEPORT=1`; improve OVMM explore/verify separately.
- [ ] **rby1 MCTS place flake**: kinematic pick succeeds but place attach/release
      can fail (partial 0.75 in floor runs). Teleport refs score 1.0 — gap is
      arm place, not find/pick.
- [ ] **SigLIP floor validate rerun** (optional): `eval_tamp_floor.py --backend
      dynagraph` GPU-exclusive; compare find/partial vs pre-fix table (see DeWorldSG
      section below).

## Manipulation / MolmoSpaces + rby1 (PR #83 follow-ups)

Offline units + scripted table smokes exist; these are the remaining **real / integration** and product gaps.

### Config / agent path
- [x] **Wire `agent.manip_mode` / `manip_collision` / `manip_planner` into `DynamemTaskExecutor`**: `emet run agent` merges the finalized chat `agent:` section (YAML / `--set agent.manip_*`) into `parameters["agent"]` before executor init; `EMET_MANIP_*` env vars still win. (Manip-only top-level `agent:` blocks are now recognized as chat-agent sections by the loader.)
- [x] **Stretch MuJoCo pick/place default (docs)**: with visual-servo off, any sim advertising `sim_set_body_pose` takes **GT teleport** (`prefer_sim_teleport_manip`; table in `docs/molmospaces.md`; loud callout in `AGENT_RUN.md`). When comparing paper/OVMM Stretch numbers to the old grasp path, pass `-V` / `--visual-servo`.
- [x] **Stretch / AnyGrasp `_pickup` / `_place` always return True**: fixed — Stretch path propagates `agent.manipulate` / `agent.place` / `GraspObjectOperation.was_successful` (and declines confirmation → False).

### Real tests (not yet green on every machine)
- [x] **MolmoSpaces ithor + rby1 CHAT tool chain** (`scene_tasks` → plan → execute,
      teleport): green 2026-08-23 (`20260823_003208_1379d8`).
- [x] **MolmoSpaces ithor + rby1 kinematic** CHAT `scene_tasks` → plan → execute
      in managed smoke gate (`PROFILE=smoke`): green 2026-08-23 (`20260823_152854_4c7766`).
- [x] **Molmo kinematic approach frame**: fixed `_world_base_xyt` + place detach/`sim_set_body_pose`/verify-before-retract; bowl→microwave kinematic PASS (grasp_err≈0.027, place_err=0; 2026-08-03).
- [x] **OVMM full** episode `molmo_ithor_rby1_s2_bowl_pp` with `manip_mode=sim` (find + teleport pick/place) — reconfirmed 2026-08-03 post base-frame/place fixes.
- [x] **Stretch teleport** CHAT plan/execute (`default_table_stretch.yaml`): green
      2026-08-23 (`20260823_013540_64c74f`).
- [ ] **Robocasa / Stretch visual-servo** pick-place smoke (`--visual-servo`) to
      lock teleport-vs-servo behavior on hardware path.
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
