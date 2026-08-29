# Environment variables

Optional process-environment toggles for simulation, ZMQ clients, and MolmoSpaces. Most apps read these at startup; export in the shell before `emet serve` / `emet run`.

## MolmoSpaces

**[MolmoSpaces environment variables](molmospaces_environment_variables.md)** — spawn, autoplace, occupancy map, navigation teleport (`EMET_MOLMOSPACES_NAV_TELEPORT`), asset paths, and related test knobs.

**[MolmoSpaces spawn metadata](molmospaces_spawn_metadata.md)** — checked-in `molmospaces_spawn.json` per robot and `emet molmospaces write-spawn-metadata` (offline; not an env var).

See also [MolmoSpaces](molmospaces.md) for install and CLI usage.

## Configuration

| Variable | Used by | Purpose |
|----------|---------|---------|
| `EMET_CONFIG` | `emet run agent`, `emet run dynagraph`, `emet run dynamem`, `emet stream`, `emet capture` | Packaged default path for unified nested YAML when `--config` is omitted **and** no connection-profile `config` applies. Explicit `--config` wins; else profile `config` (named `--connection` or active profile); else this env / `configs/emet/default.yaml`. See [emet_config.md](emet_config.md) and [cli.md](cli.md) (`emet connect`). |

## Benchmarks

Paper benchmark runbook: [paper_benchmarks.md](paper_benchmarks.md). **Overnight smoke + diagnostics:** [evaluation.md](evaluation.md).

| Variable | Where used | Notes |
|----------|------------|-------|
| `EMET_DISABLE_TTS` | `DynamemController` init | Skip Piper TTS (`1`/`true`). OVMM find-phase sets this by default (no audio in batch eval; avoids Piper wedging under Robocasa+VL). |
| `EMET_SKIP_HEAD_SWEEP` | `DynamemController.look_around` | `1` — skip Stretch-style head pans (single `update()`). Non-Stretch robots (rby1) skip by default. |
| `EMET_FORCE_HEAD_SWEEP` | `DynamemController.look_around` | `1` — force head pans even on rby1 / GenericZmqClient. |
| `EMET_EVAL_EXPORT_MAP` | Habitat / OVMM / SQA3D episode bundles | Write `topdown_map.png` (default on). YAML: `eval.export_map`. Alias: `HABITAT_EQA_EXPORT_MAP`. |
| `EMET_EVAL_EXPORT_MAP_OVERLAY` | Habitat episode bundles | `topdown_map_overlay.png` (GT navmesh + agent map + trajectory; default on). YAML: `eval.export_map_overlay`. |
| `EMET_EVAL_EXPORT_MAP_VIDEO` | Same | `topdown_exploration.mp4` timelapse from stride map frames (default on). YAML: `eval.export_map_video`. |
| `EMET_EVAL_MAP_VIDEO_STRIDE` | Same | Steps between map frames when `EMET_EVAL_MAP_STRIDE=0` (default `5`). YAML: `eval.map_video_stride`. |
| `EMET_EVAL_EXPORT_VIDEO` | Same | Write `episode_rgb.mp4` (head camera; manifest `head_camera_mp4`). YAML: `eval.export_video`. Alias: `HABITAT_EQA_EXPORT_VIDEO`. |
| `EMET_EVAL_EXPORT_VIDEO_SUBSTEPS` | Habitat episode bundles | One RGB frame per Habitat `sim.step` during nav/rotate (default on). YAML: `eval.export_video_substeps`. |
| `EMET_EVAL_VIDEO_MOTION_PACED` | Same | Motion-paced MP4 encoding (default on). YAML: `eval.video_motion_paced`. |
| `EMET_EVAL_EXPORT_FRAMES` | Same | Save RGB frames under `frames/`. |
| `EMET_EVAL_EXPORT_OBSTACLE_GRIDS` | Same | `obstacles_2d.npy`, `explored_2d.npy`, `grid_meta.json` (default on). |
| `EMET_HABITAT_PAD_OBSTACLES` | `emet_habitat.runner._configure_habitat_mapping` | Habitat-only obstacle dilation radius in grid cells (default `0` = off, temporary doorway-entry probe). Non-zero also restores `filters.smooth_kernel_size=1`. |
| `EMET_EVAL_EXPORT_TRAJECTORY` | Same | `trajectory.jsonl` (default on). |
| `EMET_EVAL_EXPORT_OBJECT_CROPS` | Same | Dynagraph object-crop mosaic when graph memory is present (default on). |
| `EMET_EVAL_MAP_STRIDE` | Same | Save intermediate maps every N steps (0 = final only). Alias: `HABITAT_EQA_MAP_STRIDE`. |
| `EMET_EVAL_EXPORT_GRAPH` | Same | Full graph checkpoint (heavy). Alias: `HABITAT_EQA_EXPORT_GRAPH`. |
| `EMET_EVAL_EXPORT_COMPACT_MEMORY` | Habitat episode bundles | Write a graph-only `compact_memory/` checkpoint with semantic graph, identities, room/events, attempts, and runtime counters, but no voxel map, dense frames, navigation pixels, or evidence-view pixels. It supports semantic inspection/reload, not post-hoc visual verification (default off). YAML: `eval.export_compact_memory`. |
| `EMET_EVAL_EXPORT_WORLD_EVIDENCE_RGB` | Same | Save full-resolution `world_evidence_views/*.png` alongside evidence metadata (default on). Disable for compact sweeps. YAML: `eval.export_world_evidence_rgb`. |
| `EMET_EVAL_EXPORT_VOXEL_HISTORY` | Habitat episode bundles | Slim `observations_history.jsonl` from in-memory voxel frames (default on in Habitat runners). |
| `EMET_EVAL_EXPORT_VOXEL_PICKLE` | Same | Optional full `voxel_debug.pkl` via `write_to_pickle`. |
| `EMET_OVMM_OUTPUT_SIM` | `eval_ovmm_find_phases.py` | OVMM sim sweep output. Default `~/runs/emet/ovmm_find_phase` (`configs/ovmm/benchmark.yaml`). |
| `EMET_OVMM_OUTPUT_FULL` | `eval_ovmm_full.py` | Full OVMM (find + pick/place) output. Default `~/runs/emet/ovmm_full`. |
| `EMET_OVMM_OUTPUT_HABITAT` | `eval_habitat_ovmm_find_phases.py` | Habitat OVMM proxy output. Default `~/runs/emet/ovmm_habitat`. |
| `EMET_OVMM_SKIP_TABLE_MAPPING_POSE` | `ovmm_find_phase.run_mapping_protocol` | Skip rby1 default-table backup + `look_front` before rotate scan (`1`/`true`). |
| `EMET_OVMM_S0_PARITY` | `ovmm_find_phase.run_episode_find_phase` | Default **on** for S0 ``default_table*`` episodes: pytest-aligned oneshot path (`DynamemTaskExecutor` + live map, no scene cache, interactive dynagraph profile, phrase-only localize). Set `0`/`false` for full OVMM find-phase harness. |
| `EMET_SQA3D_OUTPUT` | `emet sqa3d run-real-sweep`, `aggregate_sqa3d_sweep.py` | Default sweep output root. Default `~/runs/emet/sqa3d` (`configs/sqa3d/benchmark.yaml`). |
| `EMET_DYNAMIC_EXPLORE_OUTPUT` | `scripts/eval_dynamic_exploration.py` | Dynamic exploration sweep output. Default `~/runs/emet/dynamic_exploration` (`configs/benchmarks/dynamic_exploration.yaml`). |
| `EMET_TAMP_CLUTTER_OUTPUT` | `scripts/eval_tamp_clutter.py` | TAMP clutter-clearance benchmark output. Default `~/runs/emet/tamp_clutter`. |
| `EMET_DYNAMIC_EXPLORE_DYNAGRAPH_TIMEOUT_S` | `dynamic_exploration_runner.py` | Override per-run ``emet run dynagraph`` subprocess timeout (seconds). Default scales with explore budget (~105 min for Robocasa K=3 on GPU). |
| `EMET_DYNAMIC_EXPLORE_HEARTBEAT_S` | `dynamic_exploration_runner.py` | Heartbeat interval while a dynagraph subprocess is running (default `120`). Writes to stderr + `progress.jsonl`. |
| `EMET_DYNAMIC_EXPLORE_STALE_LOG_S` | `dynamic_exploration_runner.py` | Warn when `dynagraph.log` mtime is older than this many seconds (default `900`). Surfaces post-VLM / EQA hangs. |
| `EMET_DYNAMIC_EXPLORE_STALE_KILL_S` | `dynamic_exploration_runner.py` | Kill the dynagraph **process group** when the log is stale this long (default `2×` `STALE_LOG_S`). Prevents leaked GPU holders after wrapper-only `kill`. |
| `EMET_VL_GENERATE_HEARTBEAT_S` | `qwen3_vl_client.py` | Print `[vl] generate heartbeat` this often while `model.generate` runs (default `30`; `0` disables). Keeps eval log mtime fresh during long vision prefill. |
| `EMET_VL_GENERATE_TIMEOUT_S` | `qwen3_vl_client.py` | Hard wall-clock cap for one VL `model.generate` (default `180`). Raises `VlGenerateTimeoutError` instead of hanging for tens of minutes (pathological SDPA / wedged CUDA). Set `0` to disable; tighten (e.g. `120`) if multi-minute hangs return. Decode uses `HardTimeStop`; prefill hangs are aborted by a worker-thread join. |
| `EMET_EQA_ANSWER_MAX_NEW_TOKENS` | `graph_memory.query_answer` | Override `eqa_vl.answer_max_new_tokens` (default `384`; `0` uses client default) — the per-call decode cap for the answer VLM. Tune per model in config rather than pinning here; `256` truncated 31/32 generations in the 2026-07-29 bal-32 run. |
| `EMET_EQA_PROMPT_MAX_TOKENS` | `graph_memory.query_answer` | Unified approximate text-token budget for HISTORY + CONFIRMED_MEMORY + SCENE_GRAPH in the EQA user prompt (default `2500`; char/4 estimator). Truncation order: oldest HISTORY → memory tail → edges → lowest-ranked node labels. `0` disables. Also `eqa_vl.eqa_prompt_max_tokens`. |
| `EMET_EQA_ANSWER_FORMAT` | `graph_memory.query_answer` / `parse_answer` | `json` or `labeled`. Default: **json** when `eqa.prompt_variant` is `hmeqa`/`mcq`, else labeled. JSON contract: `{"reasoning","answer","confidence","action","confidence_reasoning"}` with labeled-field fallback. Also `eqa.answer_format`. |
| `EMET_EQA_LOCATION_OVERRIDE_EQUIP_GATE` | `GraphEQAMemory` location MCQ | Default **on** (`eqa.location_override_equip_gate`). When on, the geometric equipment-distance guess must not override a confident VLM A–D letter (full-113 2026-08-13 scoring fix). Set `0`/`false` to restore legacy always-override for A/B. |
| `EMET_EQA_LOCATION_OVERRIDE_IMAGE_GATE` | `GraphEQAMemory` location MCQ | Default **on** (`eqa.location_override_image_gate`). When on, image-label mapping must not override a confident VLM A–D letter. Default flipped to on after the 2026-08-14 live re-run showed the image branch is the main source of `[memory-location]` overrides that clobber confident-correct VLM letters (q44/q14/q25/q41/q47). Set `0`/`false` to restore the legacy image override (corrects memory-steered VLM guesses). |
| `EMET_EQA_INCLUDE_IMAGE_DESCRIPTIONS` | `graph_memory.query_answer` | `1`/`0` — include the legacy `IMAGE_DESCRIPTIONS` text block for attached obs (labels+coords). Default **off** (`eqa_vl.include_image_descriptions: false`): RGB frames + `SCENE_GRAPH` already carry the signal; the dump mostly duplicated graph nodes and invited model `Caption:` loops. When on, labels already present on SCENE_GRAPH Image tags are omitted from the dump. Set `1` only for A/B vs the legacy prompt. |
| `EMET_EQA_AGENTIC_VERIFY` | GraphEQA / dynagraph EQA | `1`/`0` — enable unified explore/navigate/verify/answer loop (`eqa.agentic_verify`). Does **not** disable per-frame graph label VLM (that caused `n_object=0` on HM-EQA holdout q104/q105). |
| `EMET_GRAPH_EQA_EXTRACT_VLM` | `SensorGraphBuilder` | `0`/`false` — opt-out of per-frame vision-VLM graph label extract (voxel/detector labels only). Default: extract enabled. Use only if mid-nav Habitat+Qwen `libcuda` faults force it. |
| `EMET_EQA_AGENTIC_ROUTER` | `AgenticEQAExecutor` | `1`/`0` — override `eqa.agentic_vlm_router`: let the shared Qwen3-VL pick tools via JSON tool calls (`0` = deterministic fallback only, for reproducible evals). |
| `EMET_EQA_AGENTIC_DECISION_POLICY` | Grounded Graph Agent / HM-EQA | `legacy` (default) or `grounded_v2`. Explicit implementation gate; `paper-router` never changes it. Also `eqa.agentic_decision_policy`. |
| `EMET_EQA_GRAPH_EVIDENCE_MODE` | Grounded Graph Agent / HM-EQA | `off` (default), `shadow`, or `agent`. Shadow collects stable evidence without policy visibility; agent makes it visible after that rollout phase lands. Also `eqa.graph_evidence_mode`. |
| `EMET_EQA_ROOM_HISTORY_MODE` | Grounded Graph Agent / HM-EQA | `off` (default), `shadow`, or `agent`. Separates room-history collection from policy visibility. Also `eqa.room_history_mode`. |
| `EMET_EQA_ROOM_TARGET_HINTS` | Grounded Graph Agent / HM-EQA | `1`/`0` — expose question-derived target-room hints. Default `1` preserves the legacy canonical-room state. Also `eqa.room_target_hints`. |
| `EMET_EQA_ROUTER_ROOM_IMAGES` | `AgenticEQAExecutor` | Nearby object RGBs attached to each router turn for `current_room` (default `3`; `0` = text-only router for speed A/B). Live robot view is always included when images &gt; 0. Trace logs `router_ms` / `n_room_images`; `emet jobs report -q ID --rooms` prints mean/max. |
| `EMET_EQA_ROOM_POLICY` | `AgenticEQAExecutor` | `canonical` (default) or `llm` — how room identity is stored for agentic explore. Canonical buckets via `normalize_current_room` / `question_target_rooms` (metrics + leave hint). LLM keeps free-text `current_room` + router `in_target_area`. Frontier picks re-ask the real Question with graph `room=`/`near=` context — no frozen MCQ-derived place brief. Also `eqa.room_policy`. |
| `EMET_EQA_ROOM_STAMP_INVESTIGATE` | `AgenticEQAExecutor` | Default **off**. When `1`, stamp room clusters after investigate from local obs labels (feeds graph room timeline / agent state). **Keep off for paper-router of-record** — stamps regressed HM-EQA letter accuracy vs explore-streak-only. Opt-in only for graph-room-evidence A/Bs ([experiments/graph_room_evidence.md](experiments/graph_room_evidence.md)). Also `eqa.room_stamp_investigate`. |
| `EMET_EQA_MERGED_MEMORY` | `GraphEQAMemory.query_answer` | Default **on**: fold CONFIRMED_MEMORY into `SCENE_GRAPH` (graph-grounded `present` tags + short tail for candidates/unobserved) instead of a separate summary block. Set `0` to restore the standalone summary block (the HM-EQA paper row pins this off via `configs/benchmarks/dynagraph.yaml`). SigLIP alone never asserts present/absent. Also `eqa.merged_memory`. |
| `EMET_EQA_ATTEMPT_LEDGER_MODE` | Grounded Graph Agent / HM-EQA | `off` (default), `shadow`, or `agent`. `off` writes no durable rows and exposes no dedicated action history; `shadow` collects the ledger/artifact while hiding it from `grounded_v2`; `agent` also exposes recent outcomes, loop flags, per-place attempt summaries, global rows, and mirrored attempt events. H2H maps non-off modes to the legacy write gate. Also `eqa.attempt_ledger_mode`; full semantics and checked-in variant configs: [attempt_ledger.md](attempt_ledger.md). |
| `EMET_EQA_ACTION_PROGRESS_MODE` | Grounded Graph Agent / HM-EQA | `off` (default), `shadow`, or `enforce`. Independent from ledger visibility. `shadow` records counterfactual suppression decisions without changing cards or execution; `enforce` omits an equivalent terminal/no-progress action while its target-local progress token is unchanged. This first policy is scoped to mostly static HM-EQA scenes, not lifelong dynamic-world memory. Also `eqa.action_progress_mode`; see [attempt_ledger.md](attempt_ledger.md#static-world-action-progress-policy). |
| `EMET_EQA_ATTEMPT_LEDGER` | `GraphEQAMemory.record_attempt` | Legacy boolean write gate, default **off**. When `1`, append structured `AttemptRecord` rows (navigate / verify / pick / place / closer_look / …) in graph memory in addition to per-node `nav_attempts` / `nav_failures`. `emet hmeqa` derives it from `EMET_EQA_ATTEMPT_LEDGER_MODE`. Also `eqa.attempt_ledger` (`true` or `{enabled: true, max_records, persist_absent_claims}`). Full reference: [attempt_ledger.md](attempt_ledger.md). |
| `EMET_EQA_AGENTIC_MAX_TOOL_ROUNDS` | HM-EQA frozen manifest | Effective agentic tool-round budget recorded for the run (default `8`; also `eqa.agentic_max_tool_rounds`). |
| `EMET_EQA_AGENTIC_MAX_NAV_STEPS` | HM-EQA frozen manifest | Effective agentic navigation budget recorded for the run (default `8`; also `eqa.agentic_max_nav_steps`). |
| `EMET_HMEQA_VARIANT_ID` | `emet hmeqa` / H2H script | Stable A/B label (default `legacy`) written to `run_manifest.json` and inherited by episodes. Also `eqa.variant_id`. |
| `EMET_HMEQA_USE_HM3D_SEMANTICS` | `emet hmeqa` / H2H script | Frozen GT-oracle axis: use HM3D semantic sensor labels (`1`) or GT-free perception (`0`, default). CLI: `--use-hm3d-semantics` / `--no-hm3d-semantics`. |
| `EMET_HMEQA_USE_ENRICH_LABELS` | `emet hmeqa` / H2H script | Frozen, independent GT-oracle axis: seed per-question GraphEQA object hints (`1`) or not (`0`, default). CLI: `--enrich-labels` / `--no-enrich-labels`. |
| `EMET_HMEQA_CONFIG_DIGEST` | `emet hmeqa` / H2H script | Internal deterministic SHA-256 of frozen axes, model, budgets, arms, question IDs, and artifact profile. Resume validates it; do not hand-edit. |
| `EMET_HMEQA_MANIFEST_PREPARED` | `emet hmeqa` → H2H script | Internal CLI-to-script handoff marker. The script validates the manifest already created or checked by the CLI without changing episode resume semantics, then unsets the marker. |
| `EMET_HMEQA_RUN_CONFIG_JSON` | `emet hmeqa` → H2H script | Internal canonical frozen-config handoff to the managed job. The script unsets it after validating/writing `run_manifest.json`. |
| `EMET_HMEQA_RUN_SOURCES_JSON` | `emet hmeqa` → H2H script | Internal map of effective config paths to their source (`command_line`, config/default, or preset). Stored in the first manifest, then unset. |
| `EMET_HMEQA_ENV_SANITIZED` | H2H internal launcher | Internal recursion guard set only after rebuilding the H2H child environment from the frozen allowlist. Do not set it manually; doing so bypasses the direct-script sanitizing re-exec. |
| `EMET_ATTEMPT_LEDGER_PERSIST_ABSENT` | `GraphEQAMemory.clear_retracted_nav_claims` | When `1`, keep close-look ABSENT claim blacklists across questions (ledger rows always persist when the ledger is on). Default **off**. Also `eqa.attempt_ledger.persist_absent_claims`. |
| `EMET_ATTEMPT_LEDGER_MAX` | `GraphEQAMemory` | Cap on stored `AttemptRecord` rows (default `512`). Also `eqa.attempt_ledger.max_records`. |
| `EMET_EQA_AGENTIC_MCQ_DEBIAS` | `AgenticEQAExecutor` | Default **on**. Unverified forced answers (budget exhaustion) run the letter-free debias (`vote_mcq_letter`: freeform + ≤2 rotation votes) before the ladder, fixing the last-option (D) bias seen in the trace audit. `0` restores the raw EQA letter. Also `eqa.agentic_mcq_debias`. |
| `EMET_EQA_AGENTIC_CLOSE_LOOK` | `AgenticEQAExecutor` | Default **on**. Per-episode close-look flag: VLM `extract_target_from_question` **OR** count/clock/state keywords (so a VLM false-negative cannot disable stay on “how many” / “what time”). Router state + `DETECTIONS_REMAIN` / close-map stay. `0` disables. Also `eqa.agentic_close_look`. Tune with other close-map knobs one at a time — [countclock_bisect.md](experiments/countclock_bisect.md#tuning-ladder-one-knob-at-a-time). |
| `EMET_EQA_LOCATION_MISSING_FIND` | `GraphEQAMemory.query_answer` | Default **on** (`eqa.location_missing_find`; env escape hatch). Location MCQs downgrade while an unattached FIND view (`_eqa_find_obs_ids` `count_mcq_only=False`) remains (mirror count, q47 `wall clock`). `0` disables (6/15 vs 4/5 canary). |
| `EMET_EQA_CLOSE_MAP_GATE` | `GraphEQAMemory.query_answer` | Default **on** (`eqa.close_map_gate`; env escape hatch). Location MCQs also require voxel `close_map resolved` (`close_map_catalog_fields` `R_M 0.55` `aimed`) before confident (AB `7/15` vs A `6/15`). `0` disables. |
| `EMET_EQA_IMG_STRICT` | `GraphEQAMemory.query_answer` | Default **off** (`eqa.img_strict`; env escape hatch). Require attached image landmark (`_location_letter_from_attached_images`) to match parsed letter before location confident (`5/15`). `1` enables. |
| `EMET_CLOSE_MAP_R_M` | `CloseDistanceMap` | Aimed camera range (m) that counts as a resolved close look (default `0.55`). Occupancy exploration is not enough for small objects. See [close_map.md](close_map.md). |
| `EMET_CLOSE_MAP_AIM_DEG` | `CloseDistanceMap` | Optical-axis cone (deg) for an “aimed” hit (default `25`). |
| `EMET_CLOSE_MAP_QUERY_RADIUS_M` | `CloseDistanceMap` | XY neighborhood radius (m) when querying a place card (default `0.35`). |
| `EMET_CLOSE_MAP_ESCAPE_ATTEMPTS` | `decide_close_look` | Agentic find: max approaches on an unresolved XY before escape (default `4`). |
| `EMET_CLOSE_MAP_CHAT_ESCAPE_ATTEMPTS` | `decide_close_look` | CHAT/TAMP tighter cap so the robot leaves unreachable furniture (default `2`). |
| `EMET_EQA_AGENTIC_NO_EARLY_UNVERIFIED` | `AgenticEQAExecutor` | Default **on**. Unverified auto-submits (bare `answerable` state) are held while rounds/nav budget remain — the harness forces the debiased ladder answer at exhaustion (q2 early-abandon fix). `0` restores early unverified submits. Also `eqa.agentic_no_early_unverified`. |
| `EMET_EQA_AGENTIC_SINGLE_VIEW_CONFIRM` | `AgenticEQAExecutor` | Default **on**. A single VLM assess that saw the target (`present=true`) and offered a letter confirms `verified` — no phrase-token or two-view corroboration needed (verified answers score ~86% vs ~35% forced guesses; raised the verification rate from ~13% of episodes). The `present` guard from the q28/q39 absence fix stays. `0` restores the phrase/two-view gate. Also `eqa.agentic_single_view_confirm`. |
| `EMET_EQA_AGENTIC_EVIDENCE_IMAGE` | `AgenticEQAExecutor` | Default **on**. Unverified final EQA answers pin the best VLM-assessed view (answerable+present) as Image 1 via `force_obs_ids` instead of a pure diversified pick. `0` restores diversified selection. Also `eqa.agentic_evidence_image`. |
| `EMET_EQA_HYP_RECALL_K` | `AgenticEQAExecutor` | Top-K evidence cards recalled for the agentic router / fallback (default `6`). Retrieval only — the VLM decides among listed `obs_id`s. |
| `EMET_EQA_ROOM_LINK_RADIUS_M` | `GraphEQAMemory` room clusters | Planar radius (m) linking object nodes into room CCs with `near` edges for the **proximity** backend (default `2.0`; also `eqa.room_link_radius_m`). |
| `EMET_EQA_ROOM_CLUSTERING_BACKEND` | `room_clustering.partition` | `proximity` (default, implemented), `occupancy_cc`, or `portal` (latter two error until implemented). Also `eqa.room_clustering.backend`. |
| `EMET_EQA_ROOM_ASSIGN_MAX_M` | `GraphEQAMemory.graph_room_at_robot` | Max distance (m) from robot XY to a cluster centroid to assign a room (default `3.0`; also `eqa.room_assign_max_m`). |
| `EMET_EQA_AGENTIC_REQUIRE_VERIFIED` | `AgenticEQAExecutor` | `1` — refuse unverified `submit_answer` (incl. fallback / round exhaust). At exhaustion the episode falls to the forced-answer ladder (see `EMET_EQA_FORCE_ANSWER`), not to a silent `Unknown`. SigLIP is high-recall FP-leaning support for verify — see [agentic_scale.md](experiments/agentic_scale.md#siglip-role-in-agentic-verify-design). |
| `EMET_EQA_FORCE_ANSWER` | `AgenticEQAExecutor` | Default **on**. At budget / verification exhaustion, run the four-image EQA and commit to best-guess semantic option text instead of returning `Unknown`. The ladder is EQA `Answer:` text (`eqa_answer`) → a view assess that *saw* the target (`vlm_suggested`) → deferred pending choice (`pending_letter`, retained as a metrics tag) → deterministic uniform prior (`uniform_prior`), recorded as `answer_provenance` with a calibrated `answer_confidence` in the episode metrics and the `forced_answer` trace row. Only the Habitat scoring adapter converts the resolved choice index to A–D. H2H summarize / `emet jobs report` print a per-channel breakdown and `accuracy_excl_uniform_prior` so a lucky prior cannot inflate the headline. `0`/`false` restores the legacy abstain (trace row `abstain_unverified`) for A/B. |
| `EMET_EQA_AGENTIC_VERIFIER` | `AgenticEQAExecutor` | Hybrid where-next backend for ranking places: `none`/`siglip` (paper-router / overnight default), `owlv2`, or `yoloe`. Answerability is Qwen `vlm_assess` on pixels (detector ABSENT/PRESENT is not fed into that prompt and does not unlock submit). |
| `EMET_EQA_ANSWERABLE_CONFIRM` | `AgenticEQAExecutor` | Hybrid confirm before submit unlock (default **on**). Raw VLM `answerable` unlocks only after phrase/inventory corroboration **or** a second agreeing semantic choice on another obs. `0`/`false` restores legacy immediate unlock. `need_more_views` always defers. Trace: `answerable_deferred` / `answerable_confirmed`. |
| `EMET_EQA_TRACE` | `AgenticEQAExecutor` | `1` — append `agentic_trace.jsonl` (SigLIP embeds + GT) for offline tuning via `scripts/tune_agentic_verify.py`. |
| `EMET_EQA_EPISODE_DIR` | Agentic EQA / OVMM find | Episode artifact directory. OVMM batch sets this to ``OUT/<episode>_<backend>/`` so traces and query PNGs land next to the run JSON. |
| `EMET_AGENTIC_QUERY_IMAGES` | `query_images.dump_query_rgb` | Default **on**. Write the RGB actually sent to Qwen (`vlm_assess` / `capture`) as PNGs. `0`/`false` disables. |
| `EMET_AGENTIC_QUERY_IMAGES_DIR` | `query_images.dump_query_rgb` | Override dump directory. Default: ``$EMET_EQA_EPISODE_DIR/images`` (or next to the agentic trace). Files: ``rgb_<obs>.png`` plus ``<kind>_r<round>_obs<id>.png``. |
| `EMET_EQA_QUESTION_TIMEOUT_S` | `controller_graph_eqa.run_eqa` | Per-question wall-clock cap for GraphEQA planning/nav loops (default `900`; `0` disables). |
| `EMET_SCENE_MAP_CACHE_DIR` | `scene_map_cache.py` | Root for prebuilt scene maps (graph + voxel). Default `~/.cache/emet/scene_maps`. |
| `EMET_USE_SCENE_MAP_CACHE` | OVMM find / dynamic explore | Load cached baseline and skip rotate/explore when present (default `1`). Set `0` or pass `--no-scene-cache`. |
| `EMET_SCENE_MAP_HF_REPO` | `scene_map_cache_hub.py` | HuggingFace dataset repo id for push/pull (e.g. `org/emet-scene-maps`). |
| `EMET_SCENE_MAP_CACHE_AUTO_PULL` | `ensure_cached_map` | On local miss, try HF pull when repo is set (default `1`). |
| `EMET_VLM_DEVICE_CHECK_MAX_PARAMS` | `vlm_device.assert_cuda_placement` | When `hf_device_map` is missing, cap parameter/buffer device sampling (default `512`). Avoids multi-hour stalls after int4 load. |
| `SQA3D_DATA_DIR` | `emet sqa3d`, `emet eval-sqa3d` | Root for SQA3D Zenodo JSON (`sqa_task/`, optional `localization_task/`). Default `~/.cache/sqa3d/data`. See [sqa3d.md](sqa3d.md). |
| `SCANNET_ROOT` | `emet sqa3d` embodied runs | ScanNet v2 download root (`scans/<scene_id>/…`). Default `~/.cache/scannet`. |
| `SCANNET_DOWNLOAD_SCRIPT` | `scripts/download_scannet_data.py` | Override path to `download-scannet.py`. |
| `SQA3D_MIN_FREE_MIB` | `scripts/run_sqa3d_gpu_sweep.sh` | Minimum free GPU MiB before starting a sweep (default `14000`). See [sqa3d_compute.md](sqa3d_compute.md). |
| `SQA3D_SCANNET_TEST_SCENE` | pytest `test_scannet_embodied_smoke` | Scene id for integration smoke (default `scene0380_00`). |
| `RUN_SQA3D_SCANNET_TESTS` | pytest | Set `1` to run embodied ScanNet integration smoke when mesh may be missing. |
| `EMET_EQA_VL_MODEL_SIZE` | EQA / GraphEQA VLM load | Override `eqa_vl.model_size` in `dynav_config.yaml` (legacy Qwen3.5 path; default EQA uses `eqa.vl_hf_model_id` **Qwen/Qwen3-VL-8B-Instruct** int4). |
| `EMET_SIGLIP_VERSION` | `get_shared_mask_siglip_encoder` (voxel map / dynagraph grounding) | Override the SigLIP checkpoint: `base`, `so400m` (default), `siglip2_base`, `siglip2_so400m`. A/B encoder upgrades without config edits. |
| `EMET_SIGLIP_DTYPE` | `SiglipEncoder` / `MaskSiglipEncoder` weight load | `float32` (default), `float16`, or `bfloat16`. Halves SigLIP VRAM (so400m: 3.5 GB → 1.75 GB); outputs are cast back to fp32 so stored features/thresholds are unchanged. int4/int8 unsupported (breaks the MaskSiglip head surgery). |
| `EMET_VLM_FRONTIER_SCORING` | Dynagraph classic EQA (`controller_graph_eqa.py` coverage path) | Set `1` to let the EQA VLM pick the exploration frontier from reachable candidate views (≤6 images/iteration, utility-ranked) before the SigLIP-nearest heuristic in `run_eqa_one_iter`. **Agentic** `explore_frontier` always tries `_vlm_frontier_choice` when the method exists (no env gate). Baseline `graph_eqa` coverage path still defaults off. |
| `EMET_DYNAGRAPH_MCQ_DEBIAS` | Habitat `emet-habitat` dynagraph harness | `1` / `0` override `mcq_debias` after harness profile (CLI: `--mcq-debias` / `--no-mcq-debias`). |
| `EMET_DYNAGRAPH_MEMORY_SUMMARY` | Habitat `emet-habitat` dynagraph harness | `1` / `0` override CONFIRMED_MEMORY block (CLI: `--memory-summary` / `--no-memory-summary`). |
| `EMET_DYNAGRAPH_EXPLORE_UNCOVERED` | Habitat `emet-habitat` dynagraph harness | `off`, `on`, or `conservative` (CLI: `--explore-when-uncovered`). Default per harness: `habitat_eqa` uses `conservative` in [`configs/benchmarks/dynagraph.yaml`](../configs/benchmarks/dynagraph.yaml). |
| `PYTORCH_CUDA_ALLOC_CONF` | PyTorch CUDA | Optional allocator hint (e.g. `expandable_segments:True`); set by `run_sqa3d_gpu_sweep.sh` and `run_sqa3d_sharded_sweep.sh` if unset. |
| `NEED_MIB` | `emet eval check/wait`, `scripts/gpu_preflight.sh` | Min free VRAM (MiB) (default `12000`). |
| `GPU_STABLE_CHECKS` | `emet eval wait` | Consecutive passing free-VRAM reads (default `3`). |
| `GPU_WAIT_INTERVAL` | `emet eval wait` | Seconds between reads (default `30`). |
| `GPU_WAIT_MAX_ROUNDS` | `emet eval wait`, `emet eval recover`, `emet jobs run --gpu-wait-max-rounds` | Maximum failed polling rounds (default `120`); timeout returns failure instead of waiting forever. CLI: `--max-rounds` / `--gpu-wait-max-rounds`. |
| `GPU_SETTLE_SEC` | `emet eval kill-stale` | Sleep after pattern kills (default `15`). |
| `GPU_KILL_STALE` | `emet_gpu_between_steps` | Set `0` to skip process cleanup between overnight phases. |
| `EMET_GPU_PROTECT_PIDS` | `emet eval kill-stale` | Space-separated PIDs never killed (plus the caller process and its ancestors). |
| `EMET_JOBS_DIR` | `emet jobs` | Directory for job registry JSON (default `~/runs/emet/jobs`). |
| `EMET_GPU_LOCK` | `emet jobs run --gpu-exclusive`, direct H2H script | Canonical host-wide `flock` path held for the full exclusive job lifetime (default `~/runs/emet/gpu.lock`; shared with v2/v3 launchers). |
| `EMET_GPU_LOCK_FILE` | `emet jobs run --gpu-exclusive`, direct H2H script | Compatibility alias for `EMET_GPU_LOCK` when the canonical variable is unset. |
| `EMET_GPU_LOCK_FD` | Nested managed H2H internal launcher | Informational descriptor number exported only after FD 9 is verified against the canonical lock inode and lock ownership. Ambient values are not trusted; do not set this manually. |
| `EMET_GPU_LOCK_TIMEOUT` | `emet jobs run --gpu-exclusive`, direct H2H script | Lock wait timeout in seconds (default `21600`, six hours); a timed-out job fails without starting its command. CLI: `--lock-timeout-sec`. |
| `EMET_JOB_ID` | smoke/queue scripts | Set by `emet jobs run`; nested H2H preserves it only when the registry record is live and its PID is the current process or an ancestor. Scripts heartbeat the validated job. Also write `OUT/progress.json` for ETA without a job id. |
| `EGL_FAIL_ABORT` | `scripts/run_hmeqa_agentic_h2h.sh` / `emet hmeqa` | Abort after N consecutive Habitat EGL/CUDA-map failures (`WindowlessContext` / `unable to find CUDA device`). Default `2`; `0` = never. |
| `NATIVE_CRASH_POLICY` | H2H / `emet hmeqa --crash-policy` | `skip` (default): settle + retry + continue; `abort`: stop batch on first native crash. |
| `NATIVE_CRASH_ABORT` | H2H (deprecated) | `1` → same as `NATIVE_CRASH_POLICY=abort`. |
| `NATIVE_CRASH_RETRIES` | H2H | Retries of the same qid after a native crash under `skip` (default `1`). |
| `NATIVE_CRASH_SETTLE_SEC` | H2H | Sleep after native crash before retry/next (default `60`). |
| `NATIVE_CRASH_STREAK_ABORT` | H2H / `emet hmeqa --streak-abort` | Under `skip`, abort after N **consecutive** native crashes (default `2`; early exit when harness is wedged). `0` = never. |
| `EMET_SKIP_CPU_AFFINITY` | H2H / `emet eval affinity` / `python -m emet.utils.cpu_affinity` / `emet jobs --cpu-safe` | Set `1` to skip pinning away from turbo P-cores (default: affinity **on** / fail-closed). |
| `EMET_EXCLUDE_CPU_MIN_MHZ` | `emet.utils.cpu_affinity` / `emet eval affinity` | Exclude logical CPUs whose sysfs `cpuinfo_max_freq` is ≥ this many MHz (default `6000`). On the i9-14900KF that removes CPUs **8–11**. Do **not** use `taskset -c 0-7,10-31`. |
| `EMET_CPU_EXCLUDE` / `EMET_CPU_ALLOW` | `emet.utils.cpu_affinity` | Optional extra exclude list, or explicit allow list (csv / ranges). |
| `EPISODE_COOLDOWN_SEC` | H2H / `emet hmeqa --cooldown` | Seconds to `sync` + sleep after each episode (default `20`; `0` disables). |
| `EPISODE_GPU_WAIT` | H2H | Re-run `gpu_preflight --wait` before each episode (default `1`). |
| `EMET_HMEQA_OUT` | `emet hmeqa resume\|status` | Override OUT resolution when no path argument is given. |

### Large paper eval orchestrator

Used by `scripts/run_large_paper_eval.sh` and `scripts/run_sqa3d_sharded_sweep.sh`. See [paper_benchmarks.md](paper_benchmarks.md) (Large paper eval queue).

| Variable | Where used | Notes |
|----------|------------|-------|
| `EMET_LARGE_EVAL_LOG_DIR` | `run_large_paper_eval.sh` | Master + per-phase logs. Default `~/runs/emet/large_eval`. |
| `EMET_SQA3D_OUTPUT` | SQA3D sweeps / sharded sweep | Same as benchmarks table above. |
| `SQA3D_GPUS` | `run_large_paper_eval.sh` | Comma-separated CUDA ids (e.g. `0,1,2,3`) → `run_sqa3d_sharded_sweep.sh` (~linear speedup). |
| `SQA3D_NO_ISOLATE` | `run_large_paper_eval.sh` | Set `1` for in-process batch (`--no-isolate-episodes`; faster, OOM risk). |
| `SQA3D_METHODS` | `run_large_paper_eval.sh` | Space-separated methods (default `dynagraph dynamem`). e.g. `SQA3D_METHODS=dynagraph`. |
| `SKIP_SQA3D_TEST` | `run_large_paper_eval.sh` | Set `1` to skip test-split SQA3D sweeps (val only). |
| `SKIP_OVMM` | `run_large_paper_eval.sh` | Set `1` to skip OVMM find replicates. |
| `SKIP_DYNAMIC_EXPLORE` | `run_large_paper_eval.sh` | Set `1` to skip dynamic exploration matrix. |
| `OVMM_CPU_ONLY` | `run_large_paper_eval.sh` | Set `1` for `--cpu-only` OVMM (overlap with SQA3D on GPU in another terminal). |
| `DYNAMIC_EXPLORE_CPU_ONLY` | `run_large_paper_eval.sh` | Set `1` for `--cpu-only` dynamic exploration runs. |
| `EMET_STATUS_LOG` | `scripts/status_log.sh` (sourced by orchestrators) | Per-checkout tail-able status log path. Default `~/runs/emet/status/<repo_basename>/STATUS.log` (not a flat shared file — sibling trees like `home_robot_v3` / `v4` must not interleave). Each run also writes `OUT/STATUS.log`. Recovery: `uv run emet status tail`. See [evaluation.md](evaluation.md#first-command-after-an-agent-death-uv-run-emet-status-tail). |
| `EMET_STATUS_DIR` | `scripts/status_log.sh` | Directory holding `STATUS.log` + `latest` symlink. Default `~/runs/emet/status/<repo_basename>`. Ignored when `EMET_STATUS_LOG` is set. |

## ZMQ and simulation (general)

| Variable | Where used | Notes |
|----------|------------|-------|
| `EMET_ZMQ_STARTUP_TIMEOUT` | ZMQ clients, `emet run molmospaces-explore` | Seconds to wait for first observation (default 60). Documented in [molmospaces_environment_variables.md](molmospaces_environment_variables.md). |
| `EMET_ZMQ_TIMING` | `BaseZmqServer` | `1` — print periodic SEND/RECV timing lines. Default off (also enabled by server `--verbose`). |
| `EMET_ZMQ_FULL_HZ` | `BaseZmqServer` | Optional maximum full RGB-D observation publish rate. Unset keeps the legacy unthrottled loop; OVMM eval subprocesses default to 5 Hz. |
| `EMET_ZMQ_STATE_HZ` | `BaseZmqServer` | Optional maximum low-level state publish rate. Unset keeps the legacy unthrottled loop; OVMM eval subprocesses default to 30 Hz. |
| `EMET_ZMQ_SERVO_HZ` | `BaseZmqServer` | Optional maximum visual-servo stream publish rate. Unset keeps the legacy unthrottled loop; OVMM eval subprocesses default to 10 Hz. |
| `EMET_NAVGRID_ASCII` | Dynamem / Dynagraph mapping | Terminal nav grid; see [dynagraph.md](dynagraph.md). |
| `EMET_NAVGRID_MAX_SIDE` | Nav grid ASCII | Default 320. |
| `EMET_NAVGRID_CONTEXTS` | Nav grid ASCII | Limit which hooks print. |
| `EMET_ROBOSUITE_POST_LOAD_DEBUG` | `robosuite_load_utils` | Post-load velocity / contact diagnostics after `RobosuiteZmqServer` startup. Set `1`/`true`/`yes`/`on`. |
| `EMET_MUJOCO_CTRL_DEBUG` | `robosuite_server` | Log stationary `ctrl` apply cycles (first N steps, then periodic). Set `1`/`true`/`yes`/`on`. |
| `EMET_MUJOCO_CTRL_DEBUG_VERBOSE` | `robosuite_server` | Full per-actuator `ctrl` lines (use with `EMET_MUJOCO_CTRL_DEBUG=1`). |
| `EMET_SIM_FALL_TILT_DEG` | `fall_detection` | Max base tilt from upright (degrees) before a red **SIM ROBOT FALLEN OVER** error. Default `55`. |
| `EMET_ROBOSUITE_AUTOPLACE` | `scene_base_spawn` | Planar base autoplace on Robocasa / default-table merges (default `1`). `0`/`false`/`no`/`off` disables. |
| `MUJOCO_GL` | MuJoCo rendering | e.g. `egl` for headless GPU cameras on Linux. RobosuiteZmqServer sets this automatically unless `--use-glx`. |
| `EMET_MARS_ONBOARD_DA3` | Innate Mars bridge | Set to `1` on the Jetson when `emet mars start --onboard-da3` runs DA3 stereo onboard (see [innate_mars_hardware.md](robots/innate_mars_hardware.md)). |
| `EMET_MARS_DA3_SPECKLE_OPEN_KERNEL` | Onboard Jetson DA3 | Morphological depth speckle filter kernel (pixels); `0` = off (default). Same helper as dynav `filters.depth_speckle_open_kernel`. |
| `EMET_MARS_DA3_SPECKLE_OPEN_ITERATIONS` | Onboard Jetson DA3 | Speckle opening repeat count (default `1`). Ignored when kernel is `0`. |
| `EMET_STREAM_VERBOSE` | `emet capture` / `emet stream` | `1`/`true` → per-step status + DA3 INFO timing (same as `--verbose`). Default off; dynav `stream.verbose` also applies. |
| `DA3_LOG_LEVEL` | Depth Anything 3 (`depth_anything_3`) | `WARN` (default during stream when not verbose), `INFO`, `ERROR`. Dynav `stream.da3_log_level` sets default via `configure_mapping_session_logging`. |
| `EMET_ROBOT_PASSWORD` | Deploy / Mars SSH | Optional password when not stored in the connection profile. |

See also [simulation_modules.md](simulation_modules.md) for maintainer-oriented module notes.

## Agent (`emet run agent`)

| Variable | Where used | Notes |
|----------|------------|-------|
| `EMET_MANIP_MODE` | `DynamemTaskExecutor` / `resolve_agent_manip_mode` | `teleport` (default) or `kinematic` (IK + attach). Overrides `agent.manip_mode`. See [molmospaces.md](molmospaces.md) mobile manipulation. |
| `EMET_MANIP_COLLISION` | kinematic pick/place | `none` (default) or `voxel` (2D obstacle map). Overrides `agent.manip_collision`. |
| `EMET_MANIP_PLANNER` | kinematic pick/place | `rrt_connect` (default), `rrt`, or `linear`. Joint-space path after IK. Overrides `agent.manip_planner`. |
| `EMET_SIM_THIRD_PERSON` | `robosuite_server` full obs | When `1`, render a **base chase** view into `third_person_image` (extra EGL cost). Used by `--record-mp4` smokes. |
| `EMET_SIM_THIRD_PERSON_CAMERA` | same | Optional body name to follow (default: robot `base_link`). |
| `EMET_SIM_THIRD_PERSON_DISTANCE` | chase cam | Orbit distance in meters (default `5.5`). |
| `EMET_SIM_THIRD_PERSON_AZIMUTH` | chase cam | Azimuth **offset** from behind the base `+X` axis, degrees (default `125` = side/rear iso). |
| `EMET_SIM_THIRD_PERSON_ELEVATION` | chase cam | Orbit elevation degrees (default `-28`). |
| `EMET_SIM_THIRD_PERSON_LOOKAT_Z` | chase cam | Lookat height above base origin, meters (default `0.75`). |
| `EMET_VL_CACHE_SYSTEM_PREFIX` | `qwen3-vl-eqa` / `Qwen3VLClient` | `1`/`0` — cache system-prompt KV across agent turns (default on via `eqa.vl_cache_system_prefix`). CLI: `--cache-vl-prefix` / `--no-cache-vl-prefix`. |
| `EMET_ALLOW_CPU_VLM` | Qwen3-VL / Gemma VLM / Qwen2.5-VL load | `1` — allow silent CPU bf16 fallback when GPU int4 load fails. **Default off**: agent refuses CPU fallback (multi-minute “Thinking…” hangs). |
| `EMET_HF_LOCAL_ONLY` | VL / SigLIP `from_pretrained` | `1` — require local HF cache only (same idea as `HF_HUB_OFFLINE=1`). Warm cache is preferred automatically even when unset. |
| `EMET_VL_PREFIX_GENERATE_TIMEOUT_S` | `Qwen3VLClient` | Soft warn + disable prefix cache after a slow prefix-KV generate (default `45`). |
| `EMET_ATTN_EAGER` | `Qwen3VLClient` | `1` — force eager attention (debug). |
| `EMET_REQUIRE_FLASH_ATTN` | `attn_impl.resolve_attn_implementation` | `1`/`0` — require Flash-Attn 2 on CUDA VL loads. **Default on** when unset (fail loud instead of silent SDPA). |
| `EMET_ALLOW_SDPA_ATTN` | same | `1` — permit PyTorch SDPA when flash-attn is missing (slower; long multi-image EQA can crawl). Overrides the default require-flash policy. |
| `EMET_FORCE_TEGRA` | `emet.utils.platform_info` | `1` — treat host as Jetson/Tegra for install hints (tests / CI). |
| `EMET_INSTALL_PROFILE` | `install.sh` | `minimal` \| `standard` \| `full` \| `jetson`. `jetson` = lean Orin install (see [jetson.md](jetson.md)). |
| `EMET_OPENAI_BASE_URL` | `OpenaiClient` / `--llm openai` | OpenAI-compatible API root including `/v1` (e.g. `http://HOST:8000/v1`). See [llm_serve.md](llm_serve.md). |
| `OPENAI_BASE_URL` | same | Fallback if `EMET_OPENAI_BASE_URL` unset. |
| `EMET_OPENAI_MODEL` | `get_llm_client("openai")` | Model id sent to the remote server (default `gpt-4o` when unset). |
| `EMET_VL_ENDPOINT` | `OpenaiVLLMClient` / `create_dynamem_vllm` | Override `eqa.vl_endpoint` (unified-7b: `openai@http://HOST:8000/v1`; dual-2b uses `:8001`). Caption/EQA only; voxels stay local. |
| `EMET_LLM_HOST` | `emet run chat --host`, `emet run agent --host`, `emet llm …`, `emet deploy llm` | LAN OpenAI host (no default). Example: your Orin hostname. Sets text+VL endpoints via `apply_llm_host`. |
| `EMET_CALIBAN_HOST` | same (compat alias) | Fallback if `EMET_LLM_HOST` unset. |
| `EMET_CALIBAN_REPO` | `emet deploy llm` / `deploy_caliban_vl.sh` | Remote checkout with `docker/jetson_llm_server.py` (default `~/src/home_robot_v3`). |
| `EMET_JETSON_LLM_IMAGE` | Jetson LLM runner | Docker image tag (default `emet-jetson-llm:r35.4.1`). |
| `EMET_JETSON_LLM_NAME` | `run_jetson_llm_container.sh` | Docker container name (default `emet-jetson-llm`; use a second name for dual-port). |
| `EMET_LLM_SERVE_PORT` | Jetson runner / serve | Host port for the container (default `8000`; dual-2b VL uses `8001`). |
| `EMET_LLM_SERVE_QUANT` | `jetson_llm_server.py` | `fp16` (only supported on JP5 Tegra image). `awq`/`int4`/`int8`/`bnb` exit with a clear error — see [llm_serve.md](llm_serve.md) § Quantization. |
| `EMET_LLM_SERVE_API_KEY` | `emet serve llm` + client | Optional Bearer token for the LAN LLM server. |
| `EMET_LLM_SERVE_DEVICE` | `emet serve llm` | Default device when `--device` omitted (`auto` / `cuda` / `cpu`). |
| `EMET_GOMP_PRELOAD_DONE` | `openai_server` | Set after aarch64 libgomp re-exec (internal). |
| `EMET_SKIP_GOMP_PRELOAD` | `openai_server` | `1` — skip Jetson libgomp LD_PRELOAD workaround. |
| `eqa.vl_image_max_side` | VL clients / `describe_scene` | Longest RGB edge before VL/detector (default `512`; `0` = no resize). Override: `--set eqa.vl_image_max_side=384`. |
| `eqa.vl_image_max_pixels` | Same | Optional `H*W` cap after side resize (`0` = off). |
| `EMET_AGENT_THINKING_STATUS` | Agent loop | `1`/`0` — emit `*Thinking…*` status lines (default on). CLI: `--thinking-status` / `--no-thinking-status`. Heartbeats every ~8s while the LLM is still running. |
| `EMET_AGENT_MODEL_DEBUG` | Agent / VL clients | `1` — model stack + prefix-cache hit logs + generate phase timings. CLI: `--debug-models`. |
| `EMET_AGENT_TOOL_DEBUG` | Agent loop | `1` — verbose tool I/O. CLI: `--debug-tools`. |
| `EMET_AGENT_CAMERA_DEBUG` | Agent tools | `1` — head-camera frame stats. CLI: `--debug-camera`. |
| `EMET_BASE_ROTATE_ONLY` | ZMQ client + agent tools + Dynamem executor | `1` — **yaw-only / in-place scan** (no XY drive). Blocks `explore` / `move_forward` / `find` / absolute nav; keeps `rotate_base`, `scan_environment`, `describe_scene`. Use when Mars is plugged in / tethered. |
| `EMET_AGENT_MOTION_STATUS` | Controllers | `1`/`0` — fine-grained terminal progress for head sweeps / rotate-in-place / explore steps (default **on**). Discord still only gets coarse milestones (`*Look around: sweeping head*`, mid/end scan). |
| `EMET_CONFIRM_NAV` | Agent / DynamemController | `1` — before executing a motion plan, show the path on the 2D map (Rerun `world/nav/plan_map` + Discord PNG) and wait for **y/n** (terminal or Discord). Same as `emet run agent --confirm-nav`. Recommended on the real robot. Scripted `-c` runs auto-accept. |

See [AGENT_RUN.md](AGENT_RUN.md).

Add new cross-cutting `EMET_*` variables here or in a topic-specific doc and link from [simulation.md](simulation.md) / [README.md](../README.md) as appropriate.
