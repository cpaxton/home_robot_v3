# Testing index

Central map of **how to run tests**, **what each suite validates**, and **where detailed write-ups live**. From the repo root, use **`uv sync`** then **`uv run emet test`** (see [cli.md](cli.md#testing)).

## Run from this repo

### Extracted shared-agent stack (October 2026)

#### Measured scan implementation and outstanding gates (2026-10-10)

- Completed replay: frozen main scored **4/4**, the full frozen candidate
  **1/4** on q12/q14/q56/q6. The three earlier EQA losses reproduced.
  Isolated #180 versus its main baseline scored **4/4 on both**. This narrows
  attribution; it does not clear #181/#182/#184 or prove accuracy neutrality.
- Completed S0: both old main and candidate failed object and receptacle find;
  runtimes were approximately **22.4 vs 16.5 minutes**. Faster failure is not
  task success. RoboCasa failed asset preflight, so it has no valid task score.
- Current-main measured route gate passed **50/50 waypoints**, unchanged
  0.02 m / 0.03 rad precision tolerances. Recorded head tilt remained near
  -30 degrees, with camera forward Z near -0.5. The older candidate's level
  head has **not** reproduced on this integrated base. The existing atomic
  simulator command-snapshot repair is a plausible explanation, not proven
  attribution. The idle-only probe never requested a tilted view.
- A remaining head-wait bug is repaired: stationary off-target is not arrival;
  tilt velocity participates in settling; missing/stale telemetry times out;
  `head_to` and `look_front` propagate success/failure. Tolerances are unchanged.
- Opt-in mapping configuration: `mapping.scan_profile: coverage`. Legacy
  `local` remains the default and preserves `mapping_rotate_steps`. Coverage
  instead requests eight absolute episode-frame headings and front/downward
  head views (up to 16 captures), independent of scene or target labels. The
  same implementation backs `look_around(profile="coverage")`.
- Coverage returns requested/measured base/head poses, camera transforms and
  intrinsics, frame sequences, map observation indices, elapsed time, command
  counts, unsupported views and partial-failure reasons. Map indices are **not
  graph object IDs**. Existing swept-footprint checks gate agent turns; no
  unknown floor is declared free. Missing head capability yields horizontal
  captures with an explicit vertical-coverage gap. Final gaze is preserved.
- The 180-second scan budget is cooperative (synchronous perception in flight
  cannot be interrupted); an earlier caller deadline can be passed. Each turn
  uses at most a 30-second requested wait. Do not claim a hard end-to-end SLA
  from these bounds; simulator adapters can scale motion waits.
- GPU validation is **blocked**, not passed: the model-free probe was cancelled
  before simulation because NVML reports a driver/library version mismatch.
  Repair the host driver before requeueing. No reboot was performed.
- Targeted control, scan, tool, OVMM contract and navigation regressions:
  **379 passed, 2 skipped** (the two simulator integration tests are opt-in).
  A Mesa software-rendering fallback booted S0 and measured head tilt near
  -30 degrees, but ran at about **0.016 simulation seconds per wall second**.
  It was stopped during posture preparation, without accepted scan captures.
  This is not a coverage pass. Its probe child needed explicit termination
  after job-wrapper cancellation; unrelated simulation processes were retained.

Resume serially, with CPU-safe affinity and GPU exclusivity:

1. Run `scripts/probe_rby1_camera.py --sim configs/sim/default_table_stretch.yaml
   --coverage-scan --output-dir <fresh-directory>` through `emet jobs run`.
   Inspect every accepted camera view/pose and require actual target visibility;
   completion alone is not acceptance. The probe has no occupancy map, so it
   tests optics/control in the known fixture, not agent swept-clearance safety.
2. Extend this model-free gate to four initial headings and an opposite-side
   fixture; score simulator masks and valid depth offline, never in policy.
   These gates and the GT visibility scorer remain outstanding.
3. Only after that gate, run paired legacy/coverage S0 with identical Qwen,
   physics, seed and find budgets; report the larger mapping image/time budget.
4. Replay EQA losses at successive #181, #182 and #184 prefixes. Keep the
   coverage profile off for this attribution; do not promote the full stack
   while its repeated loss remains unexplained.
5. Preflight installed RoboCasa/Molmo assets before the small paired pilots.
   Keep TAMP to regression checks; no Habitat-OVMM sweep in this change.

#### Repaired-head pilot and review handoff (2026-10-09)

- Current-main acceptance sandbox incorporates the reconciled slices, #190's
  telemetry repair and isolated evidence output. Combined checks: **801 passed,
  one previously reproduced main missing-RGB failure**. Queue/registry checks:
  **28 passed** after correcting dependency waits to precede GPU-lock acquisition.
  Requeued waiting jobs use this ordering; the active pilot was not interrupted.
- Old-main S0 completed with object and receptacle find both false (valid metrics,
  about 22 minutes). Candidate S0 is ongoing. Its saved assessment image contains
  sky and floor rather than the table. A passive state sample shows an upright
  base (up dot Z about 0.999994), level head and camera forward near +X. The table
  is toward -Y; four mapping views request only three +45-degree increments, and
  the scan currently ignores motion return values. Coverage/heading must be
  checked before blaming object grounding. Do not claim the earlier stationary
  telemetry probe validated object visibility: its image is also horizon-only.
- **Merged with explicit user approval:** #176 (`cee9b093`) and #180 (`2ad47882`).
  The isolated #180 EQA job remains queued and is now post-merge evidence, not a
  passed prerequisite. Prior approval/gate notes below describe the earlier state.
- Staged current-main reconciliation branches (not extra PRs or merged changes):
  `review/providers-current-main` at `e1fadc97` passes 100 provider/dialogue tests;
  `review/navigation-current-main` at `f69006d6` passes 213 navigation/TAMP tests;
  `review/inspection-current-main` at `10de62a9` passes 334 recovery/API tests.
  Navigation conflict resolution retains main's measured-time progress window,
  finite-pose guards, differential-drive execution, conservative footprint and
  existing diagnostic reason names, while adding phase-handoff progress and
  unknown-floor diagnostic metadata. Running pilot heads remain frozen. Reconcile
  acceptance/telemetry next, then update public dependent heads after validation.
- User-supplied review of **#176** found no blocking issues: restoring conversation
  history and iteration count in `finally` covers current serialized reset-based
  perception calls, including failure and nested contexts. Independent merge
  candidate after maintainer approval; not blocked by the upper-stack EQA losses.
  Accept the current shallow snapshot for this scoped fix: in-place mutation of
  existing message dictionaries is not protected. Track a Qwen-specific
  reset-context regression and reassess deep copying if mutation is introduced.
  The conditional-context style suggestion is optional. No merge performed here.
- User approved **#180** after comment resolution. Repair `394caa3a` distinguishes
  missing semantic verification from absent evidence, releases newly proposed
  candidates after failed instance admission, and documents once-per-view/query
  admission. Validation: 264 passes plus the known main missing-RGB failure;
  on current main, 305 passes plus the same failure including TAMP API/bridge
  checks. Isolated paired EQA on q12/q14/q56/q6 is queued before merging; approval
  does not waive the regression gate. Direct view admission remains explicitly
  non-idempotent; the executor owns duplicate-attempt prevention.
- Frozen `bf8b635d` EQA-12 seed 0 completed with valid metrics for all twelve
  questions: **5/12**, versus **8/12** on frozen main `a9a5f1d1`. Paired losses:
  q12, q14, q56; no gains. Startup is repaired, but accuracy acceptance is **not**
  cleared. The three losses committed a VLM-suggested answer after one
  investigation; their traces report successful navigation with displacement.
  This is not evidence that the selected view was adequate or the answer grounded.
- The old dev loop reused `cli_episode_qNNNN` diagnostic directories across arms.
  JSONL scores remain separate, but baseline visual evidence was overwritten or
  mixed with later output. Do not treat those directories as a trustworthy paired
  visual comparison. The repaired driver records a unique invocation prefix in
  `diagnostic_tags.txt` and supplies seed/question-specific bundle tags.
- The queued replay uses q12/q14/q56 plus unchanged-success q6, paired seed 0,
  separate diagnostic tags, unchanged model/budgets, and serial execution after
  the small-OVMM battery. Retain all outcomes; do not select a favorable repeat.
  If losses repeat, test cumulative stack prefixes to identify the owning slice.
  If attribution is stochastic, add paired seeds 1 and 2 before making claims.
- Current main advanced to `a7805b63` with #177/#178. The frozen comparison does
  not validate those integrations. Reconcile navigation, mapping, simulator and
  dispatch overlaps in a separate worktree, preserving main's safety contracts.
- Review #176 and #180 independently now. Land only reviewed slices with their
  own passing gates; upper slices remain draft if they regress. Review order is
  #180 → #181 → #182 → #184 → #185. Telemetry is extracted onto current main in
  **#190** (four focused snapshot/forwarding tests pass); remove its duplicate
  from #185 when restacking, not while the experiment checkout is active.
- Before promoting the reconciled stack: offline contracts with no new failures;
  five unchanged-tolerance navigation repeats; paired current-main EQA-12;
  matched-provider small-OVMM regression checks; and minimal existing admitted
  TAMP smoke. Different-provider comparisons are complete-system comparisons,
  not isolated navigation ablations. Find success is not manipulation acceptance.

#### Startup/telemetry repairs and bottom-up review (2026-10-09)

- Confirmed the extracted EQA crash with a serial one-frame comparison: old
  `2d8b30b1` imported MuJoCo through A* → `base_goal_rank` → `voxel_arm_collision`
  and aborted with OpenGL `InvalidValue` (exit 134). Repair `f8d8666a` separates
  shared grid coordinates from arm FK; `12e4c426` rendered a 480×640 Habitat frame
  successfully under the same launch environment. Job `20261009_154414_7eb1e7`.
  The repair is now included in #182 as `1bdb2949`, not only the acceptance tip.
- Stretch omitted body orientation and actuator targets from its state schema.
  Repair `12e4c426` samples them in the physics process and carries named controls
  through IPC/ZMQ. Unknown legacy fields remain unknown. The stationary live
  retest receives both fields. Route job `20261009_154446_722088` completed all ten
  waypoints with complete telemetry in all 11 frames: worst receipt errors 0.009731 m
  and 0.029891 rad (limits 0.02 m / 0.03 rad); minimum body-up dot world-Z 0.999994.
  This is one diagnostic repeat, not the configured five-repeat navigation gate.
- Repaired-stack offline result: **732 passed, one existing main failure** (the
  missing-RGB voxel-upgrade test described below). Focused repair pack: 39 passed.
- #176 is the independent human-review candidate (three focused restoration tests
  rerun); #180 has 102 focused grounding/find tests passing. #181 remains opt-in;
  #182 needs repaired-head navigation/EQA evidence; #184/#185 depend on those gates.
  No PR has been merged by this review pass. Do not merge #183 wholesale or absorb
  #169's deferred wheel/pregrasp work into these repairs. The TAMP stack remains
  separately owned (#177 → #178 → #179 → #186 → #187).
- Main `a9a5f1d1` EQA-12 scored 8/12. The original extracted run produced no metrics
  (12 native startup crashes), not 0/12 accuracy. Earlier `eb1e0890`/`17d96365`
  paired EQA runs both scored 6/12 with identical correctness. A one-frame EGL pass
  is not an EQA accuracy gate; a repaired paired rerun is still required.

Review order: #180 current-view grounding → #181 opt-in providers → #182 measured
navigation → #184 bounded inspection/recovery → acceptance tooling. #176 remains
an independent conversation-state fix. Do not merge the legacy #167/#168 containers
wholesale. The [frozen source inventory](plans/shared_stack_inventory.json) records
every changed source file and commit, including partial extractions and explicitly
deferred work. Blob equality is accounting, not semantic or performance acceptance.

Run `scripts/run_extracted_acceptance.sh eqa` or `room` **only through**
`emet jobs run --cpu-safe --gpu-exclusive --need-mib 12000`. Set `SOURCE_SHA` to the
full committed checkout SHA, `OUT_DIR` to a fresh artifact directory, and `EMET_PY`
and `HABITAT_BIN` to the installed environments when using a frozen worktree.
The driver rejects dirty/untracked source, records inputs, and never enables nav
teleports. Do not bypass the runner's 10-GB free-space gate or launch hardware.

The EQA phase uses the existing seeded 12-question development loop; the room phase
uses S0, RoboCasa and Molmo find diagnostics, not full manipulation acceptance.
`DEV_SEED` and `FIND_EPISODES` select matched repeats/subsets. The room preset is
opt-in, uses perfect simulator depth, and keeps cheap YOLOE/SAM2 proposals separate
from final VLM verification. It is **not** real-perception evidence or a new default.

Acceptance order: offline contracts and minimal TAMP regression → measured
stationary/known-route smoke on the actual robot/controller → paired main/candidate
EQA and S0 → RoboCasa/Molmo. Keep model, scene, seed, depth, budgets and physics
identical within each pair; if testing the provider preset, apply it to both arms.
Main cannot be assumed to support candidate-only configuration. Diagnose tool/control
failures separately before attributing differences to models or memory.

Retain per-episode correctness, object/receptacle results, measured pose and head
feedback, floor recovery events, images/traces, and runtime. Exit zero means the run
completed, **not** that tasks passed. Missing metrics/native crashes are infrastructure
failures. A 12-question run is a regression signal, not a paper-quality improvement
claim. Source-branch results do not validate extracted heads. Room pilots at
`eb1e0890` and `17d96365` failed both find phases in S0 and RoboCasa; no room acceptance
is claimed. Full TAMP, manipulation/physics changes, and Habitat OVMM remain separate.

Combined offline check (2026-10-09): **726 passed, one known main failure**,
`test_voxel_sim_upgrades_full_frame_absent_to_present` (missing RGB; expected
PRESENT, got ABSENT). Minimal MCTS and TAMP bridge tests are included. Reproduce:

```bash
mapfile -t checks < <(git diff --name-only a9a5f1d1 2d8b30b1 -- src/test)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
.venv/bin/python -m pytest --noconftest -q --tb=short -o addopts='' -p no:cacheprovider \
  "${checks[@]}" src/test/scripts/test_extracted_acceptance.py \
  src/test/motion/test_mcts_tamp_planning.py src/test/controller/task/tamp/test_agent_bridge.py \
  src/test/memory/test_agentic_mixins.py src/test/memory/test_habitat_ovmm_agentic_find.py \
  src/test/memory/test_ovmm_agentic_routing.py src/test/eval/test_agentic_*.py
```

Use **`uv run emet …`** (or `source .venv/bin/activate` then bare `emet`) from the **project root** so commands pick up **this checkout’s** code and virtualenv.

| Symptom | Fix |
|---------|-----|
| `No such option: --explore-loop` on `emet run dynagraph` | Wrong `emet` on PATH (another clone or old install). Run `which emet`; use **`uv run emet`** from this repo. Confirm with `uv run python -m emet.app.run_dynagraph --help` (should list `--explore-loop`) |
| Sim logs show `Sim navigation: driving toward` (no `[sim_nav]`) | **`emet` from another checkout** (e.g. `home_robot_v4`). From this repo: `cd ~/src/home_robot_v2 && emet serve …` (yellow “re-running: uv run emet”) or `./scripts/emet-v2 serve …` |
| rotate_in_place sends goals hundreds of meters away | Usually stale sim + wrong `nav_world`; run [test_rotate_in_place_robocasa_nav.py](../src/test/simulation/test_rotate_in_place_robocasa_nav.py) after nav changes |
| Tests skip sim unexpectedly | Default runs include sim; set `RUN_SIM_TESTS=0` or `uv run emet test --no-sim` to skip |
| Missing MuJoCo / Robocasa | `uv sync` with default groups, or `./install.sh --sim` |

## Quick commands

| Goal | Command |
|------|---------|
| **Simulation smoke battery (7 tracks)** | `./scripts/run_simulation_smoke_battery.sh` — see [simulation_testing_plan.md](simulation_testing_plan.md) |
| Motion planning (offline) | `uv run emet test src/test/motion/algo/test_rrt.py src/test/motion/test_arm_rrt.py src/test/motion/test_voxel_obstacle_planning.py -q` — see [motion_planning.md](motion_planning.md) |
| **No-NN sim pick/place** | `EMET_SIM_NAV_TELEPORT=1 uv run python scripts/scripted_sim_pick_place.py --start-sim --sim configs/sim/default_table_rby1.yaml --manip-mode kinematic` — GT+MuJoCo only ([motion_planning.md](motion_planning.md#no-neural-nets-smoke-sim-only)) |
| **Molmo grasp oracle MP** | `uv run emet test src/test/perception/grasps/ src/test/motion/test_arm_manip_profile.py -q`; optional smoke `scripts/scripted_molmo_grasp_mp.py` ([motion_planning.md](motion_planning.md#molmospaces-grasp-oracle-multi-robot)) |
| Full suite (sim on by default) | `uv run emet test` |
| Skip sim (faster CI-style) | `uv run emet test --no-sim` |
| Verbose | `uv run emet test -v` |
| Memory backend smokes | `uv run emet test src/test/memory/test_memory_backends_smoke.py -v` |
| Multi-robot Dynagraph floor E2E | `uv run python src/test/app/run_dynagraph_multi_robot_e2e.py` |
| Dynagraph unit (explore loop, graph memory) | `uv run emet test src/test/app/test_dynagraph_explore.py src/test/memory/test_graph_eqa_memory.py -v` |
| GraphObjectFusion + GT export (fast) | `uv run emet test src/test/memory/test_graph_object_fusion.py src/test/simulation/test_mujoco_gt_objects.py -v` |
| Dynagraph benchmark smoke (unit) | `uv run emet test src/test/app/test_dynagraph_benchmark_smoke.py -v` |
| Dynagraph staleness / disappearance | `uv run emet test src/test/memory/test_dynagraph_staleness_disappearance.py -v` |
| Unified Dynagraph eval CLI | `uv run emet eval-dynagraph --episode /tmp/export` |
| SQA3D benchmark (unit) | `uv run emet test src/test/benchmarks/sqa3d/ -v` |
| SQA3D ScanNet embodied smoke | `uv run python scripts/run_sqa3d_scannet_smoke.py` |
| SQA3D EM@1 scoring | `uv run emet eval-sqa3d -p preds.jsonl --split val` |
| Graph fusion calibration (one scene) | `emet export-sim-gt` → `emet run dynagraph --calibration-export` → `emet tune-graph-fusion` (see [dynagraph.md](dynagraph.md#object-gt-export-and-graphobjectfusion-calibration)) |
| GraphEQA human-answer formatter | `uv run emet test src/test/memory/test_graph_eqa_human_answer.py -v` |
| Manual Dynagraph EQA + export (Robocasa) | See [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md#single-eqa-question-manual-per-robot) |
| Dynagraph Robocasa question CLI (CI) | `uv run emet test src/test/app/test_dynagraph_robocasa_question_cli.py -v` |

Environment: **`RUN_SIM_TESTS=0`** skips sim integration tests. Heavy VLLM download tests use marker **`vllm_load`** and are excluded from default runs — see [plans/TESTING_VLLM_LOAD.md](plans/TESTING_VLLM_LOAD.md).

---

## Documentation by topic

| Topic | Doc | What it covers |
|-------|-----|----------------|
| **Motion planning (base + arm)** | [motion_planning.md](motion_planning.md) | RRT / A\* on voxel maps, kinematic arm RRT-Connect, offline tests |
| **Memory backends (SVM, DynaMem, GraphEQA)** | [plans/TESTING_BACKENDS.md](plans/TESTING_BACKENDS.md) | Test matrix, red-cylinder / Robocasa spin integration, backend unit tests |
| **Dynagraph multi-robot Robocasa E2E** | [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md) | Floor-area parity (stretch / innate_mars / galaxea_r1), spawner maps, artefact paths |
| **GraphEQA manual Robocasa** | [graph_eqa.md](graph_eqa.md#testing-graph-eqa-in-robocasa) | Interactive questions in kitchen sim (Gemini / encoder) |
| **MolmoSpaces** | [plans/2025-03-10_molmospaces_testing.md](plans/2025-03-10_molmospaces_testing.md), [molmospaces.md](molmospaces.md) | Wrapper venv, CLI smoke, optional integration; **multi-robot matrix below** |
| **Simulation smoke battery (paper)** | [simulation_testing_plan.md](simulation_testing_plan.md) | Seven-track sequential Habitat + Robocasa + Molmo + SQA3D validation |
| **Multi-robot sim plumbing** | [plans/MULTI_ROBOT_TESTING.md](plans/MULTI_ROBOT_TESTING.md) | Registry, MJCF load, GenericZmqClient (partially superseded by unit tests) |
| **CLI / install smoke** | [cli.md](cli.md#testing), [simulation.md](simulation.md#2-test-the-setup) | `emet test`, serve smoke |
| **SQA3D + ScanNet EQA** | [sqa3d.md](sqa3d.md) | Situated QA loaders, EM@1 eval, Open3D mesh replay |
| **Agent / LLM** | [AGENT_RUN.md](AGENT_RUN.md#testing), [llm_agent.md](llm_agent.md#testing-the-llm-agent) | Agent loop, component tests |
| **Refactor logs (not test specs)** | [logs/README.md](logs/README.md) | Historical change notes |

There is **no other single “master” file** today; this page is the hub. Feature-specific docs link back here where useful.

---

## Automated test map (by validation goal)

### Geometry / navigation / floor coverage

| Validates | Test / harness | Notes |
|-----------|----------------|-------|
| Offline RRT / arm RRT on voxel-shaped grids | [test_voxel_obstacle_planning.py](../src/test/motion/test_voxel_obstacle_planning.py), [test_arm_rrt.py](../src/test/motion/test_arm_rrt.py) | Unit; [motion_planning.md](motion_planning.md) |
| Explored floor vs spawner walkable (3 robots, Robocasa) | [run_dynagraph_multi_robot_e2e.py](../src/test/app/run_dynagraph_multi_robot_e2e.py) | [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md); **does not** assert graph nodes or EQA |
| Nav world ↔ session frame | [test_nav_xyt_session.py](../src/test/utils/test_nav_xyt_session.py) | Unit |
| **rotate_in_place** world (x,y) stays at spawn (innate_mars + Robocasa) | [test_rotate_in_place_robocasa_nav.py](../src/test/simulation/test_rotate_in_place_robocasa_nav.py) | Sim; asserts `spawn_compose` goals, not `nav_world` |
| **VLM-free multi-env nav/explore** (Robocasa L1, OVMM Robocasa L2, Molmo iTHOR 0) | [test_multi_env_nav_explore_smoke.py](../src/test/simulation/test_multi_env_nav_explore_smoke.py) | Sim; no Qwen/VL worker — `rotate_in_place` + one `run_exploration`; asserts `|Δxy|≥0.08m` or explored-cell growth. Helper: [`nav_explore_smoke.py`](../src/emet/eval/nav_explore_smoke.py). Run when no other MuJoCo/EGL job is live: `uv run emet test -v src/test/simulation/test_multi_env_nav_explore_smoke.py` |
| Robocasa freejoint spawn (Stretch / Galaxea, collision autoplace) | [test_robocasa_freejoint_spawn.py](../src/test/simulation/test_robocasa_freejoint_spawn.py) | Sim; `EMET_ROBOSUITE_AUTOPLACE` + `find_robocasa_freejoint_xyz` (no startup nav rollback) |
| Robosuite nav frame unit (spawn compose, jump guard) | [test_robosuite_nav_world_clamp.py](../src/test/simulation/test_robosuite_nav_world_clamp.py) | Unit |
| Floor metrics export | [test_floor_metrics.py](../src/test/memory/test_floor_metrics.py) | Unit |
| Red cylinder localize (DynaMem, default scene) | [test_red_cylinder_in_sim.py](../src/test/mapping/test_red_cylinder_in_sim.py) | Sim; stretch + innate_mars |
| Robocasa memory after spin | [test_robocasa_memory_after_spin.py](../src/test/simulation/test_robocasa_memory_after_spin.py) | Sim |

### Scene graph / object labels (model-derived)

| Validates | Test / harness | Notes |
|-----------|----------------|-------|
| **OpenVocab scene graph** in Robocasa (dedup, kitchen labels) | [test_scene_graph_robocasa.py](../src/test/scene_graph/test_scene_graph_robocasa.py) | Uses `SceneGraphProcessor` + `cpu_scene_graph`; **not** GraphEQAMemory / Dynagraph path |
| Open-vocab unit behaviour | [test_open_vocab_scene_graph.py](../src/test/scene_graph/test_scene_graph_in_sim.py) | Default scene spin |
| GraphEQA memory unit (merge, staleness, query) | [test_graph_eqa_memory.py](../src/test/memory/test_graph_eqa_memory.py) | Mock EQA clients; package map: [graph_memory.md](graph_memory.md) |
| GraphEQA + default scene (red/blue) | [test_graph_eqa_default_scene_sim.py](../src/test/memory/test_graph_eqa_default_scene_sim.py) | Sim RGB + **injected** graph labels; mocked `query_answer` |
| Per-frame detections on export | Dynagraph `--export` → `graph/frames/detections_*.json` | Manual review; see [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md#assessing-semantic--eqa-quality) |

### EQA / answering questions

| Validates | Test / harness | Notes |
|-----------|----------------|-------|
| Mocked EQA on constructed graph | [test_graph_eqa_default_scene_sim.py](../src/test/memory/test_graph_eqa_default_scene_sim.py) | Known scene; labels not from full VL pipeline |
| Interactive Robocasa EQA | Manual — [graph_eqa.md](graph_eqa.md#testing-graph-eqa-in-robocasa) | Requires API keys / models |
| **Dynagraph graph build + NL answer on known scene** | [`test_dynagraph_benchmark_smoke.py`](../src/test/app/test_dynagraph_benchmark_smoke.py) (unit); sim harness [`run_dynagraph_benchmark_smoke.py`](../src/test/app/run_dynagraph_benchmark_smoke.py) | Question bank + `eqa_results.json`; full sim via `RUN_DYNAGRAPH_BENCHMARK_SMOKE=1` |

### Infrastructure / CLI / controllers

| Validates | Test / harness |
|-----------|----------------|
| `emet` CLI, `run dynagraph` help | [test_cli.py](../src/test/cli/test_cli.py) |
| Controller smoke (SVM, DynaMem, GraphEQA, Dynagraph imports) | [test_controller_smoke.py](../src/test/controller/test_controller_smoke.py) |
| Multi-robot registry / MJCF | [test_multi_robot.py](../src/test/simulation/test_multi_robot.py) |
| Dynagraph explore loop helper | [test_dynagraph_explore.py](../src/test/app/test_dynagraph_explore.py) |

### MolmoSpaces multi-robot (serve, merge, navigation)

Robot registry and aliases: [robots/supported_robots.md](robots/supported_robots.md). MolmoSpaces install, merge, and troubleshooting: [molmospaces.md](molmospaces.md#quick-verification-developers).

**Quick serve smoke** (repo root, sim + wrapper installed; first iTHOR load can take 1–2 minutes):

```bash
# Merge + ZMQ load; expect no MJCF errors and autoplace log (not stuck at origin)
for r in stretch rby1 innate_mars xlerobot; do
  echo "=== $r ==="
  timeout 120 uv run emet serve mujoco --scene ithor --robot "$r" --headless
done
```

| Robot | `--robot` | Server stack | Serve / merge tests | Nav / mapping tests |
|-------|-----------|--------------|---------------------|---------------------|
| **Stretch** | `stretch` | `MujocoZmqServer` (Stretch stack) | `RUN_MOLMOSPACES_TESTS=1` → [test_molmospaces_ithor_base_settle.py](../src/test/molmospaces/test_molmospaces_ithor_base_settle.py) (rby1-oriented settle; stretch uses same merge path) | `RUN_STRETCH_MOLMO_DYNAMEM=1` → [test_stretch_molmospaces_dynamem_floor_map.py](../src/test/molmospaces/test_stretch_molmospaces_dynamem_floor_map.py) |
| **Galaxea R1** | `rby1`, `galaxea_r1` | `RobosuiteZmqServer` | [packages/emet_molmospaces/tests/test_rby1_scene.py](../packages/emet_molmospaces/tests/test_rby1_scene.py) (merge + step); `RUN_MOLMOSPACES_TESTS=1` → [test_molmospaces_ithor_base_settle.py](../src/test/molmospaces/test_molmospaces_ithor_base_settle.py) | `RUN_MULTI_ROBOT_NAVGRID=1` → [test_multi_robot_molmospaces_navgrid_similarity.py](../src/test/molmospaces/test_multi_robot_molmospaces_navgrid_similarity.py) |
| **Innate Mars** | `innate_mars`, `maurice` | `RobosuiteZmqServer` (planar) | [test_multi_robot.py](../src/test/simulation/test_multi_robot.py) (spec/MJCF); wrapper merge: `test_merge_innate_mars_into_minimal_scene_no_double_floor` | Robocasa Dynagraph E2E: [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md); Molmo navgrid: `RUN_MULTI_ROBOT_NAVGRID=1` (above); `EMET_NAVGRID_COMPARE_ROBOTS=stretch,rby1,innate_mars` + [scripts/tier4_multi_robot_navgrid_compare.py](../scripts/tier4_multi_robot_navgrid_compare.py) |
| **XLeRobot** | `xlerobot` | `RobosuiteZmqServer` (planar) | wrapper `test_merge_xlerobot_into_minimal_scene`, `test_merge_stretch_into_scene_with_memory` (merge `memory` + robot `njmax` guard) | `RUN_XLEROBOT_DYNAMEM=1` → [test_xlerobot_dynamem_default_table_smoke.py](../src/test/molmospaces/test_xlerobot_dynamem_default_table_smoke.py); `RUN_XLEROBOT_MOLMO_DYNAMEM=1` → [test_xlerobot_molmospaces_dynamem_floor_map.py](../src/test/molmospaces/test_xlerobot_molmospaces_dynamem_floor_map.py); `RUN_STRETCH_XLEROBOT_NAVGRID=1` → [test_stretch_xlerobot_navgrid_similarity.py](../src/test/molmospaces/test_stretch_xlerobot_navgrid_similarity.py) |

**Cross-robot Dynagraph / DynaMem nav benchmark** (default table + Robocasa + MolmoSpaces tiers):

```bash
EMET_SIM_NAV_TELEPORT=1 uv run python src/test/app/run_dynagraph_nav_benchmark.py --robot xlerobot --dynamem --default --molmo
```

See [dynagraph_nav_benchmark.md](dynagraph_nav_benchmark.md). Molmo iTHOR GT query on train index 0 is **sink** (not sofa).

**Dynagraph + Stretch on MolmoSpaces iTHOR** (manual): terminal A `emet serve mujoco --scene ithor --robot stretch`, terminal B `emet run dynagraph`. If you see `Timeout waiting for navigation step` while the sim teleports, check branch `fix/stretch-molmo-nav-wait` (relative `move_base_to` must not double-compose with server `nav_relative`). Rerun `world/robot` uses `gps`/`compass` + `navigation_origin_xyt`; if the base marker is frozen but the voxel map moves, confirm ZMQ obs `gps` is updating and session has `navigation_origin_xyt`.

**Wrapper-only pytest** (no live iTHOR assets):

```bash
uv run pytest packages/emet_molmospaces/tests/ -q
```

---

## Known gap: graph + EQA on a known scene (Dynagraph)

**Status (2026-07):** Unit coverage landed in [`test_dynagraph_known_scene_attach.py`](../src/test/memory/test_dynagraph_known_scene_attach.py) — red cylinder / blue cube instance items and fusion detections must attach as object nodes with allowlisted labels (no GPU). World-change invalidation: [`test_dynagraph_staleness_disappearance.py`](../src/test/memory/test_dynagraph_staleness_disappearance.py) + [`test_lifelong_checkpoint_invalidate.py`](../src/test/eval/test_lifelong_checkpoint_invalidate.py). Operator notes: [experiments/dynagraph_dynamic_memory.md](experiments/dynagraph_dynamic_memory.md). Full Robocasa/MuJoCo E2E with live YoloE remains a stronger integration check.

**Remaining E2E gap:** Explore-only exports can still show **`Nodes (0)`** in `scene_graph_report.txt` when detections exist but the instance→graph hook did not run. Prefer graph-health fields in Habitat `metrics.json` / dynamic-explore cycle rows (`graph_health`) and `uv run python scripts/summarize_graph_health.py …`.

| Step | Detail |
|------|--------|
| Unit (CI) | `uv run emet test src/test/memory/test_dynagraph_known_scene_attach.py --no-sim` |
| Scene E2E | Default MuJoCo table or Robocasa seed 0 with `--export` + `--question` |
| Assert graph | `graph_health.n_object ≥ 2` or allowlisted labels in `scene_graph_report.txt` |
| Health triage | `scripts/summarize_graph_health.py` → `blowup` / `fragmentation` / `empty_graph` / `ok` |

Until full E2E is green in CI, use **manual** `--question` runs (documented in [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md#assessing-semantic--eqa-quality)) and inspect **`detections_*.json`** + **`scene_graph_report.txt`**.

---

## See also

- [Dynagraph](dynagraph.md) — CLI, explore loop, export layout
- [GraphEQA](graph_eqa.md) — graph memory design and manual Robocasa testing
- [plans/README.md](plans/README.md) — design plans (architecture, GraphEQA plan, mapping refactor)
