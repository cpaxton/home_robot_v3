# Shared-agent paper update and figure plan

## Claim boundary

The intended contribution is one agent and one evidence lifecycle across EQA,
OVMM, and multi-step TAMP. Robot capabilities, reach limits, calibration, and
approach geometry belong in adapters/configuration, not robot-specific reasoning
policies. That is the target architecture, not a claim that the current prototype
has already demonstrated learned end-to-end manipulation on every platform.

Distinguish view evidence, weak search candidates, freshly grounded objects, and
action outcomes. A navigation anchor is not object geometry; an assisted simulator
attachment is not a physically verified grasp. Keep oracle-assisted TAMP controls
separate from learned retrieval-to-manipulation results.

## Next paper edits, after the limited pilot

### Bounded cross-benchmark acceptance (no sweep)

Use the existing random-16 EQA IDs as a development pilot, not a new benchmark
search. Freeze the common budget and resolved configs before rerunning the three
primary rows: DynaMem, arrival-only lazy memory, and query-driven memory. Reuse
previous results only when their source/configuration actually matches. Report
paired outcomes; 16 questions cannot establish broad superiority.

For OVMM, freeze six cases spanning visible, occluded, repeated/ambiguous, unseen,
absent, and moved/revisited targets before evaluating the same three rows. Check
positive grounding and appropriate negative behavior, not just exit codes.
Unsupported attributes must remain explicit failures/abstentions. Do not tune a
separate query admission policy for EQA versus OVMM to make these cases pass.

For TAMP, retain the table, decoy, and furnished-scene gates below as execution
controls. Add a learned retrieval -> fresh grounding -> capability adapter ->
pick/place -> fresh observation sequence, with independent outcome measurement.
The current query-mode manipulation adapter does not yet provide that live gate
for rby1/Sourccey; do not substitute the GT task-handle run for it. One bounded
multi-object relocation case should also exercise replanning after a move.

For Herman, collect a short stationary RGB-D/pose sequence, validate timestamps,
depth support and projection, then replay it through the same memory/query path.
Demonstrate at least one supported visible-object query and one absent/ambiguous
query with saved image/map evidence. Replay success is stationary perception proof
of life, not navigation or manipulation transfer. Use no base (including yaw),
arm, or gripper commands. Only head tilt is authorized if needed; a passive
subscriber is preferred for initial capture. Do not use the documented
`EMET_BASE_ROTATE_ONLY` mode as a safety boundary because it still allows yaw.

Acceptance requires evidence of the shared lifecycle working in both EQA and
OVMM, bounded/reasonable memory without losing relevant evidence, correct
rejection/invalidation, and honestly separated manipulation controls. Report the
whole selected set, including failures. Run a targeted ablation only for a
remaining causal question; do not expand into a full parameter sweep.

1. Update `paper/sections/03_method.tex`: describe the shared evidence lifecycle
   and conditional graph admission. The current every-step graph-write description
   does not describe the opt-in lazy/query-driven prototype. Document grounding,
   rejection, invalidation, and capability-based action dispatch explicitly.
2. Update `paper/sections/04_experiments.tex`: freeze paired episode IDs, source
   revision, model, exploration/action budgets, seeds, and resolved row configs.
   Compare complete systems first (voxel-only/DynaMem, arrival-only, query-driven);
   use targeted ingestion/fusion ablations to investigate causes. Do not infer
   causality from rows changing multiple settings or select a paper winner solely
   from the development random-16.
3. Update `paper/sections/05_results.tex`: separate unit/integration gates,
   development smokes, paired task results, and oracle execution controls. Report
   answer/find quality and retained recall alongside memory growth, grounding
   success, stale-target rejection, model calls, and runtime. Include failed cases.
4. Update `paper/sections/appendix/09_tamp_clutter.tex` and robot-platform appendix:
   list adapter differences and GT exposure, navigation snaps, latch/attachment,
   and placement assistance per row. Preserve PR #160's known cabinet-placement
   and Innate Mars navigation/validity failures until specifically retested.
5. Revise abstract/conclusion last, limiting claims to completed paired evidence.
   Release runnable commands, configs, provenance, and artifact-generation steps.

## Figures from recorded evidence

| Figure | Evidence and purpose |
| --- | --- |
| Shared architecture | Repo-native diagram of one agent, evidence tiers, and capability adapters; mark unimplemented paths. |
| Exploration map | Top-down occupancy and actual trajectory, start/goal, query/arrival events, and scale; distinguish simulator GT overlays from learned memory. |
| Evidence lifecycle | Synchronized RGB, mask/depth grounding, candidate and object markers; include absent/ambiguous rejection and stale-target invalidation. |
| Manipulation sequence | Chase, head, and available wrist/overhead frames for approach, grasp, re-approach, and placement; label oracle assistance. |
| Quality versus memory | Paired task outcomes, retained recall, graph size, runtime and uncertainty; never substitute bounded size for quality. |
| Failure panel | Failed grounding, cabinet placement, or navigation with matching map/frame evidence. |

Use real run frames, not generated illustrations, for empirical panels. Record
source revision, command/resolved config, scene/episode, seed, frame/event IDs,
camera name, and assistance settings in the artifact manifest. Use consistent
colors and legends for weak candidates, grounded objects, and rejected targets.
Preserve raw videos/logs and regenerate cropped panels reproducibly; do not imply
that unmatched frames come from the same episode. Review dataset/assets licensing
before redistributing imagery. Missing camera streams must be labeled unavailable.

## September 5 local gates

### Instrumented grounding diagnosis

The post-fix three-row Habitat diagnostic at `39640791` scored DynaMem 1/2,
lazy arrival 0/2, query-driven 0/2. This supersedes the earlier miswired DynaMem
agentic control, which had no router client. It does not establish superiority.

Instrumented query repeat `20260905_230122_e483a6` at `c5ff2347` also scored 0/2
(97 s). Its immutable pre-admission cache is under
`/tmp/emet-query-grounding-evidence-20260905/evidence/grounding`:

- The first arrival shows a living room, with 57 detections; the VLM returned no
  matching regions for the lamp-on-bed referent.
- The second shows a window/wall close-up, with 30 detections. The VLM selected
  region 20, labeled lamp, at confidence 0.0302 with 589 valid mask points. The
  configured 0.12 confidence floor rejected it. Visual review did not establish
  a lamp or bed in that frame. Lowering admission would risk admitting a false
  positive rather than repair retrieval/viewpoint selection.

Cache JSON links to matching RGB PNG and depth/mask NPZ files, records the raw
semantic decision, retrieval score, source/frame IDs, and admission configuration.
The existing explicit `EMET_EQA_EPISODE_DIR` enables these diagnostic artifacts;
`query_memory.grounding_cache_dir` can select a separate directory. Replay with
`replay_grounding_admission(record, config)` only measures admission on a fixed
observed detection set. It cannot estimate retrieval recall, recreate detections
below the detector's emission floor, or predict changed navigation trajectories.
The cache is never read by live action grounding. Add independently checked
labels before using these records to choose thresholds; VLM acceptance is not GT.

MuJoCo find now exposes `--backend lazy_graph --query-driven-memory`, using the
same policy helper as Habitat. Red/blue comparison `20260905_230647_10f8d9` at
`3621d63e` runs DynaMem then query-driven on
`default_table_rby1_s0_distinct_recep`, four mapping views, agentic find 12 rounds /
8 nav steps, with scene-cache reuse disabled. Artifacts:
`/tmp/emet-redblue-query-20260905`. Results are pending; do not mark this gate passed.

Source: `903b0a87`, branch `feat/query-driven-memory`. Runs are sequential with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, loopback simulator
port offset 180, `MUJOCO_GL=egl`, and `EMET_SIM_NAV_TELEPORT=1`. No Herman hardware
commands were issued. No full benchmark sweep was launched.

- Grounding/manipulation contract and TAMP bridge: 30 tests passed.
- rby1 default-table kinematic pick/place: passed; reported displacement 0.102 m.
  Artifacts: `/tmp/emet-tamp-rby1-table-20260905`.
- Sourccey default-table kinematic pick/place: passed using its side approach;
  reported displacement 0.102 m. Artifacts: `/tmp/emet-tamp-sourccey-table-20260905`.
- rby1 mixed grasps: rejected both unreachable decoys (IK errors 2.391/3.395),
  selected reachable candidate 2, and completed pick/place.
  Artifacts: `/tmp/emet-tamp-rby1-decoys-20260905`.
- Recorded Sourccey table repeat: passed, with `third_person.mp4` and start/final
  front, front-right, wrist, overhead, and third-person PNGs under
  `/tmp/emet-tamp-sourccey-recorded-20260905`. The inspected chase frame is a debug
  render; rendering/framing needs review before publication.
- Additional TAMP planning/helpers, script contracts, query configuration, and
  OVMM-find regressions: 47 tests passed (77 tests across the two invocations).
- rby1 furnished iTHOR scene agent-tools gate: passed `scene_tasks` ->
  `plan_pick_place(task:1)` -> `execute_pick_place_plan(plan:1)` using the same
  commands as `scripts/run_tamp_agent_tools_gate.sh`'s kinematic item. The actual
  task handle selected bowl -> dishwasher (not the CLI's microwave hint).
  Reported displacement 2.2144 m, placement error 0.0200 m. The first launch
  failed before simulator startup because this worktree lacks a local
  MolmoSpaces environment; retrying with the existing provisioned environment's
  `bin` directory on `PATH` succeeded. No source workaround was needed.

The learned retrieval-to-manipulation sequence and paired EQA/OVMM pilot remain
open. This small gate set does not rerun the full clutter/navigation battery or
resolve the previously reported PR #160 failures.

These use GT object/receptacle geometry, synthetic COM grasps, base snapping,
and simulator attachment/placement assistance. `--skip-oracle` skips the grasp
service; it does **not** make the experiment GT-free. They are execution gates,
not evidence of learned OVMM success or real-robot grasp robustness.

Reproduce a table gate from a provisioned environment with the repository on
`PYTHONPATH`:

```bash
python scripts/scripted_tamp_pick_place.py --start-sim \
  --sim configs/sim/default_table_rby1.yaml --port-offset 180 \
  --manip-mode kinematic --skip-oracle --cpu-only \
  --figures-dir /tmp/emet-tamp-table
```

For Sourccey use `configs/sim/default_table_sourccey.yaml`. Add
`--plant-infeasible-grasps` for the rejection gate, or `--record-mp4 --video-fps 4`
for low-rate video and camera stills. Run only one simulator at a time.
