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
