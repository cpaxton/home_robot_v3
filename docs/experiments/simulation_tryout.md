# Trying the review branch in simulation

Active review: `fix/grounding-robot-validation` in `/tmp/emet-query-memory`.
The original `/home/cpaxton/src/home_robot_v3` checkout has unrelated changes;
do not switch/reset it to try this branch. Mars is out of scope.

## Local environment

On this workstation, use the existing interpreter with the review source path.
The top-level `emet` CLI can re-execute from the current checkout, so use the
application module explicitly when testing this worktree:

```bash
cd /tmp/emet-query-memory
export PYTHONPATH=/tmp/emet-query-memory/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl EMET_ALLOW_SDPA_ATTN=1 TOKENIZERS_PARALLELISM=false
```

These are workstation-specific instructions, not a portable installation recipe.
The review tree currently uses local MolmoSpaces/Robocasa environment symlinks;
they are intentionally not committed. Run one heavy workload at a time and wait
for queued evaluation jobs to finish before launching an interactive demo.

## Integrated planning/control smoke

This exercises the live CHAT tool registry, semantic task/plan handles and
execution on rby1/native MolmoSpaces. It is **scripted and GT-assisted**, not a
learned grasping demonstration. It uses the Galaxea proxy asset, synthetic
grasps, simulator attachment and base snapping.

```bash
EMET_SIM_NAV_TELEPORT=1 /home/cpaxton/src/home_robot_v3/.venv/bin/python \
  scripts/scripted_sim_pick_place.py \
  --start-sim --sim configs/sim/molmospaces_ithor_train_0.yaml \
  --manip-mode kinematic --object bowl --receptacle microwave --cpu-only \
  --tool-calls-json '[{"name":"scene_tasks","arguments":{"object_filter":"bowl","robot":"rby1"}},{"name":"plan_pick_place","arguments":{"task_ref":"task:1"}},{"name":"execute_pick_place_plan","arguments":{"plan_ref":"plan:1"}}]'
```

Add `--record-mp4 --video-out /tmp/emet-demo-third-person.mp4` for a chase-view
artifact. The fresh integrated gate `20260908_120415_6467fb` passed and saved
`/tmp/emet-integrated-tamp-chat-video-20260908/third_person.mp4` (22.7 s).
The inspected chase view is heavily occluded by scene geometry; recording is
functional, but this is not a paper-ready visualization.
Consult the validation ledger rather than assuming every new revision is live-tested.

## Interactive model entry point

The application supports the same harness with an explicit robot and simulator:

```bash
/home/cpaxton/src/home_robot_v3/.venv/bin/python -m emet.app.run_agent \
  --robot rby1 --start-sim \
  --sim-config /tmp/emet-query-memory/configs/sim/molmospaces_ithor_train_0.yaml \
  --headless --no-discord --command "describe the scene"
```

This command's interface has been checked, but this interactive model invocation
has not yet been validated end-to-end at the latest revision. It defaults to the
Qwen3.5-4B text tool router; the bounded OVMM pilot instead uses its managed
Qwen3-VL worker. Do not compare those as identical model settings.

Learned find/localization remains under diagnosis. A returned XYZ is not itself
a successful scored localization or a verified object. In particular, the last
completed Stretch control was 1/2. The final native rby1/Molmo retest at
`9e2d9c4c` was 0/2 in 122.5 s, with two graph nodes; Stretch remained 1/2 in
956.3 s, without a verified close-look view. Captured-view identity now works
independently of graph insertion, but rby1 approach sampling and Stretch waypoint
timeouts still prevent reliable targeted search. These are not passing model demos.
Use [the validation ledger](robot_grounding_validation.md) for jobs, budgets,
artifacts and new outcomes. No full EQA/OVMM/TAMP paper comparison is claimed.
