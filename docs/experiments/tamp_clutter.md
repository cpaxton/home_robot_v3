# TAMP clutter-clearance benchmark

Assess **task & motion planning** (TAMP): the ability of the robot to interact with the
world as part of a plan. Each episode starts the robot **surrounded by small floor
objects** that must be moved (picked up and relocated to a drop receptacle / bin) to
complete a task.

Two modes:

| Mode | Command | Goal |
|------|---------|------|
| `cleanup`  | "clean up the room" | Relocate all scattered objects to a bin |
| `nav_goal` | "get to the sofa" | Clear a path of objects, then navigate to a scene landmark |

**Manipulation = pick-and-relocate** (no push primitive). Each grasp follows a **latch
contract**: the end-effector reaches the object's grasp frame, the gripper closes, and
the sim `attach` welds the object to the gripper (no physics-grasp requirement). This
isolates TAMP (what to move, where, in what order) from grasp robustness.

**Manip modes:**

| Mode | Meaning |
|------|---------|
| `latch` | Kinematic IK + sim attach (`KinematicPickPlaceExecutor`) — real approach + close + latch |
| `sim` | Teleport oracle (upper-bound reference) |
| `attempt` | Real visual-servo grasp (strict; optional) |

The large registry sets **rby1 → latch** and **stretch / innate_mars / nori → sim**.
Nori's model arm bottoms out at z≈0.29 m, so floor objects (z≈0.02 m) need the
teleport oracle or a raised drop; pass `--manip-mode latch` for a latch experiment.

## Paper references

- Body: `paper/sections/04_experiments.tex` → `sec:tamp_clutter`
- Results: `paper/sections/05_results.tex` → `tab:tamp_clutter`
- Appendix: `paper/sections/appendix/09_tamp_clutter.tex`

## Primary metrics

| Column | Meaning |
|--------|---------|
| `task_success` | cleanup: all objects relocated; nav_goal: goal reached, post-clear route open (`nav_path_open`), and not `skipped_invalid` |
| `skipped_invalid` | cluttered nav_goal whose GT probe did **not** show a blocked route (excluded from the scored success rate) |
| `goal_reached` | nav_goal: base reached the landmark within `success_radius_m` |
| `nav_path_open` | after TAMP, densified teleport-chord samples (~2 cm) are footprint-clear of leftover clutter/furniture (otherwise the runner refuses to snap) |
| `n_cleared` / `n_relocated` | how many scattered objects were moved to the bin |
| `manip_success_rate` | relocated / total |
| `motion_failures` | grasp/place execution failures |
| `planning_wall_s` / `manip_wall_s` | planner vs arm-execution time |
| `episode_valid` | GT probe: was the route blocked by the clutter |

## Episode generation

`configs/ovmm/clutter_episodes_large.yaml` is an **offline template** (no live scatter).
At eval time the runner scatters N pickable objects in a ring
(`sim_set_body_pose`) and runs the **GT validity probe**
(`emet.eval.tamp_clutter.clutter_blocks_path`). Cleanup uses an open ring
(n∈{3,6}). Scored **nav_goal** uses a **tight closed ring** (n=8 @ 0.5 m,
`tight_ring: true`) so the probe is expected to report blocked; if it does not,
the row is `skipped_invalid` and is excluded from the success denominator.

The 24-episode GT+MCTS **battery** (`--test-battery`) is the blocked-route gate:
8-object tight ring (`navblocked`) vs n=0 pure nav (`navclear`). See
[tamp_clutter_testing.md](tamp_clutter_testing.md).

## Robots and scenes

The episode schema is robot-first: `robot` ∈ {`rby1`, `stretch`, `innate_mars`, `nori`},
and each episode carries a `scene_index` (MolmoSpaces iTHOR FloorPlan index) that
overrides the base sim config. Live kinematic latch on the robosuite server is opt-in
via `RobotSpec.advertise_kinematic_manip` (rby1, innate_mars, nori).

## Episode registry (large)

`scripts/generate_tamp_clutter_registry.py` writes
`configs/ovmm/clutter_episodes_large.yaml` with **200 deterministic templates**: rby1
across iTHOR train indices 0..21, Stretch, Innate Mars and Nori across 0..5, cleanup +
blocked nav-goal, and goal landmarks rotated across the set (sofa, fridge, table, bed, …)
plus `auto` samples.

```bash
uv run python scripts/generate_tamp_clutter_registry.py --rby1-scenes 22
uv run python scripts/eval_tamp_clutter.py \
  --episodes configs/ovmm/clutter_episodes_large.yaml --dry-run
```

## Run

```bash
# Unit tests (no GPU)
uv run emet test src/test/eval/test_tamp_clutter_config.py -q

# Dry run (config matrix only)
uv run python scripts/eval_tamp_clutter.py --dry-run

# Fast gate (rby1 iTHOR, N=3 cleanup)
uv run python scripts/eval_tamp_clutter.py --smoke

# Large registry (all robots; prefer emet jobs for GPU)
NEED_MIB=8000 uv run emet jobs run --name tamp-clutter --need-mib 8000 -- \
  uv run python scripts/eval_tamp_clutter.py --episodes configs/ovmm/clutter_episodes_large.yaml

# Problem-set build: resolve scatter + validity probe into a snapshot YAML
uv run python scripts/eval_tamp_clutter.py --generate --output-dir ~/runs/emet/tamp_clutter/gen

# Visualize (GUI): stream the scene + scattered objects + robot to the Rerun viewer
# (http://localhost:9090?url=ws://localhost:9877) — any robot (rby1, innate_mars, nori,
# stretch) and any scene index.
uv run python scripts/eval_tamp_clutter.py --episode-id ithor_cleanup_s1_bin_n3 --rerun
```

Output: per-episode JSON + `aggregate_tamp_clutter.csv` under `EMET_TAMP_CLUTTER_OUTPUT`
(default `~/runs/emet/tamp_clutter`). The runner prints `scored=` / `skipped_invalid=`.

## Remaining experiments (not merge blockers)

Queue on GPU via `emet jobs` (never as an agent turn). Full checklist: `TODO.md` (TAMP clutter).

```bash
# Re-run GT+MCTS battery after chord-collision / reachable-landmark (old 24/24 is 2026-08-28)
NEED_MIB=8000 uv run emet jobs run --name tamp-gt-battery --need-mib 8000 -- \
  uv run python scripts/eval_tamp_clutter.py --test-battery \
  --battery-robots nori --battery-scenes 0,1

# Large 200-episode registry
NEED_MIB=8000 uv run emet jobs run --name tamp-clutter --need-mib 8000 -- \
  uv run python scripts/eval_tamp_clutter.py \
  --episodes configs/ovmm/clutter_episodes_large.yaml

# rby1 latch smoke (small YAML)
NEED_MIB=8000 uv run emet jobs run --name tamp-latch-rby1 --need-mib 8000 -- \
  uv run python scripts/eval_tamp_clutter.py --episode-id ithor_cleanup_s1_bin_n3
```

Fill `tab:tamp_clutter` from `aggregate_tamp_clutter.csv` (scored denominator, drop
`skipped_invalid`).

## LLM-agent mode

The deterministic chain (`emet.controller.task.tamp.clutter_chain.plan_clear_clutter`)
is the reproducible baseline. An LLM-agent driver parses the command
(`e.phrase()`) and resolves the landmark / bin via the CHAT skills (`scene_tasks`,
`plan_pick_place`, `execute_pick_place_plan`, `find_objects`) — see
`docs/plans/2026-08-22_tamp_agent_tools.md` for the tool seam.

## Tests

```bash
uv run emet test src/test/eval/test_tamp_clutter_config.py -q
```

See `docs/paper_benchmarks.md` and `docs/experiments/README.md` for the experiment index.
