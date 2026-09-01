# Full OVMM benchmark (FindObj + Pick + FindRec + Place)

Extends the [find-phase harness](ovmm_find_phase_benchmark.md) with **Pick** and **Place** phases aligned with the [OVMM](https://ovmm.github.io/) task structure. Scoring uses MuJoCo `sim_object_placements` GT deltas (not official HSSD minival yet).

## Phases

| Phase | Metric | Find-phase equivalent |
|-------|--------|------------------------|
| FindObj | `find_object_success` | same |
| Pick | `pick_success` | GT: object left start recep or moved ≥ threshold |
| FindRec | `find_recep_success` | same |
| Place | `place_success` | GT: object within radius of goal recep **and closer than at manip start** (when `placements_before` is set) |
| Aggregate | `ovmm_full_partial` | mean of active phases (2 or 4) |
| Full task | `ovmm_full_success` | AND of all four (when manip enabled) |

## Manip modes

| Mode | Behavior |
|------|----------|
| `skip` | Find phases only (same as `eval_ovmm_find_phases.py`) |
| `oracle` (fallback) | Pick/place success copied from find success (harness smoke / upper bound); full runs use each episode's `manip_mode` unless `--manip-mode` overrides it |
| `sim` | MuJoCo freejoint teleport via ZMQ `sim_set_body_pose` (sim E2E) |
| `attempt` | AnyGrasp pick/place on real robot; **auto uses sim teleport when `is_simulation`** |
| `mcts` | **Kinematic arm** via [`plan_pick_place_mcts`](../src/emet/controller/task/tamp/task_search.py): distance-heuristic UCT search picks the (object, receptacle) assignment, then `execute_task_plan` drives approach → grasp → place. Scoring uses the same GT placement deltas as `sim`, so MCTS-vs-teleport is directly comparable. Requires a kinematic-capable server (`kinematic_manip: true`, e.g. rby1 / Galaxea). |

`attempt` needs a working AnyGrasp socket on real hardware. In sim, `attempt` and `sim` both use body teleport (no AnyGrasp). `mcts` is the arm-driving path; see the [MCTS/TAMP plan](../plans/2026-08-13_agent_mcts_tamp.md).

**Action-outcome ledger:** when `eqa.attempt_ledger` / `EMET_EQA_ATTEMPT_LEDGER` is on and the agent has `graph_memory`, scored pick/place phases in `emet.eval.ovmm_full` call `record_manip_attempt` (`pick` / `place` rows). Default **off**. See [attempt_ledger.md](attempt_ledger.md).

**Servers:** Stretch MuJoCo and robosuite (e.g. **rby1** / MolmoSpaces merges) advertise `capabilities.sim_set_body_pose`. Molmo iTHOR objects are freejoint roots (`…_1_0_0`) with mesh children (`…_1_1_0`); teleport resolves the freejoint ancestor when the GT body is the child.

### Floor pick/place (TAMP floor suite)

`full_episodes.yaml` has episodes with `floor_object: true`: the harness drops the GT object to the floor (`drop_object_to_floor`, z ≈ 0.02 m) before the pick phase, so the robot must **pick something off the floor** (or just find it while exploring) to complete the task. See the [experiment plan](../../docs/plans/2026-08-13_agent_mcts_tamp.md) and runner:

```bash
# Fast agent integration (~5–8 min; rby1 MCTS, dynagraph only):
uv run python scripts/eval_tamp_floor.py --smoke

# Floor-only TAMP suite (uses each episode's mcts/sim/skip mode; overnight)
NEED_MIB=8000 uv run emet jobs run --name tamp-floor-suite --need-mib 8000 -- \
  uv run python scripts/eval_tamp_floor.py --backend dynagraph
uv run python scripts/eval_tamp_floor.py --manip-mode sim    # override: teleport reference
```

Episodes: `robocasa_rby1_floor_to_counter_mcts` (MCTS kinematic; use for the
optional targeted OVMM smoke),
`robocasa_sourccey_floor_to_cab_sim` / `robocasa_stretch_floor_to_counter_sim` (teleport
references; Stretch agentic find is slow due to head sweeps),
`robocasa_floor_find_only_explore` (find-only explore; very slow on Stretch).

**Sourccey kinematic row** (`robocasa_sourccey_counter_to_cab_mcts`, `manip_mode: mcts`):
the updated official `sourccey-hardware` arm URDF reaches table/counter height but its
workspace bottoms out around z≈0.36 m, so floor objects are out of reach — use the
counter-top pick as the sourccey kinematic row, not a floor pick. Offline IK ranking
and execution share `write_offline_mjcf_base_xyt` (planar `RobotSpec.planar_base_joint_names`
or a freejoint); MCTS tries each distinct arm via `kinematic_arm_sides`.

The TAMP agent-tools gate's default `PROFILE=smoke` is the faster rby1 CHAT
kinematic contract test; run this floor smoke explicitly when changing
find-phase/OVMM behavior. `PROFILE=full` runs the four-episode matrix. See
[`scripts/run_tamp_agent_tools_gate.sh`](../scripts/run_tamp_agent_tools_gate.sh) and
[`docs/plans/2026-08-22_tamp_agent_tools.md`](../plans/2026-08-22_tamp_agent_tools.md).

### OVMM `--manip-mode` ≠ chat `agent.manip_mode`

These are **separate namespaces**. OVMM full scoring does **not** read chat-agent YAML or `EMET_MANIP_*`.

| Surface | Knob | Values | Who executes pick/place |
|---------|------|--------|-------------------------|
| OVMM full (`emet ovmm full` / `eval_ovmm_full.py`) | `--manip-mode` / `FindPhaseRunConfig.manip_mode` | `skip` \| `oracle` \| `sim` \| `attempt` \| `mcts` | Harness teleports GT bodies (`sim` / sim-`attempt`), drives the kinematic arm (`mcts`), or `agent.manipulate`/`place` on the find-phase **controller** (nonsim `attempt`) |
| Chat agent (`emet run agent`) | `agent.manip_mode` / `EMET_MANIP_MODE` | `teleport` \| `kinematic` | [`DynamemTaskExecutor`](../src/emet/controller/task/dynamem/dynamem_task.py) (env vars still override YAML) |

Wiring chat `agent.manip_*` into the executor (so operators need not set `EMET_MANIP_*`) does **not** change OVMM `--manip-mode sim` behavior. Chat **kinematic** IK/RRT is a different path; see [motion_planning.md](motion_planning.md#two-manip_mode-namespaces).

## Quick start

Prefer **`emet ovmm full`** (scripts remain thin wrappers). Multi-env paper path: `emet ovmm sweep --preset molmo-robocasa`.

```bash
uv run emet test src/test/memory/test_ovmm_full_metrics.py -q

# S0 distinct recep, oracle manip (fast GT smoke)
uv run emet ovmm full \
  --episodes configs/ovmm/full_episodes.yaml \
  --episode-id default_table_s0_distinct_recep \
  --backend ground_truth \
  --not-rotate --cpu-only \
  --manip-mode oracle \
  --output-dir ~/runs/emet/ovmm_full/smoke

# Sim E2E (find + sim pick/place; GPU for perception mapping)
uv run emet ovmm full \
  --episodes configs/ovmm/full_episodes.yaml \
  --episode-id default_table_s0_distinct_recep \
  --backend dynagraph \
  --manip-mode attempt \
  --output-dir ~/runs/emet/ovmm_full/e2e

# MolmoSpaces iTHOR + rby1 (teleport manip; needs .venv-molmospaces + iTHOR assets)
uv run emet ovmm full \
  --episodes configs/ovmm/full_episodes.yaml \
  --episode-id molmo_ithor_rby1_s2_bowl_pp \
  --backend ground_truth \
  --not-rotate --cpu-only \
  --manip-mode sim \
  --output-dir ~/runs/emet/ovmm_full/molmo_rby1_smoke
```

Episodes: `configs/ovmm/full_episodes.yaml`. Outputs default to `~/runs/emet/ovmm_full` (`EMET_OVMM_OUTPUT_FULL` or `configs/ovmm/benchmark.yaml`).

CI / local smoke (find-phase + full oracle + unit tests):

```bash
uv run python scripts/smoke_ovmm_benchmark.py --skip-habitat
```

## Relation to find-phase

- Shared runner: `run_episode_find_phase()` with `FindPhaseRunConfig.manip_mode != "skip"`.
- Fair-default flags (`use_sensor_perception`, `prefer_voxel`, timing split) apply unchanged.
- Habitat full OVMM (HSSD minival) is not wired; use Habitat find-phase proxy for memory-only ablations.

See also: [paper_benchmarks.md](paper_benchmarks.md).
