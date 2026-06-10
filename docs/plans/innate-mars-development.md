# Innate Mars development — resume notes

Last updated: 2026-06-03 (after bridge Nav2 + sim experiment matrix land on `main`).

## What landed

- **`innate_mars_bridge`:** ZMQ `xyt` → Nav2 `navigate_to_pose`; `at_goal` on state port; 10-DoF `joint_positions` (base + arm + gripper mimic).
- **Sim profiles:** [`configs/sim/innate_mars.yaml`](../../configs/sim/innate_mars.yaml), [`innate_mars_robocasa.yaml`](../../configs/sim/innate_mars_robocasa.yaml).
- **Docs:** [experiments/innate_mars.md](../experiments/innate_mars.md), [robots/innate_mars_hardware.md](../robots/innate_mars_hardware.md).
- **Audit:** `uv run python scripts/audit_innate_os_topics.py` (pin: innate-os `main`).
- **Dynagraph fusion default:** `attach_graph_object_fusion` loads `default_graph_object_fusion.yaml` when parameters omit fusion.
- **Hardware depth:** `dynav_innate_mars.yaml` → `da3_stereo: true`.

## Where to resume

### 1. Hardware validation (Mars not connected yet)

When the robot is on the network:

1. innate-os up (`maurice_nav` in navigation/mapfree mode).
2. `ros2 launch innate_mars_bridge server.launch.py`
3. `uv run python scripts/audit_innate_os_topics.py` — all expected `/mars/*` + `/odom` present.
4. Follow [innate_mars_hardware.md](../robots/innate_mars_hardware.md) through Dynagraph export.

**Open questions:** Confirm Nav2 goal frame (`map` vs `odom`) on real stack; tune `_pick_nav_frame()` in `remote/modules/nav.py` if goals are rejected.

### 2. innate-os sim harness (pre-hardware)

```bash
cd innate-os && ./innate setup && ./innate sim up
# colcon build innate_mars_bridge; ros2 launch innate_mars_bridge server.launch.py
EMET_ZMQ_STARTUP_TIMEOUT=120 uv run emet run dynamem --robot innate_mars \
  --robot-ip 127.0.0.1 --dynav-config dynav_innate_mars.yaml -S
```

Optional: `RUN_INNATE_OS_BRIDGE_TESTS=1` integration smoke (not wired in CI yet).

### 3. Sim experiment matrix — refresh baselines

Re-run and update [`dynagraph_eval_reference.json`](../../src/test/fixtures/baselines/dynagraph_eval_reference.json):

| Run | Command sketch |
|-----|----------------|
| Default table GT | `emet serve mujoco --sim-config configs/sim/innate_mars.yaml` + `emet run dynagraph --ground-truth` |
| Robocasa seed 0 | `--sim-config configs/sim/innate_mars_robocasa.yaml` + `emet eval-dynagraph` |
| Fusion cal | `./scripts/run_fusion_calibration_loop.sh innate_mars` |
| Short explore graph nodes | Re-check `graph.node_count` after fusion-default fix (was 0 on short smokes) |

Full matrix: [experiments/innate_mars.md](../experiments/innate_mars.md).

### 4. Follow-up code (not done)

| Item | Files / notes |
|------|----------------|
| Arm / head ZMQ actions | `handle_action`: `head_to`, posture — wire to `maurice_arm` if APIs exist |
| VPI depth on ZMQ | Subscribe `maurice_cam` depth topic; set `depth` in session when available |
| URDF ↔ innate-os drift | Periodic diff `maurice.urdf` vs innate-os `maurice_sim/urdf/` |
| EQA on Robocasa | `--question-file dynagraph_questions.yaml` + real LLM; add `eqa.accuracy` to reference JSON |
| `create_model` / IK | `InnateMarsBackend.create_model` still `NotImplementedError` |
| Pinocchio model | Only needed for manip-heavy experiments |

### 5. Tests to run before large experiments

```bash
uv run emet test src/test/bridge/ src/test/robots/test_innate_mars_backend.py -q
uv run emet test src/test/memory/test_graph_object_fusion_default_yaml.py -q
RUN_DA3_TESTS=1 uv run emet test src/test/mapping/test_innate_mars_da3_sim.py -q  # slow, GPU
```

## Paper alignment

- Emet sim spot-checks: [`paper/sections/04_experiments.tex`](../../paper/sections/04_experiments.tex), appendix [`02_innate_mars.tex`](../../paper/sections/appendix/02_innate_mars.tex).
- Cross-robot Robocasa: compare innate_mars vs stretch/rby1 in [`dynagraph_robocasa_e2e.md`](../dynagraph_robocasa_e2e.md).

## Related branches

- `feature/stretch-robocasa-robosuite` — Stretch on RobosuiteZmqServer (separate from Mars).
- `feature/dynagraph-robocasa-explore` — fusion calibration / eval-dynagraph benchmarks.
