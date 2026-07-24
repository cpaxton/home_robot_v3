# Innate Mars experiments (sim + hardware)

Paper-aligned reproduction for **Emet sim** (MuJoCo) and **real Mars** (innate-os + `innate_mars_bridge`). See also [README.md](README.md) (master experiment index), [dynagraph_benchmarks.md](../dynagraph_benchmarks.md), and [robots/innate_mars.md](../robots/innate_mars.md).

## Sim experiment matrix

Use **sensor depth** in MuJoCo (`dynav_config.yaml`). Use **`dynav_innate_mars.yaml`** only for hardware-like DA3 stacks.

| Experiment | Server | Agent | Eval |
|------------|--------|-------|------|
| Default table GT | `emet serve mujoco --sim-config configs/sim/innate_mars.yaml` | `emet run dynagraph --ground-truth --robot innate_mars --export /tmp/mars_gt` | `emet eval-dynagraph --episode /tmp/mars_gt` |
| Robocasa explore | `emet serve mujoco --sim-config configs/sim/innate_mars_robocasa.yaml` | `emet run dynagraph --robot innate_mars --export /tmp/mars_rc` | `emet eval-dynagraph --episode /tmp/mars_rc` |
| Fusion calibration | Robocasa server | `emet run dynagraph --calibration-export /tmp/mars_cal.jsonl --export /tmp/mars_cal` | `emet eval-calibration --gt … --frames /tmp/mars_cal.jsonl` |
| Fusion A/B | Robocasa | `./scripts/run_dynagraph_fusion_ab.sh innate_mars 0 20` | compare JSON outputs |
| DA3 (hardware-like) | Robocasa | `--dynav-config dynav_innate_mars.yaml` | compare vs sensor-depth run |
| EQA (optional LLM) | any | `--question-file dynagraph_questions.yaml` | `eqa.accuracy` in eval JSON |

### Quick commands

```bash
# Terminal 1 — default table
uv run emet serve mujoco --sim-config configs/sim/innate_mars.yaml

# Terminal 2 — GT graph smoke
uv run emet run dynagraph --robot innate_mars --robot-ip 127.0.0.1 \
  --ground-truth --export /tmp/mars_gt_default --no-rerun

uv run emet eval-dynagraph --episode /tmp/mars_gt_default
```

```bash
# Robocasa seed 0 (cross-robot parity)
uv run emet serve mujoco --sim-config configs/sim/innate_mars_robocasa.yaml

uv run emet run dynagraph --robot innate_mars --robot-ip 127.0.0.1 \
  --export /tmp/mars_robocasa_s0 --no-rerun --explore-max-iters 15

uv run emet eval-dynagraph --episode /tmp/mars_robocasa_s0
```

### Reference thresholds

See [`src/test/fixtures/baselines/dynagraph_eval_reference.json`](../../src/test/fixtures/baselines/dynagraph_eval_reference.json):

- Robocasa innate_mars: `explored_fraction_min` ≥ 0.08, fusion `spatial_recall_min` ≥ 0.5
- Default table GT: `gt_graph_completeness_min` ≥ 0.9
- Calibration JSONL: `spatial_recall` ≈ 1.0 (geometry), `label_recall` diagnostic

### CI smokes

```bash
uv run emet test src/test/robots/test_innate_mars_backend.py -q
uv run emet test src/test/bridge/test_innate_mars_bridge_joint_layout.py -q
uv run python scripts/audit_innate_os_topics.py --skip-ros
```

## Real hardware (innate-os + bridge)

See [hardware bring-up checklist](../robots/innate_mars_hardware.md).

### Sim vs hardware visualization (same CLI)

After `git pull origin main` and `uv sync`, use the **same Rerun tools** as MuJoCo sim. Only the ZMQ host and depth preset differ:

| Tool | Sim (localhost) | Hardware (`herman`) |
|------|-----------------|---------------------|
| Cameras montage | `emet preview-cameras --source local` | `emet preview-cameras --source zmq --connection herman` |
| One-shot map | `emet capture --backend voxel_only` | `emet capture --connection herman --backend voxel_only` |
| Live Rerun (cameras + MJCF mesh) | `emet stream --cameras-only` | `emet stream --cameras-only --connection herman` |
| Live voxel map (DynaMem) | `emet stream --backend dynamem` | `emet stream --connection herman` (remote default) |
| Live graph (Dynagraph) | `emet stream --backend dynagraph` | `emet stream --connection herman --backend dynagraph` |
| GraphEQA baseline | `emet stream --backend graph_eqa` | `emet stream --connection herman --backend graph_eqa` |
| Full session | `emet run dynamem --robot innate_mars` | `emet run dynamem --robot innate_mars --robot-ip herman --dynav-config dynav_innate_mars.yaml` |
| Graph export | `emet run dynagraph --export /tmp/mars` | same with `--robot-ip herman --dynav-config dynav_innate_mars.yaml` |
| Discord chat + explore | (sim: `--robot innate_mars --start-sim`) | `emet run agent --connection herman --config configs/agent_innate_mars.yaml --name Herman` (needs `DISCORD_TOKEN`; see [hardware Discord section](../robots/innate_mars_hardware.md#discord-chat--explore-herman)) |

**Depth:** Sim ZMQ carries sensor depth → default `dynav_config.yaml`. Hardware has no ZMQ depth → `stream`/`capture` auto-select `dynav_innate_mars.yaml` (DA3 stereo) when the host is not localhost. Override with `--dynav-config` anytime.

**Connection profile:** `emet connect save herman --user jetson1 --robot innate_mars` so `--connection herman` sets host + robot on `capture`, `stream`, and `preview-cameras`. Details: [zmq_obs.md](../zmq_obs.md).

Rerun viewer: `http://localhost:9090?url=ws://localhost:9877` (SSH tunnel ports 9090/9877 when working over VPN). See [rerun.md](../rerun.md).

### Stationary smoke (no base / head motion)

Use these when the robot should stay put (mapping from current pose only):

```bash
# Bridge (if not already up)
emet mars start --connection herman

# Cameras only
uv run emet preview-cameras --source zmq --connection herman

# One ZMQ frame + single voxel-only update (no nav commands)
uv run emet capture --connection herman --backend voxel_only --no-rerun

# Repeated stationary updates (no rotate, no explore, no xyt)
uv run emet stream --connection herman --backend voxel_only --max-steps 3 --headless

uv run emet stream --connection herman --backend dynagraph --max-steps 1 --headless
```

**Known issue:** stationary `dynagraph` stream can inflate graph node count (dedup fails with DA3 depth / label noise). See [known_issues.md](../known_issues.md#dynagraph-graph-node-explosion-on-stationary-hardware-stream). Prefer `--backend voxel_only` for voxel-only hardware smoke until fixed.

**Avoid (these move the head or base):**

| Command | Motion |
|---------|--------|
| `emet run dynamem` (default) | `rotate_in_place` at start |
| `emet run dynamem --manipulation-only` | skips rotate; still interactive |
| `emet run dynagraph` without `-N` | `rotate_in_place` at start |
| `emet run dynagraph --explore-loop` | frontier navigation (`xyt`) |
| `emet run dynagraph --calibration-export` + `--export` | calibration path forces rotate even with `-N` |

For interactive dynagraph without initial rotate: `emet run dynagraph -N …`. For export with updates and no rotate, prefer `emet stream --backend dynagraph --max-steps N` then export from a future `--input-path` workflow, or add `--calibration-steps` only after a code fix honors `-N` on the calibration path.

## innate-os sim harness (ROS fidelity)

Genesis + Docker ROS from [innate-os](https://github.com/innate-inc/innate-os) — validates **bridge topics/TF/Nav2**, not MuJoCo Dynagraph numbers.

```bash
# Host: innate-os sim
cd innate-os && ./innate setup && ./innate sim up

# Colcon workspace with symlink to src/innate_mars_bridge
colcon build --packages-select innate_mars_bridge && source install/setup.bash
ros2 launch innate_mars_bridge server.launch.py

# Audit topics (on robot/sim host with ROS sourced)
uv run python scripts/audit_innate_os_topics.py

# Emet client
EMET_ZMQ_STARTUP_TIMEOUT=120 uv run emet run dynamem --robot innate_mars \
  --robot-ip 127.0.0.1 --dynav-config dynav_innate_mars.yaml -S
```

**Smoke checklist:** head stereo in Rerun, odom updates, `xyt` nav goals accepted, `at_goal` on state port, voxel map grows.
