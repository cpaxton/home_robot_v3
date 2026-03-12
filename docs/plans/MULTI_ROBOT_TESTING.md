# Multi-Robot Support — Testing Plan

## What changed

This branch adds support for multiple robots beyond Stretch:

| Component | File(s) | Description |
|-----------|---------|-------------|
| **Rename** | `zmq_client.py`, 31 consumer files | `HomeRobotZmqClient` → `StretchZmqClient` (backward-compat alias kept) |
| **RobotSpec extensions** | `robots/base.py` | Added `mjcf_path`, `actuator_names`, `base_link_name` fields |
| **Galaxea R1 MJCF** | `assets/robot/galaxea_r1/` | Hand-tuned MJCF with 26 actuators, 34 meshes, camera sites |
| **GalaxeaR1Backend** | `robots/galaxea_r1/__init__.py` | Full `RobotSpec`, joint/actuator mapping, camera config |
| **GenericZmqClient** | `controller/generic_zmq_client.py` | Robot-agnostic ZMQ client driven by `RobotSpec` |
| **RobosuiteZmqServer** | `simulation/robosuite_server.py` | ZMQ server wrapping MuJoCo directly for non-Stretch robots |
| **Parameterized robocasa_gen** | `simulation/stretch_mujoco/robocasa_gen.py` | `robot=` param; non-Stretch keeps robosuite robot in scene |
| **CLI `--robot` flag** | `cli.py`, `simulation/mujoco_server.py` | `emet serve mujoco --robot galaxea_r1` etc. |

## Test matrix

### Unit / smoke tests (no sim required)

All in `src/test/simulation/test_multi_robot.py`. Run with:

```bash
uv run python -m pytest src/test/simulation/test_multi_robot.py -v
```

| Test | What it checks |
|------|---------------|
| `test_galaxea_r1_spec` | `GalaxeaR1Backend.get_spec()` returns correct DOF (26), joint count, actuator count, camera names, MJCF path, footprint |
| `test_stretch_spec` | `StretchBackend.get_spec()` still works (no regression) |
| `test_robot_registry` | `ROBOT_REGISTRY` contains `stretch` and `galaxea_r1` |
| `test_galaxea_r1_mjcf_loads` | MJCF loads in MuJoCo, has 33 qpos / 26 actuators, all actuator names resolve, sim steps without error |
| `test_generic_zmq_client_import` | `GenericZmqClient` imports cleanly |
| `test_stretch_zmq_client_backward_compat` | `HomeRobotZmqClient` alias resolves to `StretchZmqClient` |
| `test_robosuite_server_import` | `RobosuiteZmqServer` imports cleanly |
| `test_robocasa_gen_robot_param` | `model_generation_wizard` signature includes `robot` parameter |
| `test_robot_spec_new_fields` | `RobotSpec` accepts and stores `mjcf_path`, `actuator_names`, `base_link_name` |
| `test_robocasa_gen_robosuite_mapping` | `_robosuite_robot_for()` maps `stretch` → `PandaMobile`, `PandaOmron` → `PandaOmron`, etc. |

### Integration tests (require sim extra + MuJoCo)

These are manual for now. They require `emet install sim` and `emet sync -e sim`.

#### 1. Stretch (regression — should still work exactly as before)

```bash
# Terminal 1: start server
emet serve mujoco

# Terminal 2: run dynamem
emet run dynamem -S

# Terminal 3: run agent
emet run agent --debug
```

Expected: Stretch spawns in default scene with red cylinder + blue cube. Agent can scan and find objects. No regressions from the rename.

#### 2. Stretch + Robocasa (regression)

```bash
# Terminal 1
emet serve robocasa

# Terminal 2
emet run dynamem -S
```

Expected: Robocasa kitchen scene with Stretch. Same behavior as before.

#### 3. Robosuite-native robot in Robocasa (new)

```bash
# PandaOmron
emet serve robocasa --robot PandaOmron

# Tiago
emet serve robocasa --robot Tiago

# GR1
emet serve robocasa --robot GR1
```

Expected: Server starts, robosuite robot appears in the Robocasa kitchen scene (not stripped and replaced). The server publishes observations over ZMQ. A `GenericZmqClient` can connect and receive joint states.

#### 4. Galaxea R1 standalone (new)

```bash
# Start R1 server with its MJCF (no robocasa scene yet)
emet serve mujoco --robot galaxea_r1
```

Expected: `RobosuiteZmqServer` starts with the R1 MJCF. The robot has 26 actuators. Joint states are published. A `GenericZmqClient` can connect.

**Note**: The Galaxea R1 does not yet have a Robocasa integration (it's not a robosuite-registered robot), so `emet serve robocasa --robot galaxea_r1` will need the R1 MJCF to be manually composed into a kitchen scene. This is future work.

#### 5. GenericZmqClient round-trip (new)

```python
# After starting any non-Stretch server:
from emet.robots.galaxea_r1 import GalaxeaR1Backend
from emet.controller.generic_zmq_client import GenericZmqClient

spec = GalaxeaR1Backend().get_spec()
client = GenericZmqClient(robot_spec=spec, robot_ip="127.0.0.1")
client.wait_for_obs(timeout=10)
q = client.get_joint_positions()
print(f"Joint positions ({len(q)} DOF): {q}")
xyt = client.get_base_pose()
print(f"Base pose: {xyt}")
```

Expected: Client connects, receives observations, joint positions have correct dimensionality (26 for R1).

## Automated CI tests

The 10 unit tests in `test_multi_robot.py` run in the standard test suite:

```bash
RUN_SIM_TESTS=0 uv run python -m pytest src/test/ -v --timeout=60
```

They do **not** require a running server or the sim extra (except `test_galaxea_r1_mjcf_loads` which needs `mujoco`).

## Known limitations / future work

1. **Galaxea R1 kinematic model**: `create_model()` raises `NotImplementedError`. IK must use MuJoCo-based or third-party solver.
2. **Navigation for non-Stretch**: `RobosuiteZmqServer` logs but does not yet implement `xyt` navigation goals. The velocity controller is Stretch-specific.
3. **Camera rendering**: `RobosuiteZmqServer` renders from MuJoCo camera sites. Camera intrinsics are computed from `fovy`; real camera calibration data is not yet available for R1.
4. **Robocasa + R1**: The R1 is not registered in robosuite's `ROBOT_CLASS_MAPPING`, so it can't be used as a robosuite robot in Robocasa scenes yet. This would require either registering it or using the strip-and-replace approach (like Stretch).
5. **DynaMem / agent integration**: The agent loop and DynaMem controller still assume Stretch. Running `emet run agent --robot galaxea_r1` requires further work to make the controller robot-agnostic.
