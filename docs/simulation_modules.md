# Simulation modules (maintainer reference)

Python modules added or extended for **Robocasa serve**, **MolmoSpaces spawn**, and **registry-robot ZMQ sim** (`RobosuiteZmqServer`). End users normally use [`emet serve`](cli.md); this page is for contributors tuning MJCF, assets, and load behavior.

Related user docs: [simulation.md](simulation.md), [molmospaces.md](molmospaces.md), [molmospaces_spawn_metadata.md](molmospaces_spawn_metadata.md).

## Overview

| Module | Purpose |
|--------|---------|
| [`fall_detection`](../src/emet/simulation/fall_detection.py) | Detect tipped/fallen base from MuJoCo `xmat` (body +Z · world +Z); red error in Stretch + Robosuite ZMQ servers. |
| [`mujoco_stationary_control`](../src/emet/simulation/mujoco_stationary_control.py) | Build full-length stationary `ctrl` from joint transmissions + spec hold buffer (used by Stretch and registry robots). |
| [`robosuite_stationary_control`](../src/emet/simulation/robosuite_stationary_control.py) | Re-export shim → use `mujoco_stationary_control`. |
| [`robosuite_load_utils`](../src/emet/simulation/robosuite_load_utils.py) | Post-load helpers for `RobosuiteZmqServer`: home keyframe, free-joint addrs, diagnostics. |
| [`mujoco_home_tune`](../src/emet/simulation/mujoco_home_tune.py) | Interactive GUI to tune robot **home** pose and print MJCF `<key ctrl="..."/>` snippets. |
| [`robocasa_assets_check`](../src/emet/simulation/robocasa_assets_check.py) | Preflight: basic fixtures, LightWheel registry, objaverse `reg_bbox` before `emet serve robocasa`. |
| [`robocasa_registry_sync`](../src/emet/simulation/robocasa_registry_sync.py) | Sync LightWheel meshes into `fixture_registry/*.yaml` (`fixtures/` + `objects/lightwheel/` accessories + cabinet door/handle mesh ids). |
| [`robocasa_objaverse_bbox`](../src/emet/simulation/robocasa_objaverse_bbox.py) | Post-process objaverse MJCFs to add `reg_bbox` geoms (wraps Robocasa `calc_object_bb_reg`). |
| [`molmospaces_spawn_metadata`](../src/emet/simulation/molmospaces_spawn_metadata.py) | Runtime load of per-robot `molmospaces_spawn.json`. |
| [`write_molmospaces_spawn_metadata`](../src/emet/app/write_molmospaces_spawn_metadata.py) | Offline measure + write spawn JSON (see [molmospaces_spawn_metadata.md](molmospaces_spawn_metadata.md)). |
| [`chase_camera`](../src/emet/simulation/chase_camera.py) | FREE chase cam off `base_link` for `--record-mp4` / `EMET_SIM_THIRD_PERSON` (raised lookat; avoids torso clip). Nadir overhead via `EMET_SIM_OVERHEAD`. |
| [`scene_base_spawn`](../src/emet/simulation/scene_base_spawn.py) | Re-export shim for spawn helpers (original import path). |
| [`spawn_geom`](../src/emet/simulation/spawn_geom.py) | Floor rays, collision clip, occupancy, exterior-tongue gates. |
| [`spawn_settle`](../src/emet/simulation/spawn_settle.py) | Free-joint Z settle / restore / foot-clearance probes. |
| [`spawn_debug`](../src/emet/simulation/spawn_debug.py) | ASCII/PNG occupancy maps when `EMET_MOLMOSPACES_SPAWN_DEBUG` is on. |
| [`spawn_molmospaces`](../src/emet/simulation/spawn_molmospaces.py) | MolmoSpaces free-joint XY search (`find_molmospaces_freejoint_xyz`). |
| [`spawn_planar`](../src/emet/simulation/spawn_planar.py) | Planar slide+yaw spawn (`find_planar_base_xyt`, `write_planar_base_xyt`). |
| [`spawn_robocasa`](../src/emet/simulation/spawn_robocasa.py) | Robocasa free-joint autoplace (`find_robocasa_freejoint_xyz`). |

Runtime wiring: [`robosuite_server.py`](../src/emet/simulation/robosuite_server.py), [`mujoco_serve_argv.py`](../src/emet/simulation/mujoco_serve_argv.py), [`scene_base_spawn.py`](../src/emet/simulation/scene_base_spawn.py) (re-exports `spawn_geom` / `spawn_settle` / `spawn_debug` / `spawn_molmospaces` / `spawn_planar` / `spawn_robocasa`).

---

## MuJoCo home pose tuning (`mujoco_home_tune`)

Tune the **home** keyframe for registry robots (Galaxea R1 / rby1, innate_mars, etc.): open a MuJoCo viewer, pose the robot, close the window, copy the printed `<key name="home" ctrl="..."/>` line into the robot MJCF.

Uses [`compute_stationary_ctrl_vector`](../src/emet/simulation/mujoco_stationary_control.py) and [`robosuite_load_utils`](../src/emet/simulation/robosuite_load_utils.py) for initial pose (`default` / `zeros` / `home`).

**Requires a display** (not for headless CI).

```bash
# Physics Simulate (full dynamics UI)
uv run python -m emet.simulation.mujoco_home_tune \
  src/emet/assets/robot/galaxea_r1/galaxea_r1.xml \
  --base-body base_link

# Kinematic viewer only (no mj_step; drag joints safely)
uv run python -m emet.simulation.mujoco_home_tune path/to/merged.xml \
  --kinematic --initial-pose home

uv run python -m emet.simulation.mujoco_home_tune --help
```

| Flag | Meaning |
|------|---------|
| `mjcf` | Robot or merged scene MJCF path (positional). |
| `--base-body` | Free-joint base body (default `base_link`). |
| `--initial-pose` | `default`, `zeros`, or `home` before tuning. |
| `--tune-base-z` | Hoist floating base to this world Z before freezing (default `0.38`). |
| `--kinematic` | Passive viewer + `mj_forward` only (no physics). |

Programmatic API: `build_tune_model()`, `run_tune_home_gui()`, `print_home_keyframe_snippet()`.

Tests: [`test_robosuite_load_utils.py`](../src/test/config/test_robosuite_load_utils.py) (`build_tune_model`, `format_key_ctrl_attr`).

---

## Stationary control (`mujoco_stationary_control`)

When the ZMQ server steps physics, non-nav joints need a **stationary** `ctrl` vector so arms/torso do not collapse. This module:

- Fills `model.nu` from actuator transmissions and current `qpos`
- Zeros wheel / velocity actuators
- Overlays `RobotSpec` hold targets from `RobosuiteZmqServer`

Per-robot policies: [`galaxea_r1/sim_stationary.py`](../src/emet/robots/galaxea_r1/sim_stationary.py), [`stretch/sim_stationary.py`](../src/emet/robots/stretch/sim_stationary.py).

---

## Robosuite load helpers (`robosuite_load_utils`)

Used during `RobosuiteZmqServer` startup:

| Function | Role |
|----------|------|
| `apply_home_keyframe_preserving_base` | Apply MJCF `home` keyframe; keep merged-scene base pose and **scene freejoints** (table cubes) when Molmo autoplace ran. |
| `apply_home_keyframe_preserving_planar_base` | Same for planar `base_x/y/yaw` robots (Sourccey / xlerobot); restores object `<freejoint/>` qpos after `mj_resetDataKeyframe`. |
| `apply_zero_joint_pose_preserving_base` | Zero articulated joints; preserve base. |
| `freejoint_qpos_qvel_addrs` | Locate free joint on `base_link` (Molmo merge). |
| `log_post_load_diagnostics` | Optional velocity / contact logging (`EMET_ROBOSUITE_POST_LOAD_DEBUG=1`; see [environment_variables.md](environment_variables.md)). |
| `probe_max_qvel_unforced_steps` | Short dynamics probe after load. |

---

## Robocasa asset tooling

### Preflight (`robocasa_assets_check`)

`emet serve robocasa` calls these before building a kitchen:

- `basic_fixtures_present` — base fixtures zip extracted
- `lightwheel_registry_ok` — `fixtures_lw` + `Sink025` in registry
- `fixture_registry_layout_ok` — required registry YAML stems exist
- `objaverse_reg_bbox_ok` — objaverse models have `reg_bbox` geoms

### Registry sync (`robocasa_registry_sync`)

The `fixtures_lw` zip adds meshes but often **replaces** `fixture_registry/` with a slim subset. Restore vendored YAML from git, then sync LightWheel entries:

```bash
git -C third_party/robocasa checkout -- robocasa/models/assets/fixtures/fixture_registry/
uv run python scripts/sync_robocasa_lightwheel_registry.py
```

Module: `sync_lightwheel_fixture_registry()`, `missing_required_registry_stems()`.

### Objaverse bbox (`robocasa_objaverse_bbox`)

One-time post-process after objaverse download:

```bash
uv run python scripts/process_robocasa_objaverse_reg_bbox.py
```

Wraps Robocasa `calc_object_bb_reg` over `third_party/robocasa/.../objaverse/`. `download_robocasa_assets.py` runs this automatically when objaverse is fetched.

---

## MolmoSpaces spawn metadata

Runtime: [`molmospaces_spawn_metadata.py`](../src/emet/simulation/molmospaces_spawn_metadata.py) loads optional `molmospaces_spawn.json` next to each robot MJCF.

Offline measurement: **`emet molmospaces write-spawn-metadata`** — full guide in [molmospaces_spawn_metadata.md](molmospaces_spawn_metadata.md).

---

## Tests

| Test file | Covers |
|-----------|--------|
| [`test_robosuite_load_utils.py`](../src/test/config/test_robosuite_load_utils.py) | Load utils, home keyframe, `mujoco_home_tune` |
| [`test_robocasa_assets_check.py`](../src/test/config/test_robocasa_assets_check.py) | Asset preflight helpers |
| [`test_molmospaces_spawn_metadata.py`](../src/test/config/test_molmospaces_spawn_metadata.py) | Spawn JSON load + `resolve_serve_robot` |

```bash
uv run emet test src/test/config/test_robosuite_load_utils.py \
  src/test/config/test_robocasa_assets_check.py \
  src/test/config/test_molmospaces_spawn_metadata.py -q
```

### Observed manipulation camera evidence

[`rgb_encoding.py`](../src/emet/simulation/rgb_encoding.py) selects the explicit
Stretch full/servo RGB wire profile: legacy JPEG by default, lossless PNG with
`EMET_SIM_RGB_CODEC=png`. The existing client decoder accepts both. The
[`rgbd_registration.py`](../src/emet/core/rgbd_registration.py) JSON sensor packet
preserves original RGB/raw-depth grids, aligned depth, calibration and acquisition
intervals using lossless compressed bytes. `EMET_SIM_RGBD_PROVENANCE=1` enables it
in the head camera manager; the status object invalidates it when either head
channel changes. The Stretch full observation and `Observations` carry it without
retiming. Reprojection is validated before diagnostic use; ownership is not inferred.

`stretch_mujoco/mujoco_server_camera_manager.py` and
`stretch_mujoco/datamodels/status_stretch_camera.py` preserve per-camera acquisition
intervals in asynchronous mode, with capture under the shared physics/render lock. RGB metadata
also identifies its separate depth interval; publication does not renew either.
The local clock identity permits explicit same-host simulation mapping through
`controller/manipulation/capture_clock.py`, not a hardware synchronization claim.
`stretch_mujoco/enums/stretch_cameras.py` supports the opt-in head image scale and
self-occlusion controls documented in [environment variables](environment_variables.md).
See [live source assembly evidence](experiments/observed_source_assembly_20261027.md).

Headless Stretch defaults to `EMET_SIM_SYNCHRONIZED_RGBD=1`.
`synchronized_capture.py` copies the physics state under that lock and renders a
private sample while physics continues. The same flag can opt the interactive
viewer in. Set `EMET_SIM_SYNCHRONIZED_RGBD=0` to use asynchronous rendering.
Camera poses and eight Stretch motor readings describe the same original sample.
A separate full-robot configuration channel is explicitly privileged kinematic
debug truth, including unmeasured passive joints; observed effects reject it.
The motor-only rigid EE estimator does not assume passive-angle sensors or
qualify fingertip/individual-telescope collision geometry. The fixture preparation
script emits separate robot-only/scene XML and a JSON launch/manifest pair; see
[the encoder/sampling evidence and rerun instructions](experiments/observed_encoder_sampling_20261029.md).

Simulated head and wrist intrinsics use their rendered pinhole FOV, dimensions
and pixel-center convention. In particular, `mujoco_server_stretch.py` publishes
480×270 wrist `ee_K`, matching the camera manager, rather than using physical
focal calibration or 1280×720 sensor metadata. Actual-model tests compare both
with MuJoCo's GL frustum. Physical camera settings are unchanged.
Pinhole `cam_intrinsic` and `cam_resolution` metadata use that same render
FOV/resolution rather than a conflicting physical focal calibration. Cameras
with declared physical sensor size retain sensor-unit conversion.
The paired
`scripts/eval_stretch_multiview_masks.py --output DIR` diagnostic renders head and
wrist from one private physics snapshot and runs independent RGBD tracking during
the privileged native control. It uses CPU OSMesa via `emet jobs`, no VLM. See
[tracking profiles, attempts and limits](experiments/observed_handoff_20261030.md).
`depth_registration.register_depth(..., return_provenance=True)` additionally
reports the original depth pixel and continuous RGB projection selected by the
nearest-sample z buffer. The default aligned-depth result is unchanged. The
paired diagnostic archives those maps, raw depth/calibration and eight motor
channels from the same frozen sample, with privileged camera localization labelled
separately. `robot_kinematic_signature` binds named tendon terms and coefficients
as well as rigid joint/camera calibration. See the
[capture/whole-robot progress record](experiments/observed_transfer_progress_20261116.md);
neither reprojection nor outer robot geometry establishes endpoint ownership.

The paired script also accepts `--stationary-captures N` (zero/default runs the
native control; otherwise N must be at least two), `--head-pan` (default -1.57),
`--head-tilt` (default -0.7), and `--robot-only-model PATH`. The last explicitly
enables rejection of possible rigid-tool interference using original depth rays;
it does not certify self ownership. Stationary mode scores no transfer.

`scripts/eval_continuous_rgbd_source.py --sim CONFIG --output DIR` runs a live ZMQ
head-camera producer while its consumer pauses. `--consumer-pause-s` defaults to
3.0 and must exceed the unchanged 1.5-second tracking gap; `--rounds` defaults to
2; `--head-tilt` defaults to -0.7. It uses an explicit operator colour seed and
known cylinder prior, eight same-sample motors and simulator camera localization.
This diagnostic does not install `RobotViewGuard`; optional original registration
provenance is archived separately with the flag below. Input archives, source
hashes, capture intervals, dispatch audit and failure feedback are retained.
It uses CPU OSMesa via managed jobs and requires no VLM. This is a stationary
candidate-source check, not observed pick/place or hardware acceptance.

The live script now accepts `--rgb-codec jpeg|png` (default JPEG) and
`--registration-audit` (default off). The latter archives the optional original
sensor packet alongside ordinary wire pixels; tracking continues to use the
ordinary decoded observation. `scripts/eval_rgbd_transport.py --recording DIR
--output FILE` compares original/wire RGB and depth from the same first two
captures, keeping model, seed pixels and tracker thresholds fixed. It rejects
bounded archive tails that no longer contain the original reference capture.
See [the transport cause and live checks](experiments/observed_rgbd_transport_20261117.md).

`eval_continuous_rgbd_source.py` also accepts the pair `--robot-only-model PATH`
and `--geometry-profile JSON`. They retain same-capture camera/encoder/gripper
geometry and assemble two visual-effect frames per successful pause window.
Both flags are required together and default off. The supplied profile is
simulation-only declared error/model assumptions; it does not enable execution.
The source never substitutes a later encoder read or treats passive joints as
measured. See [the coherent-frame audit](experiments/observed_visual_frames_20261117.md).
