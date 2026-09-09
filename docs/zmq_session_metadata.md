# ZMQ `emet_session` metadata

Every full-observation, state, and (where applicable) servo message from Emet’s MuJoCo ZMQ servers includes a static dict under the key `emet_session`. Clients should treat unknown keys as forward-compatible and must not assume the block is present (older servers).

## Required shape (schema version 1)

| Key | Type | Meaning |
|-----|------|--------|
| `schema_version` | `int` | Must be `1` for this document. |
| `runtime_kind` | `str` | e.g. `robosuite_sim`, `stretch_mujoco_sim`; hardware bridges should use a distinct value (e.g. `hardware_stretch`). |
| `is_simulation` | `bool` | Same meaning as the top-level `is_simulation` flag on observations. |
| `emet_robot_id` | `str` | Duplicate of top-level `emet_robot_id` for a single merge point. |
| `capabilities` | `dict` | Declared features, e.g. `teleport_base` (bool), `depth`, `num_cameras`, `dof`, `zmq_obs_slim` (bool, Innate Mars: one JPEG per camera on the wire), `zmq_lidar_f32` (bool, `lidar_points` published as float32 N×2), `zmq_image_scaling` (float), `zmq_obs_metadata_only` (bool, 4401 omits JPEG), `zmq_images_on_port` (int, usually 4404), `video_streams` (dict name → RTSP URL), `zmq_video_h264` (bool), `zmq_h264_port` (int, default 4405), `zmq_webp_images` (bool). Innate Mars hardware bridge also advertises `onboard_da3` and `onboard_dinov3` (bool) when the corresponding `--onboard-*` flags are set. |
| `environment` | `dict` | Optional scene identity; see below. |

Optional: `mjcf_model_name`, `scene_source_basename` (short debugging strings).

### `sim_object_placements` (simulation only)

When the MuJoCo server knows object poses at load time, it may include:

```json
"sim_object_placements": {
  "apple_main": {"cat": "apple", "pos": [0.1, -0.5, 0.9], "quat": [1, 0, 0, 0]},
  "object2": {"cat": "red cylinder", "pos": [0.08, -0.55, 0.6], "quat": [1, 0, 0, 0]}
},
"sim_object_placements_frame": "mujoco_world"
```

- **Robocasa**: from the scene wizard (`object_placements_info`), **pos refreshed from MuJoCo `body_xpos` at server start** when the sim model is loaded.
- **Full scene (Robocasa)**: fixture-level scan (sink, counter, cabinets, …) merged with wizard manipulable objects; each entry may include world **axis-aligned `bounds`** `[[min_xyz],[max_xyz]]` from MuJoCo mesh/collision geoms.
- **Default table scene**: fixed table + red cylinder + blue cube from `scene_environment.xml`, **overlaid with live body poses** when the server has `model`/`data`.
- **MolmoSpaces merged MJCF**: body scan at server start (non-robot bodies with geoms; capped count on iTHOR-scale scenes).

**Frame:** every `pos` is **absolute MuJoCo world XYZ** (meters). Same as `camera_pose`. On Robosuite servers, `gps`/`compass` are **episode-relative**; use `navigation_origin_xyt` + `nav_xyt_to_world_xyt` for robot base in world. Do **not** subtract gps from GT positions.

Clients use this for Dynagraph **`--ground-truth`** mode and dev alignment checks (`mujoco_align`). Keys starting with `_emet_` are internal (e.g. spawn hints) and are not graph objects.

**Limitations:** placements are fixed at server start (not live physics). See [dynagraph.md — Ground-truth limitations](dynagraph.md#limitations).

## `environment` kinds

- `molmospaces`: `scene`, `split`, `index` identify the dataset row used to build the merged MJCF.
- `robocasa`: `task`, `style`, `layout` when the Robocasa wizard was used.
- `default_table` / `stretch_default_scene`: generic packaged scenes.

## Hardware / third-party publishers

If you implement a bridge that speaks the same ZMQ observation protocol:

1. Set `is_simulation` to `false` and pick a `runtime_kind` that is not `*_sim`.
2. Populate `capabilities` honestly (e.g. `teleport_base: false` unless you implement teleport semantics).
3. Include `schema_version: 1` and `emet_robot_id` consistent with top-level `emet_robot_id`.
4. Omit sim-only top-level keys such as ``sim_to_real_ratio``.

## Top-level sim state: ``sim_to_real_ratio``

Stretch MuJoCo ZMQ **state** messages may include top-level ``sim_to_real_ratio`` (float): simulated seconds produced per wall-clock second. Clients use it to stretch motion-wait timeouts when the sim runs slower than real time (`motion_wait_timeout_scale` in [`zmq_protocol.py`](../src/emet/core/zmq_protocol.py)). Hardware and older servers omit the key; absence must leave real-robot wait budgets unchanged. Cap is ``EMET_ZMQ_SIM_WAIT_SCALE_MAX`` (10×).

## Client API

- `GenericZmqClient.get_emet_session()` and `StretchZmqClient.get_emet_session()` return a shallow copy of the latest block, or `None`.
