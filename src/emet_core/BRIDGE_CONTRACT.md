# Bridge contract (ZMQ server)

Any **robot bridge** or **simulator bridge** that talks to the Emet agent over ZMQ must implement the same contract. Robot bridges: **Stretch** → `stretch_ros2_bridge`, **Innate Mars** → `innate_mars_bridge`. Simulator example: MuJoCo server. This keeps the agent backend-agnostic.

## Base class

Subclass `emet.core.server.BaseZmqServer` and implement the abstract methods.

## Ports (defaults)

- **send_port** 4401: full observations (get_full_observation_message)
- **recv_port** 4402: actions (handle_action)
- **send_state_port** 4403: lightweight state (get_state_message)
- **send_servo_port** 4404: servo/visual stream (get_servo_message)
- **send_h264_port** 4405 (optional): H.264 NAL side channel when `EMET_ZMQ_H264=1` (Innate Mars)

## Publish rate caps (optional)

Set maximum publish rates via env (not floors): `EMET_ZMQ_FULL_HZ`, `EMET_ZMQ_STATE_HZ`, `EMET_ZMQ_SERVO_HZ`. Unset = legacy unthrottled loop.

## Image wire format (Innate Mars defaults)

- **4401** full obs: half-res JPEG (default `EMET_ZMQ_IMAGE_SCALING=0.5`) with scaled intrinsics; optional metadata-only when `EMET_ZMQ_OBS_INCLUDE_IMAGES=0`.
- **4404** servo: scaled JPEG preview when full obs omits images or when `EMET_ZMQ_SERVO_INCLUDE_IMAGES=1`; Mars launch defaults to **no** servo JPEG when 4401 carries images (`EMET_ZMQ_SERVO_INCLUDE_IMAGES=0`) to avoid double encode on Jetson.
- Optional codecs: `EMET_ZMQ_WEBP_IMAGES`, `EMET_ZMQ_TURBOJPEG`, `EMET_ZMQ_JPEG_QUALITY`.
- RTSP side channel: `EMET_MARS_VIDEO_RTSP=1` advertises `capabilities.video_streams` (viewer: `emet comm video`).
- Metadata-only mode: `capabilities.zmq_obs_metadata_only`, `capabilities.zmq_images_on_port` (4404); clients merge servo JPEG into full obs (`merge_servo_images_into_full_obs`).

## Required methods

| Method | Returns | Purpose |
|--------|---------|--------|
| `get_control_mode()` | `str` | `"manipulation"`, `"navigation"`, or `"none"` |
| `get_full_observation_message()` | `dict` | Full obs (images, pose, joints, etc.) for logging / heavy clients |
| `get_state_message()` | `dict` | Small state (pose, joints, at_goal, etc.) for fast loop |
| `get_servo_message()` | `dict` | Images + poses + robot config for policy/visual servoing |
| `handle_action(action: dict)` | None | Execute one action from the client |
| `is_running()` | `bool` | True while the server should keep running |

## Message shapes (conventions)

Clients expect these keys when present. All bridges should use the same keys so the agent can treat backends interchangeably.

### get_state_message()

- `base_pose`: (x, y, theta) or 4x4 matrix
- `ee_pose`: 4x4 end-effector pose in world/map frame
- `joint_positions`: 1D array
- `joint_velocities`: 1D array
- `control_mode`: str
- `step`: int (optional)
- `at_goal`: bool (optional)
- `is_homed` / `is_runstopped`: bool (optional)

### get_servo_message()

- `ee/pose`: 4x4
- `robot/config`: joint position array
- `step`: int
- `head_cam/color_image`: compressed (e.g. jpg bytes)
- `head_cam/color_camera_K`: 3x3 numpy
- `head_cam/pose`: 4x4
- `head_cam/color_image/shape`: (H, W, C)
- `head_cam/image_scaling`: float
- Same pattern for `head_cam_left/`, `head_cam_right/`, `ee_cam/` when the robot has multiple head cams or an EE camera.
- Depth keys (`head_cam/depth_image`, `head_cam/depth_camera_K`, etc.) when depth is available.

### get_full_observation_message()

- At least: images (e.g. `rgb`, `depth` or per-cam), `camera_pose` or per-cam poses, `joint`, `joint_velocities`, `ee_pose`, `base_pose` or `gps`/`compass`, `step`, `recv_address`.
- Onboard perception fields (optional, per robot):
  - `depth`: JPEG2000 (JP2) bytes of meter depth ×1000 as uint16 (Innate Mars onboard DA3; `EMET_MARS_ONBOARD_DA3=1`).
  - `dinov3_head`: `list[float]` head-left DINOv3 vits16 embedding (Innate Mars onboard; `EMET_MARS_ONBOARD_DINOV3=1`). Optional — clients must tolerate absence.
- `lidar_points` / `lidar_timestamp`: optional 2D scan as N×2 **float32** array + int timestamp (Stretch hardware, Innate Mars `/scan`, innate_mars MuJoCo sim). Servers should publish float32 (`slim_zmq_obs_lidar`); clients accept float32 or legacy float64.
- Slim image wire format (`capabilities.zmq_obs_slim`): publish canonical keys only (`head_cam_left/image`, `head_cam_right/image`, `ee_cam/image`); clients expand legacy aliases (`rgb`, `rgb_right`, `rgb_tertiary`) via `emet.core.zmq_obs_codec`.
- Capability flags are advertised in `session["capabilities"]` (see `docs/zmq_session_metadata.md`): e.g. `onboard_da3`, `onboard_dinov3`, `zmq_obs_slim`, `teleport_base`.

### handle_action(action)

- Common keys: `xyt`, `nav_relative`, `nav_blocking`, `joint`, `gripper`, `head_to`, `posture`, `control_mode`, `say` / `say_sync`, `base_velocity`. Bridges ignore unsupported keys.

## Adding a new backend

1. Depend only on **emet-core** (and your backend SDK: ROS2, mujoco, etc.).
2. Subclass `BaseZmqServer`, implement the six methods above.
3. Map your backend’s observations into the same dict keys; map action keys to your backend’s commands.
4. No need for torch, perception, or LLM in the bridge process.
