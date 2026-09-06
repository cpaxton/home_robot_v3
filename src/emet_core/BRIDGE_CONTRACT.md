# Bridge contract (ZMQ server)

Any **robot bridge** or **simulator bridge** that talks to the Emet agent over ZMQ must implement the same contract. Robot bridges: **Stretch** → `stretch_ros2_bridge`, **Innate Mars** → `innate_mars_bridge`. Simulator example: MuJoCo server. This keeps the agent backend-agnostic.

## Base class

Subclass `emet.core.server.BaseZmqServer` and implement the abstract methods.

## Ports (defaults)

- **send_port** 4401: full observations (get_full_observation_message)
- **recv_port** 4402: actions (handle_action)
- **send_state_port** 4403: lightweight state (get_state_message)
- **send_servo_port** 4404: servo/visual stream (get_servo_message)

## Required methods

| Method | Returns | Purpose |
|--------|---------|--------|
| `get_control_mode()` | `str` | `"manipulation"`, `"navigation"`, or `"none"` |
| `get_full_observation_message()` | `dict` | Full obs (images, pose, joints, etc.) for logging / heavy clients |
| `get_state_message()` | `dict` | Small state (pose, joints, at_goal, etc.) for fast loop |
| `get_servo_message()` | `dict` | Images + poses + robot config for policy/visual servoing |
| `handle_action(action: dict)` | None | Execute one action from the client |
| `is_running()` | `bool` | True while the server should keep running |
| `start_navigation_command(action)` | `dict` | Start once, without waiting for arrival; return frozen resolved goal, frame and actual motion mode |
| `navigation_command_result(context)` | `None` or `(status, result)` | Poll this goal, combining controller outcome with measured pose error |
| `cancel_navigation_command()` | `bool` | Bounded stop; return true only after confirming cancellation/rest |

## Command protocol v2

Both workstation clients require v2. Deploy core and the bridge together; there
is no legacy fallback. Each outgoing observation/state/servo dictionary contains
`command_protocol` (version and server boot ID), `command_receipts`, and the most
recent identity-scoped `command_error`. `step` is telemetry only; `at_goal` is a
diagnostic, not an acknowledgement or command-completion authority.

Actions include a `command` envelope with `version`, `server_boot_id`,
`client_session_id`, and nonnegative monotonic `sequence`. Retries reuse the same
identity **and** payload. JSON-normalized payloads are hashed. Identical retries
return the latest receipt without dispatch; conflicting, stale, or evicted
commands never execute. Relative goals are resolved once, at server dispatch.

Navigation is a standalone `xyt` command, with boolean `nav_relative`,
`nav_world`, `nav_teleport`, and `nav_blocking` flags and a positive finite
`nav_timeout_s`. Relative and world flags cannot both be true. Frames and actual
motion modes are recorded in receipts; a requested simulator teleport cannot
silently fall back to velocity driving. Hardware rejects teleport requests.

Receipts progress from accepted to running to succeeded/failed/cancelled;
busy commands are rejected. Terminal outcomes are immutable. Only one navigation
goal is active, and mutating posture/joint/velocity commands cannot interrupt it.
Use `cancel_navigation: {client_session_id: ..., sequence: ...}` to target that
goal. Expiry triggers cancellation even if the client disconnects. An unconfirmed
stop locks navigation until inspection/restart; it is not a successful arrival.
Controller tolerances are included with measured arrival errors, not hidden by
changing benchmark thresholds. Non-navigation receipts currently acknowledge
adapter dispatch; they do not verify physical manipulation success.

Receipts are bounded to 256 entries. Explicit `release_control: true` is accepted
only with no active/uncertain navigation. Client shutdown attempts cancellation
and release before closing telemetry. Released sessions can never execute again;
after 256 retired sessions, restart is required to acquire control. A lost client
that cannot release also requires inspection/restart. Restart changes boot ID:
clients report unknown outcomes and never replay automatically into a new boot.

Canonical runtime sources are in `src/emet/core`. After changing them, run
`python scripts/sync_core_runtime.py` to update the standalone robot distribution;
`--check` and `test_runtime_package_parity.py` prevent divergent implementations.

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

### handle_action(action)

- Common keys: `xyt`, `nav_relative`, `nav_blocking`, `joint`, `gripper`, `head_to`, `posture`, `control_mode`, `say` / `say_sync`, `base_velocity`. Bridges ignore unsupported keys.

## Adding a new backend

1. Depend only on **emet-core** (and your backend SDK: ROS2, mujoco, etc.).
2. Subclass `BaseZmqServer`, implement the six methods above.
3. Map your backend’s observations into the same dict keys; map action keys to your backend’s commands.
4. No need for torch, perception, or LLM in the bridge process.
