# Innate Mars (Maurice)

**Simulation:** `emet serve mujoco --robot innate_mars` loads the vendored Maurice-style MJCF under `src/emet/assets/robot/innate_mars/` merged with `scene_default.xml`. See `NOTICE.md` in that directory for asset provenance and license.

**Real robot:** On the compute that runs ROS 2, start the bridge (e.g. `ros2 launch innate_mars_bridge server.launch.py`). The bridge publishes the same ZMQ keys as other Emet robots, including `rgb`, `rgb_right`, `gps`, `compass`, `camera_K`, `camera_pose`, `camera_K_right`, `camera_pose_right`, and `emet_robot_id` (`innate_mars`). Depth is absent on the wire; use `depth_source: auto` or `da3` in DynaMem config so **Depth Anything 3** fills depth (two-view when both head images and poses are present).

**Agent / DynaMem:** `emet run agent --robot innate_mars` and `emet run dynamem` pass `allow_missing_depth` for this robot so RGB-only ZMQ messages are accepted.

**Session metadata:** Messages include schema v1 ``emet_session`` (see [zmq_session_metadata](../zmq_session_metadata.md)): ``runtime_kind`` = ``innate_mars_ros2_bridge``, stereo/camera/dof capabilities, and ``is_simulation: false``.
