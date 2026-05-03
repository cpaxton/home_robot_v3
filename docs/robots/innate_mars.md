# Innate Mars (Maurice)

**Reference URDF:** ``src/emet/assets/robot/innate_mars/maurice.urdf`` matches innate-os ``maurice_sim/urdf/maurice.urdf`` with ``package://…/meshes/`` rewritten to ``meshes/``. Use it for RViz / TF parity checks; **MuJoCo** still uses ``innate_mars.xml``. Link placements and arm mesh orientations follow the URDF; **base** and **head** geoms keep an extra MuJoCo roll where STL frames still disagree with MuJoCo’s mesh convention.

**Simulation (Emet / MuJoCo):** `emet serve mujoco --robot innate_mars` loads the vendored Maurice-style MJCF under `src/emet/assets/robot/innate_mars/` merged with the shared **`scene_environment.xml`** (table room + props). The robot MJCF is **robot-only** (no scene floor or world lights). See `NOTICE.md` in that directory for asset provenance and license. To inspect the model: `emet view-mujoco --robot innate_mars` adds optional grid/lights via `innate_mars_visual_extras.xml`; use `--merge-scene` or `--no-extras` as needed (see CLI docs).

**Innate OS Docker (upstream reference):** The public **innate-os** tree (e.g. ``~/src/innate-os``) documents simulation in the root **README** and **SIMULATION_MODE.md**: build with ``docker compose -f docker-compose.dev.yml build``, run ``up -d``, enter the container with ``exec innate zsh -l``, then ``./scripts/launch_sim_in_tmux.zsh``. That starts the ROS 2 stack (Zenoh, RWS, navigation, manipulation, etc.); **RViz** is the usual robot visualization (often via noVNC at ``http://localhost:8080/vnc.html``). The **canonical robot geometry** there is ``ros2_ws/src/maurice_bot/maurice_sim/urdf/maurice.urdf`` (via ``robot_state_publisher`` / MoveIt), **not** the MJCF file. Upstream ``maurice_sim/mjcf/maurice.mjcf`` matches **base + arm + floor** in MuJoCo but **does not include a head body**; optional Python MuJoCo drivers only reference ``camera_base`` and ``camera_arm``. Emet’s ``innate_mars.xml`` extends that line with portable ``meshdir``, head link + ``head.STL``, and stereo cameras; for Emet MuJoCo merges, the **room floor** comes from ``scene_environment.xml``, not the robot file. **Link** transforms follow the vendored URDF, with base/head mesh rolls only where needed for MuJoCo STLs.

**Real robot:** On the compute that runs ROS 2, start the bridge (e.g. `ros2 launch innate_mars_bridge server.launch.py`). The bridge publishes the same ZMQ keys as other Emet robots, including `rgb`, `rgb_right`, `gps`, `compass`, `camera_K`, `camera_pose`, `camera_K_right`, `camera_pose_right`, and `emet_robot_id` (`innate_mars`). Depth is absent on the wire; use `depth_source: auto` or `da3` in DynaMem config so **Depth Anything 3** fills depth (two-view when both head images and poses are present).

**DynaMem + Depth Anything 3:** Hardware Mars does not publish depth on ZMQ. The `depth-anything-3` library is a default project dependency; run the client with the packaged preset:

```bash
emet run dynamem --robot innate_mars --robot-ip 127.0.0.1 -S --dynav-config dynav_innate_mars.yaml
```

That YAML sets `depth_source: da3` and defaults to a **faster** checkpoint (`DA3-SMALL`, lower `da3_process_res`) for interactive use; override `da3_model_id` for maximum metric quality. For MuJoCo sim you can keep `dynav_config.yaml` (`depth_source: sensor`) to use rendered depth, or use `dynav_innate_mars.yaml` to validate DA3 point clouds against the same scene as the real robot.

**Fast DA3 sanity check (Rerun):** With `emet serve mujoco --robot innate_mars` running, use `emet debug-da3-depth --robot innate_mars` to log left RGB, colormapped depth, and a strided point cloud under `da3/…` (same `resolve_depth_map` path as DynaMem). Add `--depth-source sensor` to compare against sim-rendered depth without running DA3.

`RobotSpec` for this robot declares `dynamem_depth_source_hint="da3"` (see `emet.robots.base` / `get_robot_spec("innate_mars")`). Depth inference uses the `depth-anything-3` package from the default install.

In **Rerun**, check `world/semantic_memory/pointcloud` for table and props aligned with the room; fix `camera_K` / `camera_pose` / `gps` if the cloud is sheared or floating.

**ZMQ observation keys (bridge ↔ client):** Full-fidelity DA3 uses head stereo when present. The generic client expects:

| Key | Role |
|-----|------|
| `rgb`, `camera_K`, `camera_pose` | Primary head camera (JPEG bytes decoded in client) |
| `depth` | Optional; omit on real Mars (`allow_missing_depth` + DA3) |
| `rgb_right`, `camera_K_right`, `camera_pose_right` | Second eye for stereo DA3 (JPEG); MuJoCo sim publishes these for `head_left`/`head_right` pairs |
| `gps`, `compass` | Base pose for voxel exploration frame |

Same schema as other Emet robots; see [zmq_session_metadata](../zmq_session_metadata.md) for session envelopes.

**Agent / DynaMem:** `emet run agent --robot innate_mars` and `emet run dynamem` pass `allow_missing_depth` for this robot so RGB-only ZMQ messages are accepted.

**Session metadata:** Messages include schema v1 ``emet_session`` (see [zmq_session_metadata](../zmq_session_metadata.md)): ``runtime_kind`` = ``innate_mars_ros2_bridge``, stereo/camera/dof capabilities, and ``is_simulation: false``.

**Sim cameras (MuJoCo):** ``RobotSpec.camera_names`` are ``head_left``, ``head_right``, ``camera_arm`` (wrist), matching named cameras in the MJCF after URDF alignment. An extra ``camera_base`` exists for debugging. Primary ZMQ RGB uses the first name (head left). ``emet serve mujoco`` merges ``scene_environment.xml`` with the robot and, when a ``meshes/`` directory exists, injects an absolute ``meshdir`` in the merge wrapper so the same STL files load as for the standalone MJCF. Stereo head cameras use fixed ``quat`` in MJCF (ROS-style mounting); base and head meshes keep an extra roll vs URDF for STL alignment.

## Debugging URDF vs MuJoCo

MuJoCo only reads MJCF; RViz / standalone URDF viewers only read URDF. Compare **maurice.urdf** side‑by‑side with ``emet view-mujoco --robot innate_mars``.

**Standalone URDF viewer (no ROS):** [**urdf-viz**](https://github.com/openrr/urdf-viz) (Rust) opens URDF + STL quickly:

```bash
# Install once: cargo install urdf-viz   # or: brew install openrr/tap/urdf-viz
cd src/emet/assets/robot/innate_mars
urdf-viz maurice.urdf
```

Meshes load relative to the URDF path (`meshes/*.STL`). Use joint sliders to sanity‑check the kinematic tree.

**ROS 2 RViz:** Source ROS Humble, publish ``robot_description`` from ``maurice.urdf``, run ``rviz2`` — matches innate-os Docker workflows.

**Quick STL sanity:** Open individual ``meshes/*.STL`` in **MeshLab** or **Blender** if you suspect a bad mesh frame (does not show joints).
