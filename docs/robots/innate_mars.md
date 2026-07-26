# Innate Mars (Maurice)

**Reference URDF:** ``src/emet/assets/robot/innate_mars/maurice.urdf`` matches innate-os ``maurice_sim/urdf/maurice.urdf`` with ``package://…/meshes/`` rewritten to ``meshes/``. Use it for RViz / TF parity checks; **MuJoCo** still uses ``innate_mars.xml``. Link placements and arm mesh orientations follow the URDF; **base** and **head** geoms keep an extra MuJoCo roll where STL frames still disagree with MuJoCo’s mesh convention.

**Simulation (Emet / MuJoCo):** `emet serve mujoco --robot innate_mars` loads the vendored Maurice-style MJCF under `src/emet/assets/robot/innate_mars/` merged with the shared **`scene_environment.xml`** (table room + props). The robot MJCF is **robot-only** (no scene floor or world lights). See `NOTICE.md` in that directory for asset provenance and license. To inspect the model: `emet view-mujoco --robot innate_mars` adds optional grid/lights via `innate_mars_visual_extras.xml`; use `--merge-scene` or `--no-extras` as needed (see CLI docs). **ZMQ `xyt` navigation** in sim drives the planar **slide X / slide Y / hinge yaw** base (`base_x`, `base_y`, `base_yaw` velocity actuators), not a floating free joint—see `RobotSpec.planar_base_joint_names` and `RobosuiteZmqServer` (same protocol as Stretch; different MuJoCo mechanics).

**Sim depth policy (sensor vs DA3):** In MuJoCo, ZMQ already carries **rendered sensor depth**. DynaMem should use **`depth_source: sensor`** with the default **`dynav_config.yaml`** unless you are deliberately reproducing a stack that infers depth. **Use Depth Anything 3 in sim only when your model or deployment config uses DA3 as the depth path** (for example hardware Mars with `dynav_innate_mars.yaml` or an agent YAML that sets `depth_source: da3`). Turning on DA3 in sim while the rest of the pipeline expects sensor depth makes debugging harder and can mislead Rerun vs map comparisons.

**Rerun vs voxel map (same fused depth):** DynaMem fuses whatever **`depth_source`** resolves to (sensor, DA3, masks, clamps). Live Rerun **`world/head_camera/points`** is built from that **same** resolved depth buffer for the head frame (not a separate “raw ZMQ only” path), so head points and **`world/point_cloud`** (from **`voxel_pcd`**) agree on depth geometry when DynaMem is running. Remaining mismatch is usually **pose or step skew** between streams, not two different depth tensors.

**`--perfect-depth` / `EMET_DYNAMEM_PERFECT_DEPTH`:** In sim, this forces **sensor** depth into mapping and skips DA3—good for checking **navigation frame, intrinsics, and base pose** without stereo inference. **What to expect:** The **voxel map** should look structurally correct (ground-truth layout). The **head point cloud** channel can still look **noisier or stranger** than the voxel cloud: it is a per-pixel unprojection of raw simulator depth (holes, flying pixels, ceiling strips), while the voxel representation **aggregates and filters** over time. A clean voxel map with a “messy” head PCD in that mode usually means the **fusion path is correct**, not that coordinates disagree. See also [DynaMem debugging](../dynamem.md) for `--perfect-depth` and sky-mask behavior.

**Innate OS Docker (upstream reference):** The public **innate-os** tree (e.g. ``~/src/innate-os``) documents simulation in the root **README** and **SIMULATION_MODE.md**: build with ``docker compose -f docker-compose.dev.yml build``, run ``up -d``, enter the container with ``exec innate zsh -l``, then ``./scripts/launch_sim_in_tmux.zsh``. That starts the ROS 2 stack (Zenoh, RWS, navigation, manipulation, etc.); **RViz** is the usual robot visualization (often via noVNC at ``http://localhost:8080/vnc.html``). The **canonical robot geometry** there is ``ros2_ws/src/maurice_bot/maurice_sim/urdf/maurice.urdf`` (via ``robot_state_publisher`` / MoveIt), **not** the MJCF file. Upstream ``maurice_sim/mjcf/maurice.mjcf`` matches **base + arm + floor** in MuJoCo but **does not include a head body**; optional Python MuJoCo drivers only reference ``camera_base`` and ``camera_arm``. Emet’s ``innate_mars.xml`` extends that line with portable ``meshdir``, head link + ``head.STL``, and stereo cameras; for Emet MuJoCo merges, the **room floor** comes from ``scene_environment.xml``, not the robot file. **Link** transforms follow the vendored URDF, with base/head mesh rolls only where needed for MuJoCo STLs.

**Real robot:** On the compute that runs ROS 2, start the bridge (e.g. `ros2 launch innate_mars_bridge server.launch.py`). The bridge publishes the same ZMQ keys as other Emet robots, including `rgb`, `rgb_right`, `gps`, `compass`, `camera_K`, `camera_pose`, `camera_K_right`, `camera_pose_right`, and `emet_robot_id` (`innate_mars`). ZMQ **`xyt`** actions are forwarded to Nav2 **`navigate_to_pose`** (innate-os `maurice_nav`). Depth is absent on the wire; use `depth_source: auto` or `da3` in DynaMem config so **Depth Anything 3** fills depth (`da3_stereo: true` in `dynav_innate_mars.yaml` when both eyes are present).

**Hardware bring-up:** [innate_mars_hardware.md](innate_mars_hardware.md). **Experiments matrix:** [experiments/innate_mars.md](../experiments/innate_mars.md).

**innate-os sim harness (ROS):** With [innate-os](https://github.com/innate-inc/innate-os) `./innate sim up`, build this bridge and run `ros2 launch innate_mars_bridge server.launch.py`; audit with `uv run python scripts/audit_innate_os_topics.py`. Use for bridge/Nav2 validation; Dynagraph **paper metrics** stay on Emet MuJoCo sim.

**DynaMem + Depth Anything 3:** Hardware Mars does not publish depth on ZMQ. The `depth-anything-3` library is included with the default install (`uv sync` / default dependency groups). Stereo RGB plus both intrinsics and poses feed **Depth Anything 3** stereo inference; the result drives voxel mapping the same way as sensor depth on other robots.

```bash
# Same default dynav as other robots (dynav_config.yaml). For hardware Mars without ZMQ depth, add:
#   --dynav-config dynav_innate_mars.yaml
emet run dynamem --robot innate_mars --robot-ip 127.0.0.1 -S
```

To use **MuJoCo rendered depth** in DynaMem (recommended in sim), keep the default ``dynav_config.yaml`` (``depth_source: sensor``) or pass its path explicitly. For the **DA3 + Mars tuning** preset (hardware without ZMQ depth, or when matching a DA3 depth path), pass ``--dynav-config dynav_innate_mars.yaml``.

Override `da3_model_id` in that YAML (or a fork) for heavier metric checkpoints when you have GPU headroom.

**DA3 voxel “walls” / sky junk:** Stereo networks often assign **finite** depth to textureless sky or bright ceiling strips; unprojection then paints tall vertical sheets into the voxel map. Mars defaults set **`da3_clip_max_m: 3.5`** and **`da3_ignore_sky_fraction_top: 0.14`** under `robots.innate_mars.mapping` in [`configs/emet/default.yaml`](../configs/emet/default.yaml). Raise/lower those keys if lighting or camera tilt still leaves phantom obstacles; `emet debug-da3-depth` supports `--sky-fraction-top` and `--clip-depth-max-m` for the same behavior outside DynaMem.

**Floating mid-air blobs:** Hardware DA3 can leave isolated depth speckles and small floating clusters in Rerun ``world/point_cloud``. Mild post-filters are **on by default** for Mars (`depth_speckle_open_kernel: 3`, `voxel_pcd_dbscan_min_samples: 6`). If thin furniture is over-pruned, set either to ``0`` or pass ``--set mapping.filters.depth_speckle_open_kernel=0``. See [dynav_config.md § DA3 post-filters](../dynav_config.md#depth--voxel-post-filters-da3-hardware-opt-in).

**Curved / bowed walls vs sim:** Sim **sensor** depth is planar; DA3 depth is learned and lighting-dependent — flat walls may look **curvy** in the point cloud even when RGB looks fine. That is usually model/intrinsics/lighting, not the speckle filters. Compare with ``emet debug-da3-depth --depth-source sensor`` in sim, or try ``da3_model_id: depth-anything/DA3METRIC-LARGE`` on hardware.

**Fast DA3 sanity check (Rerun):** With `emet serve mujoco --robot innate_mars` running, use `emet debug-da3-depth --robot innate_mars` to log left RGB, colormapped depth, and a strided point cloud under `da3/…` (same `resolve_depth_map` path as DynaMem). Add `--depth-source sensor` to compare against sim-rendered depth without running DA3.

`RobotSpec` for this robot sets `dynamem_depth_source_hint="da3"` as **documentation** for **hardware** (no ZMQ depth). **DynaMem defaults** still load ``dynav_config.yaml`` (sensor depth in sim) unless you pass ``--dynav-config dynav_innate_mars.yaml``. The hint is **not** a recommendation to use DA3 in MuJoCo; in sim, keep the default unless you are deliberately matching a DA3-only stack.

### DynaMem exploration (sim or hardware)

Prerequisites: project env with default groups (includes **dynamem** and **da3**); first DA3 run may download model weights.

1. **MuJoCo server** (table scene + stereo ZMQ), from repo root:

   ```bash
   uv run emet serve mujoco --robot innate_mars
   ```

2. **DynaMem client** (same default ``dynav_config.yaml`` as Stretch — **sensor depth in sim**; add ``--dynav-config dynav_innate_mars.yaml`` only for **hardware** Mars without ZMQ depth, or when intentionally matching a **DA3** depth path):

   ```bash
   uv run emet run dynamem --robot innate_mars -S
   ```

   With Rerun enabled (default), open the viewer (browser or native) as for other `emet run dynamem` sessions—for example `http://localhost:9090?url=ws://localhost:9877` when the log shows the websocket URL. Use `--no-rerun` for headless mapping.

3. **LLM agent + Discord (same memory stack):** use [`configs/agent_innate_mars.yaml`](../../configs/agent_innate_mars.yaml) (`robot: innate_mars`, Discord + EQA on). Hardware: [Discord chat + explore](innate_mars_hardware.md#discord-chat--explore-herman). Example:

   ```bash
   export DISCORD_TOKEN=...
   uv run emet run agent --connection herman --config configs/agent_innate_mars.yaml --name Herman
   ```

4. **Optional integration test** (slow; GPU recommended; caches weights after first run):

   ```bash
   RUN_DA3_TESTS=1 uv run emet test src/test/mapping/test_innate_mars_da3_sim.py -q
   ```

In **Rerun**, check `world/semantic_memory/pointcloud` for table and props aligned with the room; fix `camera_K` / `camera_pose` / `gps` if the cloud is sheared or floating.

**ZMQ observation keys (bridge ↔ client):** Full-fidelity DA3 uses head stereo when present. The generic client expects:

| Key | Role |
|-----|------|
| `rgb`, `camera_K`, `camera_pose` | Primary head camera (JPEG bytes decoded in client) |
| `depth` | Optional; omit on real Mars (`allow_missing_depth` + DA3) |
| `rgb_right`, `camera_K_right`, `camera_pose_right` | Second eye for stereo DA3 (JPEG); MuJoCo sim publishes these for `head_left`/`head_right` pairs |
| `rgb_tertiary`, `camera_name_tertiary`, `camera_K_tertiary`, `camera_pose_tertiary` | Optional third camera (JPEG + intrinsics/pose); **MuJoCo sim** when MJCF exposes a distinct third cam (see note below); typically absent on real bridge |
| `gps`, `compass` | Base pose for voxel exploration frame |

Same schema as other Emet robots; see [zmq_session_metadata](../zmq_session_metadata.md) for session envelopes.

**Sim tertiary camera (`rgb_tertiary`, …):** See table row above. The Mars ROS bridge normally omits these; they appear when running **`emet serve mujoco --robot innate_mars`** with a current `RobosuiteZmqServer` that attaches the third MJCF cam (`camera_arm`).

### Camera diagnostics and head nod preview

Use **`emet preview-cameras`** (see [CLI: preview-cameras](../cli.md#emet-preview-cameras-options)) for a quick **montage** from either the merged default scene (`--source local`) or one live ZMQ frame (`--source zmq`, port 4401).

- **Local render** matches `RobosuiteZmqServer` RGB handling (same table scene as `emet serve mujoco --robot innate_mars`).
- **`--nod`** (local only): sweeps **`joint_head`** (**sim hinge +head X**) while stereo looks ~**−world Y** (table‑front); nod pitches gaze. Writes **`montage_0000.png` …** plus optional **`--nod-video`**. Default motion is **`bounce`** (down–up–down) in `--nod-frames` steps; use **`--nod-motion once`** for a single stroke.
- **`head_to`** actions in sim map Stretch-style **`tilt`** to **`joint_head`** (**`pan`** is ignored until a second head DOF exists in MJCF).

**Agent / DynaMem:** `emet run agent --robot innate_mars` and `emet run dynamem` pass `allow_missing_depth` for this robot so RGB-only ZMQ messages are accepted.

**Session metadata:** Messages include schema v1 ``emet_session`` (see [zmq_session_metadata](../zmq_session_metadata.md)): ``runtime_kind`` = ``innate_mars_ros2_bridge``, stereo/camera/dof capabilities, and ``is_simulation: false``.

**Sim cameras (MuJoCo):** ``RobotSpec.camera_names`` are ``head_left``, ``head_right``, ``camera_arm`` (wrist). An extra ``camera_base`` exists for debugging. Primary ZMQ RGB uses the first list entry (``head_left``). **`robosuite_rgb_depth_ops`** includes **`flipud`** so Renderer buffers match upright OpenCV/display row order with chained **`camera_K`**. Stereo cameras share identical ``fovy`` and the same head stereo ``quat`` in MJCF. **joint_head** body origin matches **maurice.urdf**. **`head_visual` body ``pos``** trims the head STL (**``Y``** toward ``head_left``; **``Z``** lowers mesh vs neck). Stereo mounts multiply URDF camera ``xyz`` by **R_z(+π/2)** in the head frame (outward offset ~**−head Y**/table‑forward; 60 mm baseline on **± X**) with a merged‑scene clearance tweak on **`y`**. Optics multiply URDF **REP** by **R_z(−π/2)** so MuJoCo **− Z** ≈ **−head Y** (table‑forward in the default merged scene). **Sim nod** uses hinge axis **``1 0 0``** (**URDF** still lists **`0 −1 0`** hardware axis). Shared **80°** ``fovy``.

**Wrist ``camera_arm`` (MJCF-only polish on top of URDF kinematics):** The vendor URDF only defines ``arm_camera_link`` as a pitched fixed child of ``link5`` (``xyz`` + ``rpy="0 0.43633 0"``); it does **not** declare an arm optical frame or CameraInfo-style axes. MuJoCo still needs a ``<camera>`` with a quaternion that maps **ROS-style optical** (**+Z** forward) to MuJoCo’s **−Z** view direction. Emet uses the same **optical→MuJoCo** mapping quaternion **relative to the pitched link** as for the head cameras, but **without** the head-only **``R_head_z(±π/2)``** “table twist” (the wrist parent frame is already pitched on the arm; the head twist exists so stereo rays aim at the merged table along **−world Y**). Two sim-only tweaks sit on that camera element: a **1 cm translation** along optical forward (implemented as **+0.01 m** along MuJoCo **−Z** in ``arm_camera_link``) so the pinhole sits slightly in front of nearby visual geoms (wrist link mesh and the orange ``ee_link`` sphere were close enough to clip inside the frustum and paint an orange ring), and an **Rz(π)** in the camera frame so the rendered image orientation matches the rest of the RGB/**``flipud``**/OpenCV pipeline. Neither offset is duplicated in the URDF because the real bridge does not publish an arm stream through this MJCF path.

**Head stereo** uses **REP** (**``fixed_*_camera_optical_frame``**) with an extra **`R_head_z(−π/2)`** in MJCF so sim RGB shares the **−Y** tabletop frame with the **intended** room layout, unlike raw TF where optical **+Z** aligns with **+head X**.

``emet serve mujoco`` merges ``scene_environment.xml`` with the robot and, when a ``meshes/`` directory exists, injects an absolute ``meshdir`` in the merge wrapper so the same STL files load as for the standalone MJCF. Base and head mesh geoms keep an extra euler vs raw URDF meshes where STL vs MuJoCo conventions disagree.

## Why MJCF camera orientation is not a literal copy of the URDF

Three separate ideas get mixed if we “just copy” TF into MJCF:

1. **MuJoCo’s camera convention**  
   A MuJoCo ``<camera>`` always looks down its body **−Z**. ROS / REP-103 optical frames use **+Z** forward, **X** right, **Y** down. Every sim camera therefore needs a **fixed** rotation (quaternion in MJCF) that is *not* optional decoration—it is the bridge between optical conventions and MuJoCo.

2. **What the URDF actually specifies**  
   - **Head:** ``maurice.urdf`` gives (a) stereo mount origins on ``head`` and (b) per-eye **``fixed_*_camera_optical_frame``** joints with ``rpy="-1.5708 0 -1.5708"`` so **+Z** optical points along **+head X**. That is correct for ROS drivers, but the merged Emet table scene is laid out so that **useful sim gaze** is roughly **−world Y**. MJCF therefore applies an extra **yaw in the head frame** on the **optics** (and a matching **yaw** on the mount **xyz** so the lenses stay in plausible places and ``mj_ray`` clears ``head_geom``).  
   - **Wrist:** The URDF only has ``fixed_arm_camera_link`` with the same **``xyz``** and **pitch on Y** as in MJCF; there is **no** arm optical link. MJCF adds the **optical→MuJoCo** quaternion **in the pitched link frame** only—**not** the head stereo’s extra **``R_z``** table correction, because that correction is specific to how the head sits on ``base_link`` and how we want binocular rays in the room.

3. **Sim-only tuning (wrist)**  
   Near-geometry clipping and pixel **up** vs downstream expectations are **renderer** concerns. The **1 cm** offset and **180°** roll on ``camera_arm`` fix self-clipping / frustum scraping against wrist visuals and align the bitmap with **``flipud``** + intrinsics; they are **not** claims about hardware CAD.

So: **link origins and arm pitch match the URDF**; **head stereo numbers are URDF geometry plus deliberate MJCF yaw for the merged scene**; **wrist camera is URDF kinematics plus standard optical mapping and small MJCF-only offsets**.

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
