# Innate Mars (Maurice) MuJoCo assets

These meshes and `innate_mars.xml` are derived from the **Maurice** simulation package in Innate’s ROS workspace (`maurice_sim/mjcf/maurice.mjcf` and `maurice_sim/meshes/`).

## Modifications in this tree

- `meshdir` in the MJCF was changed from a developer-specific absolute path to `meshdir="meshes"` (relative to this directory).
- The ground-plane geom was renamed from `floor` to `maurice_floor` so the model can be merged with `scene_default.xml` without duplicate default geom names.
- **Body order** under `base_link` matches the upstream Maurice MJCF: `link1` (full arm) first, then the **`head`** subtree. Upstream’s `maurice.mjcf` has no head; the head block is an Emet extension from `maurice.urdf` and is listed after the arm to mirror Innate’s arm-first MJCF and the URDF’s joint ordering.

## Innate OS Docker simulation (reference)

In the **innate-os** repository (e.g. `~/src/innate-os`), the documented flow is: `docker compose -f docker-compose.dev.yml build`, `up -d`, `exec innate zsh -l`, then `./scripts/launch_sim_in_tmux.zsh` (see root `README.md` and `SIMULATION_MODE.md`). That stack runs ROS 2 (Zenoh, RWS/rosbridge, nav, manipulation, etc.); **RViz** is the usual 3D view (often via noVNC on `http://localhost:8080/vnc.html`). The **authoritative** robot shape for that stack is `maurice_sim/urdf/maurice.urdf` (plus SRDF for the arm), not the MJCF. The upstream `maurice_sim/mjcf/maurice.mjcf` is for optional MuJoCo nodes (e.g. `maurice_sim/sim.py`) and historically contains **no `head` body**—only `camera_base` and `camera_arm`. Emet’s `innate_mars.xml` adds the head, head mesh, and `head_left` / `head_right` from the same URDF so our MuJoCo sim can render stereo and stay consistent with hardware/bridge camera names.

## License

Confirm redistribution terms with Innate Robotics before publishing this repository or distributing these files. Upstream `package.xml` in some trees lists license as TODO.

## Alignment with reference URDF

Geometries and camera placements in ``innate_mars.xml`` were checked against ``maurice_sim/urdf/maurice.urdf`` (Innate Maurice package): head link + stereo camera positions, link2 offset (z), gripper finger origin (joint6), EE link, and arm camera link pose. The MJCF remains planar-base + velocity actuators for teleop-style sim (not identical joint limits to URDF).


### Geometry sources

- **Kinematic tree (body ``pos`` between links):** Matches **`maurice.urdf`** (same file innate-os ships as ``maurice_sim/urdf/maurice.urdf``). The legacy Innate ``maurice.mjcf`` had several joint translations that **did not** match that URDF (e.g. ``joint2`` z, ``joint4`` z sign). Emet’s MJCF follows the **URDF** so MuJoCo matches RViz / innate-os TF.
- **Arm link meshes (``link1``–``link62``):** URDF visuals use **identity** ``rpy`` on each mesh. Those geoms use **default orientation** (no ``euler`` / ``quat`` on the mesh geom), matching RViz. The old Maurice MJCF used large per-geom ``euler`` hacks that fought the URDF and made links look twisted.
- **Base / head meshes:** URDF also uses identity ``rpy`` on ``base`` and ``head`` visuals, but the exported STL frames still need a single roll fix in MuJoCo; ``base_geom`` and ``head_geom`` keep ``euler="-1.5708 0 0"`` (same convention as many ROS→MuJoCo pipelines).
- **Arm / wrist camera:** ``arm_camera_link`` pose and ``camera_arm`` orientation match URDF ``fixed_arm_camera_link``.
- **Gripper equality:** Matches URDF mimic on ``joint6M``: both hinges use axis ``(0,0,-1)``, with ``polycoef="0 -1 0 0 0"`` so ``q_joint6M = -q_joint6`` (URDF ``multiplier="-1"``). The legacy MJCF used axis ``+Z`` on ``joint6M`` and ``polycoef`` ``+1`` to fake the same motion—confusing for URDF comparisons.
- **Merge wrapper:** When merging with ``scene_default.xml``, an absolute ``meshdir`` may be injected (see ``mujoco_server``).
