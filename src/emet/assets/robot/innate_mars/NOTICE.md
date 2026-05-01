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

- **Arm / gripper / wrist camera:** Kinematic tree and per-geom ``euler`` values follow the **upstream** ``maurice_sim/mjcf/maurice.mjcf`` (Innate). Those differ slightly from numeric origins in ``maurice.urdf`` because the MJCF mesh rotations were tuned for MuJoCo visualization.
- **Head:** The ``head`` body origin matches URDF ``joint_head`` from ``base_link``. ``head_geom`` uses ``euler="-1.5708 0 0"`` (same STL→MuJoCo convention as ``base_geom``). After that rotation, ``head_geom`` uses ``pos="0 0 0.02175"`` so the head shell meets the torso without intersection (see mesh overlap check in history). ``emet serve mujoco`` merges ``scene_default.xml`` with the robot in a small wrapper; when a ``meshes/`` directory exists, the merge injects an **absolute** ``meshdir`` so the same STL files are used as when loading the robot MJCF alone (avoids edge cases with include path resolution on some installs).
