# Innate Mars (Maurice) MuJoCo assets

These meshes and `innate_mars.xml` are derived from the **Maurice** simulation package in Innate’s ROS workspace (`maurice_sim/mjcf/maurice.mjcf` and `maurice_sim/meshes/`).

## Modifications in this tree

- `meshdir` in the MJCF was changed from a developer-specific absolute path to `meshdir="meshes"` (relative to this directory).
- The ground-plane geom was renamed from `floor` to `maurice_floor` so the model can be merged with `scene_default.xml` without duplicate default geom names.

## License

Confirm redistribution terms with Innate Robotics before publishing this repository or distributing these files. Upstream `package.xml` in some trees lists license as TODO.

## Alignment with reference URDF

Geometries and camera placements in ``innate_mars.xml`` were checked against ``maurice_sim/urdf/maurice.urdf`` (Innate Maurice package): head link + stereo camera positions, link2 offset (z), gripper finger origin (joint6), EE link, and arm camera link pose. The MJCF remains planar-base + velocity actuators for teleop-style sim (not identical joint limits to URDF).


### Geometry sources

- **Arm / gripper / wrist camera:** Kinematic tree and per-geom ``euler`` values follow the **upstream** ``maurice_sim/mjcf/maurice.mjcf`` (Innate). Those differ slightly from numeric origins in ``maurice.urdf`` because the MJCF mesh rotations were tuned for MuJoCo visualization.
- **Head:** Mounted at the URDF ``joint_head`` translation from ``base_link``. The ``head_geom`` uses ``euler="-1.5708 0 0"`` — the same STL→MuJoCo convention as ``base_geom``. If the head still looks one axis off in your viewer, try matching ``link1_geom`` instead (``euler="0 0 1.5708"``).
