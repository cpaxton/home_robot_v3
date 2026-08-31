# Sourccey assets — provenance & license

The MJCF (`sourccey.xml`), `arm_frag.xml`, and meshes in this directory are derived
from the open-source Sourccey hardware CAD by Vulcan Robotics:

- **Source**: https://github.com/vulcan-forge/sourccey-hardware
- **Arm kinematics / inertials / meshes**: converted from the **updated official**
  `URDF/ArmLeft/ArmLeft.urdf` (the left arm is canonical; the right arm is the
  code-side X-mirror, because the two official `ArmLeft`/`ArmRight` exports are
  asymmetric). Vendored URDF source lives under `urdf/` (`ArmLeft/`, `ArmRight/`,
  plus the legacy `Arm.urdf` reference).
- **Base / dome / lift / wheels**: assembled from the STEP CAD parts (see
  `scripts/robot_assets/`); meshes are rendered from `step_to_stl.py`.
- **Hardware license**: CERN Open Hardware Licence Version 2 - Strongly
  Reciprocal ([CERN-OHL-S-2.0](https://github.com/vulcan-forge/sourccey-hardware/blob/main/LICENSE)).
- **Robot specs**: https://vulcanrobotics.ai/specs (414 mm footprint, 1030 mm tall,
  15.88 kg, 4 mecanum wheels, 100 N linear lift, dual 5-DOF + gripper arms).

## Regeneration

Everything under `meshes/`, `arm_frag.xml`, and `sourccey.xml` is generated, not
hand-edited. The updated official arm chain converts with the main emet venv (the
URDF visual origins already place the Unity-exported meshes, so the mesh-map uses
`offset_mm: [0, 0, 0]`):

```bash
# 1. mesh-map: URDF mesh basename -> vendored STL (arm_l_<name>.stl), no offset.
#    (ArmLeft URDF meshes are copied to meshes/ as arm_l_<name>.stl)

# 2. URDF arm -> MJCF fragment (mass-scale to land near the 15.88 kg real robot)
uv run python scripts/robot_assets/urdf_to_mjcf.py \
    src/emet/assets/robot/sourccey/urdf/ArmLeft/ArmLeft.urdf \
    --mesh-map /tmp/mesh_map.json --mass-scale 0.30 --out /tmp/arm_left_frag.xml

# 3. recenter the fragment so the shoulder_pan pivot sits at the arm_root origin,
#    then save as src/emet/assets/robot/sourccey/arm_frag.xml

# 4. full robot MJCF (instances the canonical fragment for both sides)
uv run python scripts/robot_assets/assemble_sourccey.py
```

See [`docs/robots/sourccey.md`](../../../../../docs/robots/sourccey.md) and
[`scripts/robot_assets/README.md`](../../../../../scripts/robot_assets/README.md).