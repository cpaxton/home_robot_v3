# Sourccey assets — provenance & license

The MJCF (`sourccey.xml`), `arm_frag.xml`, and meshes in this directory are derived
from the open-source Sourccey hardware CAD by Vulcan Robotics:

- **Source**: https://github.com/vulcan-forge/sourccey-hardware
- **Arm kinematics / inertials / meshes**: converted from the **updated official**
  `URDF/ArmLeft/ArmLeft.urdf` (the left arm is canonical; the right arm is the
  code-side X-mirror, because the two official `ArmLeft`/`ArmRight` exports are
  asymmetric). Vendored URDF source lives under `urdf/` (`ArmLeft/` + `ArmRight/`
  STLs; Unity `.asset`/`.meta` sidecars are not vendored).
- **Base / dome / lift / wheels**: assembled from the STEP CAD parts (see
  `scripts/robot_assets/`); meshes are rendered from `step_to_stl.py`.
- **Hardware license**: CERN Open Hardware Licence Version 2 - Strongly
  Reciprocal ([CERN-OHL-S-2.0](https://github.com/vulcan-forge/sourccey-hardware/blob/main/LICENSE)).
- **Robot specs**: https://vulcanrobotics.ai/specs (414 mm footprint, 1030 mm tall,
  15.88 kg, 4 mecanum wheels, 100 N linear lift, dual 5-DOF + gripper arms).

## Regeneration

Everything under `meshes/arm_l_*.stl` (sim copies), `arm_frag.xml`, and `sourccey.xml`
is generated, not hand-edited. `urdf/ArmLeft/meshes/` is the official URDF copy;
`urdf/ArmRight/` is provenance-only (runtime uses the code-side X-mirror). The
checked-in `mesh_map.json` maps each ArmLeft STL basename to `meshes/arm_l_<stem>.stl`.

```bash
# 1. URDF arm -> MJCF fragment, recentered so shoulder_pan sits at arm_root
uv run python scripts/robot_assets/urdf_to_mjcf.py \
    src/emet/assets/robot/sourccey/urdf/ArmLeft/ArmLeft.urdf \
    --mesh-map src/emet/assets/robot/sourccey/mesh_map.json \
    --mass-scale 0.30 --recenter-joint shoulder_pan --wrap-body arm_root \
    --out src/emet/assets/robot/sourccey/arm_frag.xml

# 2. full robot MJCF (instances the canonical fragment for both sides)
uv run python scripts/robot_assets/assemble_sourccey.py
```

See [`docs/robots/sourccey.md`](../../../../../docs/robots/sourccey.md) and
[`scripts/robot_assets/README.md`](../../../../../scripts/robot_assets/README.md).
