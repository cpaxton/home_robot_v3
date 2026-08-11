# Sourccey assets — provenance & license

The MJCF (`sourccey.xml`) and meshes in this directory are derived from the
open-source Sourccey hardware CAD by Vulcan Robotics:

- **Source**: https://github.com/vulcan-forge/sourccey-hardware
- **Arm kinematics / inertials**: converted from the `Arm.urdf` shipped in
  https://github.com/vulcan-forge/lerobot-vulcan (HuggingFace LeRobot fork).
- **Hardware license**: CERN Open Hardware Licence Version 2 - Strongly
  Reciprocal ([CERN-OHL-S-2.0](https://github.com/vulcan-forge/sourccey-hardware/blob/main/LICENSE)).
- **Robot specs**: https://vulcanrobotics.ai/specs (414 mm footprint, 1030 mm tall,
  15.88 kg, 4 mecanum wheels, 100 N linear lift, dual 5-DOF + gripper arms).

## Regeneration

Everything under `meshes/` and `sourccey.xml` is generated, not hand-edited.
The STEP→STL manifest (`step_to_stl.py --manifest`) lives alongside this repo's
asset tooling; the mesh-map and STEP paths are recorded in
`scripts/robot_assets/` docs. Re-run:

```bash
# STEP -> STL (needs a cadquery venv, see scripts/robot_assets/README.md)
/path/to/cadquery-venv/bin/python scripts/robot_assets/step_to_stl.py \
    --manifest /tmp/opencode/convert_all.py.json --out-dir src/emet/assets/robot/sourccey/meshes

# URDF arm -> MJCF fragment (main emet venv is fine)
uv run python scripts/robot_assets/urdf_to_mjcf.py \
    /path/to/Arm.urdf --mesh-map /path/to/mesh_map.json \
    --mass-scale 0.27 --out src/emet/assets/robot/sourccey/arm_frag.xml

# full robot MJCF
uv run python scripts/robot_assets/assemble_sourccey.py
```

See [`docs/robots/sourccey.md`](../../../../../docs/robots/sourccey.md) and
[`scripts/robot_assets/README.md`](../../../../../scripts/robot_assets/README.md).
