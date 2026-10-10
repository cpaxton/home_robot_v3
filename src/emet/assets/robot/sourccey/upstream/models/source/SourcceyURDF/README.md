# Sourccey Mk.V robot URDF

Open `SourcceyMkV.urdf` from this folder so its relative `meshes/` paths resolve.
The model was generated from `NewArmFiles/Sourccey Mk.V Assembly.f3z` through
Fusion's `FullRobotCapture` script.
`UrdfStaging/convert_full.py` removes detached duplicate body clusters when a
matching copy in the same occurrence is anchored to a joint. This removes the
extra gripper shells and motor caps from the full assembly capture without
changing the source Fusion file.

## Kinematics

- Root: `base_link` at the Fusion assembly origin.
- Both arms use the meshes, mounting positions, and joint axes captured from
  the full Fusion assembly. Each has the same six joint roles, effort, and
  velocity values as its corresponding standalone arm URDF:
  `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`,
  and `gripper`, prefixed `left_` or `right_`.
- The shoulder-lift ranges match the working full-control scene: left
  -120 to +120 degrees and right -120 to +110 degrees. Both gripper joints
  range from -5 degrees closed to +60 degrees open, matching the visible
  full-assembly fingers. Other arm limits retain their standalone URDF values.
- Four wheel joints are continuous: `front_left_wheel`,
  `front_right_wheel`, `rear_left_wheel`, and `rear_right_wheel`.
- `linear_actuator` is prismatic, with the captured Fusion pose as zero and
  a 0.315 m extension range toward the lower CAD limit. Both shoulder mounts
  are descendants of this joint, so the actuator raises and lowers both arms.
- The two wrist mounts and 12 unjointed shell/accessory components have fixed
  attachments. The shell components are attached to the chassis.

Fusion provided the link masses and inertia tensors before detached meshes were
filtered. Their total remains 169.042 kg; recapture per-body mass properties and
check CAD materials before using dynamic simulation results quantitatively.
Meshes serve as both visual and collision
geometry. The URDF describes articulation; wheel control, mecanum traction,
actuator control, and simulator plugins depend on the target simulation system.

To regenerate after a CAD change, run `FullRobotCapture` in Fusion with the
full assembly open, then from the repository root run:

```sh
python UrdfStaging/convert_full.py
python UrdfStaging/validate_full.py
```

The generator reads arm joint settings from `Assets/Urdf/ArmLeft/ArmLeft.urdf`
and `Assets/Urdf/ArmRight/ArmRight.urdf`, with the shoulder-lift and gripper overrides
above. Validation checks those settings,
both arms' mesh provenance, and each CAD joint axis against its servo horns.
