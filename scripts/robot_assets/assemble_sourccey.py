#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Assemble the Sourccey full-robot MJCF from the generated arm fragment + base/dome/lift.

Run with the main emet venv (no cadquery needed)::

    uv run python scripts/robot_assets/assemble_sourccey.py

Writes ``src/emet/assets/robot/sourccey/sourccey.xml``.

Kinematics source of truth:
- Arm chain (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper):
  ``vulcan-forge/lerobot-vulcan`` ``Arm.urdf``, converted by ``urdf_to_mjcf.py``.
- Base: planar ``base_x/base_y/base_yaw`` on ``base_root`` (matches RoboCasa planar autoplace).
- Lift: vertical prismatic ``lift``; dome + cameras ride the lift carriage.
- 4 cameras: ``front_left``/``front_right`` (dome), ``wrist_left``/``wrist_right`` (grippers).
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_ASSETS = REPO / "src" / "emet" / "assets" / "robot" / "sourccey"

ARM_FRAG = SRC_ASSETS / "arm_frag.xml"
OUT_XML = SRC_ASSETS / "sourccey.xml"

# Base / body geometry (meters). Sourccey: 414 mm footprint, 1030 mm tall.
WHEEL_R = 0.048  # 96 mm mecanum wheels
WHEEL_Y = 0.17  # wheel lateral offset from body centerline
WHEEL_X = 0.17  # wheel longitudinal offset
L1_HALF = 0.125  # level-1 plate ~250 mm square
L1_Z = 0.10
L2_HALF = 0.104
L2_Z = 0.335  # level-2 plate ~ above level-1 walls (220 mm)
L3_HALF = 0.092
L3_Z = 0.570  # level-3 plate above level-2 walls
DOME_HALF = 0.104
DOME_BASE_Z = 0.795  # dome level above level-3 walls
DOME_TOP = 0.995

# Wall vertical extent (each ~220 mm, center of each wall panel).
WALL_1_Z = L1_Z + 0.11  # level-1 walls center
WALL_2_Z = L2_Z + 0.11  # level-2 walls center
WALL_3_Z = L3_Z + 0.11  # level-3 walls center

# Lift travel (prismatic range, meters). Dome + arms ride up on the carriage.
LIFT_MIN = 0.0
LIFT_MAX = 0.20

# Arm shoulder mount, relative to the lift carriage (which sits above the dome level).
# Mount quat rotates the URDF arm frame so its ``-Y`` extension points outward
# (world -X for left, +X for right) while keeping the shoulder-pan axis world-vertical.
# Mount sits above the dome top so tucked arms don't clip the dome plates.
ARM_MOUNT_Y = 0.22  # shoulder base above the carriage top (dome top ~0.20)
ARM_MOUNT_X = DOME_HALF + 0.02
# R_z(±90deg) in MJCF quat (w x y z): left -> -90 (cos -1/2? no): use exact values below.
ARM_MOUNT_QUAT = {"left": "0.7071068 0 0 -0.7071068", "right": "0.7071068 0 0 0.7071068"}

# Default "home" pose for navigation: arms tucked so they don't self-collide / clip.
# Matches the arm joint order in the MJCF: shoulder_pan, shoulder_lift, elbow_flex,
# wrist_flex, wrist_roll, gripper. The left fragment is X-mirrored (sagittal), so the
# left/right joint values must be OPPOSITE sign for the poses to be mirror images.
ARM_HOME = (1.2, -0.6, 0.8, 0.0, 0.0, 0.8)
LIFT_HOME = 0.05


def _mirror_x_attrs(line: str) -> str:
    """Mirror an MJCF line across the sagittal plane: negate x in pos/quat/axis attributes.

    Used for the left arm so left/right are true mirror images (the URDF arm is a
    single-hand chain; mirroring X of positions/axes + quat x-component flips it).
    """
    import re

    def negate_first_three(m: re.Match) -> str:
        vals = m.group(1).split()
        vals[0] = f"{-float(vals[0]):.8g}"
        return f"{' '.join(vals)}"

    def flip_quat_mirror(m: re.Match) -> str:
        # mirror across the sagittal (x=0) plane conjugates by a 180-deg rotation about x,
        # which negates the w,x quaternion components (equivalently y,z up to sign).
        vals = m.group(1).split()
        vals[0] = f"{-float(vals[0]):.8g}"
        vals[1] = f"{-float(vals[1]):.8g}"
        return f"{' '.join(vals)}"

    vec_pat = re.compile(r"((?:-?[\d.eE+-]+ ){2}-?[\d.eE+-]+)")
    quat_pat = re.compile(r"((?:-?[\d.eE+-]+ ){3}-?[\d.eE+-]+)")

    def sub_attr(pat: re.Pattern, repl, line: str, attr: str) -> str:
        return re.sub(rf'{attr}="([^"]*)"', lambda m: f'{attr}="{repl(m)}"', line)

    line = sub_attr(vec_pat, lambda m: negate_first_three(m), line, "pos")
    line = sub_attr(vec_pat, lambda m: negate_first_three(m), line, "axis")
    line = sub_attr(quat_pat, lambda m: flip_quat_mirror(m), line, "quat")
    return line


def prefix_arm_fragment(fragment: str, side: str) -> str:
    """Prefix every ``<body name=``/``<joint name=`` in the arm fragment with ``<side>_``.

    The URDF arm fragment is emitted once and instanced for both sides; MuJoCo requires
    globally unique body/joint names, and actuators must reference the prefixed joints.
    """
    import re

    out = []
    for line in fragment.splitlines():
        m = re.search(r'(<(?:body|joint) name=")([^"]+)(")', line)
        if m:
            line = f"{line[: m.start(1)]}{m.group(1)}{side}_{m.group(2)}{m.group(3)}{line[m.end(3) :]}"
        if side == "left":
            line = _mirror_x_attrs(line)
        out.append(line)
    return "\n".join(out)


def build() -> str:
    arm = ARM_FRAG.read_text()
    indent_arm = "\n".join(("        " + ln) if ln.strip() else "" for ln in arm.splitlines())

    w: list[str] = []
    a = w.append
    a('<?xml version="1.0" encoding="utf-8"?>')
    a('<mujoco model="sourccey">')
    a('  <compiler angle="radian" meshdir="./meshes/" autolimits="true" discardvisual="false"/>')
    a('  <option timestep="0.005" iterations="50" tolerance="1e-10" solver="Newton" gravity="0 0 -9.81"/>')
    a("  <visual>")
    a('    <global offwidth="1024" offheight="1024"/>')
    a("  </visual>")
    a("")
    a("  <asset>")
    a('    <material name="plastic_white" rgba="0.93 0.93 0.93 1" specular="0.15" shininess="0.2"/>')
    a('    <material name="plastic_dark" rgba="0.15 0.15 0.15 1" specular="0.1" shininess="0.1"/>')
    a('    <material name="servo_dark" rgba="0.08 0.08 0.1 1" specular="0.2" shininess="0.3"/>')
    a('    <material name="wheel_rubber" rgba="0.05 0.05 0.05 1"/>')
    a('    <material name="dome_cyan" rgba="0.7 0.85 0.95 0.9"/>')
    # base / body meshes
    for name in (
        "base_plate_l1",
        "base_plate_l2",
        "base_plate_l3_f",
        "base_plate_l3_b",
        "base_plate_l3_l",
        "base_plate_l3_r",
        "base_plate_dome_l",
        "base_plate_dome_r",
        "base_plate_dome_rear",
        "dome_front",
        "dome_back",
        "dome_left",
        "dome_right",
        "dome_top",
        "eye_platform",
        "camera_holder",
        "lift_motor",
        "lift_rack",
        "lift_base",
        "wheel_holder_fl",
        "wall_l1_front",
        "wall_l1_back",
        "wall_l2_front",
        "wall_l2_left",
        "wall_l2_right",
        "wall_l3_front",
        "wall_l3_left",
        "wall_l3_right",
    ):
        a(f'    <mesh name="{name}" file="{name}.stl" scale="0.001 0.001 0.001"/>')
    # arm meshes
    for name in (
        "arm_base",
        "arm_shoulder",
        "arm_bicep_l",
        "arm_forearm",
        "arm_wrist",
        "gripper_base",
        "gripper_front",
        "gripper_finger",
    ):
        a(f'    <mesh name="{name}" file="{name}.stl" scale="0.001 0.001 0.001"/>')
    a("  </asset>")
    a("")
    a("  <default>")
    # All robot geoms are visual-only (contype=0/conaffinity=0), matching innate_mars and the
    # codebase spawn pattern: Robocasa planar autoplace probes stay O(1) (the first-candidate
    # hint is accepted instantly). Spawn safety comes from the planar clip guards + footprint;
    # motion-planning collision is delegated to external planners (RobotModel / pinocchio).
    a('    <default class="robot_collision">')
    a('      <geom type="mesh" density="80" friction="0.9" group="1" contype="0" conaffinity="0"/>')
    a("    </default>")
    a('    <default class="arm_collision">')
    a('      <geom type="mesh" density="100" friction="0.9" group="1" contype="0" conaffinity="0"/>')
    a("    </default>")
    a('    <default class="robot_visual">')
    a('      <geom type="mesh" material="plastic_white" group="2" contype="0" conaffinity="0" density="0"/>')
    a("    </default>")
    a("  </default>")
    a("")
    a("  <worldbody>")
    a('    <light directional="true" diffuse="0.9 0.9 0.9" dir="-1 -1 -1.2" pos="0 0 1.2"/>')
    a('    <camera name="preview_front" pos="1.2 -2.2 1.0" xyaxes="1 0 0 0 0.64 0.77" fovy="50"/>')
    a('    <camera name="preview_top" pos="0 0 3.0" xyaxes="1 0 0 0 1 0" fovy="50"/>')
    a('    <geom type="plane" size="4 4 0.02" material="plastic_dark"/>')
    # ---- base_root: planar base joints (slide x, slide y, hinge yaw) ----
    a('    <body name="base_root" pos="0 0 0">')
    # chassis inertial: battery + electronics + base structure (real robot 15.88 kg total;
    # arms ~3.5 kg, so the mobile base carries ~11 kg).
    a('      <inertial pos="0 0 0.05" mass="8.5" diaginertia="0.2 0.2 0.3"/>')
    # armature on the planar base joints keeps the heavy chassis numerically stable under
    # velocity actuators (matches how the nav P-controller drives them).
    a('      <joint name="base_x" type="slide" axis="1 0 0" pos="0 0 0" damping="5" armature="0.5"/>')
    a('      <joint name="base_y" type="slide" axis="0 1 0" pos="0 0 0" damping="5" armature="0.5"/>')
    a('      <joint name="base_yaw" type="hinge" axis="0 0 1" pos="0 0 0" damping="5" armature="0.5"/>')
    a("")
    # ---- wheels (visual; planar base carries motion) ----
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        a(f'      <body name="wheel_{sx}_{sy}" pos="{sx * WHEEL_X} {sy * WHEEL_Y} {WHEEL_R}">')
        a(
            f'        <geom type="cylinder" size="{WHEEL_R} {WHEEL_R * 0.35}" fromto="0 0 {-WHEEL_R} 0 0 {WHEEL_R}" material="wheel_rubber" group="1" class="robot_collision"/>'
        )
        a("      </body>")
    a("")
    # ---- lower body (fixed plates + walls) ----
    # Plates and walls are visual-only (dense meshes would make Robocasa spawn contact
    # probes slow); a single box collider around the body provides the spawn footprint.
    a('      <body name="body_collider" pos="0 0 0.33">')
    a('        <geom type="box" size="0.125 0.125 0.28" class="robot_collision" friction="0.9"/>')
    a("      </body>")
    for i, (_half, z, name) in enumerate(
        [
            (L1_HALF, L1_Z, "base_plate_l1"),
            (L2_HALF, L2_Z, "base_plate_l2"),
        ]
    ):
        a(f'      <body name="plate_{i}" pos="0 0 {z}">')
        a(f'        <geom type="mesh" mesh="{name}" class="robot_visual" pos="0 0 0"/>')
        a("      </body>")
    # level-1 walls (front/back), level-2 walls (front/left/right), level-3 walls
    wall_layout = [
        ("wall_l1_front", 0, L1_HALF + 0.07, WALL_1_Z),
        ("wall_l1_back", 0, -(L1_HALF + 0.07), WALL_1_Z),
        ("wall_l2_front", 0, L2_HALF + 0.07, WALL_2_Z),
        ("wall_l2_left", -(L2_HALF + 0.07), 0, WALL_2_Z),
        ("wall_l2_right", L2_HALF + 0.07, 0, WALL_2_Z),
        ("wall_l3_front", 0, L3_HALF + 0.07, WALL_3_Z),
        ("wall_l3_left", -(L3_HALF + 0.07), 0, WALL_3_Z),
        ("wall_l3_right", L3_HALF + 0.07, 0, WALL_3_Z),
    ]
    for name, dx, dy, z in wall_layout:
        a(f'      <body name="{name}" pos="{dx} {dy} {z}">')
        a(f'        <geom type="mesh" mesh="{name}" class="robot_visual"/>')
        a("      </body>")
    a("")
    # ---- linear actuator column (fixed to base) ----
    a('      <body name="lift_column" pos="0 0 0">')
    a(f'        <geom type="mesh" mesh="lift_motor" class="robot_visual" pos="0 0 {L1_Z - 0.02}"/>')
    a("      </body>")
    # ---- lift carriage (prismatic z) rides on the column ----
    a('      <body name="lift_carriage" pos="0 0 0">')
    a(
        f'        <joint name="lift" type="slide" axis="0 0 1" pos="0 0 {L1_Z}" range="{LIFT_MIN} {LIFT_MAX}" damping="15" frictionloss="1.0" armature="0.2"/>'
    )
    a('        <inertial pos="0 0 0.12" mass="0.6" diaginertia="0.005 0.005 0.008"/>')
    # rack visual
    a(f'        <geom type="mesh" mesh="lift_rack" class="robot_visual" pos="0 0 {L1_Z + 0.05}"/>')
    # dome-level plates ride on carriage
    for name, dx, dy in (
        ("base_plate_dome_l", -DOME_HALF, 0),
        ("base_plate_dome_r", DOME_HALF, 0),
        ("base_plate_dome_rear", 0, -DOME_HALF),
    ):
        a(f'        <body name="dome_plate_{name}" pos="{dx} {dy} {DOME_BASE_Z}">')
        a(f'          <geom type="mesh" mesh="{name}" class="robot_visual"/>')
        a("        </body>")
    # ---- head / dome ----
    a(f'        <body name="head" pos="0 0 {DOME_BASE_Z}">')
    a('          <inertial pos="0 0 0.10" mass="1.2" diaginertia="0.01 0.01 0.015"/>')
    # The four dome panels are curved quarter-shells (~207 mm). Recentered on bbox centroid,
    # so stacking them at the head center forms the rounded head. The dome_top cap sits on top.
    for name, dx, dy in (
        ("dome_front", 0, 0.0),
        ("dome_back", 0, 0.0),
        ("dome_left", 0, 0.0),
        ("dome_right", 0, 0.0),
    ):
        a(f'          <geom type="mesh" mesh="{name}" class="robot_visual" pos="{dx} {dy} 0.02"/>')
    a(f'          <geom type="mesh" mesh="dome_top" class="robot_visual" pos="0 0 {DOME_TOP - DOME_BASE_Z - 0.02}"/>')
    a('          <geom type="mesh" mesh="eye_platform" class="robot_visual" pos="0 0.06 0.08"/>')
    a('          <geom type="mesh" mesh="camera_holder" class="robot_visual" pos="0 0.09 0.10"/>')
    # cameras (front eyes, 20 deg down — MuJoCo camera looks along -z of its frame).
    # Placed proud of the dome outer surface (y≈0.104) so they see the scene, not the dome interior.
    a('          <camera name="front_left" pos="-0.04 0.12 0.10" xyaxes="1 0 0 0 0.342 0.940" fovy="70"/>')
    a('          <camera name="front_right" pos="0.04 0.12 0.10" xyaxes="1 0 0 0 0.342 0.940" fovy="70"/>')
    a("        </body>")
    a("")
    # ---- arms (left / right) ----
    for side, arm_sx in (("left", -1.0), ("right", 1.0)):
        a(f'      <body name="arm_mount_{side}" pos="{arm_sx * ARM_MOUNT_X} 0 {ARM_MOUNT_Y + DOME_BASE_Z}">')
        a(f'        <geom type="mesh" mesh="arm_base" class="robot_visual" pos="{arm_sx * 0.03} 0 -0.05"/>')
        # wrist camera looks outward along the arm (left -X, right +X), level (world up = +Z)
        cam_xy = "0 1 0 0 0 1" if side == "left" else "0 -1 0 0 0 1"
        a(f'        <camera name="wrist_{side}" pos="0 0.10 0" xyaxes="{cam_xy}" fovy="70"/>')
        a(f'        <body name="arm_{side}" pos="{arm_sx * 0.03} 0 -0.05" quat="{ARM_MOUNT_QUAT[side]}">')
        a(prefix_arm_fragment(indent_arm, side))
        a("        </body>")
        a("      </body>")
    a("")
    a("    </body>")  # lift_carriage
    a("    </body>")  # base_root
    a("  </worldbody>")
    a("")
    a("  <actuator>")
    # planar base uses velocity transmissions (nav stack drives base via velocity ctrl,
    # like innate_mars / xlerobot); arms/gripper/lift are position actuators.
    # kv tuned with base-joint armature so a nav P-controller goal converges (~6 s to 1.8 m).
    for base in ("base_x", "base_y", "base_yaw"):
        a(f'    <velocity name="{base}_act" joint="{base}" kv="300"/>')
    a('    <position name="lift_act" joint="lift" kp="200" ctrlrange="0 0.20"/>')
    for side in ("left", "right"):
        for jname in (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ):
            a(f'    <position name="{side}_{jname}_act" joint="{side}_{jname}" kp="50" ctrlrange="-3.1416 3.1416"/>')
    a("  </actuator>")
    a("")
    # keyframe: robot home (arms tucked) — robosuite_load_utils looks for ``sourccey_home``
    a("  <keyframe>")
    qpos = [0.0, 0.0, 0.0, LIFT_HOME]
    ctrl = [0.0, 0.0, 0.0, LIFT_HOME]
    # left arm uses +ARM_HOME; right arm is the sagittal mirror so uses negated values
    qpos.extend(ARM_HOME)
    ctrl.extend(ARM_HOME)
    qpos.extend(-v for v in ARM_HOME)
    ctrl.extend(-v for v in ARM_HOME)
    a(
        f'    <key name="sourccey_home" qpos="{" ".join(f"{v:.6g}" for v in qpos)}" ctrl="{" ".join(f"{v:.6g}" for v in ctrl)}"/>'
    )
    a("  </keyframe>")
    a("</mujoco>")
    return "\n".join(w) + "\n"


def main() -> None:
    SRC_ASSETS.mkdir(parents=True, exist_ok=True)
    xml = build()
    OUT_XML.write_text(xml)
    print(f"Wrote {OUT_XML} ({len(xml)} bytes)")


if __name__ == "__main__":
    main()
