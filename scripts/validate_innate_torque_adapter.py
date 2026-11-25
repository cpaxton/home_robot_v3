#!/usr/bin/env python3
"""Smoke-test the torque -> position-servo adapter against innate-os's MuJoCo sim.

This script is deliberately an **external integration test**. It does not vendor
or import innate-os into Emet; it expects an innate-os checkout on the machine
and imports ``mars_sim_driver.core.VirtualMars`` from it.

The test covers the parts of the proposed Mars interface that can be validated
without a robot:

- a torque command mapped through ``PositionServoAdapter`` reaches roughly the
  same joint motion as applying that torque directly in MuJoCo;
- torque and position-target limits clip;
- velocity-damping feedforward compensates the servo's own damping;
- a torque command can be applied while the gripper is in contact, and the
  contact force is observable;
- the Emet multi-arm profile/action-vector layout matches Mars (5 arm joints,
  gripper separate).

Setup (once):

    git clone https://github.com/innate-inc/innate-os.git /tmp/opencode/innate-os
    cd /tmp/opencode/innate-os
    ./innate-sim assets
    cd sim && uv sync --group notebook

Run from the Emet checkout:

    cd /tmp/opencode/innate-os/sim
    MUJOCO_GL=egl .venv/bin/python /path/to/emet/scripts/validate_innate_torque_adapter.py

Or use ``scripts/run_innate_torque_validation.sh``.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _install_emet_import_stubs() -> None:
    """Import the contract without executing emet.controller/__init__ (torch)."""
    root = _repo_root()
    for name, rel in (
        ("emet", "src/emet"),
        ("emet.controller", "src/emet/controller"),
        ("emet.controller.manipulation", "src/emet/controller/manipulation"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(root / rel)]
        sys.modules[name] = module


def _import_innate(sim_dir: Path) -> None:
    package = sim_dir.parent / "ros2_ws" / "src" / "mars_bot" / "mars_sim_driver"
    if not package.is_dir():
        raise SystemExit(f"mars_sim_driver not found under {sim_dir}")
    sys.path.insert(0, str(sim_dir))
    sys.path.insert(0, str(package))


def _profile(sim_profile: bool):
    from emet.controller.manipulation.wbc_profiles import example_mars_profile

    profile = example_mars_profile()
    group = profile.joint_group("primary_arm")
    if sim_profile:
        # Match mars_sim_driver.core's PD servo for a true apples-to-apples test.
        from mars_sim_driver.core import EFFORT_LIMIT, KD_JOINT, KP_JOINT

        joints = tuple(
            replace(joint, kp_eff=KP_JOINT, kd_eff=KD_JOINT, max_lead=0.2, torque_limit=EFFORT_LIMIT)
            for joint in group.joints
        )
    else:
        # Hardware-like bounds: the actual numbers are bench placeholders, but
        # this exercises the same clipping path the robot-side adapter uses.
        joints = tuple(
            replace(joint, kp_eff=1000.0, kd_eff=1.0, max_lead=0.05, torque_limit=2.0)
            for joint in group.joints
        )
    return replace(profile, joint_groups=(replace(group, joints=joints),))


def _observation(sim, profile):
    from emet.controller.manipulation.wbc_contract import (
        ArmObservation,
        GripperObservation,
        WbcObservation,
    )

    group = profile.joint_group("primary_arm")
    encoder = sim.encoder_positions()
    q = tuple(float(encoder[name]) for name in group.joint_names)
    qd = tuple(float(sim.data.qvel[sim._joints[name][1]]) for name in group.joint_names)
    grip = float(sim.data.qpos[sim._joints["joint6"][0]])
    return WbcObservation(
        base_pose=(0.0, 0.0, 0.0),
        base_twist=(0.0, 0.0),
        arms=(
            ArmObservation(
                name="primary",
                joint_position=q,
                joint_velocity=qd,
                gripper=GripperObservation(position=(grip,)),
            ),
        ),
        occupancy=None,
    )


def _action(profile, *, joint4_torque: float = 0.0, gripper: float = 0.0):
    from emet.controller.manipulation.wbc_contract import ArmAction, GripperCommand, WbcAction

    group = profile.joint_group("primary_arm")
    torques = [0.0] * len(group.joints)
    torques[3] = float(joint4_torque)  # wrist flex, a low-gravity joint
    return WbcAction(
        base_velocity=(0.0, 0.0),
        arms=(
            ArmAction(
                name="primary",
                joint_torques=tuple(torques),
                gripper=GripperCommand(values=(float(gripper),)),
            ),
        ),
    )


class _TorqueMars:
    """VirtualMars subclass that accepts direct joint torque for comparison."""

    def __new__(cls, *args, **kwargs):
        from mars_sim_driver.core import VirtualMars

        class TorqueMars(VirtualMars):
            def __init__(self):
                super().__init__()
                self._torques: dict[str, float] = {}

            def set_joint_torque(self, name: str, torque: float) -> None:
                self._torques[name] = float(torque)

            def _apply_control(self):
                super()._apply_control()
                for name, torque in self._torques.items():
                    _qadr, dadr, _target = self._joints[name]
                    limit = self._servo[dadr][1]
                    self.data.qfrc_applied[dadr] = float(np.clip(torque, -limit, limit))

        return TorqueMars(*args, **kwargs)


def _run_adapter(sim, profile, *, joint4_torque: float, steps: int, gravity=None):
    from emet.controller.manipulation.torque_adapter import PositionServoAdapter

    adapter = PositionServoAdapter(profile, gravity_torques=gravity)
    group = profile.joint_group("primary_arm")
    for _ in range(steps):
        command = adapter.command(_action(profile, joint4_torque=joint4_torque), _observation(sim, profile))
        for name in group.joint_names:
            sim.set_joint_target(name, command.joint_positions[name])
        sim.step(0.01)


def _run_direct(sim, profile, *, joint4_torque: float, steps: int):
    group = profile.joint_group("primary_arm")
    for _ in range(steps):
        for name in group.joint_names:
            sim.set_joint_torque(name, joint4_torque if name == "joint4" else 0.0)
        sim.step(0.01)


def _contact_forces(sim) -> list[dict]:
    import mujoco

    out = []
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        first = sim.model.geom(contact.geom1).name
        second = sim.model.geom(contact.geom2).name
        if "cube" not in first and "cube" not in second:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(sim.model, sim.data, i, force)
        out.append(
            {
                "geoms": [first, second],
                "force_n": float(np.linalg.norm(force[:3])),
                "normal_n": float(force[0]),
            }
        )
    return out


def _setup_contact(sim) -> None:
    sim.step(1.0)
    ee = sim.data.xpos[sim.model.body("robot_ee_link").id]
    if not sim.drop_prop_at("cube", float(ee[0]), float(ee[1])):
        raise RuntimeError("could not place the contact cube")
    sim.step(1.0)
    sim.set_joint_target("joint6", -0.6)  # close until the cube stops the jaw
    sim.step(1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--innate-sim",
        type=Path,
        default=Path("/tmp/opencode/innate-os/sim"),
        help="path to the external innate-os/sim checkout",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sim_dir = args.innate_sim.expanduser().resolve()
    if not (sim_dir / "assets").is_dir():
        raise SystemExit(
            f"{sim_dir}/assets is missing. From the innate-os checkout run `./innate-sim assets`."
        )
    _install_emet_import_stubs()
    _import_innate(sim_dir)

    from mars_sim_driver.core import VirtualMars

    from emet.controller.manipulation.torque_adapter import PositionServoAdapter
    from emet.controller.manipulation.wbc_contract import action_from_vector, action_vector

    profile = _profile(sim_profile=True)
    hardware_profile = _profile(sim_profile=False)

    report: dict[str, object] = {"innate_sim": str(sim_dir), "profile": profile.name}
    failures: list[str] = []

    # 1. Adapter vs direct torque, same command.
    adapter_sim = VirtualMars()
    adapter_sim.step(0.2)
    _run_adapter(adapter_sim, profile, joint4_torque=0.5, steps=100)
    direct_sim = _TorqueMars()
    direct_sim.step(0.2)
    _run_direct(direct_sim, profile, joint4_torque=0.5, steps=100)
    group = profile.joint_group("primary_arm")
    a = _observation(adapter_sim, profile).arm("primary")
    d = _observation(direct_sim, profile).arm("primary")
    q_index = group.joint_names.index("joint4")
    adapter_q4, direct_q4 = a.joint_position[q_index], d.joint_position[q_index]
    adapter_qd4, direct_qd4 = a.joint_velocity[q_index], d.joint_velocity[q_index]
    adapter_tau = adapter_sim.data.qfrc_applied[adapter_sim._joints["joint4"][1]]
    direct_tau = direct_sim.data.qfrc_applied[direct_sim._joints["joint4"][1]]
    report["adapter_vs_direct"] = {
        "adapter_q4": adapter_q4,
        "direct_q4": direct_q4,
        "q4_delta": abs(adapter_q4 - direct_q4),
        "adapter_qd4": adapter_qd4,
        "direct_qd4": direct_qd4,
        "adapter_applied_torque": adapter_tau,
        "direct_applied_torque": direct_tau,
    }
    if abs(adapter_q4 - direct_q4) > 0.03:
        failures.append("adapter/direct joint4 position differs by > 0.03 rad")
    if abs(adapter_qd4 - direct_qd4) > 0.03:
        failures.append("adapter/direct joint4 velocity differs by > 0.03 rad/s")

    # 2. Hardware-like clipping.
    hardware_adapter = PositionServoAdapter(hardware_profile)
    hardware_obs = _observation(adapter_sim, hardware_profile)
    command = hardware_adapter.command(_action(hardware_profile, joint4_torque=100.0), hardware_obs)
    lead = abs(command.joint_positions["joint4"] - hardware_obs.arm("primary").joint_position[q_index])
    report["limit_clipping"] = {
        "commanded_torque": command.commanded_torques["joint4"],
        "lead_rad": lead,
        "torque_limit": hardware_profile.joint_group("primary_arm").joints[3].torque_limit,
        "max_lead": hardware_profile.joint_group("primary_arm").joints[3].max_lead,
    }
    if abs(command.commanded_torques["joint4"]) > 2.0 + 1e-9:
        failures.append("commanded torque exceeded the 2.0 N*m validation limit")
    if lead > 0.05 + 1e-9:
        failures.append("position lead exceeded max_lead")

    # 3. Gravity feedforward remains bounded at the home pose.
    gravity_sim = VirtualMars()
    gravity_sim.step(0.2)
    start = np.array([gravity_sim.encoder_positions()[name] for name in group.joint_names])
    gravity = lambda _obs: {
        name: float(gravity_sim.data.qfrc_bias[gravity_sim._joints[name][1]]) for name in group.joint_names
    }
    _run_adapter(gravity_sim, profile, joint4_torque=0.0, steps=100, gravity=gravity)
    end = np.array([gravity_sim.encoder_positions()[name] for name in group.joint_names])
    drift = float(np.max(np.abs(end - start)))
    report["gravity_hold"] = {"max_joint_drift_rad": drift}
    if drift > 0.02:
        failures.append("gravity-feedforward hold drifted > 0.02 rad")

    # 4. Contact telemetry while a torque is applied.
    contact_sim = VirtualMars()
    _setup_contact(contact_sim)
    before = _contact_forces(contact_sim)
    _run_adapter(contact_sim, profile, joint4_torque=0.2, steps=100)
    after = _contact_forces(contact_sim)
    report["contact"] = {
        "before": before,
        "after": after,
        "gripper_position": float(contact_sim.joint_positions()["joint6"]),
        "cube_pose": contact_sim.object_poses().get("cube"),
    }
    if not after:
        failures.append("no cube contact was measured after the torque command")

    # 5. Multi-arm schema / vector layout.
    vector = action_vector(_action(profile), profile)
    rebuilt = action_from_vector(vector, profile)
    report["schema"] = {
        "arms": list(profile.arm_names),
        "arm_dof": len(group.joints),
        "gripper_dof": profile.arm("primary").gripper.dof,
        "action_dim": int(vector.size),
        "rebuilt_arms": list(rebuilt.arm_names) if hasattr(rebuilt, "arm_names") else [a.name for a in rebuilt.arms],
    }
    if vector.size != 8:
        failures.append(f"Mars action vector should be 8, got {vector.size}")

    report["failures"] = failures
    report["ok"] = not failures
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
