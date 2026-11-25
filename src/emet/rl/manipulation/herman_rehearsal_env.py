"""Offline captured-room scene adapter; cannot create a robot connection."""

import hashlib
import json
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.env import ManipulationEnv
from emet.rl.manipulation.herman_rehearsal import finite_vector, limit_violations
from emet.rl.manipulation.models import build_manipulation_model


def camera_calibration(model, capture):
    from emet.rl.manipulation.herman_rehearsal import rectified_intrinsics

    sample = capture["samples"]["depth"]
    width, height = sample["width"], sample["height"]
    info = capture["samples"]["head_left_info"]
    k, r = rectified_intrinsics(info, width, height)
    cid = model.camera("head_left").id
    model.cam_resolution[cid] = [width, height]
    model.cam_sensorsize[cid] = [width, height]
    # MuJoCo 3.x stores principal offsets in sensor coordinates, rather than
    # OpenCV's absolute upper-left-origin pixel coordinates. Exact frustum
    # projection is checked by the renderer test, including off-center points.
    model.cam_intrinsic[cid] = [k[0, 0], k[1, 1], width / 2 - k[0, 2], k[1, 2] - height / 2]
    model.cam_ipd[cid] = 0
    model.cam_fovy[cid] = np.rad2deg(2 * np.arctan(height / (2 * k[1, 1])))
    matrix = r.T @ np.diag([1.0, -1.0, -1.0])
    model.cam_quat[cid] = Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    return {
        "head_left": {"intrinsics": k.tolist(), "rectification": r.tolist(), "resolution": [width, height]},
        "wrist": "model-only optics; no observed wrist image or optical TF",
    }


def scene_factory(
    capture,
    scene,
    *,
    forearm_collision="cad_hull",
    head_range_rad=None,
    grip_range_rad=None,
    scene_payload_half_size=None,
):
    def build(robot, **kwargs):
        model, fingerprint = build_manipulation_model(
            robot,
            scene_geoms=scene,
            hide_training_fixtures=True,
            forearm_collision=forearm_collision,
            head_range_rad=head_range_rad,
            grip_range_rad=grip_range_rad,
            scene_payload_half_size=scene_payload_half_size,
            **kwargs,
        )
        calibration = camera_calibration(model, capture)
        identity = hashlib.sha256((fingerprint + json.dumps(calibration, sort_keys=True)).encode()).hexdigest()
        return model, identity

    return build


class HermanRehearsalEnv(ManipulationEnv):
    """Reset an explicitly labeled admitted fixture; no runtime physics injection."""

    def __init__(self, config, *, initial_arm, initial_grip, initial_head, model_factory):
        self.initial_arm = finite_vector(initial_arm, 5).copy()
        self.initial_grip = float(initial_grip)
        self.initial_head = float(initial_head)
        if not np.isfinite([self.initial_grip, self.initial_head]).all():
            raise ValueError("finite measured auxiliary posture required")
        # Admission happens before any physical reset/step. In particular, the
        # actual captured posture is never silently clamped into the model.
        from emet.rl.manipulation.profiles import MARS

        if config.robot != MARS.robot:
            raise ValueError("Herman rehearsal requires Mars")
        configured = replace(
            config,
            scenario="reach",
            randomization_scale=0,
            initialization_attempts=1,
            reach_hold_probability=0,
            reach_local_probability=0,
            fixed_base_probability=0,
            certify_reach_paths=False,
            dynamic_references=False,
            retarget_probability=0,
            reach_push_probability=0,
            grasp_lift_probability=0,
            carry_hold_probability=0,
        )
        super().__init__(configured, mesh_storage="binary", model_factory=model_factory)
        limits = {
            name: list(self.model.joint(name).range.copy())
            for name in [*(f"joint{i}" for i in range(1, 7)), "joint_head"]
        }
        values = {
            **dict(zip((f"joint{i}" for i in range(1, 6)), self.initial_arm, strict=True)),
            "joint6": self.initial_grip,
            "joint_head": self.initial_head,
        }
        violations = limit_violations(values, limits)
        if violations:
            raise ValueError("rehearsal initial state exceeds model limits: " + json.dumps(violations))

    def _initial_arm_posture(self, sampled):
        # Called only during the parent's reset, before initialize()/history.
        self.data.qpos[self.model.joint("joint_head").qposadr[0]] = self.initial_head
        self.data.ctrl[self.model.actuator("joint_head").id] = self.initial_head
        return self.initial_arm.copy()

    def _initial_gripper_posture(self, sampled):
        return self.initial_grip

    def _make_reference(self):
        # The caller installs its admitted task-pose path through the normal
        # public set_task_trajectory API after reset. No unrelated random goal.
        self.control.set_reference(self.control.positions(), phase="reach")


def geometry_diagnostic(model, data):
    """Report active forearm clearance and a source CAD comparison.

    Direct geomDistance avoids broad-phase changes from flipping collision bits
    after compilation. This diagnoses model geometry, not physical clearance.
    """
    import mujoco

    base = model.geom("rl_base_link").id
    forearm = model.geom("rl_link3")
    distances = {
        name: float(mujoco.mj_geomDistance(model, data, geom, base, 0.5, np.zeros(6)))
        for name, geom in (
            ("active_signed_distance_m", forearm.id),
            ("cad_hull_signed_distance_m", model.geom("link3_geom").id),
        )
    }
    distances["forearm_collision"] = "cad_hull" if forearm.type[0] == mujoco.mjtGeom.mjGEOM_MESH else "proxy_box"
    return distances


def audit_parked_transition(model, data, measured, published_limits, *, settling_reserve_rad=0.0):
    """Review the vendor-style projection without resetting the recorded pose."""
    import mujoco

    from emet.rl.manipulation.profiles import MARS
    from emet.rl.manipulation.safety import ContactMonitor
    from emet.robots.innate_mars.startup_recovery import ENCODER_STEP_RAD, project_parked_hold

    try:
        target = project_parked_hold(measured, published_limits, settling_reserve_rad=settling_reserve_rad)
    except ValueError as exc:
        return {"state": "unavailable", "reason": str(exc), "hardware_admitted": False}
    deltas = {joint: target[joint] - measured[joint] for joint in target}
    if not any(deltas.values()):
        return {"state": "already_in_range", "target_rad": target, "hardware_admitted": False}

    size = max(map(abs, deltas.values()))
    if size > np.pi:
        return {
            "state": "unavailable",
            "reason": "outside the calibrated single-turn range",
            "hardware_admitted": False,
        }
    samples = int(np.ceil(size / ENCODER_STEP_RAD)) + 1
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    monitor = ContactMonitor(model, MARS)
    rows = []
    for fraction in np.linspace(0, 1, samples):
        for joint, delta in deltas.items():
            scratch.qpos[model.joint(joint).qposadr[0]] = measured[joint] + fraction * delta
        mujoco.mj_forward(model, scratch)
        pair_distances = {}
        for a, b, distance in zip(scratch.contact.geom1, scratch.contact.geom2, scratch.contact.dist, strict=True):
            if distance >= 0 or (a not in monitor.robot_geoms and b not in monitor.robot_geoms):
                continue
            if (a in monitor.floor and b in monitor.support_geoms) or (
                b in monitor.floor and a in monitor.support_geoms
            ):
                continue
            pair = tuple(sorted((model.geom(a).name, model.geom(b).name)))
            pair_distances[pair] = min(pair_distances.get(pair, 0), float(distance))
        rows.append(
            {
                "fraction": float(fraction),
                "forearm_chassis_gap_m": geometry_diagnostic(model, scratch)["active_signed_distance_m"],
                "contacts": [[*pair, dist] for pair, dist in pair_distances.items()],
            }
        )
    return {
        "state": "needs_contact_review" if any(row["contacts"] for row in rows) else "clear_in_static_model",
        "start_rad": {joint: measured[joint] for joint in target},
        "target_rad": target,
        "joint_delta_rad": deltas,
        "settling_reserve_rad": settling_reserve_rad,
        "hand_pitch_change_rad": sum(deltas[name] for name in ("joint2", "joint3", "joint4")),
        "static_sweep": rows,
        "hardware_admitted": False,
    }


def select_start_proposal(model, data, values, head):
    """Explicit synthetic staging search; retains attempted candidates, no motion."""
    import mujoco

    from emet.eval.physical_tamp import make_scene_checker
    from emet.rl.manipulation.coupled_tracking import reference_support_margin
    from emet.rl.manipulation.profiles import MARS

    checker = make_scene_checker(model, {"object_body": "object1", "gripper_bodies": MARS.finger_bodies})
    limits = np.array([model.joint("joint" + str(i)).range.copy() for i in range(1, 6)])
    observed = np.array([values["joint" + str(i)] for i in range(1, 6)])
    bounded = np.clip(observed, limits[:, 0] + 0.04, limits[:, 1] - 0.04)
    variations = sorted(
        (
            (shoulder, elbow)
            for shoulder in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
            for elbow in (0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6)
        ),
        key=lambda pair: pair[0] ** 2 + pair[1] ** 2,
    )
    attempts = []
    for shoulder, elbow in variations:
        q = bounded.copy()
        q[1] += shoulder
        q[2] += elbow
        row = {"joint_positions_rad": q.tolist(), "kind": "synthetic_staging_proposal"}
        if np.any(q < limits[:, 0] + 0.01) or np.any(q > limits[:, 1] - 0.01):
            attempts.append({**row, "admitted": False, "reason": "joint_margin"})
            continue
        mujoco.mj_resetData(model, data)
        for name, value in {
            **values,
            **{"joint" + str(i): v for i, v in enumerate(q, 1)},
            "joint6M": -values["joint6"],
            "joint_head": head,
        }.items():
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        collision = checker.configuration_collides(model, data)
        clearance = geometry_diagnostic(model, data)
        support = reference_support_margin(model, data, MARS)
        accepted = not collision and clearance["active_signed_distance_m"] >= 0.01 and support >= 0.01
        attempts.append(
            {
                **row,
                "admitted": accepted,
                "reason": "accepted" if accepted else "collision_or_clearance",
                "geometry": clearance,
                "support_margin_m": float(support),
                "contacts": checker.last_contacts.copy(),
            }
        )
        if accepted:
            return q, attempts
    return None, attempts
