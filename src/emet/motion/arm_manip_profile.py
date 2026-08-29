# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Per-robot arm / torso profiles for kinematic MuJoCo pick-place."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from emet.motion.mujoco_arm_ik import (
    RBY1_LEFT_ARM_JOINTS,
    RBY1_LEFT_EE_BODY,
    RBY1_RIGHT_ARM_JOINTS,
    RBY1_RIGHT_EE_BODY,
)


@dataclass(frozen=True)
class ArmManipProfile:
    """Joint / EE / actuator wiring for :class:`KinematicPickPlaceExecutor`."""

    robot_ids: tuple[str, ...]
    ee_body: str
    joint_names: tuple[str, ...]
    link_bodies: tuple[str, ...]
    actuator_names: tuple[str, ...]
    home_cmd: tuple[float, ...]
    base_freejoint_name: str = "base_freejoint"
    arm: str = "left"
    home_arm_q: tuple[float, ...] = field(default_factory=tuple)
    gripper_bodies: tuple[str, ...] = field(default_factory=tuple)

    def gripper_contact_bodies(self) -> tuple[str, ...]:
        """Bodies used to test object contact: the fingers/jaws when annotated,
        otherwise the end-effector body itself."""
        return self.gripper_bodies or (self.ee_body,)

    @staticmethod
    def for_robot(robot_id: str, *, arm: str = "left") -> ArmManipProfile:
        key = str(robot_id).lower().strip()
        arm_l = str(arm).lower().strip()
        for profile in _all_profiles():
            if key in profile.robot_ids and profile.arm == arm_l:
                return profile
        # Fall back to the robot's own spec + vendored MJCF so any registry robot with
        # an arm gets motion planning without a hardcoded table. Resolution order:
        #   1. declarative ``RobotSpec.arm_chain`` (curated per-robot)
        #   2. backend ``build_arm_manip_profile`` hook (code fallback)
        #   3. MJCF auto-discovery heuristic
        spec = _spec_for_robot_id(key)
        if spec is not None:
            chains = getattr(spec, "arm_chains", None) or {}
            chain = chains.get(arm_l) or getattr(spec, "arm_chain", None)
            if chain is not None:
                found = _profile_from_arm_chain(spec, chain, arm=arm_l)
                if found is not None:
                    return found
            backend = _backend_for_robot_id(key)
            if backend is not None:
                built = backend.build_arm_manip_profile(arm=arm_l)
                if built is not None:
                    return built
            found = ArmManipProfile.discover_from_spec(spec, arm=arm_l)
            if found is not None:
                return found
        raise KeyError(
            f"no ArmManipProfile for robot={robot_id!r} arm={arm!r}; known={[p.robot_ids for p in _all_profiles()]}"
        )

    @classmethod
    def discover_from_spec(
        cls, spec, *, arm: str = "left", robot_ids: tuple[str, ...] | None = None
    ) -> ArmManipProfile | None:
        """Build an :class:`ArmManipProfile` from a ``RobotSpec`` + vendored MJCF.

        Discovers the arm joint chain by side convention (``left_``/``right_`` prefix,
        ``*_L``/``*_R`` suffix, or a single un-suffixed arm) plus the deepest terminal
        link body (the end-effector). No per-robot hardcoding — works for sourccey,
        xlerobot, innate_mars, franka_fr3, … provided the MJCF exposes an arm chain.

        Returns ``None`` when the MJCF/spec does not expose a discoverable arm (callers
        keep the existing ``KeyError`` path so behavior is unchanged for unknown robots).
        """
        if spec is None or not getattr(spec, "mjcf_path", None):
            return None
        mjcf = Path(spec.mjcf_path)
        if not mjcf.is_file():
            return None
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(mjcf))
        arm_l = str(arm).lower().strip()
        joints = _discover_arm_joints(model, arm_l)
        if not joints:
            return None
        arm_root = _find_arm_root(model, joints)
        candidates = _arm_chain_bodies(model, arm_root) if arm_root else []
        if not candidates:
            return None
        ee_body = _pick_ee_body(candidates, arm_l, joints)
        link_bodies = tuple(b for b in candidates if b != arm_root) or (ee_body,)
        grippers = _pick_gripper_bodies(candidates, ee_body)
        jset = set(joints)
        act_names = tuple(a for a in getattr(spec, "actuator_names", ()) if _actuator_joint(spec, a) in jset) or ()
        home_arm_q = _home_q_from_mjcf(model, joints)
        n_act = len(getattr(spec, "actuator_names", ()))
        home_cmd = tuple(float(home_arm_q[j]) if j < len(joints) else 0.0 for j in range(n_act)) if home_arm_q else ()
        rid = tuple((robot_ids or ()) + ((spec.name,) if getattr(spec, "name", None) else ()))
        return cls(
            robot_ids=rid,
            ee_body=ee_body,
            joint_names=tuple(joints),
            link_bodies=link_bodies,
            actuator_names=act_names,
            home_cmd=home_cmd,
            arm=arm_l,
            home_arm_q=home_arm_q,
            gripper_bodies=grippers,
        )


def _spec_for_robot_id(robot_id: str):
    """Return a ``RobotSpec`` for a registry robot id, or None (unknown)."""
    try:
        from emet.robots import get_robot_spec
    except Exception:
        return None
    try:
        return get_robot_spec(robot_id)
    except Exception:
        return None


def _backend_for_robot_id(robot_id: str):
    """Return the ``RobotBackend`` for a registry robot id, or None (unknown)."""
    try:
        from emet.robots import get_robot_backend
    except Exception:
        return None
    try:
        return get_robot_backend(robot_id)
    except Exception:
        return None


def _profile_from_arm_chain(spec, chain, *, arm: str) -> ArmManipProfile | None:
    """Build an :class:`ArmManipProfile` from a declarative ``RobotSpec.arm_chain``.

    Returns ``None`` when the spec lacks a usable MJCF (the chain then stays inactive,
    e.g. Stretch before the merged-MJCF robosuite path lands).
    """
    mjcf = getattr(spec, "mjcf_path", None)
    if not mjcf or not Path(str(mjcf)).is_file():
        return None
    joints = list(getattr(chain, "joint_names", ()) or ())
    if not joints:
        return None
    ee_body = str(getattr(chain, "ee_body", "") or "")
    if not ee_body:
        return None
    act_names = list(getattr(chain, "actuator_names", ()) or ()) or [j for j in joints if _actuator_joint(spec, j)]
    if not act_names:
        act_names = [j for j in joints if _actuator_joint(spec, j)]
    if not act_names:
        return None

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    home_arm_q = list(getattr(chain, "home_arm_q", ()) or ()) or list(_home_q_from_mjcf(model, joints))
    # home_cmd aligned to the arm actuators actually driven (executor indexes home_cmd[i]
    # by profile.actuator_names order).
    home_cmd: list[float] = []
    for aname in act_names:
        jn = _actuator_joint(spec, aname)
        if jn in joints:
            j = joints.index(jn)
            home_cmd.append(float(home_arm_q[j]) if j < len(home_arm_q) else 0.0)
        else:
            home_cmd.append(0.0)

    link_bodies = list(getattr(chain, "link_bodies", ()) or ())
    if not link_bodies:
        # Derive the arm subtree bodies so collision sampling is not EE-only.
        root = _find_arm_root(model, joints)
        if root is not None:
            chain_bodies = _arm_chain_bodies(model, root)
            link_bodies = [b for b in chain_bodies if b != root] or [ee_body]
        else:
            link_bodies = [ee_body]
    gripper_bodies = list(getattr(chain, "gripper_bodies", ()) or ()) or [ee_body]
    return ArmManipProfile(
        robot_ids=(str(getattr(spec, "name", "") or ""),),
        ee_body=ee_body,
        joint_names=tuple(joints),
        link_bodies=tuple(link_bodies),
        actuator_names=tuple(act_names),
        home_cmd=tuple(home_cmd),
        base_freejoint_name=str(getattr(chain, "base_freejoint_name", "base_freejoint")),
        arm=arm,
        home_arm_q=tuple(home_arm_q),
        gripper_bodies=tuple(gripper_bodies),
    )


# Side tokens for arm-joint discovery: (prefix, suffix). A joint matches the
# requested arm when its name contains the side as a leading ``<side>_`` segment
# (sourccey ``left_shoulder_pan``, rby1 ``left_arm_joint1``) OR a trailing
# ``_<SIDE>`` segment (xlerobot ``Rotation_L``).
_SIDE_PREFIX = {"left": "left_", "right": "right_"}
_SIDE_SUFFIX = {"left": "_L", "right": "_R"}
# Single-arm robots (innate_mars, franka_fr3): joints without any side token, when
# the robot exposes exactly one arm chain. A joint is "arm-like" if its name contains
# a segment hint (arm/joint/shoulder/elbow/wrist/pitch/roll/servo).
_ARM_HINTS = ("shoulder", "elbow", "wrist", "arm", "pitch", "roll", "joint", "servo")


def _discover_arm_joints(model, arm: str) -> list[str]:
    """Return hinge joints belonging to the requested arm, in MJCF order.

    Matches ``<side>_`` prefix, ``_<SIDE>`` suffix, or (for single-arm robots) joints
    with arm-like names when no side token exists in the model at all.
    """
    import mujoco

    hinge = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    pref = _SIDE_PREFIX.get(arm, "")
    suff = _SIDE_SUFFIX.get(arm, "")
    if pref:
        matched = [n for n in hinge if n.startswith(pref)]
        if matched:
            return matched
    if suff:
        matched = [n for n in hinge if n.endswith(suff)]
        if matched:
            return matched
    # No side token anywhere -> single-arm robot (innate_mars, franka_fr3).
    has_any_side = any(n.startswith(("left_", "right_")) or n.endswith(("_L", "_R")) for n in hinge)
    if has_any_side:
        return []
    return [n for n in hinge if any(h in n.lower() for h in _ARM_HINTS)]


def _find_arm_root(model, joints: list[str]) -> str | None:
    """Pick the arm root body: the first joint's parent body (walk up to a body that
    is not itself driven by an arm joint)."""
    import mujoco

    for jname in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        bid = int(model.jnt_bodyid[jid])
        # root = the body this joint rotates; its parent is fixed/base (no joint)
        parent = int(model.body_parentid[bid])
        parent_joint = int(model.body_jntadr[parent])
        if parent_joint < 0:
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or None
    # fallback: first joint's body
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joints[0])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[jid])) if jid >= 0 else None


def _arm_chain_bodies(model, root_body: str) -> list[str]:
    """List body names in the kinematic subtree of *root_body*, parent-first (BFS)."""
    import mujoco

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    if root_id < 0:
        return []
    names: list[str] = []
    queue = [root_id]
    while queue:
        bid = queue.pop(0)
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if nm:
            names.append(nm)
        for c in range(model.nbody):
            if int(model.body_parentid[c]) == bid:
                queue.append(c)
    return names


def _pick_ee_body(candidates: list[str], arm: str, joints: tuple[str, ...] | list[str] = ()) -> str:
    """Choose the end-effector body from a chain: deepest gripper/jaw/finger/ee/hand leaf."""
    score: dict[str, int] = {}
    for name in candidates:
        low = name.lower().replace("-", "_")
        tokens = low.split("_")
        if tokens[0] in ("ee", "tcp", "tool") or "gripper" in tokens or "jaw" in tokens or "finger" in tokens:
            score[name] = 5
        elif "hand" in tokens:
            score[name] = 4
        elif "wrist" in tokens:
            score[name] = 3
        elif "link" in tokens and low[-1].isdigit():
            score[name] = 2
        else:
            score[name] = 1
    # prefer higher score; ties -> later (deeper) in the chain (candidates are BFS order)
    best = max(candidates, key=lambda n: (score.get(n, 0), candidates.index(n)))
    return best


# Tokens marking a body as part of the gripper (fingers/jaws). Ordered so that the
# two symmetric finger/jaw bodies (per arm) are preferred; the EE body is the fallback.
_GRIPPER_TOKENS = ("finger", "jaw", "gripper", "claw")
# Bodies that are *not* a grasp surface even though they carry a gripper token
# (camera holders, mounts, sensor pods).
_GRIPPER_EXCLUDE = ("camera", "holder", "mount", "sensor", "controller", "battery")


def _pick_gripper_bodies(candidates: list[str], ee_body: str) -> tuple[str, ...]:
    """Return finger/jaw bodies for contact testing, deepest/leaf-most preferred.

    Prefers bodies carrying a ``finger``/``jaw``/``gripper``/``claw`` token that are
    not camera/mount bodies; falls back to the EE body when the arm has no annotated
    gripper (e.g. innate_mars, franka_fr3). BFS order is parent-first, so the deepest
    two are the two symmetric fingers on dual-finger grippers.
    """
    grips = [
        b
        for b in candidates
        if any(t in b.lower().replace("-", "_").split("_") for t in _GRIPPER_TOKENS)
        and not any(t in b.lower() for t in _GRIPPER_EXCLUDE)
    ]
    if not grips:
        return (ee_body,)
    return tuple(grips[-2:])


def _actuator_joint(spec, actuator_name: str) -> str:
    """Return the joint an actuator drives, if inferable from the spec MJCF.

    Some specs store *joint* names in ``actuator_names``; accept those directly.
    """
    mjcf = getattr(spec, "mjcf_path", None)
    if not mjcf:
        return ""
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(mjcf))
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(actuator_name)) >= 0:
            return str(actuator_name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, str(actuator_name))
        if aid < 0:
            return ""
        jid = int(model.actuator_trnid[aid, 0])
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
    except Exception:
        return ""


def _home_q_from_mjcf(model, joint_names: tuple[str, ...] | list[str]) -> tuple[float, ...]:
    """Read qpos0 (compiled defaults) for the arm joints."""
    import mujoco

    q = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            q.append(0.0)
            continue
        q.append(float(model.qpos0[int(model.jnt_qposadr[jid])]))
    return tuple(q)


_PROFILES: list[ArmManipProfile] | None = None


def _galaxea_profile(*, arm: str) -> ArmManipProfile:
    from emet.robots.galaxea_r1 import R1_ACTUATOR_NAMES
    from emet.robots.rby1 import Rby1Backend

    mjcf = Path(Rby1Backend().get_spec().mjcf_path)
    if not mjcf.is_file():
        raise FileNotFoundError(f"missing Galaxea MJCF: {mjcf}")
    if arm == "right":
        joints = tuple(f"torso_joint{i}" for i in range(1, 5)) + RBY1_RIGHT_ARM_JOINTS
        links = tuple(f"right_arm_link{i}" for i in range(3, 7))
        ee = RBY1_RIGHT_EE_BODY
        grippers = ("right_gripper_finger_link1", "right_gripper_finger_link2")
    else:
        joints = tuple(f"torso_joint{i}" for i in range(1, 5)) + RBY1_LEFT_ARM_JOINTS
        links = tuple(f"left_arm_link{i}" for i in range(3, 7))
        ee = RBY1_LEFT_EE_BODY
        grippers = ("left_gripper_finger_link1", "left_gripper_finger_link2")
    home_arm = (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0)
    # Matches galaxea_r1.xml key name="home" ctrl (26 actuators).
    home_ctrl = (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.5,
        -0.5,
        0,
        0,
        0,
        0.04,
        0.04,
        0,
        0.5,
        -0.5,
        0,
        0,
        0,
        0.04,
        0.04,
    )
    return ArmManipProfile(
        robot_ids=("rby1", "galaxea_r1"),
        ee_body=ee,
        joint_names=joints,
        link_bodies=links,
        actuator_names=tuple(R1_ACTUATOR_NAMES),
        home_cmd=home_ctrl,
        arm=arm,
        home_arm_q=home_arm,
        gripper_bodies=grippers,
    )


def _all_profiles() -> list[ArmManipProfile]:
    global _PROFILES
    if _PROFILES is None:
        _PROFILES = [_galaxea_profile(arm="left"), _galaxea_profile(arm="right")]
    return _PROFILES


def resolve_manip_mode_for_robot(robot: object, *, manip_mode: str = "auto") -> str:
    """Return ``kinematic`` or ``teleport`` from session capabilities."""
    mode = str(manip_mode or "auto").lower().strip()
    sess: dict = {}
    if hasattr(robot, "get_emet_session"):
        try:
            sess = robot.get_emet_session() or {}
        except Exception:
            sess = {}
    caps = sess.get("capabilities") or {}
    if mode == "auto":
        if caps.get("kinematic_manip"):
            return "kinematic"
        if caps.get("sim_set_body_pose"):
            return "teleport"
        raise RuntimeError("auto manip: server advertises neither kinematic_manip nor sim_set_body_pose")
    if mode == "kinematic":
        if not caps.get("kinematic_manip"):
            raise RuntimeError("manip_mode=kinematic but server lacks kinematic_manip")
        return "kinematic"
    if mode == "teleport":
        if not caps.get("sim_set_body_pose"):
            raise RuntimeError("manip_mode=teleport but server lacks sim_set_body_pose")
        return "teleport"
    raise ValueError(f"unknown manip_mode={manip_mode!r}")


def home_arm_q_array(profile: ArmManipProfile) -> np.ndarray:
    q = profile.home_arm_q
    if len(q) != len(profile.joint_names):
        raise ValueError(f"home_arm_q len {len(q)} != joints {len(profile.joint_names)}")
    return np.asarray(q, dtype=np.float64)


def has_arm_manip_profile(robot_id: str, *, arm: str = "left") -> bool:
    """True when :meth:`ArmManipProfile.for_robot` would succeed."""
    try:
        ArmManipProfile.for_robot(robot_id, arm=arm)
        return True
    except KeyError:
        return False


def robot_id_from_client(robot: object) -> str:
    """Best-effort robot id from GenericZmqClient / Stretch client.

    Raises ``ValueError`` when the id cannot be resolved (do not guess ``rby1``).
    """
    spec = getattr(robot, "_spec", None)
    if spec is not None and getattr(spec, "name", None):
        return str(spec.name)
    for attr in ("robot_id", "robot_name", "_robot_id"):
        v = getattr(robot, attr, None)
        if v:
            return str(v)
    get_sess = getattr(robot, "get_emet_session", None)
    if callable(get_sess):
        try:
            sess = get_sess() or {}
        except Exception:
            sess = {}
        if isinstance(sess, dict):
            for key in ("emet_robot_id", "robot_id"):
                v = sess.get(key)
                if v:
                    return str(v)
            caps_rid = (sess.get("capabilities") or {}).get("emet_robot_id")
            if caps_rid:
                return str(caps_rid)
    raise ValueError("cannot resolve robot id from client (missing _spec / session emet_robot_id)")
