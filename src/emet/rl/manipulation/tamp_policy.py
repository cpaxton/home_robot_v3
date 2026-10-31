"""Checkpointed policy backend for the independently stepped Mars TAMP simulator.

Scene compatibility is deliberately narrow: the robot must retain its compiled
indices, geometry, dynamics, actuators and solver settings. Fixed supports and
primitive payloads may differ. This is compatibility admission, not a safety or
feasibility certificate; the TAMP planner and measured execution guards still run.
"""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from emet.rl.manipulation.models import build_manipulation_model
from emet.rl.manipulation.profiles import MARS


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def array_identity(value):
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": array.dtype.str, "sha256": sha256(array.tobytes())}


def robot_identity(model):
    """Conservative compiled Mars identity, including mesh bytes and couplings.

    All actuators, equalities, meshes and explicit contact overrides are bound.
    Extra actuated objects or mesh assets therefore require another admission
    path. Keeping robot indices fixed avoids ambiguous remapping of couplings.
    """
    root = model.body(MARS.base_body).id
    bodies = {root}
    for i in range(root + 1, model.nbody):
        if int(model.body_parentid[i]) in bodies:
            bodies.add(i)
    body_ids = np.array(sorted(bodies))
    selectors = {
        "body_": body_ids,
        "jnt_": np.flatnonzero(np.isin(model.jnt_bodyid, body_ids)),
        "dof_": np.flatnonzero(np.isin(model.dof_bodyid, body_ids)),
        "geom_": np.flatnonzero(np.isin(model.geom_bodyid, body_ids)),
    }
    values = {"body_ids": body_ids.tolist(), "robot": "innate_mars"}
    for name in dir(model):
        # Adding a fixed scene geom shifts the global BVH allocation, while
        # mesh topology/vertices and transforms below remain exactly bound.
        if name == "mesh_bvhadr":
            continue
        if name.startswith(tuple(selectors)):
            prefix = next(p for p in selectors if name.startswith(p))
            values[name] = array_identity(np.asarray(getattr(model, name))[selectors[prefix]])
        elif name.startswith(("actuator_", "eq_", "mesh_", "pair_", "exclude_", "tendon_", "wrap_")):
            values[name] = array_identity(getattr(model, name))
    joint_ids = selectors["jnt_"]
    qpos_ids = []
    for j in joint_ids:
        start = int(model.jnt_qposadr[j])
        width = 7 if model.jnt_type[j] == 0 else 4 if model.jnt_type[j] == 1 else 1
        qpos_ids.extend(range(start, start + width))
    values["qpos0"] = array_identity(model.qpos0[qpos_ids])
    values["qpos_spring"] = array_identity(model.qpos_spring[qpos_ids])
    for name in dir(model.opt):
        value = getattr(model.opt, name)
        if not name.startswith("_") and not callable(value):
            values["option/" + name] = array_identity(value)
    # Names are consumed by control and collision checking, not just rendering.
    for kind, ids in (("body", body_ids), ("joint", joint_ids), ("geom", selectors["geom_"])):
        values[kind + "_names"] = [getattr(model, kind)(int(i)).name for i in ids]
    values["actuator_names"] = [model.actuator(i).name for i in range(model.nu)]
    return sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode())


def verify_checkpoint(directory, model, evidence):
    """Bind a legacy checkpoint to archived training sources and this robot.

    The original full-scene fingerprint remains checked against a reconstructed
    training model. It is never disabled to admit a new scene.
    """
    from emet.rl.manipulation.carry_benchmark import source_hashes

    directory, evidence = Path(directory), Path(evidence)
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads(evidence.read_text())
    root = Path(__file__).resolve().parents[4]
    from emet.rl.artifacts import resolve_artifact

    archive = resolve_artifact(root / summary["full_report_archive"])
    raw = gzip.decompress(archive.read_bytes())
    if sha256(raw) != summary["uncompressed_sha256"]:
        raise ValueError("training evidence checksum mismatch")
    report = json.loads(raw)
    if not any(c.get("manifest") == manifest for c in report["checkpoints"].values()):
        raise ValueError("checkpoint is not bound to training evidence")
    sources, current = report["run"]["source_hashes"], source_hashes()
    required = {
        "rl/manipulation/models.py",
        "rl/manipulation/fixtures.py",
        "rl/manipulation/profiles.py",
        "rl/manipulation/policy.py",
        "rl/artifacts.py",
        "rl/manipulation/control.py",
        "rl/manipulation/turn_guard.py",
        "rl/manipulation/payload_assets.py",
        "rl/model.py",
        "robots/innate_mars/chassis.py",
        "@runtime",
    } | {k for k in current if k.startswith("assets/robot/")}
    if manifest.get("environment", {}).get("feedback_policy", False):
        required |= {
            "rl/manipulation/feedback_policy.py",
            "rl/manipulation/normalization.py",
            "rl/manipulation/observation_layout.py",
            "rl/manipulation/control.py",
            "rl/manipulation/tracking.py",
            "rl/pose_targets.py",
        }
    if any(k not in sources or sources[k] != current[k] for k in required):
        raise ValueError("training robot sources or runtime changed")
    cfg = manifest["environment"]
    if manifest["robot"] != "innate_mars" or manifest.get("action_mode") != "direct_velocity":
        raise ValueError("TAMP policy backend requires direct-velocity Mars policy")
    baseline, fingerprint = build_manipulation_model(
        manifest["robot"],
        **{k: cfg[k] for k in ("diagnostics", "payload_asset_set", "contact_cone", "contact_impratio", "two_supports")},
    )
    if fingerprint != manifest["fingerprint"]:
        raise ValueError("training model fingerprint changed")
    identity = robot_identity(baseline)
    if identity != robot_identity(model):
        raise ValueError("compiled robot or physics mismatch")
    if sha256((directory / "policy.zip").read_bytes()) != manifest["weights_sha256"]:
        raise ValueError("policy checksum mismatch")
    return manifest, {
        "format": "mars-tamp-policy-admission-v1",
        "robot_sha256": identity,
        "weights_sha256": manifest["weights_sha256"],
        "training_fingerprint": fingerprint,
        "training_evidence_sha256": sha256(raw),
        **({"policy_transform": manifest["calibration"]} if "calibration" in manifest else {}),
        "simulation_only": True,
        "safety_promoted": False,
    }


class CheckpointTampPolicy:
    """Fixed, deterministic actor; no teacher fallback or privileged inputs."""

    def __init__(self, directory, env, *, evidence):
        from emet.rl.manipulation.env import ManipulationConfig
        from emet.rl.manipulation.policy import ManipulationPolicy

        self.directory = Path(directory)
        manifest, self.admission = verify_checkpoint(directory, env.model, evidence)
        self._manifest_bytes = (self.directory / "manifest.json").read_bytes()
        self.policy = ManipulationPolicy(directory, robot=env.profile.robot, fingerprint=manifest["fingerprint"])
        from emet.rl.manipulation.feedback_policy import FeedbackActorCriticPolicy, GoalFeedbackFeatures

        feedback = isinstance(self.policy.model.policy, FeedbackActorCriticPolicy)
        if feedback != manifest["environment"].get("feedback_policy", False):
            raise ValueError("feedback policy metadata mismatch")
        if feedback and type(self.policy.model.policy.features_extractor) is not GoalFeedbackFeatures:
            raise ValueError("feedback feature contract mismatch")
        if feedback and self.policy.model.policy.features_extractor.orientation_scale != manifest["environment"].get(
            "feedback_orientation_scale", 1.0
        ):
            raise ValueError("feedback angular scale mismatch")
        self.calls = 0
        control = env.control
        if control.stopped or env._physics_invalid:
            raise ValueError("cannot attach policy to a stopped or invalid environment")
        control.retention_context_enabled = manifest["environment"].get("retention_context", False)
        execution = ManipulationConfig(**manifest["environment"])
        # Checkpoint settings may add protection, never disable an executor guard.
        control.turn_progress_guard_enabled |= execution.base_turn_progress_guard
        control.base_velocity_feedback_gain = execution.wheel_feedback_gain
        current_jerk = None if control.physical_wheel_jerk_limits is None else tuple(control.physical_wheel_jerk_limits)
        if (
            control.physical_wheel_slew_enabled != execution.physical_wheel_slew
            or current_jerk != execution.wheel_jerk_limits
        ):
            control.set_physical_wheel_slew(execution.physical_wheel_slew, jerk_limits=execution.wheel_jerk_limits)
        control.preview_enabled = control.pose_only_targets = True
        # Attachment happens once, before execution; never clear history at a
        # grasp/carry/place boundary or reset live physics to fit a checkpoint.
        control.history = None

    def __call__(self, observation):
        self.calls += 1
        return self.policy(observation)

    def verify_unchanged(self):
        if (self.directory / "manifest.json").read_bytes() != self._manifest_bytes or sha256(
            (self.directory / "policy.zip").read_bytes()
        ) != self.admission["weights_sha256"]:
            raise ValueError("checkpoint changed during execution")
