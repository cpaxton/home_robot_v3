"""Explicit simulator-research calibration of a fixed learned feedback actor.

This changes checkpoint weights, never the runtime controller or safety bounds.
It adds no PPO training, teacher action, Jacobian prior, or release authority.
Every output has a new checksum and still requires independent task validation.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from emet.rl.manipulation.feedback_policy import FeedbackAction, GoalFeedbackFeatures


def scale_hand_feedback(head, profile, factor):
    """Scale hand-pose coefficients on arm rows; preserve base/grip/damping.

    The supported feedback tail is two (base SE(2), hand position/rotation6)
    samples, followed by measured joint and six-dimensional base velocities.
    This is the existing GoalFeedbackFeatures contract, not arbitrary layouts.
    """
    if isinstance(factor, bool) or not np.isfinite(factor) or not 0.25 <= factor <= 4:
        raise ValueError("finite hand feedback factor in [0.25, 4] required")
    width = 2 * (3 + 9) + profile.arm_size + 6
    if (
        not isinstance(head, FeedbackAction)
        or head.feedback_width != width
        or head.motion_size != profile.action_size - 1
    ):
        raise ValueError("unsupported learned feedback layout")
    offset = 2 if profile.wheel_bodies else 0
    rows = slice(offset, offset + profile.arm_size)
    columns = [*range(3, 12), *range(15, 24)]
    with torch.no_grad():
        weight = head.gain.weight.reshape(head.motion_size, width, -1)
        bias = head.gain.bias.reshape(head.motion_size, width)
        weight[rows, columns, :] *= factor
        bias[rows, columns] *= factor
    return {"arm_action_rows": list(range(offset, offset + profile.arm_size)), "feedback_columns": columns}


def calibrate_checkpoint(source, evidence_path, destination, *, factor):
    """Create an opt-in new bundle from an admitted source; leave source intact."""
    from emet.rl.manipulation.carry_benchmark import source_hashes
    from emet.rl.manipulation.env import ManipulationConfig
    from emet.rl.manipulation.hold_experiment import evidence
    from emet.rl.manipulation.models import build_manipulation_model
    from emet.rl.manipulation.policy import ManipulationPolicy, export
    from emet.rl.manipulation.profiles import PROFILES
    from emet.rl.manipulation.tamp_policy import verify_checkpoint

    source, evidence_path, destination = Path(source), Path(evidence_path), Path(destination)
    original = (source / "manifest.json").read_bytes()
    m = json.loads(original)
    cfg = ManipulationConfig(**m["environment"])
    model, fingerprint = build_manipulation_model(
        cfg.robot,
        **{
            k: getattr(cfg, k)
            for k in ("diagnostics", "payload_asset_set", "contact_cone", "contact_impratio", "two_supports")
        },
    )
    _, admission = verify_checkpoint(source, model, evidence_path)
    bundle = ManipulationPolicy(source, robot=cfg.robot, fingerprint=fingerprint)
    if not cfg.feedback_policy or not isinstance(bundle.model.policy.features_extractor, GoalFeedbackFeatures):
        raise ValueError("calibration requires a direct learned feedback checkpoint")
    selection = scale_hand_feedback(bundle.model.policy.action_net, PROFILES[cfg.robot], factor)
    # A transformed actor is a new experiment. Do not preserve optimizer
    # moments belonging to the untransformed weights for a later fine-tune.
    policy = bundle.model.policy
    policy.optimizer = policy.optimizer_class(
        policy.parameters(), lr=policy.optimizer.param_groups[0]["lr"], **policy.optimizer_kwargs
    )
    destination.mkdir(parents=True, exist_ok=False)
    normalizer = SimpleNamespace(
        obs_rms={
            key: SimpleNamespace(mean=mean.copy(), var=var.copy(), count=m["normalization"][key]["count"])
            for key, (mean, var) in bundle.normalization.items()
        },
        epsilon=m["epsilon"],
        clip_obs=m["clip"],
    )
    export(
        bundle.model, normalizer, destination, cfg, fingerprint, steps=m["steps"], held_estimator=bundle.held_estimator
    )
    metadata = {
        "kind": "hand_pose_feedback_gain",
        "factor": float(factor),
        **selection,
        "source_weights_sha256": m["weights_sha256"],
        "source_manifest_sha256": hashlib.sha256(original).hexdigest(),
        "source_admission": admission,
        "base_and_gripper_heads_unchanged": True,
        "measured_velocity_feedback_unchanged": True,
        "ppo_steps_added": 0,
        "optimizer_reset": True,
        "safety_promoted": False,
    }
    path = destination / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["calibration"] = metadata
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    run = {
        "env": json.loads(json.dumps(asdict(cfg))),
        "fingerprint": fingerprint,
        "source_hashes": source_hashes(),
        "operation": "post_training_gain_calibration",
        "source_checkpoint": str(source),
        "source_training_evidence": str(evidence_path),
        "calibration": metadata,
        "ppo_steps_added": 0,
    }
    (destination / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    report = evidence(destination, cfg, {"calibrated": manifest}, training_run=run)
    if (source / "manifest.json").read_bytes() != original or hashlib.sha256(
        (source / "policy.zip").read_bytes()
    ).hexdigest() != m["weights_sha256"]:
        raise RuntimeError("source checkpoint changed during calibration")
    return destination, report
