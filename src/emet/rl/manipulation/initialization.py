"""Explicit training-only weight transfer, with tightly scoped physics changes.

This is a fresh optimization experiment, never admission of an old checkpoint
to another runtime model. The new policy still needs independent validation.
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from emet.rl.manipulation.models import build_manipulation_model
from emet.rl.manipulation.policy import ManipulationPolicy
from emet.rl.manipulation.profiles import PROFILES
from emet.rl.manipulation.tamp_policy import robot_identity


def _model(robot, config):
    return build_manipulation_model(
        robot,
        **{
            "diagnostics": config.get("diagnostics", False),
            "payload_asset_set": config.get("payload_asset_set", "training"),
            "two_supports": config.get("two_supports", False),
            "contact_cone": config.get("contact_cone", "pyramidal"),
            "contact_impratio": config.get("contact_impratio", 1.0),
        },
    )


def _load_source(directory, manifest, fingerprint, robot):
    """Stage only the known v6→v7 feedback upgrade for training, never runtime."""
    expected = json.loads(json.dumps(PROFILES[robot].contract(manifest["action_mode"])))
    old = manifest["contract"]
    if old == expected:
        return ManipulationPolicy(directory, robot=robot, fingerprint=fingerprint), {}
    allowed = {"version", "wheel_velocity_feedback"}
    if not (
        old.get("version") == 6
        and expected.get("version") == 7
        and old.get("wheel_velocity_feedback") is None
        and expected.get("wheel_velocity_feedback") == "bounded_pi_v1"
        and {k: v for k, v in old.items() if k not in allowed}
        == {k: v for k, v in expected.items() if k not in allowed}
    ):
        raise ValueError("unsupported transfer control contract change")
    changes = {key: {"from": old.get(key), "to": expected.get(key)} for key in sorted(allowed)}
    # The original bundle stays immutable and continues to fail normal runtime
    # loading. Its bytes/checksums remain the recorded source of this new actor.
    with tempfile.TemporaryDirectory(prefix="emet-training-transfer-") as temporary:
        staged = Path(temporary)
        for path in Path(directory).iterdir():
            if path.is_file():
                shutil.copyfile(path, staged / path.name)
        (staged / "manifest.json").write_text(json.dumps({**manifest, "contract": expected}))
        policy = ManipulationPolicy(staged, robot=robot, fingerprint=fingerprint)
    return policy, changes


def transfer_actor(model, normalizer, directory, config, fingerprint, *, prior_count=1024):
    """Copy a compatible actor/critic, append zero-influence retention columns.

    Only contact cone/impedance-ratio and the declared v6→v7 wheel-feedback
    contract upgrade are allowed. Robot
    geometry, actuators, joints, coupling and every other solver option must
    match. Existing normalizations are not rewritten in place.
    """
    if not np.isfinite(prior_count) or prior_count <= 0:
        raise ValueError("normalization prior count must be finite and positive")
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    source_model, source_fp = _model(config.robot, manifest["environment"])
    if source_fp != manifest["fingerprint"]:
        raise ValueError("transfer source model cannot be reconstructed")
    from dataclasses import asdict

    target_model, target_fp = _model(config.robot, asdict(config))
    if target_fp != fingerprint:
        raise ValueError("transfer target model mismatch")
    changes = {}
    for name in ("cone", "impratio"):
        old, new = getattr(source_model.opt, name), getattr(target_model.opt, name)
        if old != new:
            changes[name] = {"from": old, "to": new}
        # Change only a private reconstructed model for identity comparison.
        setattr(source_model.opt, name, new)
    if config.robot != "innate_mars" or robot_identity(source_model) != robot_identity(target_model):
        raise ValueError("transfer robot or unsupported physics mismatch")
    old, contract_changes = _load_source(directory, manifest, source_fp, config.robot)
    from emet.rl.manipulation.feedback_policy import GoalFeedbackFeatures
    from emet.rl.manipulation.policy import ManipulationMLPFeatures

    new_feedback_head = (
        isinstance(model.policy.features_extractor, GoalFeedbackFeatures)
        and type(old.model.policy.features_extractor) is ManipulationMLPFeatures
    )
    if (
        old.action_mode != config.action_mode
        or config.action_mode != "direct_velocity"
        or (
            type(old.model.policy.features_extractor) is not type(model.policy.features_extractor)
            and not new_feedback_head
        )
        or ("preview" in old.normalization) != config.trajectory_preview
        or not manifest.get("pose_only_targets", False)
        or (manifest["environment"].get("retention_context", False) and not config.retention_context)
    ):
        raise ValueError("transfer observation/action semantics mismatch")
    source, target = old.model.policy.state_dict(), model.policy.state_dict()
    old_shape, new_shape = old.mean.shape, normalizer.obs_rms["history"].mean.shape
    old_restoring = getattr(old.model.policy, "restoring_yaw_feedback", False)
    new_restoring = getattr(model.policy, "restoring_yaw_feedback", False)
    if old_restoring and not new_restoring:
        raise ValueError("cannot silently remove restoring yaw feedback during transfer")
    source_keys = {k for k in source if not (new_feedback_head and k.startswith("action_net."))}
    target_keys = {k for k in target if not (new_feedback_head and k.startswith("action_net."))}
    if new_restoring and not old_restoring:
        target_keys -= {"action_net.turn_gain.weight", "action_net.turn_gain.bias"}
    if source_keys != target_keys or old_shape[0] != new_shape[0] or new_shape[1] - old_shape[1] not in (0, 2):
        raise ValueError("transfer policy layout mismatch")
    for name, value in source.items():
        if new_feedback_head and name.startswith("action_net."):
            continue
        if value.shape == target[name].shape:
            target[name] = value.clone()
        elif name.endswith("project.weight") and value.shape[1] == old_shape[1]:
            target[name].zero_()
            target[name][:, : old_shape[1]] = value
        elif name in {"mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"}:
            steps = min(4, old_shape[0])
            expanded = torch.zeros_like(target[name])
            expanded[:, : steps * new_shape[1]].reshape(value.shape[0], steps, new_shape[1])[:, :, : old_shape[1]] = (
                value[:, : steps * old_shape[1]].reshape(value.shape[0], steps, old_shape[1])
            )
            tail = value[:, steps * old_shape[1] :]
            expanded[:, steps * new_shape[1] : steps * new_shape[1] + tail.shape[1]] = tail
            target[name] = expanded
        else:
            raise ValueError(f"unsupported transfer parameter: {name}")
    scale_change = None
    if isinstance(old.model.policy.features_extractor, GoalFeedbackFeatures) and not new_feedback_head:
        old_scale = old.model.policy.features_extractor.orientation_scale
        new_scale = model.policy.features_extractor.orientation_scale
        if old_scale != new_scale:
            # A declared feature-units migration preserves the deterministic
            # actor/critic functions before training. Both context and mean
            # gain inputs use this tail; compensate both, with a fresh optimizer.
            ratio = new_scale / old_scale
            angular = [*range(6, 12), *range(18, 24)]
            width = model.policy.features_extractor.feedback_width
            for name in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
                target[name][:, [target[name].shape[1] - width + i for i in angular]] *= ratio
            for name in ("action_net.gain.weight", "action_net.gain.bias"):
                values = target[name].reshape(model.action_space.shape[0] - 1, width, -1)
                values[:, angular] *= ratio
            scale_change = {"from": old_scale, "to": new_scale, "function_preserving": True}
    model.policy.load_state_dict(target)
    for key, (mean, var) in old.normalization.items():
        stats = normalizer.obs_rms[key]
        stats.mean.fill(0)
        stats.var.fill(1)
        stats.mean[..., : mean.shape[-1]] = mean
        stats.var[..., : var.shape[-1]] = var
        stats.count = min(prior_count, manifest.get("normalization", {}).get(key, {}).get("count", manifest["count"]))
    normalizer.epsilon, normalizer.clip_obs = manifest["epsilon"], manifest["clip"]
    if config.categorical_features_unscaled:
        normalizer.pin_command_features()
    return {
        "source_weights_sha256": manifest["weights_sha256"],
        "source_fingerprint": source_fp,
        "target_fingerprint": target_fp,
        "contact_solver_changes": changes,
        "control_contract_changes": contract_changes,
        "source_manifest_sha256": hashlib.sha256((Path(directory) / "manifest.json").read_bytes()).hexdigest(),
        "optimizer_restored": False,
        "feedback_mean_head_reinitialized": new_feedback_head,
        "restoring_yaw_feedback_change": {
            "from": old_restoring,
            "to": new_restoring,
            "function_preserving": old_restoring == new_restoring,
        },
        "feedback_orientation_scale_change": scale_change,
        "wheel_feedback_gain_change": {
            "from": manifest["environment"].get("wheel_feedback_gain", 2.0),
            "to": config.wheel_feedback_gain,
        },
        "physical_wheel_slew_change": {
            "from": manifest["environment"].get("physical_wheel_slew", False),
            "to": config.physical_wheel_slew,
            "dynamics_preserving": manifest["environment"].get("physical_wheel_slew", False)
            == config.physical_wheel_slew,
        },
        "wheel_jerk_limits_change": {
            "from": manifest["environment"].get("wheel_jerk_limits"),
            "to": config.wheel_jerk_limits,
            "dynamics_preserving": manifest["environment"].get("wheel_jerk_limits")
            == (None if config.wheel_jerk_limits is None else list(config.wheel_jerk_limits)),
        },
        "normalization_prior_count": prior_count,
        "base_turn_progress_guard_change": {
            "from": manifest["environment"].get("base_turn_progress_guard", False),
            "to": config.base_turn_progress_guard,
        },
        "categorical_features_unscaled": config.categorical_features_unscaled,
        "runtime_admitted": False,
        "safety_promoted": False,
    }
