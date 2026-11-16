"""Actor-only recurrent-window controller bundles; no privileged critic state."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from emet.rl.manipulation.base_navigation_feedback import BaseNavigationFeedback
from emet.rl.manipulation.profiles import PROFILES
from emet.rl.manipulation.whole_body_actor import WholeBodyActor
from emet.rl.manipulation.whole_body_observation import ACTOR_KEYS

FORMAT = "emet-whole-body-v2"
CONFIGURED_FORMAT = "emet-whole-body-v3"
MOTION_CONTRACT = "world_base_xy_yaw_and_hand_linear_angular_velocity_v1"


def export_actor(actor, destination, environment, fingerprint, *, steps, training=None):
    navigation = actor.base_navigation_feedback
    if navigation is not None:
        navigation.validate_tolerances(environment.base_xy_tolerance, environment.base_heading_tolerance)
    navigation_record = None if navigation is None else navigation.record()
    # Older v2 runtimes ignore unknown option fields. A changed feedback law
    # therefore needs a new format they reject, even with identical tensors.
    bundle_format = CONFIGURED_FORMAT if navigation is not None or not actor.base_rate_feedback else FORMAT
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
        directory = Path(temp)
        torch.save(
            {
                "format": bundle_format,
                "robot": actor.robot,
                "hidden": actor.hidden,
                "base_rate_feedback": actor.base_rate_feedback,
                "base_navigation_feedback": navigation_record,
                "state_dict": actor.state_dict(),
            },
            directory / "policy.zip",
        )
        manifest = {
            "format": bundle_format,
            "robot": actor.robot,
            "steps": steps,
            "environment": asdict(environment),
            "fingerprint": fingerprint,
            "contract": PROFILES[actor.robot].contract("direct_velocity"),
            "action_mode": "direct_velocity",
            "weights_sha256": hashlib.sha256((directory / "policy.zip").read_bytes()).hexdigest(),
            "actor_inputs": list(ACTOR_KEYS),
            "history_steps": PROFILES[actor.robot].history_steps,
            "motion_contract": MOTION_CONTRACT,
            "base_rate_feedback": actor.base_rate_feedback,
            "base_navigation_feedback": navigation_record,
            "critic_exported": False,
            "hardware_used": False,
            "safety_promoted": False,
            "training": training or {},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        policy = WholeBodyBundle(directory, robot=actor.robot, fingerprint=fingerprint)
        os.rename(directory, destination)
    return policy


class WholeBodyBundle:
    def __init__(self, directory, *, robot=None, fingerprint=None):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        m = self.manifest
        self.robot = m.get("robot")
        if (
            m.get("format") not in {FORMAT, CONFIGURED_FORMAT}
            or self.robot not in PROFILES
            or m.get("critic_exported") is not False
        ):
            raise ValueError("invalid actor-only whole-body bundle")
        if robot is not None and robot != self.robot or fingerprint is not None and fingerprint != m["fingerprint"]:
            raise ValueError("whole-body robot/model mismatch")
        if m.get("contract") != json.loads(json.dumps(PROFILES[self.robot].contract("direct_velocity"))):
            raise ValueError("whole-body actuator contract mismatch")
        if m.get("actor_inputs") != list(ACTOR_KEYS) or m.get("action_mode") != "direct_velocity":
            raise ValueError("whole-body input/action contract mismatch")
        if (
            m.get("history_steps") != PROFILES[self.robot].history_steps
            or m.get("motion_contract") != MOTION_CONTRACT
            or not m.get("environment", {}).get("retention_context")
            or not m.get("environment", {}).get("trajectory_preview")
        ):
            raise ValueError("whole-body history/reference contract mismatch")
        self.action_mode = "direct_velocity"
        raw = (directory / "policy.zip").read_bytes()
        if hashlib.sha256(raw).hexdigest() != m["weights_sha256"]:
            raise ValueError("whole-body policy checksum mismatch")
        data = torch.load(directory / "policy.zip", map_location="cpu", weights_only=True)
        if data.get("format") != m["format"] or data.get("robot") != self.robot:
            raise ValueError("whole-body weights metadata mismatch")
        if m["format"] == CONFIGURED_FORMAT and any(
            key not in record for record in (m, data) for key in ("base_rate_feedback", "base_navigation_feedback")
        ):
            raise ValueError("configured whole-body format requires explicit feedback options")
        if type(data.get("hidden")) is not int or not 1 <= data["hidden"] <= 1024:
            raise ValueError("invalid whole-body hidden width")
        # Missing fields are legacy bundles with the original rate path. Both
        # files must agree so an edited manifest cannot change actor behavior.
        rate = data.get("base_rate_feedback", True)
        declared_rate = m.get("base_rate_feedback", True)
        if type(rate) is not bool or type(declared_rate) is not bool or rate != declared_rate:
            raise ValueError("whole-body base feedback contract mismatch")
        navigation = BaseNavigationFeedback.from_record(data.get("base_navigation_feedback"))
        declared_navigation = BaseNavigationFeedback.from_record(m.get("base_navigation_feedback"))
        if navigation != declared_navigation:
            raise ValueError("whole-body base navigation contract mismatch")
        if navigation is not None and m["format"] != CONFIGURED_FORMAT:
            raise ValueError("base navigation feedback requires the configured v3 format")
        if navigation is not None:
            navigation.validate_tolerances(
                m["environment"].get("base_xy_tolerance"), m["environment"].get("base_heading_tolerance")
            )
        self.actor = WholeBodyActor(
            self.robot, data["hidden"], base_rate_feedback=rate, base_navigation_feedback=navigation
        )
        self.actor.load_state_dict(data["state_dict"], strict=True)
        if any(not torch.isfinite(value).all() for value in self.actor.state_dict().values()):
            raise ValueError("nonfinite whole-body weights")
        self.actor.eval()
        self.reference = None

    def bind_reference(self, reference):
        if not callable(reference):
            raise ValueError("announced reference velocity callback required")
        self.reference = reference

    def __call__(self, observation):
        obs = dict(observation)
        if set(obs) == {"history", "preview"} and self.reference is not None:
            obs["motion"] = self.reference()
        p = PROFILES[self.robot]
        fields = self.actor.encoder
        shapes = {"history": (p.history_steps, fields.width), "preview": (7, 18), "motion": (9,)}
        if set(obs) != set(ACTOR_KEYS):
            raise ValueError("sensor-only actor inputs required; privileged fields are forbidden")
        tensors = {}
        for name, shape in shapes.items():
            a = np.asarray(obs[name], dtype=np.float32)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError("invalid whole-body observation: " + name)
            tensors[name] = torch.from_numpy(a.copy())[None]
        with torch.no_grad():
            action = self.actor(tensors)[0].numpy().copy()
        if not np.isfinite(action).all() or np.any(abs(action) > 1):
            raise ValueError("invalid whole-body action")
        return action
