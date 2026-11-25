"""Opt-in WBC factory for the robot-local Mars bridge."""

import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from emet.robots.innate_mars.startup_recovery import JOINT_NAMES

VERIFICATION_KEYS = (
    "feedback_si_units_verified",
    "motor_read_validity_verified",
    "elbow_load_percent_verified",
)


def read_joint_limits(path):
    """Read the commissioned robot_description captured from this robot."""
    limits = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        name = joint.get("name")
        if name not in JOINT_NAMES:
            continue
        limit = joint.find("limit")
        if name in limits or limit is None or not {"lower", "upper"} <= limit.attrib.keys():
            raise ValueError(f"expected one joint limit for {name}")
        lower, upper = (float(limit.attrib[key]) for key in ("lower", "upper"))
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError(f"invalid joint limit for {name}")
        limits[name] = (lower, upper)
    if missing := set(JOINT_NAMES) - limits.keys():
        raise ValueError(f"robot_description missing joint limits: {', '.join(sorted(missing))}")
    return limits


def load_verified_actor(directory, verified_weights_sha256):
    from emet.rl.manipulation.env import ManipulationConfig
    from emet.rl.manipulation.models import build_manipulation_model
    from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle

    policy = WholeBodyBundle(directory, robot="innate_mars")
    if policy.manifest["weights_sha256"] != verified_weights_sha256:
        raise ValueError("the selected Mars actor has no matching hardware commissioning record")
    if policy.manifest.get("wheel_feedback") is not None:
        raise ValueError("this Mars transport does not implement the bundle's wheel feedback contract")
    source = ManipulationConfig(**policy.manifest["environment"])
    _, fingerprint = build_manipulation_model(
        source.robot,
        diagnostics=source.diagnostics,
        payload_asset_set=source.payload_asset_set,
        contact_cone=source.contact_cone,
        contact_impratio=source.contact_impratio,
        two_supports=source.two_supports,
        mesh_storage="binary",
    )
    if fingerprint != policy.manifest["fingerprint"]:
        raise ValueError("Mars actor source model fingerprint mismatch")
    return policy


def create_executor(node):
    """Build the selected executor from a local commissioning record."""
    import mujoco

    from emet.rl.manipulation.control import ManipulationControl
    from emet.rl.manipulation.models import build_manipulation_model
    from emet.rl.manipulation.profiles import MARS
    from emet.robots.innate_mars.wbc_runtime import MarsPoseExecutor

    config_path = Path(os.environ["EMET_MARS_WBC_CONFIG"])
    config = json.loads(config_path.read_text())
    limits = read_joint_limits(config_path.parent / config["robot_description"])
    verification = {key: config.get(key) is True for key in VERIFICATION_KEYS}
    policy = None
    if "actor" in config:
        policy = load_verified_actor(config_path.parent / config["actor"], config.get("verified_actor_weights_sha256"))
    model, _ = build_manipulation_model(
        MARS.robot,
        hide_training_fixtures=True,
        forearm_collision="cad_hull",
        grip_range_rad=limits["joint6"],
    )
    control = ManipulationControl(model, mujoco.MjData(model), MARS)
    transport = node.create_wbc_transport(driver_verification=lambda: verification.copy())
    return MarsPoseExecutor(transport=transport, control=control, published_limits=limits, policy=policy)
