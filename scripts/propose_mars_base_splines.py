"""Generate and measure spatially stratified Mars base splines in simulation."""

import argparse
import json
from pathlib import Path

import numpy as np
from measure_mars_workspace import measure

from emet.rl.manipulation.arm_spline_benchmark import save
from emet.rl.manipulation.coupled_tracking import COARSE_SUITE, CoupledTrackingEnv
from emet.rl.manipulation.whole_body_splines import JointWheelPath

CASES = Path(__file__).resolve().parents[1] / "paper/data/whole_body_control/arm_only_mix_v1/cases.json"
RADIUS_M = 0.20
MAX_DURATION_S = 30.0
SEED = 20261120
PROPOSALS = 192


def fixture():
    cases = json.loads(CASES.read_text())
    case = next(case for case in cases if case["mode"] == "base_only" and case["split"] == "train")
    env = CoupledTrackingEnv("innate_mars", [case], suite_path=COARSE_SUITE)
    try:
        env.reset()
        control = env.unwrapped.control
        return (
            case,
            control.base_pose().copy(),
            control.axle_offset[:2].copy(),
            control.positions().copy(),
            control.ee_pose().copy(),
            control.profile,
        )
    finally:
        env.close()


def propose(workspace, *, seed=SEED):
    case, anchor, offset, joints, hand, profile = fixture()
    rng = np.random.default_rng(seed)
    speed = 0.40 * np.r_[profile.base_speed, profile.base_yaw_speed, profile.speed]
    acceleration = 0.50 * np.r_[profile.base_acceleration, profile.base_yaw_acceleration, profile.acceleration]
    episodes, accepted, rejected = [], [], {}
    for index in range(PROPOSALS):
        # Sample equal-angle sectors and equal-area radial bands.
        bearing = 2 * np.pi * (index % 16 + rng.random()) / 16 - np.pi
        band = index // 16 % 4
        travel = (RADIUS_M - 0.01) * np.sqrt((band + rng.random()) / 4)
        reverse = index // 64 % 2 == 1
        if reverse:
            bearing = np.arctan2(np.sin(bearing + np.pi), np.cos(bearing + np.pi))
            travel = -travel
        yaw = np.array([0.0, bearing, bearing, bearing + rng.uniform(-0.6, 0.6)])
        distance = np.array([0.0, 0.0, travel, travel])
        wheel_delta = yaw * profile.wheel_separation / 2
        wheels = np.column_stack((distance - wheel_delta, distance + wheel_delta))
        try:
            reference = JointWheelPath(
                np.tile(joints, (4, 1)),
                wheels,
                anchor,
                offset,
                profile.wheel_separation,
                speed,
                acceleration,
                4 * acceleration,
                initial_hold_s=0.0,
            )
        except ValueError:
            rejected[str(index)] = "timing"
            continue
        if reference.end > MAX_DURATION_S:
            rejected[str(index)] = "duration"
            continue
        base = np.array([reference.coordinates(t)[0] for t in np.linspace(0, reference.end, 201)])
        peak_radius = float(np.max(np.linalg.norm(base[:, :2] - anchor[:2], axis=1)))
        if peak_radius > RADIUS_M:
            rejected[str(index)] = "radius"
            continue
        identity = f"base-spline-proposal:{index}"
        accepted.append(
            {
                "id": identity,
                "reverse_requested": reverse,
                "wheel_travel_waypoints_m": wheels.tolist(),
                "duration_s": reference.end,
                "peak_sampled_radius_m": peak_radius,
            }
        )
        episodes.append({"id": identity, "base": base.tolist(), "hand": np.tile(hand, (len(base), 1)).tolist()})
    if not accepted:
        raise ValueError("No base spline proposals met the duration and sampled radius limits")
    return {
        **measure({**workspace, "episodes": episodes, "source_scope": "seeded differential-drive spline proposals"}),
        "scope": "geometric proposals, without physics or teacher admission",
        "attempted": PROPOSALS,
        "seed": seed,
        "radius_m": RADIUS_M,
        "max_duration_s": MAX_DURATION_S,
        "fixture_case": case,
        "accepted": accepted,
        "rejected": rejected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Workspace input with the baseline base grid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = json.loads(args.workspace.read_text())["families"]["base"]
    if workspace["frame"] != "base_start":
        raise ValueError("Expected a base_start workspace")
    result = propose(workspace)
    save(args.output, result)
    print(
        len(result["accepted"]),
        result["goal_coverage"]["occupied_bins"],
        result["path_coverage"]["occupied_bins"],
    )


if __name__ == "__main__":
    main()
