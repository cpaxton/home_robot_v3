#!/usr/bin/env python3
"""Exercise a live-updated Mars waypoint stream in MuJoCo."""

import argparse
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from emet.core.command_client import command_receipt
from emet.core.pose_stream import PoseStream
from emet.rl.manipulation.command_runtime import TaskPoseCommandRuntime
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.tracking import TaskFrameFK
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.manipulation.whole_body_observation import pose_rotation


def run(config, policy, seed, *, coupled):
    env = ManipulationEnv(config)
    try:
        env.reset(seed=seed)
        runtime = TaskPoseCommandRuntime(env, policy=policy)
        client = SimpleNamespace(_act_lock=threading.Lock(), _iter=0, _state=runtime.command_message({}))

        def dispatch(action):
            runtime.dispatch_command(action)
            client._state = runtime.command_message({})

        client.send_message = dispatch
        stream = PoseStream(client)
        control = env.control
        base, joints = control.base_pose().copy(), control.positions().copy()
        fk = TaskFrameFK(env.model, env.data, control)
        receipts = []

        def advance(action, steps):
            for _ in range(steps):
                runtime.advance()
                client._state = runtime.command_message({})
                receipt = command_receipt(client, action)
                if receipt["status"] != "running":
                    break
            return receipt

        for delta in (0.015, 0.04):
            target = joints.copy()
            target[2] += delta
            target_base = base.copy()
            if coupled:
                target_base[:2] += delta * np.array([np.cos(base[2]), np.sin(base[2])])
            pose = fk(target_base, target)
            quat = Rotation.from_matrix(pose_rotation(pose)).as_quat()
            waypoint = {"arms": {"arm": [*pose[:3], *quat]}}
            if coupled:
                waypoint["base"] = target_base.tolist()
            action = stream.send(waypoint, duration_s=2 if coupled else 1)
            receipt = advance(action, 8)
            receipts.append(receipt["status"])
            if receipt["status"] != "running":
                break
        if receipt["status"] == "running":
            stream.finish()
            receipt = advance(action, 280)
        result = receipt.get("result") or {}
        return {
            "waypoint_receipts": receipts,
            "status": receipt["status"],
            "stop_confirmed": result.get("stop_confirmed"),
            "result": result,
            "base_xy_drift_m": float(np.linalg.norm(control.base_pose()[:2] - base[:2])),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, help="Optional actor-only bundle to compare with the teacher")
    parser.add_argument("--seed", type=int, default=1900000)
    parser.add_argument("--coupled", action="store_true", help="Stream a 4 cm base translation with the arm")
    args = parser.parse_args()
    policy = WholeBodyBundle(args.actor, robot="innate_mars") if args.actor else None
    config = (
        ManipulationConfig(**policy.manifest["environment"])
        if policy
        else ManipulationConfig(
            robot="innate_mars", scenario="reach", action_mode="direct_velocity", trajectory_preview=True
        )
    )
    config = replace(config, scenario="reach", randomization_scale=0, fixed_base_contract=not args.coupled)
    report = {
        "seed": args.seed,
        "hardware": False,
        "coupled": args.coupled,
        "teacher": run(config, None, args.seed, coupled=args.coupled),
    }
    if policy:
        report["actor"] = run(config, policy, args.seed, coupled=args.coupled) | {
            "weights_sha256": policy.manifest["weights_sha256"],
            "safety_promoted": policy.manifest.get("safety_promoted") is True,
        }
    print(json.dumps(report, indent=2))
    return (
        0
        if all(
            case["status"] == "succeeded" and case["stop_confirmed"] is True
            for name, case in report.items()
            if name in ("teacher", "actor")
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
