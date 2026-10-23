"""Isolate payload/support contact settling from robot tracking and grasping."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from emet.rl.manipulation.models import build_manipulation_model


def audit(*, seconds=10.0, window=2.0, tilt=0.05):
    if not np.isfinite([seconds, window, tilt]).all() or not 0 < window < seconds or not 0 <= tilt <= 0.2:
        raise ValueError("invalid support audit settings")
    rows = []
    for name in ("object1", "object2"):
        model, fingerprint = build_manipulation_model("innate_mars")
        data = mujoco.MjData(model)
        # Initialization only: move the fixture away from the robot. There are
        # no object pose edits, disturbances, or robot commands during scoring.
        model.body_pos[model.body("table").id] = [1, 0, 0.1]
        for payload in ("object1", "object2"):
            address = int(model.jnt_qposadr[model.body(payload).jntadr[0]])
            data.qpos[address : address + 3] = [1, 0, 0.231] if payload == name else [3, 3, 0.2]
            if payload == name:
                data.qpos[address + 3 : address + 7] = [np.cos(tilt / 2), np.sin(tilt / 2), 0, 0]
        address = int(model.body(name).dofadr[0])
        peak = np.zeros(6)
        trace = []
        for index in range(round(seconds / model.opt.timestep)):
            mujoco.mj_step(model, data)
            velocity = data.qvel[address : address + 6].copy()
            if data.time >= seconds - window:
                peak = np.maximum(peak, abs(velocity))
            if index % 50 == 0:
                trace.append({"time": float(data.time), "velocity": velocity.tolist()})
        rows.append(
            {
                "payload": name,
                "fingerprint": fingerprint,
                "initial_tilt_rad": tilt,
                "seconds": seconds,
                "last_window_seconds": window,
                "peak_linear_velocity_m_s": float(max(peak[:3])),
                "peak_angular_velocity_rad_s": float(max(peak[3:])),
                "settled": bool(np.isfinite(peak).all() and max(peak) < 0.02),
                "trace": trace,
            }
        )
    return {
        "scope": "isolated simulation support contact; no grasp or policy",
        "cases": rows,
        "passed": all(row["settled"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": [{k: v for k, v in row.items() if k != "trace"} for row in report["cases"]],
            }
        )
    )


if __name__ == "__main__":
    main()
