#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Write GT scene JSON from a running sim's ``emet_session.sim_object_placements``."""

from __future__ import annotations

import argparse
import sys

from emet.app.robot_cli import create_robot_client_from_cli
from emet.controller.generic_zmq_client import GenericZmqClient
from emet.memory.graph_eqa.sim_ground_truth_graph import read_sim_object_placements
from emet.robots import get_robot_spec
from emet.simulation.mujoco_gt_objects import (
    build_gt_scene_payload_from_session_placements,
    write_gt_scene_json,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", default="innate_mars")
    p.add_argument("--robot-ip", default="127.0.0.1")
    p.add_argument("--port-offset", type=int, default=0)
    p.add_argument("-o", "--out", required=True, help="Output .json path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layout", type=int, default=1)
    p.add_argument("--style", type=int, default=1)
    args = p.parse_args()

    spec = get_robot_spec(args.robot)
    if spec is None:
        print(f"Unknown robot {args.robot!r}", file=sys.stderr)
        return 1
    # Robocasa merged kitchens use the generic ZMQ client (innate_mars, galaxea_r1, stretch).
    if spec.name in ("innate_mars", "galaxea_r1", "rby1") or args.robot.lower().replace("-", "_") == "stretch":
        robot = GenericZmqClient(
            robot_ip=args.robot_ip,
            port_offset=args.port_offset,
            robot_spec=spec,
            enable_rerun_server=False,
            start_immediately=True,
        )
    else:
        robot = create_robot_client_from_cli(
            args.robot,
            args.robot_ip,
            port_offset=args.port_offset,
            enable_rerun_server=False,
            start_immediately=True,
        )
    if hasattr(robot, "wait_for_obs"):
        if not robot.wait_for_obs(timeout=60.0):
            print("Timed out waiting for first observation.", file=sys.stderr)
            return 1

    session = robot.get_emet_session()
    placements = read_sim_object_placements(session)
    if not placements:
        print("No sim_object_placements in emet_session.", file=sys.stderr)
        return 1

    payload = build_gt_scene_payload_from_session_placements(
        placements,
        robot=args.robot,
        seed=args.seed,
        layout=args.layout,
        style=args.style,
    )
    dest = write_gt_scene_json(args.out, payload)
    n = len(payload.get("objects", []))
    print(f"Wrote {n} manipulable GT objects -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
