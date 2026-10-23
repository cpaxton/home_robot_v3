#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded MuJoCo named-view reproduction, with no emet imports or IK/physics calls.

Run in an isolated managed job under GDB. CPU affinity is intentionally supplied
by the caller and recorded; this script never changes the host configuration.
"""
import argparse
import importlib
import json
import os
import time
from pathlib import Path

import mujoco


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--scene', type=Path, default=Path(__file__).resolve().parents[1] / 'src/emet/assets/robot/scene.xml')
    parser.add_argument('--joints', nargs='+', help='Explicit named-access order; default is all limited joints in model order')
    parser.add_argument('--import-module', action='append', default=[], help='Optional extension-import isolation control')
    args = parser.parse_args()
    for module in args.import_module:
        importlib.import_module(module)
    if not 0 < args.seconds <= 60:
        parser.error('--seconds must be in (0, 60]')
    start = time.monotonic()
    lookups = 0
    print(json.dumps({'pid': os.getpid(), 'affinity': sorted(os.sched_getaffinity(0)),
                      'mujoco': mujoco.__version__, 'scene': str(args.scene)}), flush=True)
    while time.monotonic() - start < args.seconds:
        model = mujoco.MjModel.from_xml_path(str(args.scene))
        data = mujoco.MjData(model)
        names = args.joints or [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                 for i in range(model.njnt) if model.jnt_limited[i]]
        for _ in range(500):
            for name in names:
                joint = model.joint(name)
                assert 0 <= joint.qposadr[0] < model.nq
                float(data.qpos[joint.qposadr[0]])
                lookups += 1
    print(json.dumps({'lookups': lookups, 'seconds': time.monotonic() - start, 'completed': True}), flush=True)


if __name__ == '__main__':
    main()
