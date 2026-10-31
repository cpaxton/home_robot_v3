"""Compute-only Mars backend parity and throughput probe; no training promotion.

First generate --cpu-reference with the validated native MuJoCo installation.
Then run --reference on the GPU host. Physics options and geometry stay intact.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np

CASES = {"hold": (0.0, 0.0), "forward": (0.03, 0.0), "left": (0.0, 0.15), "right": (0.0, -0.15)}
THRESHOLDS = {"base_position_m": 0.001, "base_orientation_rad": 0.01, "joint_position": 0.02}


def load(directory):
    state = json.loads((directory / "state.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(directory / "model.xml"))
    if int(model.opt.noslip_iterations) != state["noslip_iterations"]:
        raise ValueError("no-slip physics changed during export")
    if not np.isclose(model.opt.timestep, state["physics_dt"], rtol=0, atol=1e-12):
        raise ValueError("physics cadence changed during export")
    return model, state


def data_for(model, state):
    data = mujoco.MjData(model)
    data.qpos[:] = state["qpos"]
    data.qvel[:] = state["qvel"]
    data.ctrl[:] = state["ctrl"]
    data.time = state["time"]
    mujoco.mj_forward(model, data)
    return data


def command(state, case, index, cadence):
    ctrl = np.asarray(state["ctrl"]).copy()
    # Smooth half-second onset, using the same low-speed commands in both engines.
    t = np.clip(index * cadence / 0.5, 0, 1)
    scale = 10 * t**3 - 15 * t**4 + 6 * t**5
    v, w = np.asarray(CASES[case]) * scale
    wheels = np.array([v - w * state["wheel_separation"] / 2, v + w * state["wheel_separation"] / 2])
    ctrl[state["wheel_act"]] = wheels / state["wheel_radius"] * np.asarray(state["wheel_gear"])
    return ctrl


def cpu_reference(model, state, ticks, substeps):
    results = {}
    start = time.perf_counter()
    for case in CASES:
        data = data_for(model, state)
        trajectory = []
        for index in range(ticks):
            data.ctrl[:] = command(state, case, index, substeps * model.opt.timestep)
            mujoco.mj_step(model, data, nstep=substeps)
            trajectory.append(data.qpos.copy().tolist())
        results[case] = trajectory
    return {"mujoco": mujoco.__version__, "cases": results, "seconds": time.perf_counter() - start}


def errors(actual, expected, state):
    a, b = np.asarray(actual), np.asarray(expected)
    q = state["base_qpos"]
    cosine = np.abs(np.sum(a[:, q + 3 : q + 7] * b[:, q + 3 : q + 7], axis=1))
    # Compare all non-base joint coordinates conservatively (including wheel travel).
    mask = np.ones(a.shape[1], dtype=bool)
    mask[q : q + 7] = False
    metrics = {
        "base_position_m": float(np.max(np.linalg.norm(a[:, q : q + 3] - b[:, q : q + 3], axis=1))),
        "base_orientation_rad": float(np.max(2 * np.arccos(np.clip(cosine, 0, 1)))),
        "joint_position": float(np.max(np.abs(a[:, mask] - b[:, mask]))),
    }
    return {"peaks": metrics, "within_probe_thresholds": all(metrics[k] <= v for k, v in THRESHOLDS.items())}


def gpu_probe(model, state, reference, ticks, substeps, batches, iterations):
    import mujoco_warp as mw
    import warp as wp

    wp.init()
    rows, traces = [], {}
    with wp.ScopedDevice("cuda:0"):
        gpu_model = mw.put_model(model)
        for worlds in batches:
            # Bounded allocation; capacity overflow invalidates the probe.
            data = mw.put_data(model, data_for(model, state), nworld=worlds, nconmax=128, njmax=512)
            mw.step(gpu_model, data)  # Compile outside graph capture and measurement.
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                for _ in range(substeps):
                    mw.step(gpu_model, data)
            for _ in range(5):
                wp.capture_launch(capture.graph)
            wp.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                wp.capture_launch(capture.graph)
            wp.synchronize()
            elapsed = time.perf_counter() - start
            if not np.isfinite(data.qpos.numpy()).all():
                raise RuntimeError("non-finite GPU state")
            contacts = int(np.max(data.nacon.numpy()))
            if contacts >= worlds * 128:
                raise RuntimeError("GPU contact capacity exhausted")
            rows.append(
                {
                    "worlds": worlds,
                    "seconds": elapsed,
                    "physics_steps_per_second": worlds * iterations * substeps / elapsed,
                    "control_steps_per_second": worlds * iterations / elapsed,
                    "per_world_realtime_factor": iterations * substeps * model.opt.timestep / elapsed,
                    "contacts": contacts,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
            del data
        for case in CASES:
            data = mw.put_data(model, data_for(model, state), nworld=1, nconmax=128, njmax=512)
            with wp.ScopedCapture() as capture:
                for _ in range(substeps):
                    mw.step(gpu_model, data)
            trajectory = []
            for index in range(ticks):
                data.ctrl.assign(command(state, case, index, substeps * model.opt.timestep)[None].astype(np.float32))
                wp.capture_launch(capture.graph)
                wp.synchronize()
                trajectory.append(data.qpos.numpy()[0].tolist())
                if int(np.max(data.nacon.numpy())) >= 128:
                    raise RuntimeError("GPU contact capacity exhausted")
            traces[case] = trajectory
    parity = {case: errors(traces[case], reference["cases"][case], state) for case in CASES}
    return {"warp": wp.__version__, "throughput": rows, "cases": traces, "parity_vs_validated_cpu": parity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--ticks", type=int, default=40)
    parser.add_argument("--substeps", type=int, default=25)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists() or min(args.ticks, args.substeps, args.iterations, *args.batches) < 1:
        parser.error("fresh output and positive counts required")
    if max(args.batches) > 256:
        parser.error("probe is bounded to 256 worlds")
    model, state = load(args.model)
    result = {
        "format": "emet-mars-parallel-probe-v1",
        "hardware_used": False,
        "training_qualified": False,
        "ticks": args.ticks,
        "substeps": args.substeps,
        "physics_dt": model.opt.timestep,
        "noslip_iterations": model.opt.noslip_iterations,
        "thresholds": THRESHOLDS,
        "model_sha256": hashlib.sha256((args.model / "model.xml").read_bytes()).hexdigest(),
        "state_sha256": hashlib.sha256((args.model / "state.json").read_bytes()).hexdigest(),
    }
    if args.cpu_reference:
        result.update(cpu_reference(model, state, args.ticks, args.substeps))
    else:
        if not args.reference:
            parser.error("GPU probe requires --reference from the validated CPU installation")
        reference = json.loads(args.reference.read_text())
        for field in ("ticks", "substeps", "physics_dt", "noslip_iterations", "model_sha256", "state_sha256"):
            if reference[field] != result[field]:
                raise ValueError(f"reference mismatch: {field}")
        result["native_cpu"] = cpu_reference(model, state, args.ticks, args.substeps)
        result["cpu_parity"] = {
            case: errors(result["native_cpu"]["cases"][case], reference["cases"][case], state) for case in CASES
        }
        os.environ.setdefault("WARP_CACHE_PATH", "/dev/shm/cpaxton-mars-warp-cache")
        result.update(gpu_probe(model, state, reference, args.ticks, args.substeps, args.batches, args.iterations))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved compute-only probe:", args.output, flush=True)


if __name__ == "__main__":
    main()
