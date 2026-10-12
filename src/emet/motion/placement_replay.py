"""Private, versioned snapshots for reproducing placement search without actuation."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from time import monotonic

import mujoco
import numpy as np
import scipy

from emet.motion.placement import plan_placement_paths
from emet.motion.placement_geometry import HeldObject, PlacementScene


def save_snapshot(directory, model, data, *, scene, payload, object_centers,
                  base_candidates, joint_names, ee_body, robot_body, contact_bodies,
                  base_writer, seed=0, rrt_max_iter=400):
    """Save before search; arbitrary known-free callbacks cannot be serialized."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    mujoco.mj_saveModel(model, str(directory / "model.mjb"))
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    np.savez_compressed(directory / "inputs.npz", state=state, boxes=scene.boxes,
                        vertices_ee=payload.vertices_ee, object_centers=np.asarray(object_centers),
                        base_candidates=np.asarray(base_candidates))
    manifest = {
        "schema_version": 1, "geometry_source": scene.source,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True).strip(),
        "geometry_digest": scene.geometry_digest,
        "workspace": None if scene.workspace is None else scene.workspace.tolist(),
        "replayable": scene.known_free is None,
        "unsupported": "known_free_callback" if scene.known_free is not None else None,
        "joint_names": list(joint_names), "ee_body": ee_body, "robot_body": robot_body,
        "contact_bodies": list(contact_bodies), "base_writer": base_writer,
        "seed": seed, "rrt_max_iter": rrt_max_iter,
        "runtime": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                    "numpy": np.__version__, "scipy": scipy.__version__},
        "sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                   for name in ("model.mjb", "inputs.npz")},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return directory


def replay_snapshot(directory):
    """Fail on incompatible data rather than dropping unknown-space constraints."""
    from emet.controller.manipulation.kinematic_pick_place import write_offline_mjcf_base_xyt

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or not manifest["replayable"]:
        raise ValueError("Unsupported placement snapshot")
    if manifest["runtime"]["mujoco"] != mujoco.__version__:
        raise ValueError("Snapshot requires the recorded MuJoCo version")
    for name in ("model.mjb", "inputs.npz"):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
            raise ValueError("Snapshot checksum mismatch")
    model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
    data = mujoco.MjData(model)
    with np.load(directory / "inputs.npz", allow_pickle=False) as inputs:
        mujoco.mj_setState(model, data, inputs["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        scene = PlacementScene(inputs["boxes"], source=manifest["geometry_source"],
                               workspace=manifest["workspace"])
        if scene.geometry_digest != manifest["geometry_digest"]:
            raise ValueError("Snapshot geometry mismatch")
        payload = HeldObject(inputs["vertices_ee"])
        started = monotonic()
        result = plan_placement_paths(
            model, data, scene=scene, payload=payload, object_centers=inputs["object_centers"],
            base_candidates=inputs["base_candidates"],
            set_base=lambda m, d, p: write_offline_mjcf_base_xyt(m, d, p, **manifest["base_writer"]),
            **{key: manifest[key] for key in ("joint_names", "ee_body", "robot_body", "contact_bodies",
                                             "seed", "rrt_max_iter")},
        )
    return {"schema_version": 1, "solutions": len(result.paths), "rejections": result.rejections,
            "path_failures": result.path_failures, "planning_wall_s": monotonic() - started,
            "geometry_source": result.geometry_source, "collision_scope": result.collision_scope}
