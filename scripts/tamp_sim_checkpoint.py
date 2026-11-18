# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Save MuJoCo integration state for restore into an isolated diagnostic simulator."""

import mujoco
import numpy as np

from emet.controller.manipulation.native_scene import model_signature

_STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def save_sim_checkpoint(path, model, data):
    state = np.empty(mujoco.mj_stateSize(model, _STATE), dtype=np.float64)
    mujoco.mj_getState(model, data, state, _STATE)
    with open(path, "wb") as stream:
        np.savez_compressed(stream, version=1, model_signature=model_signature(model), state=state)


def restore_sim_checkpoint(path, model):
    """Return private data. The caller must use the original fixture and runtime."""
    with np.load(path, allow_pickle=False) as saved:
        if saved["version"].item() != 1 or saved["model_signature"].item() != model_signature(model):
            raise ValueError("incompatible_sim_checkpoint")
        state = saved["state"]
    if state.shape != (mujoco.mj_stateSize(model, _STATE),) or not np.isfinite(state).all():
        raise ValueError("invalid_sim_checkpoint")
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state, _STATE)
    mujoco.mj_forward(model, data)
    # Forward computes derived fields. Restore any integration fields it updated.
    mujoco.mj_setState(model, data, state, _STATE)
    return data
