"""Viewer storage changes preserve the complete compiled model's numerical data."""

import numpy as np
import pytest
import mujoco

from emet.rl.manipulation.mesh_storage import binary_mesh_assets
from emet.rl.manipulation.models import build_manipulation_model


@pytest.mark.sim
@pytest.mark.parametrize("robot", ["innate_mars", "stretch"])
def test_binary_viewer_mesh_storage_preserves_native_model_arrays_and_identity(robot):
    inline, fingerprint = build_manipulation_model(robot)
    binary, other = build_manipulation_model(robot, mesh_storage="binary")
    assert fingerprint == other
    count = 0
    for name in dir(inline):
        if name.startswith("_") or name == "mesh_pathadr":
            # Only source-file offsets differ: previously inline meshes now
            # have VFS paths. Paths do not affect physics or rendered geometry.
            continue
        a = getattr(inline, name)
        if isinstance(a, np.ndarray):
            count += 1
            np.testing.assert_equal(a, getattr(binary, name), err_msg=name)
    assert count > 400
    a, b = mujoco.MjData(inline), mujoco.MjData(binary)
    for _ in range(10):
        mujoco.mj_step(inline, a)
        mujoco.mj_step(binary, b)
        for name in ("qpos", "qvel", "actuator_force"):
            left, right = np.asarray(getattr(a, name)).copy(), np.asarray(getattr(b, name)).copy()
            assert np.isfinite(left).all() and np.isfinite(right).all()
            np.testing.assert_array_equal(left, right, err_msg=name)


@pytest.mark.parametrize("attribute", ['vertex="0 0 nan" face="0 0 0"', 'vertex="0 0 0" face="1 1 1"', 'vertex="0 0 0" face="0.2 0 0"', 'vertex="0 0 0"', 'vertex="0 0 0" face="0 0 0" normal="0 0 1"'])
def test_malformed_or_unsupported_mesh_storage_fails_closed(attribute):
    with pytest.raises(ValueError):
        binary_mesh_assets(f'<mujoco><asset><mesh name="test" {attribute}/></asset></mujoco>')
