# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Diagnostic observation must preserve the verifier's result and exception."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/diagnose_native_verification.py"
spec = importlib.util.spec_from_file_location("native_verifier_observer", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("raises", [False, True])
def test_observer_never_replaces_verification_result(monkeypatch, tmp_path, raises):
    from emet.controller.task.tamp.native_adapter import NativeTableAdapter

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="obj" pos="0 0 .2"><freejoint/><geom type="box" size=".02 .02 .02"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def sync(_data):
        pass

    adapter = SimpleNamespace(model=model, data=data, synchronize=sync)
    plan = SimpleNamespace(object_body="obj", goal=np.array([0, 0, 0.22]), placement_region=None)

    def original(self, plan, **kwargs):
        self.synchronize(self.data)
        if raises:
            raise RuntimeError("same failure")
        return False

    monkeypatch.setattr(NativeTableAdapter, "_verify", original)
    output = tmp_path / "trace.jsonl"
    with module.verification_trace(output):
        if raises:
            with pytest.raises(RuntimeError, match="same failure"):
                NativeTableAdapter._verify(adapter, plan, released=True, seconds=2)
        else:
            assert NativeTableAdapter._verify(adapter, plan, released=True, seconds=2) is False
    assert NativeTableAdapter._verify is original
    assert adapter.synchronize is sync
    row = json.loads(output.read_text())
    assert row["passed"] is (None if raises else False)
    assert len(row["samples"]) == 1
