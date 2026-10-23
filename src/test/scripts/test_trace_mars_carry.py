import runpy
from pathlib import Path

import pytest


def script():
    return runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/trace_mars_carry.py"))


def test_trace_rejects_non_mars_units_before_building_a_model():
    with pytest.raises(ValueError, match="Mars"):
        script()["trace_case"]({"config": {"robot": "stretch"}}, {})


def test_mixed_3d_and_error_plots_export_offline(tmp_path):
    pytest.importorskip("plotly")
    row = {
        "time_s": 0,
        "hand_commanded": [0, 0, 0.2],
        "hand_measured": [0, 0, 0.201],
        "payload_world_m": [0, 0, 0.19],
        "peak_retention_translation_m": 0.001,
        "peak_retention_rotation_rad": 0.11,
        "peak_wheel_gap_m": [0, 0, 0],
        "contacts": [],
    }
    report = {
        "case": {"id": "100g:orientation:10"},
        "final": {"code": "retention_lost"},
        "trace": [row, {**row, "time_s": 1}],
    }
    path = tmp_path / "trace.html"
    script()["export_html"](report, path)
    html = path.read_text()
    assert "0.1 rad retention limit" in html
    assert "Measured hand" in html and "retention_lost" in html
    assert "Plotly.newPlot" in html


def test_inactive_contact_slots_are_not_read(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    namespace = script()
    active = SimpleNamespace(geom=[0, 1], pos=np.zeros(3), frame=np.eye(3).ravel(), dist=-0.001)
    # This slot intentionally has no contact fields: reading it is a failure.
    data = SimpleNamespace(ncon=1, contact=[active, object()])
    model = SimpleNamespace(geom_bodyid=[1, 2], body=lambda i: SimpleNamespace(name=str(i)))
    monkeypatch.setattr(namespace["mujoco"], "mj_contactForce", lambda model, data, index, force: force.fill(1))
    contacts = namespace["active_payload_contacts"](model, data, 2)
    assert len(contacts) == 1 and contacts[0]["bodies"] == ["1", "2"]
