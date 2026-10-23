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
