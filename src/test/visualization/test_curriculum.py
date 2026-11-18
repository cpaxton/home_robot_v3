"""Curriculum media identities, source binding, coverage and local serving."""

import json
from pathlib import Path

import numpy as np
import pytest

from emet.visualization.curriculum_cases import FAMILIES, cases_for
from emet.visualization.curriculum_coverage import coverage_statistics
from emet.visualization.curriculum_gallery import CurriculumAssets
from emet.visualization.curriculum_record import sha
from emet.visualization.curriculum_video import frame_index


@pytest.mark.parametrize("family", FAMILIES)
def test_grids_have_distinct_case_seed_identities_and_explicit_scope(family):
    cases = cases_for(family)
    assert len(cases) == 16
    assert len({(c["id"], c["seed"]) for c in cases}) == 16
    assert len({c["seed"] for c in cases}) == 16
    assert {c["split"] for c in cases} <= {"train", "audit", "visualization"}
    if family.startswith("arm_splines"):
        assert {c["posture"] for c in cases} == {"ready", "raised", "left_reach", "folded_right"}
        assert {c["split"] for c in cases} == {"train", "audit"}
    if family in {"base", "tamp"}:
        assert any(c["split"] == "visualization" for c in cases)
        assert all(c["seed"] >= 480_000_000 for c in cases if c["split"] == "visualization")


def test_frame_selection_never_interpolates_unrecorded_physics():
    times = np.array([0, 0.05, 0.1])
    assert [frame_index(times, t) for t in [-1, 0, 0.049, 0.05, 0.099, 0.1, 2]] == [0, 0, 0, 1, 1, 2, 2]


def test_coverage_reports_occupancy_without_workspace_percentage():
    case = {"id": "a", "seed": 1, "split": "train", "posture": "ready"}
    hand = [[0.1, 0.2, 0.3, 1, 0, 0, 1, 0, 0], [0.2, 0.3, 0.5, 1, 0, 0, 1, 0, 0]]
    report = coverage_statistics([{"case": case, "hand": hand}, {"case": case, "rejected": True}])
    assert report["requested_paths"] == 2 and report["admitted_paths"] == 1 and report["rejected_paths"] == 1
    np.testing.assert_allclose(report["position_span_m"], [0.1, 0.1, 0.2])
    assert report["occupied_2cm_voxels"] == 2 and report["orientation_from_first_max_deg"] == 0
    assert "workspace percentage" in report["interpretation"]


def test_catalog_rejects_unpublished_changed_and_escaping_assets(tmp_path):
    video = tmp_path / "example.mp4"
    video.write_bytes(b"0123456789")
    (tmp_path / "index.json").write_text(
        json.dumps({"format": "curriculum-gallery-v1", "videos": [], "assets": {"example.mp4": sha(video)}})
    )
    assets = CurriculumAssets(tmp_path)
    assert "assets" not in assets.state()
    assert assets.asset("example.mp4") == (video, "video/mp4")
    for name in ["../example.mp4", "/example.mp4", "scene.mjb", "not-published.mp4"]:
        with pytest.raises(ValueError):
            assets.asset(name)
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        assets.asset("example.mp4")


def test_catalog_rejects_symlink_even_with_matching_bytes(tmp_path):
    target = tmp_path / "target.mp4"
    target.write_bytes(b"x")
    (tmp_path / "link.mp4").symlink_to(target)
    (tmp_path / "index.json").write_text(
        json.dumps({"format": "curriculum-gallery-v1", "assets": {"link.mp4": sha(target)}})
    )
    with pytest.raises(ValueError):
        CurriculumAssets(tmp_path).asset("link.mp4")


def test_visualization_assets_do_not_use_hardware_or_actor_inference():
    from emet.visualization import curriculum_video

    source = Path(curriculum_video.__file__).read_text()
    assert "mj_step(" not in source and "WholeBodyBundle(" not in source


def test_curriculum_routes_stream_only_catalogued_media(tmp_path):
    import http.client
    import threading

    from emet.visualization.herman_rehearsal import make_server

    class Replay:
        def state(self):
            return {"hardware_motion_enabled": False}

    video = tmp_path / "base.mp4"
    video.write_bytes(b"0123456789")
    (tmp_path / "index.json").write_text(
        json.dumps({"format": "curriculum-gallery-v1", "videos": [], "assets": {"base.mp4": sha(video)}})
    )
    server = make_server(Replay(), port=0, curriculum=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def get(path, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.request("GET", path, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read(), dict(response.headers)
        finally:
            connection.close()

    try:
        assert get("/curriculum/")[0] == 200
        assert json.loads(get("/api/state")[1])["curriculum_available"]
        assert json.loads(get("/curriculum/api/state")[1])["videos"] == []
        status, body, headers = get("/curriculum/assets/base.mp4", {"Range": "bytes=2-5"})
        assert status == 206 and body == b"2345" and headers["Content-Type"] == "video/mp4"
        assert get("/curriculum/assets/../base.mp4")[0] == 400
        assert get("/curriculum/assets/scene.mjb")[0] == 400
        assert get("/curriculum/api/state", {"Host": "outside.example"})[0] == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_base_video_reports_base_error_and_wraps_heading():
    from emet.visualization.curriculum_video import tracking_readout

    values = {"base_target": np.array([[0.03, 0.04, -np.pi + 0.01]]), "base": np.array([[0, 0, np.pi - 0.01]])}
    kind, position, angle = tracking_readout(values, 0, "base")
    assert kind == "BASE" and position == pytest.approx(50)
    assert angle == pytest.approx(np.rad2deg(0.02))
    pose = np.array([[0.1, 0.2, 0.3, 1, 0, 0, 1, 0, 0]])
    kind, position, angle = tracking_readout({"hand": pose, "hand_target": pose}, 0, "hand")
    assert kind == "EE" and position == 0 and angle == 0


def test_base_only_hand_sweeps_do_not_inflate_hand_target_coverage():
    case = {"id": "active", "seed": 1, "split": "train"}
    pose = [0.1, 0.2, 0.3, 1, 0, 0, 1, 0, 0]
    active = {"case": case, "hand": [pose, pose], "hand_active": True}
    passive = {"case": case, "hand": [[100, 100, 100, *pose[3:]]], "hand_active": False}
    result = coverage_statistics([active, passive])
    assert result["admitted_paths"] == 2 and result["active_hand_paths"] == 1 and result["passive_hand_sweeps"] == 1
    np.testing.assert_allclose(result["position_span_m"], [0, 0, 0])
    assert result["occupied_2cm_voxels"] == 1
