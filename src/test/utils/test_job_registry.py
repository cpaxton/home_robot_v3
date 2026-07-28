# Copyright (c) Chris Paxton 2026

from __future__ import annotations

import os

from emet.utils import job_registry as jr


def test_register_list_update_cancel(tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_JOBS_DIR", str(tmp_path / "jobs"))
    job = jr.register_job(
        name="unit-test-job",
        cmd="echo hi",
        out_dir=tmp_path / "out",
        status="queued",
        wait_pids=[1],
    )
    assert job.id
    assert (tmp_path / "jobs" / f"{job.id}.json").is_file()

    active = jr.list_jobs(include_terminal=False)
    assert any(j.id == job.id for j in active)

    updated = jr.update_job(job.id, status="running", pid=os.getpid())
    assert updated.status == "running"
    assert updated.pid == os.getpid()

    # cancel must not kill the pytest process (pid == self); still marks cancelled
    cancelled = jr.cancel_job(job.id, grace_s=0.1)
    assert cancelled.status == "cancelled"
    assert os.getpid() > 0  # still alive

    active2 = jr.list_jobs(include_terminal=False)
    assert not any(j.id == job.id for j in active2)
    all_jobs = jr.list_jobs(include_terminal=True)
    assert any(j.id == job.id and j.status == "cancelled" for j in all_jobs)


def test_refresh_marks_dead_pid_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_JOBS_DIR", str(tmp_path / "jobs"))
    job = jr.register_job(name="dead", status="running", pid=999999991)
    refreshed = jr.refresh_job_liveness(jr.load_job(job.id))  # type: ignore[arg-type]
    assert refreshed is not None
    assert refreshed.status == "failed"


def test_looks_like_gpu_job_and_active_pids(tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_JOBS_DIR", str(tmp_path / "jobs"))
    gpu = jr.register_job(
        name="hmeqa-bal32",
        cmd="env EMET_ALLOW_SDPA_ATTN=1 ./scripts/run_hmeqa_agentic_h2h.sh OUT",
        status="running",
        pid=os.getpid(),
    )
    cpuish = jr.register_job(name="docs-build", cmd="make docs", status="running", pid=os.getpid())
    assert jr.looks_like_gpu_job(gpu)
    assert not jr.looks_like_gpu_job(cpuish)
    pids = jr.active_gpu_job_pids()
    assert os.getpid() in pids
    assert jr.active_gpu_job_pids(exclude_job_id=gpu.id) == []


def test_format_job_row_columns():
    job = jr.JobRecord(
        id="20260721_140000_abc123",
        name="demo",
        status="queued",
        pid=None,
        out_dir="/tmp/out",
    )
    header = jr.format_job_header()
    row = jr.format_job_row(job)
    assert "STATUS" in header and "PID" in header and "PROGRESS" in header
    assert "queued" in row
    assert "demo" in row
    # PID column should be a bare dash, not "pid=-"
    assert "pid=" not in row
    assert "     -" in row or row.split()[2] == "-"
    # No progress yet → placeholder dash in PROGRESS column
    assert " - " in row or row.count("-") >= 2


def test_compute_job_progress_eta_from_meta():
    now = 1_000_000.0
    job = jr.JobRecord(
        id="j1",
        name="h2h",
        status="running",
        created_at=now - 800.0,  # 800s elapsed, 8 units → 100s/unit
        meta={"units_done": 8, "units_total": 64, "phase": "classic", "current_id": "17"},
    )
    prog = jr.compute_job_progress(job, now=now)
    assert prog.units_done == 8
    assert prog.units_total == 64
    assert prog.phase == "classic"
    assert prog.current_id == "17"
    assert prog.rate_s_per_unit == 100.0
    assert prog.eta_s == 5600.0  # 56 remaining * 100s
    assert prog.source == "meta"
    brief = jr.format_progress_brief(prog)
    assert "8/64" in brief
    assert "classic" in brief
    assert "q17" in brief
    assert "ETA" in brief


def test_progress_file_overlays_meta(tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_JOBS_DIR", str(tmp_path / "jobs"))
    out = tmp_path / "run"
    job = jr.JobRecord(
        id="j2",
        name="h2h",
        status="running",
        out_dir=str(out),
        created_at=1_000_000.0 - 100.0,
        meta={"units_done": 1, "units_total": 10, "phase": "classic"},
    )
    jr.write_progress_file(out, units_done=4, units_total=10, phase="agentic", current_id="21")
    prog = jr.compute_job_progress(job, now=1_000_000.0)
    assert prog.units_done == 4
    assert prog.phase == "agentic"
    assert prog.current_id == "21"
    assert prog.source == "meta+file"

    registered = jr.register_job(name="x", out_dir=out, status="running")
    updated = jr.update_job(
        registered.id,
        units_done=5,
        units_total=10,
        phase="agentic",
        current_id="22",
    )
    assert updated.meta["units_done"] == 5
    disk = jr.read_progress_file(out)
    assert disk["units_done"] == 5
    assert disk["current_id"] == "22"

    detail = jr.format_job_detail(
        jr.JobRecord(
            id="x",
            name="n",
            status="running",
            pid=1,
            out_dir=str(out),
            meta={"units_done": 5, "units_total": 10, "phase": "agentic"},
            created_at=1_000_000.0 - 50.0,
        )
    )
    assert "progress:" in detail
    assert "5/10" in detail


def test_summarize_eval_cmd_extracts_script_and_out():
    cmd = (
        "python scripts/eval_dynamic_exploration.py --out-dir "
        "/home/cpaxton/runs/emet/dynagraph_fix_verify_20260721_114903/eqa_smoke "
        "--port-offset-base 220"
    )
    script, out = jr.summarize_eval_cmd(cmd)
    assert script == "eval_dynamic_exploration.py"
    assert "eqa_smoke" in out
    assert "port-offset" not in out

    detail = jr.format_job_detail(jr.JobRecord(id="x", name="n", status="running", pid=1))
    assert "id:        x" in detail
    assert "status:    running" in detail


def test_resolve_report_job_prefers_running(tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_JOBS_DIR", str(tmp_path / "jobs"))
    waiting = jr.register_job(name="wait", status="waiting", pid=os.getpid())
    running = jr.register_job(name="run", status="running", pid=os.getpid())
    # Make waiting look newer so preference is by status rank, not time.
    jr.update_job(waiting.id, status="waiting")
    picked = jr.resolve_report_job(None)
    assert picked is not None
    assert picked.id == running.id
    assert jr.resolve_report_job(waiting.id).id == waiting.id  # type: ignore[union-attr]


def test_format_job_report_episode_table(tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_JOBS_DIR", str(tmp_path / "jobs"))
    out = tmp_path / "hmeqa"
    out.mkdir()
    bundle15 = out / "bundles" / "agentic_q15"
    bundle15.mkdir(parents=True)
    (bundle15 / "agentic_summary.json").write_text(
        '{"answer":"D","confidence":false,"verified":true}\n', encoding="utf-8"
    )
    (out / "agentic_q15.jsonl").write_text(
        '{"question_id":15,"correct":true,"predicted_answer":"D",'
        '"gold_answer_letter":"D","planning_steps":62,"confident":false,'
        f'"debug_bundle_dir":"{bundle15}"}}\n',
        encoding="utf-8",
    )
    (out / "agentic_q56.jsonl").write_text(
        '{"question_id":56,"correct":false,"predicted_answer":"A",'
        '"gold_answer_letter":"C","planning_steps":14,"confident":true}\n',
        encoding="utf-8",
    )
    job = jr.register_job(
        name="holdout8",
        status="running",
        pid=os.getpid(),
        out_dir=out,
        cmd="HOLDOUT_IDS=15,56,65,68 env ./scripts/run_hmeqa_agentic_h2h.sh OUT",
        meta={"units_done": 2, "units_total": 8, "phase": "agentic", "current_id": "65"},
    )
    text = jr.format_job_report(job)
    assert "2/8" in text
    assert "ok" in text and "FAIL" in text
    assert "D/D" in text and "A/C" in text
    assert "v=Y e=N" in text
    assert "e=Y" in text
    assert "v=verify-gate" in text
    assert "next: 65, 68" in text
    assert "crashes: none" in text
    payload = jr.job_report_dict(job)
    assert payload["n_correct"] == 1
    assert payload["n_incorrect"] == 1
    assert payload["remaining_ids"] == [65, 68]
    assert payload["episodes"][0]["confident"] is False
    assert payload["episodes"][0]["verified"] is True


def test_format_job_detail_lists_viz_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_JOBS_DIR", str(tmp_path / "jobs"))
    out = tmp_path / "hmeqa"
    picks = out / "bundles" / "agentic_q104" / "frontier_picks"
    maps = out / "bundles" / "agentic_q104" / "maps"
    picks.mkdir(parents=True)
    maps.mkdir(parents=True)
    (picks / "iter_00.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (picks / "iter_01.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (maps / "step_0000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (out / "bundles" / "agentic_q104" / "topdown_map.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (out / "figures").mkdir()
    (out / "figures" / "hmeqa_agentic_h2h.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    job = jr.JobRecord(
        id="jviz",
        name="frontierviz",
        status="done",
        out_dir=str(out),
    )
    detail = jr.format_job_detail(job)
    assert "frontier_picks" in detail
    assert "2 png" in detail
    assert "feh:" in detail
    assert "iter_*.png" in detail
    report = jr.format_job_report(job)
    assert "frontier_picks" in report
    assert "feh:" in report
    arts = jr.discover_out_viz_artifacts(out)
    kinds = {a.kind for a in arts}
    assert "frontier_picks" in kinds
    assert "maps" in kinds
    assert "figures" in kinds
    assert "topdown" in kinds


def test_episode_conf_cell_formats_gate_and_eqa():
    cell = jr.EpisodeScore(arm="agentic", question_id=1, confident=False, verified=True).conf_cell()
    assert cell == "v=Y e=N"
    assert jr.EpisodeScore(arm="classic", question_id=2, confident=True).conf_cell() == "e=Y"
    assert jr.EpisodeScore(arm="agentic", question_id=3).conf_cell() == "-"


def test_analyze_agentic_trace_flags_stale_and_phrase():
    rows = [
        {"tool": "inspect_graph", "picked_by": "loop"},
        {
            "event": "router_room",
            "round": 0,
            "current_room": "patio",
            "current_room_vlm": "outdoor",
            "current_room_graph": "patio",
            "rooms_line": "Rooms: patio(3), kitchen(8)",
            "question_target_rooms": ["dining_room", "kitchen", "living_room"],
            "prefer_explore_reason": "room_mismatch",
            "tool_calls": ["investigate"],
        },
        {
            "event": "prefer_explore_redirect",
            "round": 0,
            "from": "investigate",
            "to": "explore_frontier",
            "from_args": {"obs_id": 13},
        },
        {
            "hypotheses": [
                {
                    "obs_id": 13,
                    "source": "graph",
                    "phrase": "kitchen island",
                    "xyz": [-16.5, -1.1, 1.0],
                }
            ],
            "n_hypotheses": 1,
        },
        {
            "tool": "investigate",
            "event": "tool_pick",
            "picked_by": "vlm",
            "router_tool_calls": ["investigate"],
            "args": {"obs_id": 13},
        },
        {
            "tool": "investigate",
            "round": 0,
            "obs_id": 13,
            "nav_success": True,
            "target_xyz": [-16.86, -1.02, 1.0],
        },
        {
            "tool": "investigate",
            "event": "station_inspect",
            "round": 0,
            "obs_id": 13,
            "closest_m": 0.41,
            "place_inspect": "investigated=1 closest=0.4m [close] recent: r0@0.4m verify=ABSENT",
        },
        {
            "tool": "investigate",
            "round": 1,
            "obs_id": 13,
            "ok": False,
            "status": "NAV_LOOP_BLOCKED",
        },
        {"tool": "nav_loop_redirect", "round": 1},
        {
            "tool": "explore_frontier",
            "round": 1,
            "ok": True,
            "source": "vlm_frontier",
            "frontier_xyz": [-19.0, -3.0, 1.0],
        },
        {
            "tool": "vlm_assess",
            "round": 0,
            "obs_id": 12,
            "present": False,
            "answerable": False,
            "phrase": "fruit bowl",
            "reason": "not visible",
        },
        {"tool": "capture_and_update", "ok": True, "obs_id": 17},
        {
            "tool": "verify_siglip",
            "obs_id": 17,
            "phrase": "sets utensils already",
            "answerable": False,
            "detector_score": 0.06,
        },
        {"tool": "submit_answer", "event": "tool_pick", "picked_by": "fallback"},
        {"tool": "submit_answer", "event": "tool_pick", "picked_by": "fallback"},
        {"tool": "submit_answer", "event": "tool_pick", "picked_by": "fallback"},
        {"tool": "capture_and_update", "ok": True, "obs_id": 17},
        {
            "tool": "verify_siglip",
            "obs_id": 17,
            "phrase": "sets utensils already",
            "answerable": False,
            "detector_score": 0.03,
        },
        {"event": "final_location_salvage", "letter": "B", "prior_answer": "unknown", "n_images": 4},
        {"tool": "abstain_unverified", "reason": "require_verified and no fused verification"},
    ]
    a = jr.analyze_agentic_trace(rows)
    assert a["n_verify"] == 2
    assert a["answerable_any"] is False
    assert a["duplicate_verify_obs"] == [17]
    assert a["fallback_submits"] == 3
    assert a["phrases"] == ["sets utensils already"]
    assert a["max_detector_score"] == 0.06
    assert a["hypotheses"][0]["phrase"] == "kitchen island"
    assert a["router_picks"] == ["investigate(obs=13)"]
    assert a["n_nav_loop_blocked"] >= 2
    assert a["close_absent"] is True
    assert a["min_closest_m"] == 0.41
    assert a["salvage"]["letter"] == "B"
    assert a["n_explore"] == 1
    assert a["n_station_inspect"] == 1
    rooms = a["rooms"]
    assert rooms["n_turns"] == 1
    assert rooms["n_mismatch"] == 1
    assert rooms["n_vlm_graph_disagree"] == 1
    assert rooms["rooms_line"] == "Rooms: patio(3), kitchen(8)"
    assert rooms["question_target_rooms"] == ["dining_room", "kitchen", "living_room"]
    assert rooms["n_prefer_explore_redirect"] == 1


def test_format_question_report_reads_row_and_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_JOBS_DIR", str(tmp_path / "jobs"))
    out = tmp_path / "hmeqa"
    bundle = out / "bundles" / "agentic_q88"
    bundle.mkdir(parents=True)
    (out / "agentic_q88.jsonl").write_text(
        '{"question_id":88,"correct":false,"predicted_answer":"",'
        '"gold_answer_letter":"C","planning_steps":20,"confident":false,'
        '"question":"How many sets of utensils are set up?",'
        f'"debug_bundle_dir":"{bundle}"}}\n',
        encoding="utf-8",
    )
    (bundle / "agentic_trace.jsonl").write_text(
        '{"event":"router_room","round":0,"current_room":"kitchen",'
        '"current_room_vlm":"kitchen","current_room_graph":"kitchen",'
        '"rooms_line":"Rooms: kitchen(4)","question_target_rooms":["kitchen"],'
        '"prefer_explore_reason":"","tool_calls":["investigate"]}\n'
        '{"hypotheses":[{"obs_id":9,"source":"graph","phrase":"table","xyz":[1,2,1]}]}\n'
        '{"tool":"investigate","event":"tool_pick","picked_by":"vlm",'
        '"router_tool_calls":["investigate"],"args":{"obs_id":9}}\n'
        '{"tool":"investigate","event":"station_inspect","round":0,"obs_id":9,'
        '"closest_m":0.35,"place_inspect":"investigated=1 closest=0.4m [close] verify=ABSENT"}\n'
        '{"tool":"verify_siglip","obs_id":9,"phrase":"sets utensils already",'
        '"answerable":false,"detector_score":0.06}\n'
        '{"tool":"verify_siglip","obs_id":9,"phrase":"sets utensils already",'
        '"answerable":false,"detector_score":0.03}\n'
        '{"tool":"abstain_unverified","reason":"require_verified exhausted"}\n',
        encoding="utf-8",
    )
    (bundle / "topdown_map.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    job = jr.register_job(name="holdout8", status="running", out_dir=out)
    text = jr.format_question_report(job, 88)
    assert "q88" in text
    assert "FAIL" in text
    assert "sets utensils already" in text
    assert "RED FLAGS" in text
    assert "stale re-verify obs [9]" in text
    assert "── rooms ──" in text
    assert "Rooms: kitchen(4)" in text
    assert "── router ──" in text
    assert "station r0" in text
    assert "close look" in text
    assert "topdown_map.png" in text
    rooms_only = jr.format_question_report(job, 88, rooms_focus=True)
    assert "── rooms ──" in rooms_only
    assert "── nav ──" not in rooms_only
    payload = jr.question_report_dict(job, 88)
    assert payload["found"] is True
    assert payload["trace"]["duplicate_verify_obs"] == [9]
    assert payload["trace"]["close_absent"] is True
    assert payload["trace"]["rooms"]["rooms_line"] == "Rooms: kitchen(4)"
    assert any("topdown_map.png" in p for p in payload["maps"])

    missing = jr.format_question_report(job, 999)
    assert "no scored jsonl for q999" in missing


def test_find_agentic_trace_prefers_out_bundle(tmp_path):
    out = tmp_path / "hmeqa"
    bundle = out / "bundles" / "agentic_q104"
    bundle.mkdir(parents=True)
    cache = tmp_path / "cache" / "h2h_agentic_q0104" / "q0104_dynagraph"
    cache.mkdir(parents=True)
    (bundle / "agentic_trace.jsonl").write_text('{"event":"router_room","current_room":"patio"}\n', encoding="utf-8")
    (cache / "agentic_trace.jsonl").write_text('{"tool":"summary"}\n', encoding="utf-8")
    row = {"debug_bundle_dir": str(cache)}
    found = jr._find_agentic_trace(out, 104, row)
    assert found is not None
    assert "bundles/agentic_q104" in str(found)


def test_format_job_report_fail_only(tmp_path, monkeypatch):
    monkeypatch.setenv("EMET_JOBS_DIR", str(tmp_path / "jobs"))
    out = tmp_path / "hmeqa"
    out.mkdir()
    (out / "agentic_q1.jsonl").write_text(
        '{"question_id":1,"correct":true,"predicted_answer":"A","gold_answer_letter":"A",'
        '"planning_steps":10,"confident":true}\n',
        encoding="utf-8",
    )
    (out / "agentic_q2.jsonl").write_text(
        '{"question_id":2,"correct":false,"predicted_answer":"B","gold_answer_letter":"D",'
        '"planning_steps":20,"confident":false}\n',
        encoding="utf-8",
    )
    job = jr.register_job(name="t", status="done", out_dir=out)
    full = jr.format_job_report(job)
    assert "  1  " in full and "  2  " in full
    assert "0%" not in full or "50%" in full
    fails = jr.format_job_report(job, fail_only=True)
    assert "fails only" in fails
    assert "  2  " in fails
    assert "ok" not in fails.split("fails only")[-1].split("conf:")[0] or "FAIL" in fails
    # ok row should not appear in the fail-only table body
    assert "pred/gold" in fails
    body = fails.split("pred/gold", 1)[1]
    assert "A/A" not in body
    assert "B/D" in body
