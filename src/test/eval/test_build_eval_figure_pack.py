"""Tests for scripts/build_eval_figure_pack.py eval smoke aggregation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from build_eval_figure_pack import (  # noqa: E402
    OVMM_METRIC_LABELS,
    OVMM_PHASE_DIFFICULTY_NOTE,
    _find_hmeqa_jsonls,
    _investigate_status,
    _ovmm_success,
    _print_ovmm_digest,
    _summarize_hmeqa,
    _summarize_ovmm,
    _summarize_sqa3d,
    build_summary,
    write_summary_csv,
)


def _write_ovmm_json(path: Path, **fields: object) -> None:
    defaults = {
        "episode_id": path.stem,
        "backend": "dynamem",
        "object_query": "lamp",
        "start_recep": "bed",
        "goal_recep": "table",
        "find_object_success": False,
        "find_recep_success": False,
        "find_partial_success": 0.0,
    }
    defaults.update(fields)
    path.write_text(json.dumps(defaults), encoding="utf-8")


def test_ovmm_success_handles_bool_and_float():
    assert _ovmm_success(True) is True
    assert _ovmm_success(False) is False
    assert _ovmm_success(1.0) is True
    assert _ovmm_success(0.5) is True
    assert _ovmm_success(0.0) is False
    assert _ovmm_success(None) is False


def test_summarize_ovmm_all_outcome_types(tmp_path: Path):
    run_id = "eval_smoke_outcomes"
    root = tmp_path / f"{run_id}_dynamem"
    root.mkdir()
    cases = [
        ("both", True, True),
        ("object_only", True, False),
        ("recep_only", False, True),
        ("neither", False, False),
    ]
    for name, obj_ok, recep_ok in cases:
        partial = 1.0 if (obj_ok and recep_ok) else (0.5 if obj_ok or recep_ok else 0.0)
        _write_ovmm_json(
            root / f"{name}_dynamem.json",
            episode_id=name,
            find_object_success=obj_ok,
            find_recep_success=recep_ok,
            find_partial_success=partial,
        )

    summary = _summarize_ovmm(run_id, tmp_path)
    stats = summary["backends"]["dynamem"]

    assert summary["primary_metric"] == "find_recep_success"
    assert summary["difficulty_note"] == OVMM_PHASE_DIFFICULTY_NOTE
    assert summary["metric_labels"] == OVMM_METRIC_LABELS
    assert stats["n"] == 4
    assert stats["find_object_success"] == 2
    assert stats["find_recep_success"] == 2
    assert stats["find_both_success"] == 1
    assert stats["find_object_only"] == 1
    assert stats["find_recep_only"] == 1
    assert stats["find_object_success_rate"] == 0.5
    assert stats["find_recep_success_rate"] == 0.5
    assert stats["find_both_success_rate"] == 0.25
    assert stats["find_object_only_rate"] == 0.25
    assert stats["find_recep_only_rate"] == 0.25

    outcomes = {ep["episode_id"]: ep["outcome"] for ep in summary["episodes"]}
    assert outcomes == {
        "both": "both",
        "object_only": "object_only",
        "recep_only": "recep_only",
        "neither": "neither",
    }


def test_summarize_ovmm_multiple_backends_and_run_id_glob(tmp_path: Path):
    run_id = "eval_smoke_multi"
    for backend in ("dynamem", "dynagraph"):
        root = tmp_path / f"{run_id}_{backend}"
        root.mkdir()
        _write_ovmm_json(
            root / f"ep_{backend}.json",
            backend=backend,
            find_object_success=True,
            find_recep_success=False,
            find_partial_success=0.5,
        )
    # Different run_id must not be picked up.
    other = tmp_path / "other_run_dynamem"
    other.mkdir()
    _write_ovmm_json(other / "skip.json", find_object_success=True, find_recep_success=True)

    summary = _summarize_ovmm(run_id, tmp_path)
    assert set(summary["backends"]) == {"dynamem", "dynagraph"}
    assert summary["backends"]["dynamem"]["find_object_only"] == 1
    assert summary["backends"]["dynagraph"]["find_object_only"] == 1


def test_summarize_ovmm_empty_run(tmp_path: Path):
    summary = _summarize_ovmm("missing_run", tmp_path)
    assert summary["backends"] == {}
    assert summary["episodes"] == []


def test_summarize_hmeqa_accuracy(tmp_path: Path):
    jsonl = tmp_path / "smoke_hmeqa_dynagraph.jsonl"
    rows = [
        {"method": "dynagraph", "correct": True},
        {"method": "dynagraph", "correct": False},
        {"method": "dynagraph", "correct": True},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out = _summarize_hmeqa([jsonl])
    assert out["methods"]["dynagraph"] == {"n": 3, "correct": 2, "accuracy": pytest.approx(2 / 3)}


def test_find_hmeqa_jsonls_matches_run_id(tmp_path: Path):
    (tmp_path / "eval_2025_hmeqa_dynagraph.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "eval_2025_hmeqa_graph_eqa.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "other_hmeqa_dynagraph.jsonl").write_text("{}\n", encoding="utf-8")

    paths = _find_hmeqa_jsonls("eval_2025", tmp_path)
    assert [p.name for p in paths] == [
        "eval_2025_hmeqa_dynagraph.jsonl",
        "eval_2025_hmeqa_graph_eqa.jsonl",
    ]


def test_find_hmeqa_jsonls_matches_subset_tags(tmp_path: Path):
    (tmp_path / "subset_tuned_paper_holdout8_qwen3_vl.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "subset_other_run_qwen3_vl.jsonl").write_text("{}\n", encoding="utf-8")

    paths = _find_hmeqa_jsonls("tuned_paper", tmp_path)
    assert [p.name for p in paths] == ["subset_tuned_paper_holdout8_qwen3_vl.jsonl"]


@pytest.mark.parametrize(
    ("hmeqa_acc", "ovmm_obj", "ovmm_recep", "expected_status"),
    [
        ([0.0], 0, 0, "INVESTIGATE"),
        ([0.5], 0, 0, "OK"),
        ([0.0], 1, 0, "OK"),
        ([0.0], 0, 1, "OK"),
        ([], 0, 0, "INVESTIGATE"),
    ],
)
def test_investigate_status(hmeqa_acc, ovmm_obj, ovmm_recep, expected_status):
    summary = {
        "hmeqa": {
            "methods": {
                f"m{i}": {"accuracy": acc}
                for i, acc in enumerate(hmeqa_acc)
            }
        },
        "ovmm": {
            "backends": {
                "dynamem": {
                    "find_object_success": ovmm_obj,
                    "find_recep_success": ovmm_recep,
                    "partial_success": 0,
                }
            }
        },
        "sqa3d": {"methods": {}},
    }
    status, investigate = _investigate_status(summary)
    assert status == expected_status
    assert investigate == (expected_status == "INVESTIGATE")


def test_investigate_status_sqa3d_only_success():
    summary = {
        "hmeqa": {"methods": {}},
        "ovmm": {"backends": {}},
        "sqa3d": {"methods": {"dynagraph": {"n": 3, "correct": 1, "em@1": 1 / 3}}},
    }
    status, investigate = _investigate_status(summary)
    assert status == "OK"
    assert investigate is False


def test_summarize_sqa3d_em_at_1(tmp_path: Path):
    run_id = "eval_smoke_sqa3d"
    out_dir = tmp_path / f"{run_id}_dynagraph"
    out_dir.mkdir()
    jsonl = out_dir / "dynagraph_val_q0-2.jsonl"
    rows = [
        {"method": "dynagraph", "em": True},
        {"method": "dynagraph", "em": False},
        {"method": "dynagraph", "em": False},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    summary = _summarize_sqa3d(run_id, tmp_path)
    assert summary["methods"]["dynagraph"]["n"] == 3
    assert summary["methods"]["dynagraph"]["em@1"] == pytest.approx(1 / 3)


def test_build_summary_artifact_tag_fallback(tmp_path: Path, capsys):
    results = tmp_path / "results"
    ovmm = tmp_path / "ovmm"
    sqa3d = tmp_path / "sqa3d"
    results.mkdir()
    ovmm.mkdir()
    sqa3d.mkdir()

    tag = "custom_tag"
    (results / f"{tag}_hmeqa_dynagraph.jsonl").write_text(
        json.dumps({"method": "dynagraph", "correct": True}) + "\n",
        encoding="utf-8",
    )

    summary = build_summary(
        "missing_run_id",
        results_root=results,
        ovmm_root=ovmm,
        sqa3d_root=sqa3d,
        artifact_tag=tag,
    )
    assert summary["hmeqa"]["methods"]["dynagraph"]["accuracy"] == 1.0
    err = capsys.readouterr().err
    assert "artifact_tag" in err


def test_build_summary_end_to_end(tmp_path: Path):
    run_id = "eval_e2e"
    results = tmp_path / "results"
    ovmm = tmp_path / "ovmm"
    sqa3d = tmp_path / "sqa3d"
    results.mkdir()
    ovmm.mkdir()
    sqa3d.mkdir()

    hmeqa = results / f"{run_id}_hmeqa_dynagraph.jsonl"
    hmeqa.write_text(
        json.dumps({"method": "dynagraph", "correct": True}) + "\n",
        encoding="utf-8",
    )
    ovmm_dir = ovmm / f"{run_id}_dynamem"
    ovmm_dir.mkdir()
    _write_ovmm_json(
        ovmm_dir / "ep.json",
        find_object_success=True,
        find_recep_success=False,
        find_partial_success=0.5,
    )

    summary = build_summary(
        run_id,
        results_root=results,
        ovmm_root=ovmm,
        sqa3d_root=sqa3d,
    )
    assert summary["run_id"] == run_id
    assert summary["status"] == "OK"
    assert summary["hmeqa"]["methods"]["dynagraph"]["accuracy"] == 1.0
    assert summary["ovmm"]["backends"]["dynamem"]["find_object_only"] == 1


def test_write_summary_csv_includes_ovmm_phase_rates(tmp_path: Path):
    summary = {
        "hmeqa": {"methods": {"dynagraph": {"n": 2, "accuracy": 0.5}}},
        "ovmm": {
            "backends": {
                "dynamem": {
                    "n": 2,
                    "find_object_success_rate": 1.0,
                    "find_recep_success_rate": 0.5,
                    "find_partial_success_rate": 1.0,
                    "find_both_success_rate": 0.5,
                    "find_object_only_rate": 0.5,
                }
            }
        },
        "sqa3d": {"methods": {"dynagraph": {"n": 2, "em@1": 0.5}}},
    }
    out = tmp_path / "summary.csv"
    write_summary_csv(summary, out)
    text = out.read_text(encoding="utf-8")
    assert "hmeqa,dynagraph,2,accuracy,0.5000" in text
    assert "ovmm,dynamem,2,find_object_success_rate,1.0000" in text
    assert "ovmm,dynamem,2,find_recep_success_rate,0.5000" in text
    assert "ovmm,dynamem,2,find_object_only_rate,0.5000" in text
    assert "sqa3d,dynagraph,2,em@1,0.5000" in text


def test_print_ovmm_digest(capsys):
    _print_ovmm_digest({"backends": {}})
    assert capsys.readouterr().out.strip() == "OVMM: no runs found"

    _print_ovmm_digest(
        {
            "backends": {
                "dynamem": {
                    "n": 3,
                    "find_object_success_rate": 1 / 3,
                    "find_recep_success_rate": 0.0,
                    "find_both_success_rate": 0.0,
                    "find_object_only_rate": 1 / 3,
                }
            }
        }
    )
    out = capsys.readouterr().out
    assert "FindObj=33%" in out
    assert "FindRec=0%" in out
    assert "object_only=33%" in out


def test_plot_ovmm_success_bars_writes_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    from build_eval_figure_pack import _plot_ovmm_success_bars

    summary = {
        "backends": {
            "dynamem": {
                "find_object_success_rate": 0.7,
                "find_recep_success_rate": 0.3,
            }
        }
    }
    path = _plot_ovmm_success_bars(summary, tmp_path)
    assert path is not None
    assert path.name == "ovmm_findobj_findrec.png"
    assert path.stat().st_size > 0
