# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Versioned Habitat proxy instructions stay explicit and reproducible."""

from pathlib import Path

import pytest
from emet_habitat.ovmm_find_runner import load_habitat_find_phase_episodes

from emet.eval.ovmm_agentic_find import ovmm_find_object_question


def test_manifest_versions_do_not_silently_relabel_old_runs():
    root = Path(__file__).resolve().parents[3]
    legacy = load_habitat_find_phase_episodes(root / "configs/ovmm/habitat_find_phase_episodes.yaml")
    corrected = load_habitat_find_phase_episodes(root / "configs/ovmm/habitat_find_phase_nearest_v2.yaml")
    assert {e.object_relation for e in legacy} == {"on"}
    assert {e.object_relation for e in corrected} == {"nearest"}
    assert not {e.id for e in legacy} & {e.id for e in corrected}
    assert [(e.scene, e.object, e.start_recep, e.goal_recep) for e in legacy] == [
        (e.scene, e.object, e.start_recep, e.goal_recep) for e in corrected
    ]


def test_unknown_relation_fails_closed():
    with pytest.raises(ValueError, match="relation"):
        ovmm_find_object_question("lamp", "bed", relation="under")


def test_single_episode_cli_passes_seed_and_query_mode(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from emet_habitat import ovmm_find_runner
    from emet_habitat.cli import main

    seen = []

    def run(episode, config, **kwargs):
        seen.append((episode, config))
        return {"seed": config.seed}

    monkeypatch.setattr(ovmm_find_runner, "run_habitat_find_phase_episode", run)
    root = Path(__file__).resolve().parents[3]
    result = CliRunner().invoke(
        main,
        [
            "run-ovmm-find-episode",
            "--episodes",
            str(root / "configs/ovmm/habitat_find_phase_nearest_v2.yaml"),
            "--episode-id",
            "hm3d_lamp_bed_00006_nearest_v2",
            "--backend",
            "lazy_graph",
            "--query-driven-memory",
            "--agentic-find",
            "--seed",
            "17",
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    episode, config = seen[0]
    assert episode.object_relation == "nearest"
    assert config.seed == 17
    assert config.query_driven_memory and config.agentic_find
