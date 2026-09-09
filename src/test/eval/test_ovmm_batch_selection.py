# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

from emet.eval.ovmm_batch import OvmmBatchOptions, run_ovmm_batch


def test_explicit_episode_path_is_not_replaced_by_benchmark(monkeypatch, tmp_path, capsys):
    import emet.eval.ovmm_find_phase as find

    root = Path(__file__).resolve().parents[3]
    selected = root / "configs/ovmm/find_phase_episodes.yaml"
    other = tmp_path / "other.yaml"
    other.write_text("episodes: []\n")
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(f"episodes:\n  sim: {other}\n")
    original = find.load_find_phase_episodes
    paths = []

    def load(path):
        paths.append(Path(path))
        return original(path)

    monkeypatch.setattr(find, "load_find_phase_episodes", load)
    monkeypatch.chdir(root)
    opts = OvmmBatchOptions(
        episodes=str(selected),
        benchmark=str(benchmark),
        episode_ids=["molmo_rby1_ithor_s2_idx0"],
        dry_run=True,
    )
    assert run_ovmm_batch(opts, repo_root=root) == 0
    assert paths == [selected]
    assert "molmo_rby1_ithor_s2_idx0" in capsys.readouterr().out


def test_empty_selection_fails_before_worker_start(capsys):
    root = Path(__file__).resolve().parents[3]
    opts = OvmmBatchOptions(
        episodes=str(root / "configs/ovmm/find_phase_episodes.yaml"),
        benchmark=str(root / "configs/ovmm/benchmark.yaml"),
        episode_ids=["does_not_exist"],
    )
    assert run_ovmm_batch(opts, repo_root=root) == 2
    assert "no episodes selected" in capsys.readouterr().err
