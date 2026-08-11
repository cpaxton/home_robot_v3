# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for OVMM sweep prep + rates (no sim)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from emet.eval.ovmm_sweep import (
    aggregate_ovmm_rates,
    load_ovmm_sweep_preset,
    prepare_multi_env_sweep,
    resolve_ovmm_sweep_preset_path,
)


def test_resolve_molmo_robocasa_preset():
    path = resolve_ovmm_sweep_preset_path("molmo-robocasa")
    assert path.is_file()
    assert path.name == "molmo_robocasa.yaml"
    data = load_ovmm_sweep_preset("molmo-robocasa")
    assert data.get("name") == "molmo-robocasa"
    assert "default_table" not in yaml.safe_dump(data)


def test_prepare_multi_env_sweep_writes_yamls(tmp_path: Path):
    prepared = prepare_multi_env_sweep(
        tmp_path / "out",
        "molmo-robocasa",
        sync_robocasa_registry=False,
    )
    assert prepared.find_episodes.is_file()
    assert prepared.full_episodes.is_file()
    find = yaml.safe_load(prepared.find_episodes.read_text(encoding="utf-8"))
    full = yaml.safe_load(prepared.full_episodes.read_text(encoding="utf-8"))
    find_eps = find["episodes"]
    full_eps = full["episodes"]
    assert len(find_eps) == 14  # 10 robocasa + 4 molmo
    assert len(full_eps) == 6  # 5 robocasa + 1 molmo rby1
    blob = yaml.safe_dump({"find": find_eps, "full": full_eps})
    assert "default_table" not in blob
    # Sim files exist and are robocasa/molmospaces only
    sim_files = list(prepared.sim_dir.glob("*.yaml"))
    assert len(sim_files) >= 11
    kinds = set()
    for p in sim_files:
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        kinds.add(d.get("kind"))
        assert d.get("kind") in ("robocasa", "molmospaces")
    assert "robocasa" in kinds and "molmospaces" in kinds
    # Episode ids stable
    ids = {e["id"] for e in find_eps}
    assert "robocasa_l1_s1_seed0_find" in ids
    assert "molmo_ithor_idx0_bowl_explore" in ids
    full_ids = {e["id"] for e in full_eps}
    assert "molmo_ithor_idx0_rby1_bowl_full" in full_ids


def test_aggregate_ovmm_rates_excludes_bind_fails(tmp_path: Path):
    out = tmp_path / "sweep"
    find_dir = out / "find"
    find_dir.mkdir(parents=True)
    (find_dir / "ok_dynagraph.json").write_text(
        json.dumps(
            {
                "episode_id": "ok",
                "find_object_success": True,
                "find_recep_success": False,
                "find_partial_success": 0.5,
                "error": "",
            }
        ),
        encoding="utf-8",
    )
    (find_dir / "bad_dynagraph.json").write_text(
        json.dumps(
            {
                "episode_id": "bad",
                "find_object_success": False,
                "find_recep_success": False,
                "find_partial_success": 0.0,
                "error": "RuntimeError: Ran _load_model() 50 times but could not initialize task!",
            }
        ),
        encoding="utf-8",
    )
    (find_dir / "bind_dynagraph.json").write_text(
        json.dumps(
            {
                "episode_id": "bind",
                "find_object_success": False,
                "find_recep_success": False,
                "find_partial_success": 0.0,
                "error": "sim did not bind port 5555",
            }
        ),
        encoding="utf-8",
    )
    rates = aggregate_ovmm_rates(out, backend="dynagraph")
    assert rates["find"]["n"] == 1
    assert rates["find"]["find_object"] == 1
    assert rates["find"]["find_recep"] == 0
    assert rates["find"]["skipped_bind_or_init"] == 2
    assert abs(rates["find"]["mean_partial"] - 0.5) < 1e-6
    assert (out / "rates.json").is_file()
