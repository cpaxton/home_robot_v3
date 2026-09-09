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
