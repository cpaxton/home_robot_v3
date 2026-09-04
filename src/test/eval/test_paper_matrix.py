# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from emet.eval.paper_matrix import load_paper_matrix, resolve_paper_row


def test_paper_matrix_declares_the_full_honest_row_set():
    matrix = load_paper_matrix()
    expected = {
        "dynamem_voxel",
        "static_no_instance",
        "lazy_arrival",
        "dynagraph_bounded_instance",
        "ground_truth_oracle",
    }
    assert expected <= set(matrix["rows"])
    for dataset in ("hmeqa", "sqa3d", "ovmm_find"):
        assert expected <= set(matrix["datasets"][dataset]["rows"])


def test_paper_matrix_resolves_explicit_non_oracle_policy():
    row = resolve_paper_row("hmeqa", "lazy_arrival")
    assert row["policy"]["use_hm3d_semantics"] is False
    assert row["ingestion"]["semantic_ingest_mode"] == "arrival_only"
