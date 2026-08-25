# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path
from unittest.mock import MagicMock

from emet.habitat.episode_debug import coerce_path_string, coerce_session_id
from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor


def test_coerce_session_id_rejects_magicmock_repr():
    mock = MagicMock()
    mock._episode_debug_dir = MagicMock()
    assert coerce_path_string(getattr(mock, "_episode_debug_dir", None)) is None
    assert coerce_session_id(getattr(mock, "_episode_debug_dir", None), fallback="session:q1") == "session:q1"
    assert coerce_session_id(1001, fallback="") == "1001"
    assert coerce_session_id("gre-q11-integrity", fallback="") == "gre-q11-integrity"


def test_agentic_executor_session_id_never_contains_magicmock():
    agent = MagicMock()
    ex = AgenticEQAExecutor(agent, "Where is the clock?", collect_trace=False)
    assert "MagicMock" not in ex._session_id
    assert ex._session_id.startswith("session:")


def test_coerce_path_string_accepts_real_paths(tmp_path: Path):
    assert coerce_path_string(str(tmp_path)) == str(tmp_path)
    assert coerce_path_string(tmp_path) == str(tmp_path)
