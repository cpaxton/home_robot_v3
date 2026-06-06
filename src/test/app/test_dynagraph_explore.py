# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from unittest.mock import MagicMock, patch

import pytest

from emet.app.dynagraph_explore import dynagraph_explore_until_terminated


def test_dynagraph_explore_max_iterations_all_success():
    agent = MagicMock()
    agent.run_exploration.side_effect = [True, True, True]

    reason, ok, nit = dynagraph_explore_until_terminated(
        agent, max_iterations=3, max_consecutive_failures=2
    )
    assert reason == "max_iterations"
    assert ok == 3
    assert nit == 3
    assert agent.run_exploration.call_count == 3


def test_dynagraph_explore_stops_on_consecutive_failures():
    agent = MagicMock()
    agent.run_exploration.side_effect = [False, False, False, True]

    reason, ok, nit = dynagraph_explore_until_terminated(
        agent, max_iterations=10, max_consecutive_failures=3
    )
    assert reason == "consecutive_failures"
    assert ok == 0
    assert nit == 3
    assert agent.run_exploration.call_count == 3


def test_dynagraph_explore_failure_streak_resets_after_success():
    agent = MagicMock()
    agent.run_exploration.side_effect = [False, False, True, False, False, False]

    reason, ok, nit = dynagraph_explore_until_terminated(
        agent, max_iterations=12, max_consecutive_failures=3
    )
    assert reason == "consecutive_failures"
    assert ok == 1
    assert nit == 6
    assert agent.run_exploration.call_count == 6


def test_dynagraph_explore_requires_positive_max_iterations():
    agent = MagicMock()
    with pytest.raises(ValueError, match="positive"):
        dynagraph_explore_until_terminated(agent, max_iterations=0)


def test_dynagraph_explore_timeout_before_second_iteration():
    agent = MagicMock()
    agent.run_exploration.return_value = True

    times = iter([100.0, 100.0, 102.0])

    def mono():
        return next(times)

    with patch("emet.app.dynagraph_explore.time.monotonic", side_effect=mono):
        reason, ok, nit = dynagraph_explore_until_terminated(
            agent,
            max_iterations=20,
            max_consecutive_failures=5,
            timeout_s=1.0,
        )
    assert reason == "timeout"
    assert ok == 1
    assert nit == 1
    assert agent.run_exploration.call_count == 1
