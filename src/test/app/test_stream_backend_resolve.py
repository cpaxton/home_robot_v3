# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""``emet stream`` backend flag resolution."""

import pytest
from click import UsageError

from emet.app.stream_agent_factory import resolve_stream_backend


def test_backend_dynamem():
    assert resolve_stream_backend(backend="dynamem", cameras_only=False) == "dynamem"


def test_backend_dynagraph():
    assert resolve_stream_backend(backend="dynagraph", cameras_only=False) == "dynagraph"


def test_backend_voxel_only():
    assert resolve_stream_backend(backend="voxel_only", cameras_only=False) == "voxel_only"


def test_cameras_only_returns_none():
    assert resolve_stream_backend(backend=None, cameras_only=True) is None


def test_cameras_only_conflicts_with_backend():
    with pytest.raises(UsageError, match="not both"):
        resolve_stream_backend(backend="dynamem", cameras_only=True)
