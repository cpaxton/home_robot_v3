# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.
#
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

"""Emote backend wiring tests.

``import emet.controller.emotes.backend`` loads ``emet.controller`` (Dynamem / optional perception).
Skip cleanly when that stack is unavailable (e.g. headless OpenCV / ultralytics).
"""

import pytest


def test_stretch_backend_returns_stretch_emotes():
    try:
        from emet.controller.emotes.backend import StretchEmoteBackend
        from emet.robots.stretch import StretchBackend
    except Exception as e:
        pytest.skip(f"emote controller stack unavailable: {e}")

    b = StretchBackend().get_emote_backend()
    assert isinstance(b, StretchEmoteBackend)


def test_rby1_backend_returns_generic_emotes():
    try:
        from emet.controller.emotes.backend import GenericEmoteBackend
        from emet.robots.rby1 import Rby1Backend
    except Exception as e:
        pytest.skip(f"emote controller stack unavailable: {e}")

    b = Rby1Backend().get_emote_backend()
    assert isinstance(b, GenericEmoteBackend)


def test_innate_mars_backend_returns_innate_emotes():
    try:
        from emet.robots.innate_mars import InnateMarsBackend
        from emet.robots.innate_mars.emote_backend import InnateMarsEmoteBackend
    except Exception as e:
        pytest.skip(f"emote controller stack unavailable: {e}")

    b = InnateMarsBackend().get_emote_backend()
    assert isinstance(b, InnateMarsEmoteBackend)
