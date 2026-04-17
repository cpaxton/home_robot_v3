# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from emet.controller.emotes.backend import GenericEmoteBackend, StretchEmoteBackend
from emet.robots.rby1 import Rby1Backend
from emet.robots.stretch import StretchBackend


def test_stretch_backend_returns_stretch_emotes():
    b = StretchBackend().get_emote_backend()
    assert isinstance(b, StretchEmoteBackend)


def test_rby1_backend_returns_generic_emotes():
    b = Rby1Backend().get_emote_backend()
    assert isinstance(b, GenericEmoteBackend)
