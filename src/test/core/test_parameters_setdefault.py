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
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

import pytest

from emet.core.parameters import Parameters


def test_parameters_setdefault_inserts_missing_top_level():
    p = Parameters(a=1)
    assert p.setdefault("b", 2) == 2
    assert p["b"] == 2
    assert p.setdefault("b", 99) == 2


def test_parameters_setdefault_rejects_slash_key():
    p = Parameters(nested={"x": 1})
    with pytest.raises(ValueError):
        p.setdefault("nested/x", 0)
