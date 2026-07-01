# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

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
