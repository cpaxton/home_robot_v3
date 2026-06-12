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

import os

import numpy as np
from innate_mars_bridge.onboard_da3 import OnboardDA3Depth, create_onboard_da3_from_env, onboard_da3_enabled


def test_onboard_da3_disabled_by_default():
    os.environ.pop("EMET_MARS_ONBOARD_DA3", None)
    assert not onboard_da3_enabled()
    assert create_onboard_da3_from_env() is None


def test_onboard_da3_enabled_from_env():
    os.environ["EMET_MARS_ONBOARD_DA3"] = "1"
    try:
        assert onboard_da3_enabled()
        assert create_onboard_da3_from_env() is not None
    finally:
        os.environ.pop("EMET_MARS_ONBOARD_DA3", None)


def test_onboard_da3_reports_import_error_when_perception_missing(monkeypatch):
    os.environ["EMET_MARS_ONBOARD_DA3"] = "1"
    try:
        da3 = OnboardDA3Depth()

        def _fail_import():
            raise ImportError("no emet on robot")

        monkeypatch.setattr(
            "innate_mars_bridge.onboard_da3.create_da3_estimator_from_parameters",
            None,
            raising=False,
        )
        import innate_mars_bridge.onboard_da3 as mod

        monkeypatch.setattr(
            mod,
            "create_da3_estimator_from_parameters",
            _fail_import,
            raising=False,
        )
        # Force re-import path inside _lazy_estimator
        import builtins

        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "emet.perception.depth.da3_estimator":
                raise ImportError("no emet on robot")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _import)
        da3._estimator = None
        da3._load_error = None
        out = da3.infer_depth_meters(np.zeros((8, 8, 3), dtype=np.uint8))
        assert out is None
        assert da3.load_error is not None
    finally:
        os.environ.pop("EMET_MARS_ONBOARD_DA3", None)
