# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Ensure chase-cam pixels survive GenericZmqClient.get_observation."""

from __future__ import annotations

import numpy as np

from emet.controller.generic_zmq_client import get_observation_from_zmq_dict
from emet.core.interfaces import Observations


def test_get_observation_from_zmq_dict_keeps_third_person() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    tp = np.full((4, 4, 3), 200, dtype=np.uint8)
    obs = get_observation_from_zmq_dict(
        {
            "rgb": rgb,
            "depth": np.zeros((4, 4), dtype=np.float32),
            "third_person_image": tp,
            "gps": np.zeros(2),
            "compass": np.zeros(1),
            "step": 1,
        }
    )
    assert obs is not None
    assert obs.third_person_image is not None
    np.testing.assert_array_equal(obs.third_person_image, tp)


def test_observations_field_defaults_none() -> None:
    o = Observations(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        gps=np.zeros(2),
        compass=np.zeros(1),
    )
    assert o.third_person_image is None
