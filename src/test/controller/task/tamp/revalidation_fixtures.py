# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Synthetic local evidence for source-lifetime/handle contract tests only."""

import numpy as np

from emet.controller.manipulation.observed_effects import CaptureInterval
from emet.motion.local_scene_store import LocalScene
from emet.motion.observed_admission import ObservedEnvironmentAdmission
from emet.motion.observed_background import ObservedBackground, StaticBackgroundAssumption
from emet.motion.observed_support import ObservedSupportPatch
from emet.motion.observed_support_contact import ObservedSupportContact
from emet.motion.observed_workspace import ObservedWorkspace


def local_scene(stamp, *, capture_id=None, missing_ray=False, object_ref="payload"):
    capture = CaptureInterval(capture_id or str(stamp), stamp, stamp + 0.001, stamp + 0.01, "clock")
    grid = ObservedWorkspace([-0.2, -0.2, -0.05], 0.05, np.zeros((8, 8, 8), np.uint8))
    background = ObservedBackground(
        grid, np.full(grid.cells.shape, stamp), "epoch", frozenset((capture.observation_id,)), capture
    )
    patch = ObservedSupportPatch(
        [[-0.1, -0.1, -0.001], [0.1, 0.1, 0.001]], [0, 0, 0], 0.0, capture.observation_id, stamp + 0.01, 0.001
    )
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.0
    mask = np.ones((21, 21), bool)
    if missing_ray:
        mask[10, 10] = False
    support = ObservedSupportContact(
        object_ref,
        "table",
        "epoch",
        "ideal-test",
        patch,
        capture,
        np.ones((21, 21)),
        [[20, 0, 10], [0, 20, 10], [0, 0, 1]],
        pose,
        mask,
        payload_motion_error_m=0.0,
    )
    return LocalScene(
        ObservedEnvironmentAdmission(
            background,
            static_assumption=StaticBackgroundAssumption("epoch", stamp - 0.01, stamp + 100.0, "isolated-test"),
        ),
        support,
    )
