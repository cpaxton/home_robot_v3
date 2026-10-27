# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Explicit local simulator clock mapping; never infer hardware clock offsets."""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval


@lru_cache(maxsize=1)
def local_monotonic_clock_id():
    """Linux boot + time namespace identify this monotonic clock, without hostnames."""
    try:
        identity = Path('/proc/sys/kernel/random/boot_id').read_text().strip() + os.readlink('/proc/self/ns/time')
    except OSError:
        return None
    return hashlib.sha256(identity.encode()).hexdigest()


def local_sim_capture_interval(timing, *, received_monotonic, observation_id):
    """Bound the RGB+depth acquisition union only for a matching local simulator.

    Images can have different exposure intervals. This does NOT certify a static
    camera, object or arm between those exposures, nor encoder interpolation.
    """
    identity = local_monotonic_clock_id()
    if (not isinstance(timing, dict) or identity is None
            or timing.get('available') is not True or timing.get('clock_id') != identity or timing.get('clock_domain') != 'mujoco_render_monotonic'
            or timing.get('source') != 'renderer_scene_capture'):
        raise ObservationUnavailable('capture_clock_mapping_unavailable')
    intervals = [timing.get('capture_interval_ns'), timing.get('depth_capture_interval_ns')]
    for interval in intervals:
        if (not isinstance(interval, (tuple, list)) or len(interval) != 2
                or any(type(value) is not int or value <= 0 for value in interval)
                or interval[1] < interval[0]):
            raise ObservationUnavailable('capture_interval_unavailable')
    earliest = min(interval[0] for interval in intervals)/1e9
    latest = max(interval[1] for interval in intervals)/1e9
    try:
        return CaptureInterval(observation_id, earliest, latest, received_monotonic, 'simulation-clock:'+identity)
    except ValueError as error:
        raise ObservationUnavailable('capture_interval_unavailable') from error
