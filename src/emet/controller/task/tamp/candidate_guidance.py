# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Apply observed, world-frame approach guidance before route certification."""
from dataclasses import replace

import numpy as np


def guide_approach(options, hint, *, observation_id):
    if set(hint)-{'direction_world', 'distance_m', 'frame_id', 'observation_id'}:
        raise ValueError('manipulation_options_unsupported')
    if hint.get('frame_id') != 'world':
        raise ValueError('invalid_manipulation_hint')
    if hint.get('observation_id') != observation_id:
        raise ValueError('stale_manipulation_hint')
    direction = hint.get('direction_world')
    if direction is not None:
        try:
            direction = np.asarray(direction, float)
        except (TypeError, ValueError) as error:
            raise ValueError('invalid_manipulation_hint') from error
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
            raise ValueError('invalid_manipulation_hint')
        direction = direction/np.linalg.norm(direction)
    distance = hint.get('distance_m')
    if distance is not None and (isinstance(distance, bool) or not isinstance(distance, (int, float))
                                 or not np.isfinite(distance) or distance <= 0):
        raise ValueError('invalid_manipulation_hint')
    result = []
    for option in options:
        pre, grasp, lift = option.grasp
        length = np.linalg.norm(np.asarray(grasp[0])-pre[0]) if distance is None else float(distance)
        axis = direction
        if axis is None:
            delta = np.asarray(grasp[0])-pre[0]
            if np.linalg.norm(delta) < 1e-8:
                raise ValueError('invalid_manipulation_hint')
            axis = delta/np.linalg.norm(delta)
        position = np.asarray(grasp[0])-axis*length
        result.append(replace(option, grasp=((position, pre[1]), grasp, lift)))
    return tuple(result)
