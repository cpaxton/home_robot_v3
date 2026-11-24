# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Independent simulator carry scoring; object-state evidence never enters policy."""
import math


def short_carry_evidence(samples, retained_pickup):
    if not samples:
        return {'short_carry_passed': False, 'final_held_horizontal_motion_m': 0.}
    first, last = samples[0], samples[-1]
    displacement = math.hypot(last['xyz_m'][0]-first['xyz_m'][0], last['xyz_m'][1]-first['xyz_m'][1])
    held_motion = displacement if last['pad_contacts'] == 2 else 0.
    return {'short_carry_passed': bool(retained_pickup and held_motion >= .05),
            'final_held_horizontal_motion_m': held_motion}
