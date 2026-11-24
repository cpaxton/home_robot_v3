# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Independent simulator scoring only; never feed object-state metrics to policy."""


def pickup_retention(samples, initial_height_m):
    """Require an elevated two-pad hold through the last half-second of trace."""
    if not samples:
        return {'retained_pickup': False, 'final_two_pad_lift_m': 0., 'retention_window_s': 0.}
    last = samples[-1]
    window = [sample for sample in samples if sample['sim_s'] >= last['sim_s']-.500001]
    duration = window[-1]['sim_s']-window[0]['sim_s']
    final_lift = last['xyz_m'][2]-initial_height_m if last['pad_contacts'] == 2 else 0.
    held = [sample['pad_contacts'] == 2 and sample['xyz_m'][2]-initial_height_m >= .04 for sample in window]
    return {'retained_pickup': bool(duration >= .4 and final_lift >= .04 and sum(held)/len(held) >= .9),
            'final_two_pad_lift_m': float(final_lift), 'retention_window_s': float(duration)}
