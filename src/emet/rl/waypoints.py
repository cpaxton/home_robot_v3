"""Reachable coordinated waypoint goals with differential-drive base headings."""

import numpy as np

from emet.rl.control import Goal


def coordinated_waypoints(start_base, start_arm, final, count):
    """Turn toward travel, translate, then align to the final requested yaw.

    Interpolating XY while holding yaw at every waypoint asks a wheeled robot
    to finish a new lateral parking maneuver for each centimeter of progress.
    Shared training/evaluation goals instead use navigation-style headings.
    Hand poses are derived by the environment's FK from each base/arm pair.
    """
    if count == 1:
        return [final]
    start_base, start_arm = np.asarray(start_base), np.asarray(start_arm)
    delta = final.base[:2] - start_base[:2]
    distance = np.linalg.norm(delta)
    yaw_delta = np.arctan2(np.sin(final.base[2] - start_base[2]), np.cos(final.base[2] - start_base[2]))
    if not final.mask[0] or distance < 0.006:
        bases = [start_base + fraction * np.r_[delta, yaw_delta] for fraction in np.linspace(1 / count, 1, count)]
    else:
        bearing = np.arctan2(delta[1], delta[0])
        relative = np.arctan2(np.sin(bearing - start_base[2]), np.cos(bearing - start_base[2]))
        if abs(relative) > np.pi / 2:
            bearing += np.pi
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        bases = [] if count == 2 else [np.r_[start_base[:2], bearing]]
        segments = count - len(bases) - 1
        bases.extend(np.r_[start_base[:2] + fraction * delta, bearing]
                     for fraction in np.linspace(1 / segments, 1, segments))
        bases.append(final.base.copy())
    goals = [Goal(np.asarray(base), start_arm + fraction * (final.arm - start_arm), final.mask.copy())
             for fraction, base in zip(np.linspace(1 / count, 1, count), bases, strict=True)]
    goals[-1] = Goal(final.base.copy(), final.arm.copy(), final.mask.copy())
    return goals
