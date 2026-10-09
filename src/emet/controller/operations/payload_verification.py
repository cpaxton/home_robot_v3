# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Visual possession checks at stationary navigation boundaries, not slip sensing."""

from dataclasses import dataclass

import numpy as np

from emet.controller.operations.query_observation import observe_lifted_query


class PayloadVerificationError(RuntimeError):
    """Possession is uncertain; do not release or start another pickup."""


@dataclass
class CarriedObject:
    query: str
    initial_xyz: np.ndarray
    relative_reference: np.ndarray
    minimum_lift_m: float
    failure: str | None = None


def verify_carried_object(agent):
    """Check a previously verified payload without moving the base or arm.

    An absent/ambiguous view does not prove a drop. Latch uncertainty so another
    navigation request cannot silently resume after a failed check. Only a
    verified pickup establishes this reference; confirmed release clears it.
    """
    carried = getattr(agent, "_carried_object", None)
    if not isinstance(carried, CarriedObject):
        return
    if carried.failure is not None:
        raise PayloadVerificationError(carried.failure)
    robot = agent.robot
    pan_tilt = None
    try:
        pan_tilt = np.asarray(robot.get_pan_tilt(), dtype=float)
        if pan_tilt.shape != (2,) or not np.isfinite(pan_tilt).all():
            pan_tilt = None
            raise ValueError("Measured head pose required for payload inspection")
        if robot.look_at_ee(blocking=True) is False:
            raise ValueError("Payload inspection head motion failed")
        observe_lifted_query(
            agent,
            robot,
            carried.query,
            initial_xyz=carried.initial_xyz,
            minimum_lift_m=carried.minimum_lift_m,
            relative_reference=carried.relative_reference,
            stage="navigation_payload_verification",
        )
    except (AttributeError, ValueError, RuntimeError) as exc:
        carried.failure = f"Payload possession unverified: {exc}"
    finally:
        if pan_tilt is not None:
            try:
                if robot.head_to(*pan_tilt, blocking=True) is False:
                    raise ValueError("Head restoration failed after payload inspection")
            except (AttributeError, ValueError, RuntimeError) as exc:
                carried.failure = carried.failure or str(exc)
    if carried.failure is not None:
        raise PayloadVerificationError(carried.failure)
