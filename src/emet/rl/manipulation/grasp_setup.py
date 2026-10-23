"""Scratch-only load-aware Mars grasp setup candidates.

Static support margins are model predictions, not dynamic or hardware safety
certificates. Candidate base/arm changes preserve the original world hand pose.
Applying a candidate here is permitted only during procedural reset; planning a
real approach trajectory is a separate responsibility of TAMP.
"""

import numpy as np

from emet.eval.physical_tamp import make_scene_checker
from emet.rl.manipulation.tracking import TaskFrameFK


def support_margin(model, data, profile, payload):
    if len(profile.wheel_bodies) != 3:
        raise ValueError("support margin currently requires the three-support Mars chassis")
    base = model.body(profile.base_body).id
    ids = {base}
    for body in range(base + 1, model.nbody):
        if int(model.body_parentid[body]) in ids:
            ids.add(body)
    ids.add(int(payload))
    ids = np.array(sorted(ids))
    weights = model.body_mass[ids].copy()
    centers = data.xpos[ids] + np.einsum("nij,nj->ni", data.xmat[ids].reshape(-1, 3, 3), model.body_ipos[ids])
    com = np.sum(centers * weights[:, None], axis=0) / weights.sum()
    polygon = np.array([data.body(name).xpos[:2] for name in profile.wheel_bodies])
    middle = polygon.mean(axis=0)
    polygon = polygon[np.argsort(np.arctan2(polygon[:, 1] - middle[1], polygon[:, 0] - middle[0]))]
    margins = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge, delta = b - a, com[:2] - a
        margins.append(float((edge[0] * delta[1] - edge[1] * delta[0]) / np.linalg.norm(edge)))
    return {"margin_m": min(margins), "loaded_com_m": com.tolist(), "support_polygon_m": polygon.tolist()}


def plan_setup(env, *, minimum_margin=0.005, azimuths=(0.0, -0.7, 0.7, -1.0, 1.0, -1.4, 1.4)):
    """Return a collision-checked candidate without mutating the live model/data."""
    c, model = env.control, env.model
    if c.profile.robot != "innate_mars" or not np.isfinite(minimum_margin) or minimum_margin < 0:
        raise ValueError("invalid Mars grasp setup request")
    base = c.base_pose().copy()
    arm = c.positions().copy()
    fk = TaskFrameFK(model, env.data, c)
    hand = fk(base, arm).copy()
    # This existing private IK solve is offline target generation, never actor input.
    end = env._vertical_reference(env.config.grasp_lift_height, rotation=env.data.xmat[c.ee].reshape(3, 3))
    initial_evidence = support_margin(model, fk.data, env.profile, env.payload)
    carry_offsets = None
    if env.config.tamp_carry_trajectories:
        from emet.rl.manipulation.carry_training import target_offsets

        # Plan the lift AND intended carry before selecting the initial setup.
        # Reserve 2 mm for reset settling; runtime admission retains its 5 mm gate.
        minimum_margin += 0.002
        carry_offsets = target_offsets(env, lift_end=end)
    candidates = []
    for azimuth in azimuths:
        q, goal = arm.copy(), base.copy()
        q[0] += azimuth
        if not c.limits[0, 0] <= q[0] <= c.limits[0, 1]:
            continue
        goal[2] = np.arctan2(np.sin(base[2] - azimuth), np.cos(base[2] - azimuth))
        trial = fk(goal, q)
        goal[:2] += hand[:2] - trial[:2]
        trial = fk(goal, q)
        if not np.allclose(trial, hand, atol=1e-6, rtol=0):
            continue
        witness = fk.data.qpos.copy()
        checker = make_scene_checker(
            model, {"object_body": env.payload_name, "gripper_bodies": env.profile.finger_bodies}
        )
        checker.set_payload(model, fk.data, env.payload_name, env.profile.ee_body)
        margins, rejected = [], False
        for fraction in np.linspace(0, 1, 9):
            sample = arm + fraction * (end - arm)
            sample[0] += azimuth
            fk(goal, sample)
            if checker.configuration_collides(model, fk.data):
                rejected = True
                break
            margins.append(support_margin(model, fk.data, env.profile, env.payload)["margin_m"])
        if rejected or min(margins) < minimum_margin:
            continue
        carry_offset = None
        if carry_offsets is not None:
            lifted = end.copy()
            lifted[0] += azimuth
            lift_margins = margins.copy()
            for offset in carry_offsets:
                destination = lifted + offset
                if np.any(destination < c.limits[:, 0] + 0.02) or np.any(destination > c.limits[:, 1] - 0.02):
                    continue
                valid = True
                margins = lift_margins.copy()
                for fraction in np.linspace(0, 1, 10):
                    fk(goal, lifted + fraction * offset)
                    if checker.configuration_collides(model, fk.data):
                        valid = False
                        break
                    margins.append(support_margin(model, fk.data, env.profile, env.payload)["margin_m"])
                if valid and min(margins) >= minimum_margin:
                    carry_offset = offset.tolist()
                    break
            if carry_offset is None:
                continue
        candidates.append(
            {
                "base": goal.tolist(),
                "arm": q.tolist(),
                "hand": hand.tolist(),
                "azimuth_offset_rad": float(azimuth),
                "minimum_sampled_margin_m": min(margins),
                "original_loaded_margin_m": initial_evidence["margin_m"],
                "qpos": witness,
                "initial_base": base.tolist(),
                "carry_joint_offset": carry_offset,
                "setup_margin_required_m": minimum_margin,
            }
        )
    if not candidates:
        return None
    nominal = next((item for item in candidates if item["azimuth_offset_rad"] == 0), None)
    disturbance = getattr(env, "_carry_push_selected", False)
    if nominal is not None and not disturbance:
        nominal["selection_reason"] = "nominal_path_margin"
        return nominal
    # Once a different setup is required, maximize its robustness margin.
    # Opposite turns have equal displacement to numerical precision, which
    # otherwise made the selected side depend on floating-point noise.
    selected = min(candidates, key=lambda item: (-item["minimum_sampled_margin_m"], abs(item["azimuth_offset_rad"])))
    selected["selection_reason"] = "maximum_margin_for_disturbance" if disturbance else "maximum_margin"
    return selected
