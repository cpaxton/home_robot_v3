"""Measured-versus-requested Mars arm diagnostics, never actor inputs or actions."""

import mujoco
import numpy as np


def hand_tracking_diagnostics(control):
    """Read current state without stepping, solving IK or changing control.

    Project the slewed joint-rate request through the measured Jacobian. This
    separates an insufficient policy request from physical tracking lag. The
    signed projection is toward the current target, not a trajectory-success
    metric: a moving reference can require transverse motion as well.
    """
    if control.profile.robot != "innate_mars" or any(len(g) != 1 for g in control.v):
        raise ValueError("hand audit currently supports the five-joint Mars arm")
    model, data = control.model, control.data
    jacobian = np.zeros((3, model.nv))
    rotation = np.zeros_like(jacobian)
    mujoco.mj_jacBody(model, data, jacobian, rotation, control.ee)
    columns = np.asarray([g[0] for g in control.v])
    requested = control.velocity.copy()
    measured = control.velocities().copy()
    predicted = jacobian[:, columns] @ requested
    actual = jacobian @ data.qvel.copy()
    error = control.hand_goal[:3] - control.ee_pose()[:3]
    direction = error / max(float(np.linalg.norm(error)), 1e-9)
    return {
        "position_error_world_m": error.tolist(),
        "slewed_joint_velocity_rad_s": requested.tolist(),
        "measured_joint_velocity_rad_s": measured.tolist(),
        "motor_target_error_rad": (control.target - control.positions()).tolist(),
        "actuator_torque_Nm": data.actuator_force[control.act].copy().tolist(),
        "predicted_arm_linear_velocity_world_m_s": predicted.tolist(),
        "measured_hand_linear_velocity_world_m_s": actual.tolist(),
        "requested_speed_toward_current_goal_m_s": float(predicted @ direction),
        "measured_speed_toward_current_goal_m_s": float(actual @ direction),
    }
