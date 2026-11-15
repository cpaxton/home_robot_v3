# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Diagnostic derivative channels; these never change rewards or acceptance."""

import numpy as np


def vector(value):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("finite world XY/yaw vector required")
    return result.copy()


def linear_magnitude(channel, name):
    jerk = np.asarray(channel["jerk"])
    return float(abs(jerk[0]) if name == "gate" else np.linalg.norm(jerk[:2]))


class DerivativeChannels:
    """Backward differences at one cadence, with explicit reset histories.

    Displacement velocities represent intervals centered half a tick before
    instantaneous Jacobian samples. Trapezoidal Jacobian velocity approximates
    that interval average; it does not replace either original measurement.
    """

    def __init__(self, pose, instantaneous_velocity, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite measurement cadence required")
        self.dt, self.pose = float(dt), vector(pose)
        self.velocities = {
            "gate": np.zeros(2),
            "displacement_vector": np.zeros(3),
            "jacobian": vector(instantaneous_velocity),
            "jacobian_zero_reset": np.zeros(3),
            "jacobian_trapezoid": np.zeros(3),
        }
        self.accelerations = {key: np.zeros_like(value) for key, value in self.velocities.items()}

    def sample(self, pose, instantaneous_velocity):
        pose, instantaneous_velocity = vector(pose), vector(instantaneous_velocity)
        delta = pose - self.pose
        delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
        values = {
            "gate": np.array([np.linalg.norm(delta[:2]) / self.dt, delta[2] / self.dt]),
            "displacement_vector": delta / self.dt,
            "jacobian": instantaneous_velocity,
            "jacobian_zero_reset": instantaneous_velocity.copy(),
            "jacobian_trapezoid": (self.velocities["jacobian"] + instantaneous_velocity) / 2,
        }
        result = {}
        for key, velocity in values.items():
            acceleration = (velocity - self.velocities[key]) / self.dt
            jerk = (acceleration - self.accelerations[key]) / self.dt
            result[key] = {"velocity": velocity.copy(), "acceleration": acceleration.copy(), "jerk": jerk.copy()}
            self.velocities[key], self.accelerations[key] = velocity.copy(), acceleration.copy()
        self.pose = pose.copy()
        return result


class NativeJerkObserver:
    """Read the exact v2 RewardMeter and native acceptance on the same steps."""

    def __init__(self):
        self.cases = {}
        self.pending = None

    def reset(self, env, case):
        from emet.rl.manipulation.whole_body_reward import RewardMeter

        if case["id"] in self.cases:
            raise ValueError("duplicate case in diagnostic replay")
        self.meter = RewardMeter(env)
        self.channels = DerivativeChannels(env.control.base_pose(), self.meter.base_velocity, env.profile.dt)
        np.testing.assert_array_equal(self.channels.pose, env.last_base_pose)
        self.current = {
            "id": case["id"],
            "reset_seed": case["reset_seed"],
            "body": env.model.body(env.control.base).name,
            "dt_seconds": env.profile.dt,
            "physics_dt_seconds": float(env.model.opt.timestep),
            "initial_time": float(env.data.time),
            "initial_qvel": env.data.qvel.copy().tolist(),
            "initial_gate_velocity": env.last_base_velocity.copy().tolist(),
            "initial_jacobian_velocity": self.meter.base_velocity.copy().tolist(),
            "initial_state_sha256": env.mastery_admission["initial_state_sha256"],
            "reference_sha256": env.mastery_admission["reference_sha256"],
            "samples": [],
        }
        self.cases[case["id"]] = self.current
        self.pending = None

    def before_step(self, env, action):
        from emet.rl.manipulation.whole_body_observation import reference_velocity

        if self.pending is not None:
            raise RuntimeError("diagnostic step has no matching after-step")
        self.pending = (float(env.data.time), reference_velocity(env.control), np.asarray(action).copy())

    def after_step(self, env, action, info):
        if self.pending is None:
            raise RuntimeError("diagnostic after-step has no matching before-step")
        before_time, motion, requested = self.pending
        np.testing.assert_array_equal(requested, action)
        physical = [np.asarray(getattr(env.data, key)).copy() for key in ("qpos", "qvel", "ctrl", "act")]
        now, previous_acceleration = float(env.data.time), self.meter.base_acceleration.copy()
        rewards = self.meter.tick(requested, motion)
        for key, before in zip(("qpos", "qvel", "ctrl", "act"), physical, strict=True):
            np.testing.assert_array_equal(before, np.asarray(getattr(env.data, key)))
        if float(env.data.time) != now:
            raise RuntimeError("diagnostic changed simulator time")
        channels = self.channels.sample(env.control.base_pose(), self.meter.base_velocity)
        np.testing.assert_array_equal(channels["gate"]["velocity"], env.last_base_velocity)
        np.testing.assert_array_equal(channels["gate"]["acceleration"], env.last_base_acceleration)
        np.testing.assert_array_equal(channels["jacobian"]["acceleration"], self.meter.base_acceleration)
        np.testing.assert_array_equal(
            channels["jacobian"]["jerk"], (self.meter.base_acceleration - previous_acceleration) / env.profile.dt
        )
        self.current["samples"].append(
            {
                "time": now,
                "actual_elapsed_seconds": now - before_time,
                "pose": env.control.base_pose().tolist(),
                "action": requested.tolist(),
                "channels": {
                    key: {name: value.tolist() for name, value in fields.items()} for key, fields in channels.items()
                },
                "reward_groups_readonly": rewards.tolist(),
                "reward_terms_readonly": self.meter.last_terms.copy(),
                "reward_base_jerk_cost": float(
                    np.sum(np.sqrt(1 + (channels["jacobian"]["jerk"] / [1, 1, 4]) ** 2) - 1)
                ),
                "code": info["code"],
                "reference_complete": bool(info["reference_complete"]),
                "force_peak_N": float(info["peak_contact_force"]),
                "native_peak_linear_jerk": float(env.peak_base_axis_jerk[0]),
                "native_peak_yaw_jerk": float(env.peak_base_axis_jerk[1]),
            }
        )
        self.pending = None

    def summarize(self):
        result = []
        for case in self.cases.values():
            samples = case["samples"]
            if not samples:
                raise ValueError("diagnostic case has no samples")
            channel_peaks = {}
            for key in samples[0]["channels"]:
                magnitudes = [linear_magnitude(row["channels"][key], key) for row in samples]
                index = int(np.argmax(magnitudes))
                peak = samples[index]
                channel_peaks[key] = {
                    "linear_jerk_peak": magnitudes[index],
                    "linear_jerk_peak_after_two_samples": max(magnitudes[2:], default=0),
                    "linear_jerk_rms": float(np.sqrt(np.mean(np.square(magnitudes)))),
                    "peak_sample": index,
                    "peak_time": peak["time"],
                    "peak_code": peak["code"],
                    "reward_proxy_at_peak": linear_magnitude(peak["channels"]["jacobian"], "jacobian"),
                    "yaw_jerk_peak": max(abs(row["channels"][key]["jerk"][-1]) for row in samples),
                }
            if not np.isclose(
                channel_peaks["gate"]["linear_jerk_peak"], samples[-1]["native_peak_linear_jerk"], rtol=0, atol=1e-12
            ):
                raise ValueError("diagnostic failed to reproduce the native jerk peak")
            result.append(
                {key: value for key, value in case.items() if key != "samples"}
                | {
                    "samples": len(samples),
                    "channel_peaks": channel_peaks,
                    "readonly_reward_group_totals": np.sum(
                        [row["reward_groups_readonly"] for row in samples], axis=0
                    ).tolist(),
                    "integrated_base_jerk_reward_component": -case["dt_seconds"]
                    * 0.01
                    * sum(row["reward_base_jerk_cost"] for row in samples),
                    "max_dt_error_seconds": max(
                        abs(row["actual_elapsed_seconds"] - case["dt_seconds"]) for row in samples
                    ),
                }
            )
        return result
