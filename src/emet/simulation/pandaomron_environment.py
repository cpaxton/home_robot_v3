# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Create RoboCasa's PandaOmron environment with native joint-position control."""

from threading import RLock

_MODEL_COMPILATION_LOCK = RLock()

def serial_mesh_compiler(xml):
    """Avoid native mesh-compiler races during repeated pinned MuJoCo resets."""
    from xml.etree import ElementTree

    root = ElementTree.fromstring(xml)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ElementTree.SubElement(root, "compiler")
    compiler.set("usethread", "false")
    return ElementTree.tostring(root, encoding="unicode")


def compile_serial_robot_model(xml_model, mode="mujoco"):
    """Apply serial compilation to robot-only models as well as scene resets."""
    import mujoco

    if mode != "mujoco":
        raise ValueError("unsupported_robot_model_mode")
    return mujoco.MjModel.from_xml_string(serial_mesh_compiler(xml_model.get_xml()))


def serial_robot_loader(env):
    # Robot.load_model compiles an intrinsic model before the scene XML
    # processors run. Scope the serial compiler to that load, including retry
    # recursion, without changing vendored source or runtime dynamics.
    load_model = env._load_model

    def load(*args, **kwargs):
        from robosuite.models.base import MujocoXML

        with _MODEL_COMPILATION_LOCK:
            previous = MujocoXML.get_model
            MujocoXML.get_model = compile_serial_robot_model
            try:
                return load_model(*args, **kwargs)
            finally:
                MujocoXML.get_model = previous

    env._load_model = load
    return env


def preserve_absolute_reset_posture(env):
    """Translate RoboCasa's reset-only neutral action into an absolute hold.

    Kitchen reset settles objects with zero actions, assuming delta control.
    Absolute zero is a different joint target. Preserve the native reset pose
    through native controller actions; normal episode commands are untouched.
    """
    import numpy as np

    reset, pre_action = env._reset_internal, env._pre_action
    resetting = [False]
    hold = [None]

    def reset_internal():
        resetting[0], hold[0] = True, None
        try:
            return reset()
        finally:
            resetting[0], hold[0] = False, None

    def apply(action, policy_step=False):
        if resetting[0]:
            if hold[0] is None:
                robot = env.robots[0]
                q = env.sim.data.qpos
                hold[0] = robot.create_action_vector({
                    "right": np.array(q[robot._ref_joint_pos_indexes], copy=True),
                    "torso": np.array(q[robot._ref_torso_joint_pos_indexes], copy=True),
                })
            action = hold[0]
        return pre_action(action, policy_step)

    env._reset_internal, env._pre_action = reset_internal, apply
    return env


CONTROL_HZ = 20


def make_pandaomron_environment(*, task, layout, style, seed):
    import robocasa  # noqa: F401  (registers environments with robosuite)
    import robosuite
    from robosuite.controllers import load_composite_controller_config, load_part_controller_config

    config = load_composite_controller_config(robot="PandaOmron")
    config["body_parts"]["right"] = {
        **load_part_controller_config(default_controller="JOINT_POSITION"),
        "input_type": "absolute",
        "kp": 150,
        "gripper": {"type": "GRIP"},
    }
    config["body_parts"]["torso"]["input_type"] = "absolute"
    env = robosuite.make(
        env_name=task,
        robots="PandaOmron",
        controller_configs=config,
        layout_and_style_ids=[[layout, style]],
        seed=seed,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=CONTROL_HZ,
        ignore_done=True,
    )
    env.set_xml_processor(serial_mesh_compiler)
    return preserve_absolute_reset_posture(serial_robot_loader(env))


def settle_absolute_environment(env, *, steps=30):
    """Let restored fixture dynamics settle while natively holding robot joints."""
    import numpy as np

    if type(steps) is not int or not 0 <= steps <= 1000:
        raise ValueError('invalid_initial_settle_steps')
    robot = env.robots[0]
    q = env.sim.data.qpos
    action = robot.create_action_vector({
        'right': np.array(q[robot._ref_joint_pos_indexes], copy=True),
        'torso': np.array(q[robot._ref_torso_joint_pos_indexes], copy=True),
        'right_gripper': np.array([-1.]),
    })
    for _ in range(steps):
        env.step(action)
