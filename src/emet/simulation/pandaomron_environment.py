# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Create RoboCasa's PandaOmron environment with native joint-position control."""

def serial_mesh_compiler(xml):
    """Avoid native mesh-compiler races during repeated pinned MuJoCo resets."""
    from xml.etree import ElementTree

    root = ElementTree.fromstring(xml)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ElementTree.SubElement(root, "compiler")
    compiler.set("usethread", "false")
    return ElementTree.tostring(root, encoding="unicode")


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
    return env
