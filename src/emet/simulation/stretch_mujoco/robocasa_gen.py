# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.
"""Modified version of robocasa's kitchen scene generation script."""

import re
import tempfile
from collections import OrderedDict
from pathlib import Path

import click
import mujoco
import mujoco.viewer
import numpy as np

# Import robocasa so its environments register with robosuite.make(); otherwise
# robosuite.make("PickPlaceCounterToCabinet") raises "Environment ... not found".
import robocasa  # noqa: F401
import robosuite
from robocasa.models.scenes.scene_registry import StyleType
from robosuite import load_part_controller_config
from robosuite.utils.transform_utils import euler2mat, mat2quat
from termcolor import colored

from emet.simulation.stretch_mujoco.utils import (
    _strip_geom_shellinertia,
    ensure_mesh_inertia,
    get_absolute_path_stretch_xml,
    insert_line_after_mujoco_tag,
    replace_xml_tag_value,
    xml_modify_body_pos,
    xml_remove_all_tags,
    xml_remove_subelement,
    xml_remove_tag_by_name,
)


def get_styles() -> OrderedDict:
    raw_styles = {item.value: item.name.lower().capitalize() for item in StyleType}
    styles = OrderedDict()
    for k in sorted(raw_styles.keys()):
        if k < 0:
            continue
        styles[k] = raw_styles[k]
    return styles


layouts = OrderedDict(
    [
        (0, "One wall"),
        (1, "One wall w/ island"),
        (2, "L-shaped"),
        (3, "L-shaped w/ island"),
        (4, "Galley"),
        (5, "U-shaped"),
        (6, "U-shaped w/ island"),
        (7, "G-shaped"),
        (8, "G-shaped (large)"),
        (9, "Wraparound"),
    ]
)


def choose_option(options, option_name, show_keys=False, default=None, default_message=None):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all tasks

    if default is None:
        default = options[0]

    if default_message is None:
        default_message = default

    # Select environment to run
    print(f"{option_name.capitalize()}s:")

    for i, (k, v) in enumerate(options.items()):
        if show_keys:
            print(f"[{i}] {k}: {v}")
        else:
            print(f"[{i}] {v}")
    print()
    try:
        s = input(f"Choose an option 0 to {len(options) - 1}, or any other key for default ({default_message}): ")
        # parse input into a number within range
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except Exception:
        if default is None:
            choice = options[0]
        else:
            choice = default
        print(f"Use {choice} by default.\n")

    # Return the chosen environment name
    return choice


def choose_layout():
    layout = choose_option(layouts, "kitchen layout", default=-1, default_message="random layouts")

    if layout == -1:
        layout = np.random.choice(range(10))
        print(colored(f"Randomly choosing layout... id: {layout}", "yellow"))

    return layout


def choose_style():
    styles = get_styles()
    style = choose_option(styles, "kitchen style", default=-1, default_message="random styles")

    if style == -1:
        style = np.random.choice(range(11))
        print(colored(f"Randomly choosing style... id: {style}", "yellow"))

    return style


def layout_from_str(layout: str) -> int:
    """Returns the index of the layout in the orderedDict"""
    return list(layouts.values()).index(layout)


def style_from_str(style: str) -> int:
    """Returns the index of the style in the orderedDict"""
    return list(get_styles().values()).index(style)


ROBOSUITE_ROBOTS = [
    "PandaOmron",
    "Tiago",
    "GR1",
    "GR1FixedLowerBody",
    "SpotWithArm",
]

# Galaxea R1 family: use PandaMobile in Robocasa (mobile-base spawn), then strip-and-replace MJCF.
_GALAXEA_R1_ROBOT_KEYS = frozenset(
    {"rby1", "rby1m", "galaxea_r1", "galaxear1", "rb_y1", "rby_1"}
)


def _normalize_robot_key(robot_name: str) -> str:
    return robot_name.lower().replace("-", "_")


def _uses_strip_placeholder_robot(robot_name: str) -> bool:
    key = _normalize_robot_key(robot_name)
    return key in (
        "stretch",
        "hello_stretch",
        "hellostretch",
        "innate_mars",
        *_GALAXEA_R1_ROBOT_KEYS,
    )


def _robosuite_robot_for(robot_name: str) -> str:
    """Map an emet robot name to the robosuite robot class used as the scene placeholder.

    For Stretch and innate_mars we use PandaMobile as a placeholder (then strip-and-replace).
    For robosuite-native robots we use the robot directly.
    """
    if robot_name in ("stretch", None):
        return "PandaMobile"
    mapping = {r.lower(): r for r in ROBOSUITE_ROBOTS}
    mapping["pandaomron"] = "PandaOmron"
    mapping["panda_omron"] = "PandaOmron"
    key = robot_name.lower().replace("-", "").replace("_", "")
    if key in mapping:
        return mapping[key]
    for rs_name in ROBOSUITE_ROBOTS:
        if key == rs_name.lower():
            return rs_name
    return robot_name


def model_generation_wizard(
    task: str = "PickPlaceCounterToCabinet",
    layout: int = None,
    style: int = None,
    write_to_file: str = None,
    robot_spawn_pose: dict = None,
    robot: str = "stretch",
) -> tuple[mujoco.MjModel, str, dict]:
    """
    Wizard/API to generate a kitchen model for a given task, layout, and style.

    Args:
        task: Robocasa task name.
        layout: Layout id (None = interactive choice).
        style: Style id (None = interactive choice).
        write_to_file: Optional path to save the generated XML.
        robot_spawn_pose: Override spawn pose ``{pos: "x y z", quat: "w x y z"}``.
        robot: Robot name. ``"stretch"``, ``"innate_mars"``, and Galaxea R1 family ids (``"rby1"``, etc.)
            use a PandaMobile placeholder in Robocasa, then strip-and-replace with the real MJCF.
            Robosuite-native names (e.g. ``"PandaOmron"``, ``"Tiago"``, ``"GR1"``) keep that robot in the scene.
    Returns:
        Tuple of (MjModel, xml_string, object_placements_info).
    """

    if layout is None:
        layout = choose_layout()

    styles = get_styles()
    if style is None:
        style = choose_style()

    robot_key = _normalize_robot_key(robot)
    use_strip_placeholder_robot = _uses_strip_placeholder_robot(robot)
    use_stretch_robot = robot_key in ("stretch", "hello_stretch", "hellostretch")
    use_galaxea_robot = robot_key in _GALAXEA_R1_ROBOT_KEYS
    rs_robot = _robosuite_robot_for(robot)

    config = {
        "env_name": task,
        "robots": "PandaMobile" if use_strip_placeholder_robot else rs_robot,
        "controller_configs": load_part_controller_config(default_controller="OSC_POSE"),
        "translucent_robot": False,
        "layout_and_style_ids": [[layout, style]],
    }

    print(colored("Initializing environment...", "yellow"))

    env = robosuite.make(
        **config,
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    print(
        colored(
            f"Showing configuration:\n    Layout: {layouts[layout]}\n    Style: {styles[style]}",
            "green",
        )
    )
    print(colored("Spawning environment...\n", "yellow"))

    model = env.sim.model._model
    xml = env.sim.model.get_xml()

    click.secho(f"\nMaking Object Placements for task [{task}]...\n", fg="yellow")
    object_placements_info = {}
    for i in range(len(env.object_cfgs)):
        obj_name = env.object_cfgs[i]["name"]
        category = env.object_cfgs[i]["info"]["cat"]
        object_placements = env.object_placements
        print(
            f"Placing [Object {i}] (category: {category}, body_name: {obj_name}_main) at "
            f"pos: {np.round(object_placements[obj_name][0], 2)} quat: {np.round(object_placements[obj_name][1], 2)}"
        )
        xml = xml_modify_body_pos(
            xml,
            "body",
            obj_name + "_main",
            pos=object_placements[obj_name][0],
            quat=object_placements[obj_name][1],
        )
        object_placements_info[obj_name + "_main"] = {
            "cat": category,
            "pos": object_placements[obj_name][0],
            "quat": object_placements[obj_name][1],
        }

    if use_strip_placeholder_robot:
        xml, remove_robot_attrib = custom_cleanups(xml)

        if hasattr(env, "init_robot_base_pos") and hasattr(env, "init_robot_base_ori"):
            pos = np.asarray(env.init_robot_base_pos)
            ori_euler = np.asarray(env.init_robot_base_ori)
            quat_xyzw = mat2quat(euler2mat(ori_euler))
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
            robot_base_fixture_pose = {
                "pos": " ".join(map(str, pos)),
                "quat": " ".join(map(str, quat_wxyz)),
            }
        elif remove_robot_attrib is not None and "pos" in remove_robot_attrib and "quat" in remove_robot_attrib:
            robot_base_fixture_pose = remove_robot_attrib
        else:
            robot_base_fixture_pose = {"pos": "0 0 0", "quat": "1 0 0 0"}

        if robot_spawn_pose is not None:
            robot_base_fixture_pose = robot_spawn_pose

        xml = ensure_mesh_inertia(xml)

        click.secho("\nMaking Robot Placement...\n", fg="yellow")
        if use_stretch_robot:
            xml = add_stretch_to_kitchen(xml, robot_base_fixture_pose)
        elif use_galaxea_robot:
            xml = add_galaxea_r1_to_kitchen(xml, robot_base_fixture_pose, robot_key=robot_key)
        else:
            xml = add_innate_mars_to_kitchen(xml, robot_base_fixture_pose)
    else:
        xml = _cleanup_for_native_robot(xml)
        xml = ensure_mesh_inertia(xml)
        click.secho(f"\nKeeping robosuite robot '{rs_robot}' in scene.\n", fg="yellow")

    # Strip keyframes from the serialized kitchen model when we swapped in a different robot:
    # ``get_xml()`` can retain Panda-sized ``<key qpos="..."/>`` vectors that no longer line up
    # with Innate Mars / Stretch ``nq``, shifting joint defaults (notably the head).
    if use_strip_placeholder_robot:
        xml = xml_remove_all_tags(xml, "key")

    model = mujoco.MjModel.from_xml_string(xml)

    if use_strip_placeholder_robot and hasattr(env, "init_robot_base_pos"):
        pos = np.asarray(env.init_robot_base_pos, dtype=float).reshape(-1)
        yaw = 0.0
        if hasattr(env, "init_robot_base_ori"):
            ori = np.asarray(env.init_robot_base_ori, dtype=float).reshape(-1)
            if ori.size >= 3:
                yaw = float(ori[2])
        if pos.size >= 2:
            object_placements_info["_emet_spawn_hint_xyt"] = [float(pos[0]), float(pos[1]), yaw]

    if write_to_file is not None:
        with open(write_to_file, "w") as f:
            f.write(xml)
        print(colored(f"Model saved to {write_to_file}", "green"))

    return model, xml, object_placements_info


def _cleanup_for_native_robot(xml: str) -> str:
    """Minimal cleanup for scenes where we keep the robosuite robot.

    We hide the debug markers but do NOT remove actuators/sensors/option
    since the robosuite robot needs them.
    """
    xml = replace_xml_tag_value(xml, "geom", "rgba", "0.5 0 0 0.5", "0.5 0 0 0")
    xml = replace_xml_tag_value(xml, "geom", "rgba", "0.5 0 0 1", "0.5 0 0 0")
    xml = replace_xml_tag_value(xml, "site", "rgba", "0.5 0 0 1", "0.5 0 0 0")
    xml = replace_xml_tag_value(xml, "site", "actuator", "0.3 0.4 1 0.5", "0.3 0.4 1 0")
    return xml


def custom_cleanups(xml: str) -> tuple[str, dict]:
    """
    Custom cleanups to models from robocasa envs to support
    use with stretch_mujoco package.
    """

    # make invisible the red/blue boxes around geom/sites of interests found
    xml = replace_xml_tag_value(xml, "geom", "rgba", "0.5 0 0 0.5", "0.5 0 0 0")
    xml = replace_xml_tag_value(xml, "geom", "rgba", "0.5 0 0 1", "0.5 0 0 0")
    xml = replace_xml_tag_value(xml, "site", "rgba", "0.5 0 0 1", "0.5 0 0 0")
    xml = replace_xml_tag_value(xml, "site", "actuator", "0.3 0.4 1 0.5", "0.3 0.4 1 0")
    # remove subelements
    xml = xml_remove_subelement(xml, "actuator")
    xml = xml_remove_subelement(xml, "sensor")

    # remove option tag element
    xml = xml_remove_subelement(xml, "option")
    # xml = xml_remove_subelement(xml, "size")

    # remove robot
    xml, remove_robot_attrib = xml_remove_tag_by_name(xml, "body", "robot0_base")

    return xml, remove_robot_attrib


def add_stretch_to_kitchen(xml: str, robot_pose_attrib: dict) -> str:
    """
    Add stretch robot to kitchen xml
    """
    print(f"Adding stretch to kitchen at pos: {robot_pose_attrib['pos']} quat: {robot_pose_attrib['quat']}")
    stretch_xml_absolute = get_absolute_path_stretch_xml(robot_pose_attrib)
    # add Stretch xml
    xml = insert_line_after_mujoco_tag(
        xml,
        f' <include file="{stretch_xml_absolute}"/>',
    )
    return xml


def add_innate_mars_to_kitchen(xml: str, robot_pose_attrib: dict) -> str:
    """Add Innate Mars MJCF to kitchen XML (strip-and-replace after PandaMobile placeholder)."""
    from emet.utils.assets import get_robot_mjcf_path

    mjcf = get_robot_mjcf_path("innate_mars")
    if mjcf is None or not mjcf.is_file():
        raise FileNotFoundError(
            "Innate Mars MJCF not found (emet package data). Cannot build Robocasa scene for innate_mars."
        )
    root_dir = mjcf.parent.resolve()
    meshes_abs = (root_dir / "meshes").resolve()
    if not meshes_abs.is_dir():
        raise FileNotFoundError(f"Innate Mars meshes directory missing: {meshes_abs}")

    text = mjcf.read_text(encoding="utf-8")
    text = text.replace('meshdir="meshes"', f'meshdir="{meshes_abs.as_posix()}"')

    def _abs_mesh_file_attr(m: re.Match) -> str:
        fname = m.group(1)
        if fname.startswith("/") or "/" in fname:
            return m.group(0)
        return f'file="{(meshes_abs / fname).resolve().as_posix()}"'

    text = re.sub(r'file="([^"]+\.(?:STL|stl))"', _abs_mesh_file_attr, text)
    # Geom `euler=` on head/base assumes `compiler eulerseq="zyx"` (see innate_mars.xml). Robocasa's
    # merged kitchen root often omits eulerseq, so MuJoCo's default reinterprets those euler angles
    # while cameras keep explicit `quat` — head cameras then look "wrong" relative to the head mesh.
    # Replace the two known Rz(-π/2) mesh tilts with an explicit wxyz quaternion (same as zyx Rz(-π/2)).
    _rz_neg_90_wxyz = "0.7071067811865476 0 0 -0.7071067811865476"
    text = text.replace('euler="-1.5708 0 0"', f'quat="{_rz_neg_90_wxyz}"')
    if robot_pose_attrib is not None:
        pos = robot_pose_attrib["pos"]
        quat = robot_pose_attrib["quat"]
        text = re.sub(
            r'<body\s+name="base_root"[^>]*>',
            f'<body name="base_root" pos="{pos}" quat="{quat}">',
            text,
            count=1,
        )
    text = ensure_mesh_inertia(text)
    text = _strip_geom_shellinertia(text)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_innate_mars_kitchen.xml",
        delete=False,
        encoding="utf-8",
    ) as fh:
        fh.write(text)
        tmp_path = fh.name
    abs_path = Path(tmp_path).resolve().as_posix()
    print(f"Adding innate_mars to kitchen via temp MJCF: {abs_path}")
    return insert_line_after_mujoco_tag(xml, f' <include file="{abs_path}"/>')


def add_galaxea_r1_to_kitchen(
    xml: str,
    robot_pose_attrib: dict,
    *,
    robot_key: str = "rby1",
) -> str:
    """Add Galaxea R1 / rby1 MJCF to kitchen XML (strip-and-replace after PandaMobile placeholder)."""
    from emet.utils.assets import get_robot_mjcf_path

    lookup = "galaxea_r1" if robot_key in ("galaxea_r1", "galaxear1") else "rby1"
    mjcf = get_robot_mjcf_path(lookup)
    if mjcf is None or not mjcf.is_file():
        raise FileNotFoundError(
            f"Galaxea R1 MJCF not found for {robot_key!r} (lookup {lookup!r}). "
            "Cannot build Robocasa scene."
        )
    root_dir = mjcf.parent.resolve()
    meshes_abs = (root_dir / "meshes").resolve()
    if not meshes_abs.is_dir():
        raise FileNotFoundError(f"Galaxea R1 meshes directory missing: {meshes_abs}")

    text = mjcf.read_text(encoding="utf-8")
    text = text.replace('meshdir="meshes"', f'meshdir="{meshes_abs.as_posix()}"')
    text = text.replace('assetdir="meshes"', f'assetdir="{meshes_abs.as_posix()}"')

    def _abs_mesh_file_attr(m: re.Match) -> str:
        fname = m.group(1)
        if fname.startswith("/") or "/" in fname:
            return m.group(0)
        return f'file="{(meshes_abs / fname).resolve().as_posix()}"'

    text = re.sub(r'file="([^"]+\.(?:STL|stl))"', _abs_mesh_file_attr, text)
    if robot_pose_attrib is not None:
        pos = robot_pose_attrib["pos"]
        quat = robot_pose_attrib["quat"]
        text = re.sub(
            r'<body\s+name="base_link"[^>]*>',
            f'<body name="base_link" pos="{pos}" quat="{quat}" gravcomp="0">',
            text,
            count=1,
        )
    text = ensure_mesh_inertia(text)
    text = _strip_geom_shellinertia(text)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_galaxea_r1_kitchen.xml",
        delete=False,
        encoding="utf-8",
    ) as fh:
        fh.write(text)
        tmp_path = fh.name
    abs_path = Path(tmp_path).resolve().as_posix()
    print(f"Adding {robot_key} (Galaxea R1) to kitchen via temp MJCF: {abs_path}")
    return insert_line_after_mujoco_tag(xml, f' <include file="{abs_path}"/>')
