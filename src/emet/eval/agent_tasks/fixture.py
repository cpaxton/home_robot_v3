"""Small MuJoCo fixture using the real rby1 MJCF and shared CHAT TAMP tools.

This is an in-process assisted control, not ZMQ, learned navigation, IK, contact
simulation, or a substitute for the existing live robot integration gate.
"""

from __future__ import annotations

import math
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace


def obstacles(scene: dict) -> list[list[float]]:
    rooms = scene["rooms"]
    xmin = min(r["bounds"][0] for r in rooms)
    xmax = max(r["bounds"][2] for r in rooms)
    ymin = min(r["bounds"][1] for r in rooms)
    ymax = max(r["bounds"][3] for r in rooms)
    result = [
        [xmin - 0.1, ymin, xmin + 0.1, ymax],
        [xmax - 0.1, ymin, xmax + 0.1, ymax],
        [xmin, ymin - 0.1, xmax, ymin + 0.1],
        [xmin, ymax - 0.1, xmax, ymax + 0.1],
    ]
    for door in scene["doors"]:
        x = door["x"]
        result.extend([[x - 0.1, ymin, x + 0.1, door["y_min"]], [x - 0.1, door["y_max"], x + 0.1, ymax]])
    for room in rooms:
        x = (room["bounds"][0] + room["bounds"][2]) / 2
        result.append([x - 0.8, -1.6, x + 0.8, -1.15])
    result.extend(item["bounds"] for item in scene.get("furniture", []))
    return result


def check_route(scene: dict, path: list[list[float]]) -> None:
    radius = scene["footprint_radius_m"]
    for a, b in zip(path, path[1:], strict=False):
        distance = math.dist(a[:2], b[:2])
        for i in range(max(1, math.ceil(distance / 0.02)) + 1):
            f = i / max(1, math.ceil(distance / 0.02))
            x, y = [a[j] + (b[j] - a[j]) * f for j in range(2)]
            if not any(
                r["bounds"][0] <= x <= r["bounds"][2] and r["bounds"][1] <= y <= r["bounds"][3] for r in scene["rooms"]
            ):
                raise ValueError("route leaves fixture floor")
            if any(
                x0 - radius <= x <= x1 + radius and y0 - radius <= y <= y1 + radius
                for x0, y0, x1, y1 in obstacles(scene)
            ):
                raise ValueError(f"footprint collision at ({x:.2f}, {y:.2f})")


def route_to(scene: dict, start: list[float], goal: list[float]) -> list[list[float]]:
    path = [start[:2], [start[0], 0.0], [goal[0], 0.0], goal[:2]]
    check_route(scene, path)
    return path


class FixtureRobot:
    """Minimal simulator adapter for the *existing* get_tools plan/execute path."""

    def __init__(self, suite: dict, output: Path, *, render: bool = True):
        import mujoco
        import numpy as np

        from emet.utils.assets import get_robot_mjcf_path

        self.suite = suite
        self.scene = suite["scene"]
        self._spec = SimpleNamespace(name="rby1", tamp_approach="front")
        self._last_step = 0
        self.session_id = uuid.uuid4().hex
        self.on_motion = None
        self.trajectory = []
        self.held: list[str] = []
        robot_path = get_robot_mjcf_path("rby1")
        if robot_path is None:
            raise FileNotFoundError("rby1 MJCF is unavailable")
        root = ET.parse(robot_path).getroot()
        camera = root.find(".//camera[@name='zed_camera']")
        pitch = math.radians(self.scene["head_pitch_degrees"])
        camera.set("xyaxes", f"0 -1 0 {math.sin(pitch)} 0 {math.cos(pitch)}")
        compiler = root.find("compiler")
        compiler.set("assetdir", str((robot_path.parent / "meshes").resolve()))
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth="960", offheight="540")
        world = root.find("worldbody")
        ET.SubElement(world, "light", pos="4 0 6", directional="true", dir="0 0 -1")
        ET.SubElement(world, "geom", name="fixture_floor", type="plane", size="15 8 .05", rgba=".88 .87 .83 1")
        for i, (x0, y0, x1, y1) in enumerate(obstacles(self.scene)):
            is_table = i >= 4 + 2 * len(self.scene["doors"])
            height = 0.25 if is_table else 1.15
            ET.SubElement(
                world,
                "geom",
                name=f"fixture_obstacle_{i}",
                type="box",
                pos=f"{(x0 + x1) / 2} {(y0 + y1) / 2} {height}",
                size=f"{(x1 - x0) / 2} {(y1 - y0) / 2} {height}",
                rgba=".48 .32 .19 1" if is_table else ".72 .77 .80 1",
            )
        for name, obj in self.scene["objects"].items():
            body = ET.SubElement(world, "body", name=name, pos=" ".join(map(str, obj["position"])))
            if obj["movable"]:
                ET.SubElement(body, "freejoint")
            ET.SubElement(
                body,
                "geom",
                type=obj["shape"],
                size=".04 .04 .04" if obj["movable"] else ".17 .17 .02",
                mass=".05",
                rgba=" ".join(map(str, obj["color"])),
            )
        scene_path = output / "fixture.xml"
        ET.ElementTree(root).write(scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        # Forward kinematics is intentional: assisted controls must not be
        # mistaken for contact-stable manipulation or wheel-driven navigation.
        self.xyt = list(self.scene["start_xyt"])
        self._write_base()
        self.renderer = mujoco.Renderer(self.model, height=360, width=640) if render else None
        self.np = np

    def _write_base(self):
        import mujoco

        joint = self.model.joint("base_freejoint")
        adr = int(joint.qposadr[0])
        x, y, yaw = self.xyt
        self.data.qpos[adr : adr + 3] = [x, y, 0.0]
        self.data.qpos[adr + 3 : adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)

    def positions(self) -> dict:
        return {name: self.data.body(name).xpos.tolist() for name in self.scene["objects"]}

    def get_emet_session(self) -> dict:
        return {
            "is_simulation": True,
            "runtime_kind": "agent_task_assisted_fixture",
            "environment": {"kind": "three_room_fixture", "scene": self.session_id},
            "capabilities": {"sim_set_body_pose": True, "kinematic_manip": False},
            "sim_object_placements_frame": "mujoco_world",
            "sim_object_placements": {
                name: {"cat": self.scene["objects"][name]["label"], "pos": pos, "quat": [1, 0, 0, 0]}
                for name, pos in self.positions().items()
            },
        }

    def move_base_to(self, xyt, blocking=True, world_frame=True):
        if not world_frame:
            raise ValueError("fixture accepts world-frame navigation only")
        goal = list(map(float, xyt))
        path = route_to(self.scene, self.xyt, goal)
        for a, b in zip(path, path[1:], strict=False):
            count = max(1, math.ceil(math.dist(a, b) / 0.25))
            yaw = math.atan2(b[1] - a[1], b[0] - a[0]) if a != b else goal[2]
            for i in range(1, count + 1):
                self.xyt = [a[j] + (b[j] - a[j]) * i / count for j in range(2)] + [yaw]
                self._write_base()
                self.trajectory.append(self.xyt[:])
                if self.on_motion:
                    self.on_motion()
        self.xyt = goal
        self._write_base()

    def relocate(self, body, position):
        from emet.simulation.sim_manipulation import set_free_body_pose

        if not set_free_body_pose(self.model, self.data, body, position):
            raise ValueError(f"cannot relocate {body}")

    def send_action(self, action, reliable=True):
        from emet.simulation.sim_manipulation import parse_sim_set_body_pose_action

        body, position, _ = parse_sim_set_body_pose_action(action.get("sim_set_body_pose"))
        if body is None or position is None:
            raise ValueError("unsupported fixture action")
        if math.dist(self.xyt[:2], position[:2]) > 0.9:
            self.move_base_to([position[0], position[1] + 0.55, -math.pi / 2])
        self.relocate(body, position)
        self.held = [body] if position[2] > 0.65 else []
        self._last_step = int(action["step"])

    def images(self) -> dict:
        import mujoco

        if self.renderer is None:
            return {}
        result = {}
        for key, camera in (("head", "zed_camera"), ("wrist", "left_camera")):
            self.renderer.update_scene(self.data, camera=camera)
            result[key] = self.renderer.render().copy()
        camera = mujoco.MjvCamera()
        xmin = min(r["bounds"][0] for r in self.scene["rooms"])
        xmax = max(r["bounds"][2] for r in self.scene["rooms"])
        camera.lookat[:] = [(xmin + xmax) / 2, 0, 0]
        camera.distance, camera.azimuth, camera.elevation = max(12, xmax - xmin), 90, -89
        self.renderer.update_scene(self.data, camera=camera)
        result["overhead"] = self.renderer.render().copy()
        return result

    def camera_metadata(self) -> dict:
        """Actual fixture camera extrinsics (MuJoCo right/up/back convention)."""
        result = {}
        for key, name in (("head", "zed_camera"), ("wrist", "left_camera")):
            camera = self.model.camera(name)
            cid = int(camera.id)
            result[key] = {
                "position_world": self.data.cam_xpos[cid].tolist(),
                "rotation_world_from_camera": self.data.cam_xmat[cid].reshape(3, 3).tolist(),
                "vertical_fov_degrees": float(self.model.cam_fovy[cid]),
                "resolution_hw": [360, 640],
                "camera_axes": "right_up_back",
            }
        return result

    def visible_objects(self) -> dict:
        """Oracle labels gated by actual head-camera segmentation pixels.

        This is explicit perception assistance: walls and occluders still hide
        objects, but recognizing the visible pixels does not require a detector.
        """
        import mujoco

        if self.renderer is None:
            raise ValueError("agent observations require rendering")
        self.renderer.update_scene(self.data, camera="zed_camera")
        self.renderer.enable_segmentation_rendering()
        try:
            mask = self.renderer.render().copy()
        finally:
            self.renderer.disable_segmentation_rendering()
        result = {}
        for name, position in self.positions().items():
            body_id = int(self.model.body(name).id)
            geom_ids = self.np.flatnonzero(self.model.geom_bodyid == body_id)
            pixels = int(
                self.np.count_nonzero(
                    self.np.isin(mask[:, :, 0], geom_ids) & (mask[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                )
            )
            if pixels >= 3:
                result[name] = {"position": position, "pixels": pixels}
        return result

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
