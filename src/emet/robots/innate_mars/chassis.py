"""Physical Mars support model derived from the pinned upstream base STL.

Only the physical manipulation model opts into this chassis. The public planar
navigation model retains its coordinate/actuator layout. Mass distribution uses
uniform material density over the CAD solids, normalized to the upstream 0.89 kg
base mass; it is an explicit simulator estimate, not hardware identification.
"""

import hashlib
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

SOURCE_SHA256 = "e725a419bbe85a9346a0187aff6b9c1fd3bcf9cf6cc1f1c4164777134b31f5ca"
WHEEL_RADIUS = 0.03722
WHEEL_SEPARATION = 0.1719
SUPPORT_BODIES = ("mars_left_wheel", "mars_right_wheel", "mars_caster_wheel")


def _text(value):
    return " ".join(f"{x:.10g}" for x in np.asarray(value).ravel())


def _inertial(body, mass, center, inertia):
    ET.SubElement(
        body,
        "inertial",
        mass=str(float(mass)),
        pos=_text(center),
        fullinertia=_text([inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]),
    )


@lru_cache(maxsize=1)
def source_geometry():
    """Extract three ground-reaching solids, leaving no wheel-spanning hull."""
    import trimesh

    path = Path(__file__).resolve().parents[2] / "assets/robot/innate_mars/meshes/base.STL"
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Mars chassis source mesh changed; regenerate support geometry audit")
    mesh = trimesh.load(path, force="mesh")
    parts = mesh.split(only_watertight=False)
    supports = [part for part in parts if part.bounds[0, 2] < 0.001 and part.bounds[1, 2] < 0.08]
    if len(supports) != 3:
        raise ValueError("expected two wheels and one caster in the pinned Mars mesh")
    wheels = sorted([p for p in supports if p.extents[1] < 0.03], key=lambda p: p.centroid[1], reverse=True)
    casters = [p for p in supports if p.extents[1] >= 0.03]
    if len(wheels) != 2 or len(casters) != 1:
        raise ValueError("Mars support component dimensions do not match the source audit")
    chassis = trimesh.util.concatenate([p for p in parts if all(p is not s for s in supports)])
    density = 0.89 / sum(p.volume for p in parts)
    return chassis, wheels, casters[0], density


def add_physical_chassis(root):
    """Replace planar coordinates with a free chassis, driven wheels and caster.

    Dimensions come from the unrotated URDF-frame STL. The caster's small wheel
    is approximated by a cylinder fitted to its lower arc; its swivel pivot is
    the concentric mounting ring in that same CAD component.
    """
    chassis, wheels, caster, density = source_geometry()
    world, asset = root.find("worldbody"), root.find("asset")
    base = root.find(".//body[@name='base_link']")
    for parent in root.iter("body"):
        if base in list(parent):
            parent.remove(base)
            break
    world.remove(world.find("body[@name='base_root']"))
    world.append(base)
    base.set("pos", "0 0 0.003")
    ET.SubElement(base, "freejoint", name="manipulation_base")
    base.remove(base.find("inertial"))
    _inertial(base, chassis.volume * density, chassis.center_mass, chassis.moment_inertia * density)

    hull = chassis.convex_hull
    ET.SubElement(
        asset,
        "mesh",
        name="mars_chassis_contact",
        vertex=_text(hull.vertices),
        face=" ".join(map(str, hull.faces.ravel())),
    )
    collision = base.find("geom[@name='rl_base_link']")
    collision.set("mesh", "mars_chassis_contact")
    collision.set("pos", "0 0 0")
    # The upstream visual embeds wheels and caster in one static STL. Split
    # those same solids for rendering so wheel spin and caster swivel agree
    # with the physical joints, without storing duplicate derived mesh files.
    visual = asset.find("mesh[@name='base_mesh']")
    visual.attrib.pop("file")
    visual.set("vertex", _text(chassis.vertices))
    visual.set("face", " ".join(map(str, chassis.faces.ravel())))

    for name, solid, actuator in zip(SUPPORT_BODIES[:2], wheels, ("left_wheel_vel", "right_wheel_vel"), strict=True):
        center = solid.bounds.mean(axis=0)
        ET.SubElement(
            asset,
            "mesh",
            name=name + "_visual",
            vertex=_text(solid.vertices - center),
            face=" ".join(map(str, solid.faces.ravel())),
        )
        body = ET.SubElement(base, "body", name=name, pos=_text(center))
        ET.SubElement(
            body,
            "geom",
            name=name + "_visual",
            type="mesh",
            mesh=name + "_visual",
            **{"class": "visual"},
            material="matt_black",
            mass="0",
        )
        ET.SubElement(body, "joint", name=name, type="hinge", axis="0 1 0", damping="0.0001")
        _inertial(body, solid.volume * density, solid.center_mass - center, solid.moment_inertia * density)
        ET.SubElement(
            body,
            "geom",
            name=name + "_contact",
            type="cylinder",
            size=f"{WHEEL_RADIUS} 0.01225",
            quat="0.7071067812 0.7071067812 0 0",
            friction="1 0.005 0.0001",
            condim="3",
            solref="0.01 1",
            mass="0",
            group="3",
            rgba="0.2 0.8 0.3 0.15",
        )
        ET.SubElement(
            root.find("actuator"),
            "velocity",
            name=actuator,
            joint=name,
            kv="1.0",
            forcelimited="true",
            forcerange="-0.05 0.05",
        )

    # CAD mounting ring: x=-0.1229, y=0, z=0.034. Lower rolling arc:
    # x=-0.1313974, z=0.0123452, radius=0.0129978, width~=0.0236 m.
    pivot = np.array([-0.1229, 0, 0.034])
    center = np.array([-0.1313974, -0.0001984, 0.0123452])
    ET.SubElement(
        asset,
        "mesh",
        name="mars_caster_visual",
        vertex=_text(caster.vertices - pivot),
        face=" ".join(map(str, caster.faces.ravel())),
    )
    fork = ET.SubElement(base, "body", name="mars_caster_swivel", pos=_text(pivot))
    ET.SubElement(
        fork,
        "geom",
        name="mars_caster_visual",
        type="mesh",
        mesh="mars_caster_visual",
        **{"class": "visual"},
        material="matt_black",
        mass="0",
    )
    ET.SubElement(fork, "joint", name="mars_caster_swivel", type="hinge", axis="0 0 1", damping="0.0001")
    mass = caster.volume * density
    wheel_mass = mass * 0.25  # Unidentified fork/wheel partition; preserve total CAD mass and COM.
    radius, half_width = 0.0129978, 0.0118
    wheel_inertia = np.diag(
        [
            wheel_mass * (3 * radius**2 + 4 * half_width**2) / 12,
            wheel_mass * radius**2 / 2,
            wheel_mass * (3 * radius**2 + 4 * half_width**2) / 12,
        ]
    )
    fork_mass = mass - wheel_mass
    fork_center = (mass * caster.center_mass - wheel_mass * center) / fork_mass

    def parallel(m, p):
        return m * (np.dot(p, p) * np.eye(3) - np.outer(p, p))

    fork_inertia = (
        caster.moment_inertia * density
        - wheel_inertia
        - parallel(wheel_mass, center - caster.center_mass)
        - parallel(fork_mass, fork_center - caster.center_mass)
    )
    if np.linalg.eigvalsh(fork_inertia).min() <= 0:
        raise ValueError("invalid CAD caster inertia partition")
    _inertial(fork, fork_mass, fork_center - pivot, fork_inertia)
    wheel = ET.SubElement(fork, "body", name=SUPPORT_BODIES[2], pos=_text(center - pivot))
    ET.SubElement(wheel, "joint", name=SUPPORT_BODIES[2], type="hinge", axis="0 1 0", damping="0.00001")
    _inertial(wheel, wheel_mass, np.zeros(3), wheel_inertia)
    ET.SubElement(
        wheel,
        "geom",
        name="mars_caster_contact",
        type="cylinder",
        size=f"{radius} {half_width}",
        quat="0.7071067812 0.7071067812 0 0",
        friction="1 0.005 0.0001",
        condim="3",
        solref="0.01 1",
        mass="0",
        group="3",
        rgba="0.2 0.8 0.3 0.15",
    )
    # Chassis/fork contact is internal to the assembly. Wheels contact the world.
    for a, b in (("base_link", "mars_caster_wheel"),):
        ET.SubElement(root.find("contact"), "exclude", body1=a, body2=b)
    for item in list(root.find("actuator")):
        if item.get("joint") in {"base_x", "base_y", "base_yaw"}:
            root.find("actuator").remove(item)
    for item in list(root.findall("keyframe")):
        root.remove(item)
    return root


def audit():
    """Report reproducible geometry/mass evidence without starting a robot client."""
    import json

    report = json.loads(
        (Path(__file__).resolve().parents[2] / "assets/robot/innate_mars/reference/chassis_support.json").read_text()
    )
    shell, wheels, caster, density = source_geometry()
    report["components"] = [
        {
            "name": name,
            "bounds_m": solid.bounds.tolist(),
            "mass_kg": float(solid.volume * density),
            "center_of_mass_m": solid.center_mass.tolist(),
        }
        for name, solid in zip(
            ("chassis", *SUPPORT_BODIES[:2], "caster_assembly"), (shell, *wheels, caster), strict=True
        )
    ]
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(audit(), indent=2))
