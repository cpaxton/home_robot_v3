"""Lossless storage conversion for generated untextured inline MuJoCo meshes."""

import xml.etree.ElementTree as ET

import numpy as np


def binary_mesh_assets(xml):
    """Move vertex/face arrays into native binary MSH VFS assets.

    MuJoCo stores mesh vertices as float32. Parse the original decimal values
    into that same representation, retaining face order, names and transforms.
    This avoids the large numeric XML attribute reader which fails on valid
    generated Mars meshes during cold viewer-worker startup with MuJoCo 3.5.
    The caller retains the original XML for canonical source identity; compiled
    model equivalence is tested independently. No files or geometry are edited.
    """
    root, assets = ET.fromstring(xml), {}
    for index, mesh in enumerate(root.findall(".//mesh")):
        vertex = mesh.get("vertex")
        if vertex is None:
            continue
        if mesh.get("face") is None or mesh.get("normal") is not None or mesh.get("texcoord") is not None:
            raise ValueError("binary mesh storage requires explicit faces and generated normals/texcoords")
        try:
            vertices = np.asarray(vertex.split(), dtype="<f4").reshape(-1, 3)
            faces = np.asarray([int(v) for v in mesh.get("face").split()], dtype="<i4").reshape(-1, 3)
        except (ValueError, OverflowError) as exc:
            raise ValueError("invalid generated mesh arrays") from exc
        if (
            not len(vertices) or not len(faces) or not np.isfinite(vertices).all()
            or np.any(faces < 0) or np.any(faces >= len(vertices))
        ):
            raise ValueError("invalid generated mesh vertices or face indices")
        filename = f"emet_generated_mesh_{index}.msh"
        assets[filename] = (
            np.array([len(vertices), 0, 0, len(faces)], dtype="<i4").tobytes()
            + vertices.tobytes() + faces.tobytes()
        )
        del mesh.attrib["vertex"], mesh.attrib["face"]
        mesh.set("file", filename)
    return ET.tostring(root, encoding="unicode"), assets
