# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Native convex geometry distance against small local occupancy cells."""

import threading
from functools import lru_cache

import mujoco
import numpy as np

_thread_data = threading.local()


@lru_cache(maxsize=128)
def _query_model(kind, dimensions, vertices, resolution):
    asset = ''
    if kind == 'convex':
        points = np.frombuffer(vertices, dtype=np.float64).reshape(-1, 3)
        values = ' '.join(format(float(value), '.17g') for value in points.ravel())
        asset = f'<asset><mesh name="surface" vertex="{values}"/></asset>'
        geometry = 'type="mesh" mesh="surface"'
    else:
        values = ' '.join(format(value, '.17g') for value in dimensions)
        geometry = f'type="{kind}" size="{values}"'
    half = resolution / 2
    xml = f'''<mujoco><compiler usethread="false"/>{asset}<worldbody>
      <body name="shape"><geom {geometry}/></body>
      <body name="cell"><geom type="box" size="{half} {half} {half}"/></body>
      </worldbody></mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


def cell_separations(solid, centers, *, resolution):
    vertices = solid.vertices.astype(np.float64).tobytes() if solid.kind == 'convex' else b''
    model = _query_model(solid.kind, solid.dimensions, vertices, resolution)
    cache = getattr(_thread_data, 'cache', None)
    if cache is None:
        cache = _thread_data.cache = {}
    if id(model) not in cache:
        if len(cache) >= 128:
            cache.clear()
        cache[id(model)] = (model, mujoco.MjData(model))
    data = cache[id(model)][1]
    # Transform mesh offsets compiled by MuJoCo as well as the requested pose.
    model_pose = np.eye(4)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[0])
    model_pose[:3, :3] = rotation.reshape(3, 3)
    model_pose[:3, 3] = model.geom_pos[0]
    pose = solid.world_from_local @ model_pose
    data.geom_xpos[0] = pose[:3, 3]
    data.geom_xmat[0] = pose[:3, :3].ravel()
    data.geom_xmat[1] = np.eye(3).ravel()
    distances = np.empty(len(centers))
    for index, center in enumerate(centers):
        data.geom_xpos[1] = center
        distances[index] = mujoco.mj_geomDistance(model, data, 0, 1, 10., None)
        if distances[index] <= 0:
            # Some convex containment cases return zero. Retain a conservative
            # signed bound once the native kernel has confirmed intersection.
            bound = solid.lower_distance(np.asarray(center)[None])[0] + solid.offset_m
            distances[index] = min(distances[index], bound - np.sqrt(3) * resolution / 2)
    return distances
