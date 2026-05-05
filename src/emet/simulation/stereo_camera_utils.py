# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Helpers for stereo camera naming (no MuJoCo import — safe for lightweight tests)."""


def stereo_right_camera_name_from_spec(camera_names: list[str]) -> str | None:
    """Second camera id for stereo DA3 when the spec lists a clear left/right head pair."""
    if len(camera_names) < 2:
        return None
    left_n, right_n = camera_names[0].lower(), camera_names[1].lower()
    if ("left" in left_n or left_n.endswith("_l")) and ("right" in right_n or right_n.endswith("_r")):
        return camera_names[1]
    return None
