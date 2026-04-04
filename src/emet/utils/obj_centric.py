# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from dataclasses import dataclass

from torch import Tensor


@dataclass
class ObjectCentricObservations:
    low_level_output_messages: list[str] = None
    scene_images: list = None
    object_images: list = None
    scene_graph: list = None


@dataclass
class ObjectImage:
    image: Tensor = None
    position: list[float] = None
    crop_id: int = None
    object_class: str = None
    instance_id: int = None
