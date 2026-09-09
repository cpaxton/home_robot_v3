# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility shim: DynaMem controller lives in ``emet.controller.dynamem``."""

from emet.controller.dynamem import (  # noqa: F401
    _DESCRIBE_SCENE_OWL_QUERIES,
    _DESCRIBE_SCENE_YOLOE_LABELS,
    DEFAULT_TABLE_MAPPING_YAW_HALF_RAD,
    DYNAMEM_HEAD_SETTLE_S,
    DYNAMEM_HEAD_SWEEP_FRAME_SETTLE_S,
    DYNAMEM_HEAD_SWEEP_MAX_WAIT_S,
    DYNAMEM_HEAD_SWEEP_MIN_MOVE_S,
    DYNAMEM_HEAD_SWEEP_PAN_TOL_RAD,
    DYNAMEM_HEAD_SWEEP_POS_DELTA_TOL,
    DYNAMEM_HEAD_SWEEP_SPEED_TOL,
    DYNAMEM_HEAD_SWEEP_STOPPED_HOLD_S,
    INIT_ARM_POS,
    INIT_HEAD_PAN,
    INIT_HEAD_TILT,
    INIT_LIFT_POS,
    INIT_WRIST_PITCH,
    INIT_WRIST_ROLL,
    INIT_WRIST_YAW,
    DynamemController,
    RobotAgent,
    default_table_mapping_relative_yaws,
)

__all__ = [
    "DEFAULT_TABLE_MAPPING_YAW_HALF_RAD",
    "DYNAMEM_HEAD_SETTLE_S",
    "DYNAMEM_HEAD_SWEEP_FRAME_SETTLE_S",
    "DYNAMEM_HEAD_SWEEP_MAX_WAIT_S",
    "DYNAMEM_HEAD_SWEEP_MIN_MOVE_S",
    "DYNAMEM_HEAD_SWEEP_PAN_TOL_RAD",
    "DYNAMEM_HEAD_SWEEP_POS_DELTA_TOL",
    "DYNAMEM_HEAD_SWEEP_SPEED_TOL",
    "DYNAMEM_HEAD_SWEEP_STOPPED_HOLD_S",
    "INIT_ARM_POS",
    "INIT_HEAD_PAN",
    "INIT_HEAD_TILT",
    "INIT_LIFT_POS",
    "INIT_WRIST_PITCH",
    "INIT_WRIST_ROLL",
    "INIT_WRIST_YAW",
    "DynamemController",
    "RobotAgent",
    "_DESCRIBE_SCENE_OWL_QUERIES",
    "_DESCRIBE_SCENE_YOLOE_LABELS",
    "default_table_mapping_relative_yaws",
]
