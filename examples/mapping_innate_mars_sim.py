# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Minimal DynaMem mapping on Innate Mars in sim (same stack as Stretch, different --robot).
#
# Terminal 1 (from repo root, with sim extra / assets):
#   emet serve mujoco --robot innate_mars --headless
#
# Terminal 2:
#   uv run python examples/mapping_innate_mars_sim.py
#
# Or use the full CLI (Rerun, optional LLM, etc.):
#   emet run dynamem --robot innate_mars --robot-ip 127.0.0.1 --skip --cpu-only

from emet.app.robot_cli import create_robot_client_from_cli
from emet.controller.task.dynamem import DynamemTaskExecutor
from emet.core.parameters import get_parameters


def main() -> None:
    parameters = get_parameters("dynav_config.yaml")
    robot = create_robot_client_from_cli(
        "innate_mars",
        "127.0.0.1",
        enable_rerun_server=False,
        start_immediately=True,
        allow_missing_depth=True,
    )
    try:
        executor = DynamemTaskExecutor(
            robot,
            parameters,
            skip_confirmations=True,
            cpu_only=True,
        )
        executor([("rotate_in_place", "")])
        vm = executor.agent.get_voxel_map()
        result = vm.localize_text("red cylinder", return_debug=True)
        point = result[0]
        print("localize_text('red cylinder'):", point)
        if point is None:
            raise SystemExit(1)
    finally:
        robot.stop()


if __name__ == "__main__":
    main()
