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


from pathlib import Path

import click
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from emet.utils.logger import Logger

logger = Logger(__name__)


@click.command()
@click.option(
    "--input-path",
    "-i",
    type=click.Path(),
    default="saved_memory",
    help="Memory directory (common format with manifest.json). Default: saved_memory.",
)
@click.option(
    "--config-path",
    "-c",
    type=click.Path(),
    default="default_planner.yaml",
    help="Path to planner config (reserved for future use).",
)
def main(input_path, config_path):
    """Load a memory directory and display it in Rerun (common format only)."""
    import sys

    input_path = Path(input_path)
    print("Loading:", input_path)

    from emet.memory.format import is_memory_directory, load_memory

    if not input_path.is_dir() or not is_memory_directory(str(input_path)):
        logger.error(
            "Only memory directory format is supported (directory with manifest.json). "
            "Use e.g. saved_memory from 'emet run create-and-print-memory' or backend.save(path)."
        )
        sys.exit(1)

    state = load_memory(str(input_path))
    from emet.visualization.rerun import RerunVisualizer

    viz = RerunVisualizer(display_robot_mesh=False, spawn_gui=True, open_browser=False)
    viz.log_memory_state(state)
    if state.obstacles_2d is not None and state.explored_2d is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(state.explored_2d)
        ax1.set_title("Explored")
        ax1.axis("off")
        ax2.imshow(state.obstacles_2d)
        ax2.set_title("Obstacles")
        ax2.axis("off")
        plt.show()
    logger.info("Memory state logged to Rerun. Rerun server continues in background.")


if __name__ == "__main__":
    """run the test script."""
    main()
