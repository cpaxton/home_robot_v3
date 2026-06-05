# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Entry point for the open-vocabulary scene graph: builds a 3D object-centric
# memory (nodes + edges) as the robot explores, using dual SigLIP/DINOv3
# embeddings and SAM3 segmentation.  Navigation and manipulation reuse the
# DynaMem task executor; the scene graph replaces the voxel-only memory for
# object queries.

import os
import sys

import click

from emet.app.robot_cli import create_robot_client_from_cli
from emet.app.run_interactive import PickPlacePromptState, run_task_executor_loop
from emet.controller.task.dynamem import DynamemTaskExecutor
from emet.core.parameters import get_parameters
from emet.llms import LLMChatWrapper, PickupPromptBuilder, get_llm_choices, get_llm_client


@click.command()
@click.option("--server_ip", "--server-ip", default="127.0.0.1", type=str)
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--explore-iter", default=3)
@click.option(
    "--use_llm",
    "--use-llm",
    is_flag=True,
    help="Set to use the language model",
)
@click.option(
    "--llm",
    default="qwen25-3B-Instruct",
    help="Client to use for language model.",
    type=click.Choice(get_llm_choices()),
)
@click.option("--debug_llm", "--debug-llm", is_flag=True, help="Debug the language model")
@click.option(
    "--use_voice",
    "--use-voice",
    is_flag=True,
    help="Set to use voice input",
)
@click.option(
    "--visual_servo",
    "--vs",
    "-V",
    "--visual-servo",
    default=False,
    is_flag=True,
    help="Use visual servoing grasp",
)
@click.option("--robot_ip", type=str, default="", help="Robot IP address (leave empty for saved default)")
@click.option(
    "--robot",
    "robot_backend",
    default="stretch",
    type=str,
    help="Robot backend (stretch, rby1, galaxea_r1, etc.). Must match emet serve mujoco --robot.",
)
@click.option("--port-offset", default=0, type=int, help="Add to default ZMQ ports (e.g. 100 → 4501-4504)")
@click.option("--target_object", type=str, default=None, help="Target object to grasp")
@click.option("--target_receptacle", "--receptacle", type=str, default=None, help="Target receptacle to place")
@click.option(
    "--skip_confirmations",
    "--skip",
    "-S",
    "-y",
    "--yes",
    is_flag=True,
    help="Skip many confirmations",
)
@click.option(
    "--input-path",
    type=click.Path(),
    default=None,
    help="Memory directory to load (common format). If not set, run rotate_in_place first.",
)
@click.option(
    "--output-path",
    type=click.Path(),
    default=None,
    help="Output directory for saving memory (open-vocab scene graph + report when used with --export or after spin)",
)
@click.option(
    "--export",
    "export_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    default=None,
    help=(
        "After spin: save open-vocab scene graph here + scene_graph_report.txt, print to stdout. "
        "Implies --no-interactive when set."
    ),
)
@click.option(
    "--no-interactive",
    is_flag=True,
    help="After initial scan/load, skip the E/L/M REPL (exit immediately if no LLM).",
)
@click.option(
    "--match-method",
    "--match_method",
    type=click.Choice(["class", "feature"]),
    default="class",
    help="Match method for visual servoing",
)
@click.option("--device_id", default=0, type=int, help="Device ID for semantic sensor")
@click.option(
    "--cpu-only",
    "--cpu",
    is_flag=True,
    help="Run everything on CPU (uses lighter models)",
)
@click.option(
    "--headless",
    is_flag=True,
    help="No auto-open browser for Rerun; open http://<this-host>:9090 manually. For native app use --rerun-native.",
)
@click.option(
    "--no-rerun",
    is_flag=True,
    help="Disable Rerun visualization entirely",
)
@click.option(
    "--rerun-native",
    is_flag=True,
    help="Use the native Rerun desktop viewer instead of the browser (needs DISPLAY).",
)
@click.option(
    "--rerun-show-panels",
    is_flag=True,
    help="Show Rerun blueprint/selection panel (useful for debugging)",
)
@click.option(
    "--rerun-debug",
    is_flag=True,
    help="Print Rerun logging status",
)
@click.option(
    "--rerun-bind",
    is_flag=True,
    help="Bind Rerun to 0.0.0.0 for remote viewing",
)
def main(
    server_ip,
    manual_wait,
    explore_iter: int = 3,
    match_method: str = "class",
    input_path: str | None = None,
    output_path: str | None = None,
    export_dir: str | None = None,
    no_interactive: bool = False,
    robot_ip: str = "",
    robot_backend: str = "stretch",
    port_offset: int = 0,
    visual_servo: bool = False,
    skip_confirmations: bool = True,
    device_id: int = 0,
    target_object: str = None,
    target_receptacle: str = None,
    use_llm: bool = False,
    use_voice: bool = False,
    debug_llm: bool = False,
    llm: str = "qwen25-3B-Instruct",
    cpu_only: bool = False,
    headless: bool = False,
    no_rerun: bool = False,
    rerun_native: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    rerun_bind: bool = False,
    **kwargs,
):
    """Run open-vocabulary scene graph exploration.

    Builds a 3D scene graph of discrete objects as the robot explores, using
    dual SigLIP + DINOv3 embeddings for open-vocabulary retrieval and
    deduplication.  Uses DynaMem's voxel map for navigation/obstacle avoidance
    underneath, but the scene graph is the primary memory for object queries.

    Examples:

      emet run scene-graph --robot-ip 127.0.0.1 -S --headless

      emet run scene-graph -S --visual-servo --cpu --headless
    """
    from emet.mapping.scene_graph.processor import SceneGraphProcessor

    print("- Load parameters")
    parameters = get_parameters("dynav_config.yaml")

    if rerun_bind:
        os.environ["RERUN_BIND_ALL"] = "1"

    if rerun_native and headless:
        raise click.UsageError("Use either --rerun-native or --headless for Rerun, not both.")

    print("- Create robot client")
    robot = create_robot_client_from_cli(
        robot_backend,
        robot_ip,
        port_offset=port_offset,
        enable_rerun_server=not no_rerun,
        rerun_headless=headless,
        rerun_native_viewer=rerun_native,
        rerun_show_panels=rerun_show_panels,
        rerun_debug=rerun_debug,
    )

    print("- Create task executor")
    executor = DynamemTaskExecutor(
        robot,
        parameters,
        visual_servo=visual_servo,
        match_method=match_method,
        device_id=device_id,
        output_path=output_path,
        server_ip=server_ip,
        skip_confirmations=skip_confirmations,
        cpu_only=cpu_only,
    )

    sg_config_name = "cpu_scene_graph" if cpu_only else "default_scene_graph"
    sg_device = "cpu" if cpu_only else None
    print(f"- Attaching scene graph processor (config={sg_config_name})")
    sg_processor = SceneGraphProcessor(config_name=sg_config_name, device=sg_device)
    executor.agent.get_voxel_map().set_scene_graph_processor(sg_processor)

    if input_path is None:
        executor([("rotate_in_place", "")])
    else:
        from emet.memory.backend import get_memory_backend

        backend = get_memory_backend(
            "scene_graph",
            scene_graph=sg_processor.scene_graph,
            text_encoder=sg_processor.text_encoder,
        )
        backend.load(input_path)
        executor._last_memory_save_path = input_path

    # Headless export: --export DIR, or --no-interactive with --output-path (same dir as executor log)
    save_dir = export_dir or (output_path if no_interactive else None)
    if save_dir:
        from emet.memory.headless_export import export_open_vocab_scene_graph_dir

        text = export_open_vocab_scene_graph_dir(sg_processor.scene_graph, save_dir)
        print(text)
        print(f"Exported open-vocab scene graph to {save_dir}")
        executor._last_memory_save_path = save_dir

    if export_dir or no_interactive:
        if no_interactive and not save_dir:
            raise click.UsageError(
                "--no-interactive requires --export DIR or --output-path (where to write the scene graph)."
            )
        from emet.memory.utils import print_memory_view_help_on_quit

        print_memory_view_help_on_quit(getattr(executor, "_last_memory_save_path", None))
        robot.stop()
        sys.exit(0)

    prompt = PickupPromptBuilder()

    llm_client = None
    pick_place = PickPlacePromptState(target_object=target_object, target_receptacle=target_receptacle)
    if use_llm:
        llm_client = get_llm_client(llm, prompt=prompt)
        chat_wrapper = LLMChatWrapper(llm_client, prompt=prompt, voice=use_voice)

    def _list_scene_objects() -> None:
        objects = sg_processor.scene_graph.list_objects()
        print(f"Scene graph objects ({len(objects)}): {objects}")

    run_task_executor_loop(
        executor,
        app_name="Scene graph",
        list_objects=_list_scene_objects,
        pick_place=pick_place,
        llm_query=(lambda: chat_wrapper.query(verbose=debug_llm)) if llm_client is not None else None,
        debug_llm=debug_llm,
        log_llm_response=(lambda resp: print("Parsed LLM Response:", resp)) if debug_llm else None,
    )

    from emet.memory.utils import print_memory_view_help_on_quit

    print_memory_view_help_on_quit(getattr(executor, "_last_memory_save_path", None))
    robot.stop()


if __name__ == "__main__":
    main()
