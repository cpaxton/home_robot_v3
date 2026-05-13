# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os

import click

from emet.app.robot_cli import create_robot_client_from_cli
from emet.controller.task.dynamem import DynamemTaskExecutor
from emet.core.parameters import get_parameters
from emet.robots import resolve_dynav_config_yaml
from emet.llms import LLMChatWrapper, PickupPromptBuilder, get_llm_choices, get_llm_client


@click.command()
# by default you are running these codes on your workstation, not on your robot.
@click.option("--server_ip", "--server-ip", default="127.0.0.1", type=str)
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--explore-iter", default=3)
@click.option("--method", default="dynamem", type=str)
@click.option("--mode", default="", type=click.Choice(["navigation", "manipulation", "save", ""]))
@click.option(
    "--use_llm",
    "--use-llm",
    is_flag=True,
    help="Set to use the language model",
)
@click.option(
    "--llm",
    default="qwen25-3B-Instruct",
    help="Client to use for language model. Recommended: gemma, openai",
    type=click.Choice(get_llm_choices()),
)
@click.option("--debug_llm", "--debug-llm", is_flag=True, help="Set to debug the language model")
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
    type=str,
    default="stretch",
    help="Robot backend (stretch, innate_mars, rby1, galaxea_r1, …). Must match emet serve mujoco --robot.",
)
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
    help="Input path with default value None",
)
@click.option(
    "--match-method",
    "--match_method",
    type=click.Choice(["class", "feature"]),
    default="class",
    help="match method for visual servoing",
)
@click.option(
    "--mllm-for-visual-grounding",
    "--mllm",
    "-M",
    is_flag=True,
    help="Use GPT4o for visual grounding",
)
@click.option("--device_id", default=0, type=int, help="Device ID for semantic sensor")
@click.option("--manipulation-only", "--manipulation", is_flag=True, help="For debugging manipulation")
@click.option(
    "--cpu-only",
    "--cpu",
    is_flag=True,
    help="Run everything on CPU",
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
    help="Print Rerun logging status (obs/servo received, step count)",
)
@click.option(
    "--rerun-bind",
    is_flag=True,
    help="Bind Rerun to 0.0.0.0 for remote viewing (Tailscale, etc.). "
    "If direct connection fails, use SSH port forwarding instead.",
)
@click.option("--port-offset", default=0, type=int, help="Add to default ZMQ ports (e.g. 100 → 4501-4504)")
@click.option(
    "--dynav-config",
    "--dynav_config",
    type=str,
    default="dynav_config.yaml",
    help="DynaMem YAML: basename under emet/config/, cwd path, or absolute path. "
    "Innate Mars: when this is the default dynav_config.yaml, emet substitutes dynav_innate_mars.yaml "
    "(DA3 stereo depth). Override with an explicit path or basename to use a different preset.",
)
def main(
    server_ip,
    manual_wait,
    explore_iter: int = 3,
    mode: str = "navigation",
    match_method: str = "class",
    input_path: str | None = None,
    output_path: str | None = None,
    robot_ip: str = "",
    robot: str = "stretch",
    visual_servo: bool = False,
    skip_confirmations: bool = True,
    device_id: int = 0,
    target_object: str = None,
    target_receptacle: str = None,
    use_llm: bool = False,
    use_voice: bool = False,
    debug_llm: bool = False,
    llm: str = "qwen25-3B-Instruct",
    manipulation_only: bool = False,
    cpu_only: bool = False,
    headless: bool = False,
    no_rerun: bool = False,
    rerun_native: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    rerun_bind: bool = False,
    port_offset: int = 0,
    dynav_config: str = "dynav_config.yaml",
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """

    print("- Load parameters")
    dynav_resolved = resolve_dynav_config_yaml(robot, dynav_config)
    if dynav_resolved != dynav_config:
        print(f"- Using {robot} DynaMem preset: {dynav_resolved}")
    parameters = get_parameters(dynav_resolved)

    if rerun_bind:
        os.environ["RERUN_BIND_ALL"] = "1"

    if rerun_native and headless:
        raise click.UsageError("Use either --rerun-native or --headless for Rerun, not both.")

    print("- Create robot client")
    depth_mode = str(parameters.get("depth_source", "sensor")).lower()
    robot_key = robot.lower().replace("-", "_")
    allow_missing_depth = depth_mode in ("da3", "auto") or robot_key == "innate_mars"
    robot_client = create_robot_client_from_cli(
        robot,
        robot_ip,
        port_offset=port_offset,
        enable_rerun_server=not no_rerun,
        rerun_headless=headless,
        rerun_native_viewer=rerun_native,
        rerun_show_panels=rerun_show_panels,
        rerun_debug=rerun_debug,
        start_immediately=False,
        allow_missing_depth=allow_missing_depth,
    )

    print("- Create task executor")
    executor = DynamemTaskExecutor(
        robot_client,
        parameters,
        visual_servo=visual_servo,
        match_method=match_method,
        device_id=device_id,
        output_path=output_path,
        server_ip=server_ip,
        skip_confirmations=skip_confirmations,
        mllm=kwargs["mllm_for_visual_grounding"],
        manipulation_only=manipulation_only,
        cpu_only=cpu_only,
    )

    if not manipulation_only:
        if input_path is None:
            executor([("rotate_in_place", "")])
        else:
            from emet.memory.backend import get_memory_backend

            backend = get_memory_backend("dynamem", voxel_map=executor.agent.get_voxel_map())
            backend.load(input_path)
            executor._last_memory_save_path = input_path  # show view help on quit

    # Create the prompt we will use to control the robot
    prompt = PickupPromptBuilder()

    # Get the LLM client
    llm_client = None
    if use_llm:
        llm_client = get_llm_client(llm, prompt=prompt)
        chat_wrapper = LLMChatWrapper(llm_client, prompt=prompt, voice=use_voice)

    # Parse things and listen to the user
    ok = True
    while ok:
        if llm_client is None:
            # Call the LLM client and parse
            explore = input(
                "Enter desired mode [E (explore and mapping) / M (Open vocabulary pick and place) / Q (quit)]: "
            ).strip()
            if explore.upper() in ("Q", "QUIT"):
                llm_response = [("quit", "")]
            elif explore.upper() == "E":
                llm_response = [("explore", None)]
            else:
                if target_object is None or len(target_object) == 0:
                    target_object = input("Enter the target object: ")
                if target_receptacle is None or len(target_receptacle) == 0:
                    target_receptacle = input("Enter the target receptacle: ")
                llm_response = [("pickup", target_object), ("place", target_receptacle)]
        else:
            # Call the LLM client and parse
            llm_response = chat_wrapper.query(verbose=debug_llm)
            if debug_llm:
                print("Parsed LLM Response:", llm_response)

        ok = executor(llm_response)
        target_object = None
        target_receptacle = None

    # At the end, disable everything
    from emet.memory.utils import print_memory_view_help_on_quit

    print_memory_view_help_on_quit(getattr(executor, "_last_memory_save_path", None))
    robot_client.stop()


if __name__ == "__main__":
    main()
