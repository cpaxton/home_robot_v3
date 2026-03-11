# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Agent main loop: start robot with logging, load memory, optional Discord, dispatch to tools.

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

import click
from termcolor import colored

from emet.agent.tools import get_tools
from emet.core import get_parameters
from emet.controller.task.dynamem import DynamemTaskExecutor
from emet.controller.zmq_client import HomeRobotZmqClient
from emet.llms import LLMChatWrapper, PickupPromptBuilder, get_llm_client
from emet.llms.discord_bot import EmetDiscordBot
from emet.memory.backend import get_memory_backend
from emet.memory.utils import print_memory_view_help_on_quit
from emet.utils.logger import Logger

logger = Logger(__name__)


def run_agent_with_robot(
    robot_ip: str = "127.0.0.1",
    input_path: Optional[str] = None,
    discord: bool = False,
    use_llm: bool = False,
    llm: str = "qwen35-4B",
    server_ip: str = "127.0.0.1",
    skip_confirmations: bool = True,
    explore_iter: int = 3,
    debug_llm: bool = False,
    **kwargs: Any,
) -> None:
    """Start robot, optional memory load, optional Discord; run command loop with tools."""
    parameters = get_parameters("dynav_config.yaml")

    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        enable_rerun_server=True,
    )

    executor = DynamemTaskExecutor(
        robot,
        parameters,
        server_ip=server_ip,
        skip_confirmations=skip_confirmations,
        explore_iter=explore_iter,
        discord_bot=None,
        **kwargs,
    )

    if input_path:
        backend = get_memory_backend("dynamem", voxel_map=executor.agent.get_voxel_map())
        backend.load(input_path)
        executor._last_memory_save_path = input_path

    memory_backend = get_memory_backend("dynamem", voxel_map=executor.agent.get_voxel_map())
    context = {
        "executor": executor,
        "robot": robot,
        "memory_backend": memory_backend,
        "discord_bot": None,
        "xyt_for_query": None,
        "planner": getattr(executor.agent, "planner", None),
    }

    def update_xyt():
        if executor.agent.robot is not None and hasattr(executor.agent.robot, "get_base_pose"):
            context["xyt_for_query"] = executor.agent.robot.get_base_pose()

    discord_bot = None
    if discord and os.environ.get("DISCORD_TOKEN"):
        class AgentPlaceholder:
            def __init__(self, exec_obj):
                self.robot = exec_obj.robot
                self.parameters = exec_obj.agent.parameters

        discord_bot = EmetDiscordBot(
            AgentPlaceholder(executor),
            task="dynamem",
            executor=executor,
            skip_confirmations=skip_confirmations,
            output_path=getattr(executor.agent, "log", "."),
            kwargs=kwargs.get("discord_kwargs", {"match_method": "feature", "mllm_for_visual_grounding": False}),
        )
        context["discord_bot"] = discord_bot
        executor.discord_bot = discord_bot
        executor.agent.discord_bot = discord_bot
        bot_thread = threading.Thread(target=discord_bot.run, daemon=True)
        bot_thread.start()
        print(colored("Discord bot started (DISCORD_TOKEN). Messages will be handled.", "green"))

    if not getattr(executor, "manipulation_only", True):
        if input_path is None:
            executor([("rotate_in_place", "")])
        # else memory already loaded above

    llm_client = None
    chat_wrapper = None
    if use_llm:
        try:
            prompt = PickupPromptBuilder()
            llm_client = get_llm_client(llm, prompt=prompt)
            chat_wrapper = LLMChatWrapper(llm_client, prompt=prompt)
        except Exception as e:
            logger.warning("LLM failed to load (%s): %s", llm, e)
            print(colored("Agent mode requires an LLM; it failed to load.", "red"))
            print(colored("Fix the LLM (e.g. --llm, device) or run with --no-llm for letter commands only.", "yellow"))
            robot.stop()
            return
    tools = get_tools(context)
    print(colored("Agent tools: " + ", ".join(t["name"] for t in tools), "yellow"))
    if use_llm and chat_wrapper is not None:
        print(colored(f"LLM enabled ({llm}). Say what you want the robot to do.", "green"))
    else:
        print(colored("Enter mode [E=explore / M=pick+place / Q=question / P=send picture / Q quit]:", "green"))
    if debug_llm:
        print(colored("Debug: full prompt, raw and parsed LLM response will be printed.", "yellow"))

    ok = True
    while ok:
        update_xyt()
        if use_llm and chat_wrapper is not None:
            llm_response = chat_wrapper.query(verbose=debug_llm)
            if llm_response is None:
                continue
            if isinstance(llm_response, list) and len(llm_response) == 1 and llm_response[0][0] == "quit":
                ok = False
                break
            ok = executor(llm_response)
            continue

        line = input(colored("You: ", "green")).strip()
        if not line:
            continue
        if line.upper() in ("Q", "QUIT"):
            ok = False
            break
        if line.upper() == "E":
            ok = executor([("explore", None)])
            continue
        if line.upper() == "P":
            send_image = next((t["func"] for t in tools if t["name"] == "send_image"), None)
            if send_image:
                print(send_image())
            else:
                print("send_image tool not available.")
            continue
        if line.upper().startswith("Q "):
            question = line[2:].strip()
            query_memory = next((t["func"] for t in tools if t["name"] == "query_memory"), None)
            if query_memory:
                answer, imgs = query_memory(question)
                print(colored("Answer:", "blue"), answer)
                if imgs and context.get("discord_bot"):
                    context["discord_bot"].push_task_to_all_channels(message=answer, content=imgs[0] if hasattr(imgs[0], "__array__") else None)
            else:
                print("query_memory not available.")
            continue
        if line.upper() == "M" or line.upper().startswith("M "):
            parts = line[2:].strip().split() if len(line) > 1 else []
            obj = input("Object to pick: ").strip() if len(parts) < 1 else parts[0]
            rec = input("Receptacle: ").strip() if len(parts) < 2 else (parts[1] if len(parts) >= 2 else "")
            ok = executor([("pickup", obj), ("place", rec)])
            continue
        if line.upper().startswith("FIND ") or line.upper().startswith("F "):
            text = line[5:].strip() if line.upper().startswith("FIND ") else line[2:].strip()
            ok = executor([("find", text)])
            continue

        # When LLM is enabled, all natural language goes through the LLM path above; we should
        # not reach here for free-form input. If we do (e.g. chat_wrapper failed to load), don't
        # treat unknown input as pickup.
        if use_llm:
            if chat_wrapper is None:
                print(colored("LLM did not load; use letter commands only: E / M / Q / P / FIND.", "yellow"))
            else:
                print(colored("Unexpected input. Use E, M, Q, P, or natural language.", "yellow"))
            continue
        ok = executor([("pickup", line), ("place", "")]) if line else True

    print_memory_view_help_on_quit(getattr(executor, "_last_memory_save_path", None))
    robot.stop()
