# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Agent main loop: start robot with logging, load memory, optional Discord, dispatch to tools.

from __future__ import annotations

import json
import os
import threading
import timeit
from typing import Any, Dict, Optional

import click
from termcolor import colored

from emet.agent.prompt import AgentPromptBuilder, parse_tool_calls_response
from emet.agent.tools import Tool, get_tools
from emet.core import get_parameters
from emet.controller.task.dynamem import DynamemTaskExecutor
from emet.controller.zmq_client import HomeRobotZmqClient
from emet.llms import get_llm_client
from emet.llms.discord_bot import EmetDiscordBot
from emet.memory.backend import get_memory_backend
from emet.memory.utils import print_memory_view_help_on_quit
from emet.utils.logger import Logger

logger = Logger(__name__)


def _dispatch_tool_calls(
    tool_calls: list[dict],
    tools_by_name: dict[str, Tool],
    executor: DynamemTaskExecutor,
    debug: bool = False,
) -> bool:
    """Execute a list of parsed tool_calls. Returns False if quit was requested."""
    executor_cmds: list[tuple[str, str]] = []

    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments") or {}
        tool = tools_by_name.get(name)
        if tool is None:
            logger.warning("Unknown tool: %s", name)
            continue

        cmds = tool.to_executor(args)
        if cmds:
            executor_cmds.extend(cmds)
        else:
            # No executor mapping: call the tool func directly (e.g. query_memory, send_image)
            try:
                result = tool.func(**args) if args else tool.func()
                if result is not None and result != "":
                    print(colored(f"[{name}]", "cyan"), result)
            except Exception as e:
                logger.warning("Tool %s failed: %s", name, e)
                print(colored(f"Tool {name} failed: {e}", "red"))

    if not executor_cmds:
        return True

    if any(c[0] == "quit" for c in executor_cmds):
        return False

    return executor(executor_cmds)


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
    agent_name: str = "Emet",
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
    context: Dict[str, Any] = {
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

    # Build tools from context
    tools = get_tools(context)
    tools_by_name = {t.name: t for t in tools}
    print(colored("Agent tools: " + ", ".join(t.name for t in tools), "yellow"))

    # Build prompt and LLM client
    llm_client = None
    prompt_builder = None
    openai_tools_param = None  # native tool schemas for OpenAI API
    if use_llm:
        try:
            prompt_builder = AgentPromptBuilder(tools=tools, name=agent_name, context=context)
            llm_client = get_llm_client(llm, prompt=prompt_builder)
            # For OpenAI clients, prepare native tools param
            from emet.llms.openai_client import OpenaiClient
            if isinstance(llm_client, OpenaiClient):
                openai_tools_param = [t.schema() for t in tools]
        except Exception as e:
            logger.warning("LLM failed to load (%s): %s", llm, e)
            print(colored("Agent mode requires an LLM; it failed to load.", "red"))
            print(colored("Fix the LLM (e.g. --llm, device) or run with --no-llm for letter commands only.", "yellow"))
            robot.stop()
            return

    if use_llm and llm_client is not None:
        print(colored(f"LLM enabled ({llm}). Say what you want the robot to do.", "green"))
    else:
        print(colored("Enter mode [E=explore / M=pick+place / Q=question / P=send picture / QUIT]:", "green"))
    if debug_llm:
        print(colored("Debug: full prompt, raw and parsed LLM response will be printed.", "yellow"))

    ok = True
    while ok:
        update_xyt()

        # --- LLM path ---
        if use_llm and llm_client is not None:
            print("-" * 60)
            user_text = input(colored("You: ", "green")).strip()
            if not user_text:
                continue
            if user_text.upper() in ("Q", "QUIT"):
                ok = False
                break

            if debug_llm:
                print(colored("[DEBUG] System prompt:", "yellow"))
                sp = str(prompt_builder) if prompt_builder else ""
                print(sp[:2000] + ("..." if len(sp) > 2000 else ""))
                print(colored("[DEBUG] User input:", "yellow"), repr(user_text))

            t0 = timeit.default_timer()
            try:
                if openai_tools_param is not None:
                    raw_response = llm_client(user_text, verbose=debug_llm, tools=openai_tools_param)
                else:
                    raw_response = llm_client(user_text, verbose=debug_llm)
            except TypeError:
                raw_response = llm_client(user_text)
            t1 = timeit.default_timer()

            if debug_llm:
                print(colored("[DEBUG] Raw LLM response:", "yellow"), repr(raw_response))

            parsed = parse_tool_calls_response(raw_response)
            tool_calls = parsed.get("tool_calls") or []
            message = parsed.get("message") or ""

            if debug_llm:
                print(colored("[DEBUG] Parsed:", "blue"), json.dumps(parsed, indent=2))
                print(colored(f"[DEBUG] Time: {t1 - t0:.2f}s", "yellow"))

            if message:
                print(colored(f"{agent_name}:", "blue"), message)

            if tool_calls:
                ok = _dispatch_tool_calls(tool_calls, tools_by_name, executor, debug=debug_llm)
            continue

        # --- Manual (no-LLM) path ---
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
            tool = tools_by_name.get("send_image")
            if tool:
                print(tool.func())
            else:
                print("send_image tool not available.")
            continue
        if line.upper().startswith("Q "):
            question = line[2:].strip()
            tool = tools_by_name.get("query_memory")
            if tool:
                answer = tool.func(question=question)
                print(colored("Answer:", "blue"), answer)
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

        if use_llm:
            print(colored("LLM did not load; use letter commands: E / M / Q / P / FIND.", "yellow"))
            continue
        ok = executor([("pickup", line), ("place", "")]) if line else True

    print_memory_view_help_on_quit(getattr(executor, "_last_memory_save_path", None))
    robot.stop()
