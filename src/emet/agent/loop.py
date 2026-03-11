# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Agent main loop: start robot with logging, load memory, optional Discord, dispatch to tools.

from __future__ import annotations

import json
import os
import threading
import timeit
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Maximum follow-up LLM calls per user turn (prevents infinite loops)
_MAX_TOOL_ROUNDS = 3


# ---------------------------------------------------------------------------
# Chat log
# ---------------------------------------------------------------------------

class ChatLog:
    """Append-only JSONL log of the conversation for debugging and training."""

    def __init__(self, log_dir: Optional[str] = None):
        if log_dir is None:
            log_dir = os.path.join("logs", "chat")
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = os.path.join(log_dir, f"chat_{stamp}.jsonl")
        self._fh = open(self.path, "a")
        logger.info("Chat log:", self.path)

    def log(self, role: str, content: str, **extra: Any) -> None:
        record = {"ts": datetime.now().isoformat(), "role": role, "content": content}
        record.update(extra)
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _dispatch_tool_calls(
    tool_calls: List[dict],
    tools_by_name: dict[str, Tool],
    executor: DynamemTaskExecutor,
    chat_log: Optional[ChatLog] = None,
    debug: bool = False,
) -> Tuple[bool, List[str], bool]:
    """Execute a list of parsed tool_calls.

    Returns (continue_running, list_of_result_strings, has_info_results).
    continue_running is False if quit was requested.
    has_info_results is True if any tool with returns_info=True produced output.
    """
    executor_cmds: List[Tuple[str, str]] = []
    results: List[str] = []
    has_info = False

    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments") or {}
        tool = tools_by_name.get(name)
        if tool is None:
            msg = f"Unknown tool: {name}"
            logger.warning(msg)
            results.append(msg)
            continue

        cmds = tool.to_executor(args)
        if cmds:
            executor_cmds.extend(cmds)
        else:
            try:
                result = tool.func(**args) if args else tool.func()
                result_str = str(result) if result is not None else "ok"
                results.append(f"[{name}] {result_str}")
                if tool.returns_info and result is not None and result != "":
                    has_info = True
                if result is not None and result != "":
                    print(colored(f"[{name}]", "cyan"), result_str)
            except Exception as e:
                err = f"Tool {name} failed: {e}"
                logger.warning(err)
                print(colored(err, "red"))
                results.append(err)

    if not executor_cmds:
        if chat_log:
            for r in results:
                chat_log.log("tool", r)
        return True, results, has_info

    if any(c[0] == "quit" for c in executor_cmds):
        return False, results, has_info

    ok = executor(executor_cmds)
    cmd_names = [c[0] for c in executor_cmds]
    results.append(f"Executor ran: {', '.join(cmd_names)} -> {'ok' if ok else 'failed/interrupted'}")

    if chat_log:
        for r in results:
            chat_log.log("tool", r)

    return ok, results, has_info


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _call_llm(
    llm_client: Any,
    text: str,
    openai_tools_param: Optional[list],
    debug: bool,
) -> Tuple[str, float]:
    """Call the LLM and return (raw_response, elapsed_seconds)."""
    t0 = timeit.default_timer()
    try:
        if openai_tools_param is not None:
            raw = llm_client(text, verbose=debug, tools=openai_tools_param)
        else:
            raw = llm_client(text, verbose=debug)
    except TypeError:
        raw = llm_client(text)
    return raw, timeit.default_timer() - t0


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

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
            llm=None,
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

    # Chat log
    chat_log = ChatLog()
    print(colored(f"Chat log: {chat_log.path}", "yellow"))

    # Build prompt and LLM client
    llm_client = None
    prompt_builder = None
    openai_tools_param = None
    if use_llm:
        try:
            prompt_builder = AgentPromptBuilder(tools=tools, name=agent_name, context=context)
            llm_client = get_llm_client(llm, prompt=prompt_builder)
            from emet.llms.openai_client import OpenaiClient
            if isinstance(llm_client, OpenaiClient):
                openai_tools_param = [t.schema() for t in tools]
        except Exception as e:
            logger.warning("LLM failed to load (%s):", llm, e)
            print(colored("Agent mode requires an LLM; it failed to load.", "red"))
            print(colored("Fix the LLM (e.g. --llm, device) or run with --no-llm for letter commands only.", "yellow"))
            robot.stop()
            chat_log.close()
            return

    if use_llm and llm_client is not None:
        print(colored(f"LLM enabled ({llm}). Say what you want the robot to do.", "green"))
    else:
        print(colored("Enter mode [E=explore / M=pick+place / Q=question / P=send picture / QUIT]:", "green"))

    # Print system prompt once at startup when debug is on
    if debug_llm and prompt_builder is not None:
        print(colored("=" * 60, "yellow"))
        print(colored("[DEBUG] System prompt (printed once):", "yellow"))
        print(str(prompt_builder))
        print(colored("=" * 60, "yellow"))

    chat_log.log("system", str(prompt_builder) if prompt_builder else "(no LLM)")

    def _send_to_discord(text: str) -> None:
        if discord_bot is not None and hasattr(discord_bot, "push_task_to_all_channels"):
            discord_bot.push_task_to_all_channels(message=f"**{agent_name}:** {text}")

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

            chat_log.log("user", user_text)
            if debug_llm:
                print(colored(f"[DEBUG] User: {user_text!r}", "yellow"))

            # --- Multi-turn tool-use loop ---
            # The LLM may call tools that return information (e.g. query_memory).
            # When that happens we feed the results back and let the LLM summarize.
            current_input = user_text
            for _round in range(_MAX_TOOL_ROUNDS):
                raw_response, elapsed = _call_llm(
                    llm_client, current_input, openai_tools_param, debug_llm,
                )

                if debug_llm:
                    print(colored(f"[DEBUG] Raw response ({elapsed:.2f}s):", "yellow"), raw_response[:500])

                parsed = parse_tool_calls_response(raw_response)
                tool_calls = parsed.get("tool_calls") or []
                message = parsed.get("message") or ""

                if debug_llm:
                    print(colored("[DEBUG] Parsed:", "blue"), json.dumps(parsed, indent=2))

                chat_log.log("assistant", message, tool_calls=tool_calls, raw=raw_response, time_s=elapsed)

                # No tool calls — this is the final answer
                if not tool_calls:
                    if message:
                        print(colored(f"{agent_name}:", "blue"), message)
                        _send_to_discord(message)
                    break

                # Print the intermediate message (e.g. "Let me check my memory.")
                if message:
                    print(colored(f"{agent_name}:", "blue"), message)

                # Execute tool calls
                ok, results, has_info = _dispatch_tool_calls(
                    tool_calls, tools_by_name, executor, chat_log=chat_log, debug=debug_llm,
                )
                if not ok:
                    break

                result_text = "\n".join(results)

                if has_info:
                    # Feed tool results back to LLM for summarization
                    followup = f"[Tool results]\n{result_text}\n\nSummarize these results for the user in your message. Do not call any more tools."
                    if hasattr(llm_client, "add_history"):
                        llm_client.add_history({"role": "assistant", "content": raw_response})
                        llm_client.add_history({"role": "user", "content": followup})
                    current_input = followup
                    if debug_llm:
                        print(colored("[DEBUG] Info tools returned results; requesting LLM follow-up", "yellow"))
                    continue
                else:
                    # Action-only tools: send the initial message and we're done
                    if message:
                        _send_to_discord(message)
                    if results and hasattr(llm_client, "add_history"):
                        llm_client.add_history({"role": "assistant", "content": raw_response})
                        llm_client.add_history({"role": "user", "content": f"[Tool results]\n{result_text}"})
                    break
            continue

        # --- Manual (no-LLM) path ---
        line = input(colored("You: ", "green")).strip()
        if not line:
            continue
        chat_log.log("user", line)
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

    chat_log.log("system", "session ended")
    chat_log.close()
    print(colored(f"Chat log saved: {chat_log.path}", "green"))
    print_memory_view_help_on_quit(getattr(executor, "_last_memory_save_path", None))
    robot.stop()
