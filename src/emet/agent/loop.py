# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Agent main loop: start robot with logging, load memory, optional Discord, dispatch to tools.

from __future__ import annotations

import importlib
import json
import logging
import os
import queue
import sys
import threading
import timeit
from collections.abc import Callable
from datetime import datetime
from typing import Any

import click
import numpy as np
from termcolor import colored

from emet.agent.env_flags import env_agent_camera_debug
from emet.agent.model_debug import print_embodied_model_report, print_llm_invoke_line
from emet.agent.prompt import DEFAULT_AGENT_NAME, AgentPromptBuilder, parse_tool_calls_response
from emet.agent.tools import Tool, get_tools
from emet.config.embodied_agent_config import load_embodied_agent_overlay
from emet.controller.task.dynamem import DynamemTaskExecutor
from emet.controller.zmq_client import StretchZmqClient
from emet.core import get_parameters
from emet.llms import get_llm_client
from emet.llms.base import AbstractVLLMClient
from emet.memory.backend import get_memory_backend
from emet.memory.utils import print_memory_view_help_on_quit
from emet.robots import ROBOT_REGISTRY
from emet.utils.logger import Logger
from emet.utils.vram_debug import print_vram_snapshot

logger = Logger(__name__)

# Maximum follow-up LLM calls per user turn (prevents infinite loops)
_MAX_TOOL_ROUNDS = 3


def _env_agent_tool_debug() -> bool:
    """True when ``EMET_AGENT_TOOL_DEBUG`` requests verbose tool I/O on the terminal."""
    v = os.environ.get("EMET_AGENT_TOOL_DEBUG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _configure_agent_terminal_output() -> None:
    """Reduce noisy tqdm / HF / Discord lines interleaved with the agent TTY."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in ("discord", "discord.gateway", "discord.http", "discord.client"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        from transformers.utils import logging as tf_logging

        tf_logging.set_verbosity_error()
    except Exception:
        pass


def parse_manual_find_command(raw: str) -> str | None:
    """Parse no-LLM find syntax: FIND x, F x, or find x (case-insensitive verb)."""
    s = raw.strip()
    u = s.upper()
    if u.startswith("FIND "):
        return s[5:].strip()
    if u.startswith("F ") and len(s) > 2:
        return s[2:].strip()
    if s.lower().startswith("find "):
        return s[5:].strip()
    return None


# ---------------------------------------------------------------------------
# Chat log
# ---------------------------------------------------------------------------


class ChatLog:
    """Append-only JSONL log of the conversation for debugging and training."""

    def __init__(self, log_dir: str | None = None):
        if log_dir is None:
            log_dir = os.path.join("logs", "chat")
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = os.path.join(log_dir, f"chat_{stamp}.jsonl")
        self._fh = open(self.path, "a")
        logger.debug("Chat log: %s", self.path)

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
    tool_calls: list[dict],
    tools_by_name: dict[str, Tool],
    executor: DynamemTaskExecutor,
    chat_log: ChatLog | None = None,
    debug: bool = False,
    verbose_tools: bool = False,
) -> tuple[bool, list[str], bool]:
    """Execute a list of parsed tool_calls.

    Returns (continue_running, list_of_result_strings, has_info_results).
    continue_running is False if quit was requested.
    has_info_results is True if any tool with returns_info=True produced output.
    """
    executor_cmds: list[tuple[str, str]] = []
    results: list[str] = []
    has_info = False

    if verbose_tools and tool_calls:
        print(colored("[tool_calls raw]", "yellow"), flush=True)
        for i, tc in enumerate(tool_calls):
            print(colored(f"  [{i}]", "yellow"), json.dumps(tc, default=str), flush=True)

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
                    if verbose_tools:
                        print(colored(f"[{name}]", "magenta"), result_str, flush=True)
                    else:
                        print(colored(f"[{name}]", "cyan"), result_str)
            except Exception as e:
                err = f"Tool {name} failed: {e}"
                logger.warning(err)
                print(colored(err, "red"))
                if verbose_tools:
                    import traceback

                    traceback.print_exc()
                results.append(err)

    if not executor_cmds:
        if chat_log:
            for r in results:
                chat_log.log("tool", r)
        return True, results, has_info

    if any(c[0] == "quit" for c in executor_cmds):
        return False, results, has_info

    if verbose_tools:
        print(colored("[executor]", "yellow"), json.dumps(executor_cmds, default=str), flush=True)
    ok = executor(executor_cmds)
    cmd_names = [c[0] for c in executor_cmds]
    results.append(f"Executor ran: {', '.join(cmd_names)} -> {'ok' if ok else 'failed/interrupted'}")
    if verbose_tools:
        print(colored("[executor summary]", "magenta"), results[-1], flush=True)

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
    openai_tools_param: list | None,
    debug: bool,
    image: np.ndarray | None = None,
) -> tuple[str, float]:
    """Call the LLM and return (raw_response, elapsed_seconds)."""
    t0 = timeit.default_timer()

    def _with(verbose: bool, **call_kw: Any) -> str:
        return llm_client(text, verbose=verbose, **call_kw)

    def _try_chain() -> str:
        if openai_tools_param is not None and image is not None:
            try:
                return _with(debug, tools=openai_tools_param, image=image)
            except TypeError:
                pass
        if openai_tools_param is not None:
            try:
                return _with(debug, tools=openai_tools_param)
            except TypeError:
                pass
        if image is not None:
            try:
                return _with(debug, image=image)
            except TypeError:
                pass
        try:
            return _with(debug)
        except TypeError:
            return llm_client(text)

    raw = _try_chain()
    return raw, timeit.default_timer() - t0


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


def run_agent_with_robot(
    robot_ip: str = "127.0.0.1",
    robot: str = "stretch",
    input_path: str | None = None,
    discord: bool = True,
    use_llm: bool = False,
    llm: str = "qwen35-9B",
    server_ip: str = "127.0.0.1",
    skip_confirmations: bool = True,
    explore_iter: int = 3,
    debug_llm: bool = False,
    tool_debug: bool = False,
    agent_name: str = DEFAULT_AGENT_NAME,
    commands: list[str] | None = None,
    port_offset: int = 0,
    agent_config: str = "dynav_config.yaml",
    device: str = "cuda",
    max_tokens: int = 1024,
    vl_include_camera: bool = False,
    eqa: bool = False,
    share_memory_vllm: bool = True,
    headless: bool = False,
    rerun: bool = False,
    rerun_native: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    shutdown_sim_subprocess: Callable[[], None] | None = None,
    **kwargs: Any,
) -> None:
    """Start robot, optional memory load, optional Discord; run command loop with tools.

    If *commands* is a non-empty list, each entry is fed as a user turn
    (LLM mode) or manual command (no-LLM mode) instead of reading stdin.
    The agent exits after all commands are consumed.

    *shutdown_sim_subprocess*, when set, is called after ``robot_client.stop()`` so a sim
    started with ``emet run agent --start-sim`` can exit as soon as the ZMQ client disconnects
    (``run_agent`` still registers a final shutdown in ``finally`` as a safety net).

    When *eqa*, *use_llm*, and *share_memory_vllm* are true (the CLI default for sharing), DynaMem defers its
    local caption VLM until after the agent LLM loads, then reuses the agent vision-language client when
    applicable; otherwise it loads the local EQA VLM from ``dynav_config.yaml``.
    """
    _configure_agent_terminal_output()
    verbose_tools = bool(tool_debug) or _env_agent_tool_debug()

    camera_debug = env_agent_camera_debug()
    parameters = get_parameters(agent_config)
    embodied_overlay = load_embodied_agent_overlay(agent_config)
    defer_eqa_vllm = bool(eqa and use_llm and share_memory_vllm)
    _exec_kwargs = {k: v for k, v in kwargs.items() if k != "defer_eqa_vllm"}
    depth_mode = str(parameters.get("depth_source", "sensor")).lower()
    robot_key = robot.lower().replace("-", "_")
    allow_missing_depth = depth_mode in ("da3", "auto") or robot_key == "innate_mars"
    if robot_key == "stretch":
        # Do not start ZMQ in __init__: DynamemTaskExecutor calls agent.start() which invokes
        # robot.start() again; double-start left orphan recv threads and led to ZMQ double-free crashes.
        robot_client = StretchZmqClient(
            robot_ip=robot_ip,
            parameters=parameters,
            enable_rerun_server=rerun,
            rerun_headless=headless,
            rerun_native_viewer=rerun_native,
            rerun_show_panels=rerun_show_panels,
            rerun_debug=rerun_debug,
            port_offset=port_offset,
            start_immediately=False,
            allow_missing_depth=allow_missing_depth,
        )
    elif robot_key in ROBOT_REGISTRY:
        mod = importlib.import_module(ROBOT_REGISTRY[robot_key])
        backend_cls = None
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and hasattr(attr, "get_spec") and attr_name != "RobotBackend":
                backend_cls = attr
                break
        if backend_cls is None:
            raise RuntimeError(f"No RobotBackend found in {ROBOT_REGISTRY[robot_key]}")
        backend = backend_cls()
        robot_client = backend.create_client(
            robot_ip=robot_ip,
            port_offset=port_offset,
            parameters=parameters,
            allow_missing_depth=allow_missing_depth,
            enable_rerun_server=rerun,
            rerun_headless=headless,
            rerun_native_viewer=rerun_native,
            rerun_show_panels=rerun_show_panels,
            rerun_debug=rerun_debug,
        )
    else:
        raise click.UsageError(
            f"Unknown robot '{robot}'. Known: stretch, {list(ROBOT_REGISTRY.keys())}. "
            "Start the server with the same robot: emet serve mujoco --robot <name>"
        )

    executor = DynamemTaskExecutor(
        robot_client,
        parameters,
        server_ip=server_ip,
        skip_confirmations=skip_confirmations,
        explore_iter=explore_iter,
        discord_bot=None,
        eqa=eqa,
        defer_eqa_vllm=defer_eqa_vllm,
        embodied_agent=embodied_overlay,
        **_exec_kwargs,
    )
    print_vram_snapshot("after_dyn_av_executor_init_siglip_detector_voxel")

    if eqa:
        print(
            colored(
                "EQA/DynaMem: SigLIP, optional SAM3/DINO, and GraphEQA Qwen3.5-VL load lazily on the first "
                "robot update—separate HF checkpoints from text --llm (each loads once per process). "
                "VRAM milestones: EMET_VRAM_DEBUG=1 or --debug-vram (with --debug-models). "
                "One Qwen3-VL for agent+captions: --llm qwen3-vl-eqa --eqa --share-memory-vllm.",
                "cyan",
            )
        )

    if input_path:
        backend = get_memory_backend("dynamem", voxel_map=executor.agent.get_voxel_map())
        backend.load(input_path)
        executor._last_memory_save_path = input_path

    _gm = getattr(executor.agent, "graph_memory", None)
    _graph_backend = None
    if _gm is not None:
        _graph_backend = get_memory_backend(
            "graph_eqa",
            graph_memory=_gm,
            voxel_map=executor.agent.get_voxel_map(),
        )
    memory_backend = get_memory_backend("dynamem", voxel_map=executor.agent.get_voxel_map())
    context: dict[str, Any] = {
        "executor": executor,
        "robot": robot_client,
        "memory_backend": memory_backend,
        "graph_memory": _gm,
        "graph_memory_backend": _graph_backend,
        "scene_graph_processor": getattr(executor.agent, "_open_vocab_sg_processor", None),
        "embodied_agent_config": embodied_overlay,
        "discord_bot": None,
        "xyt_for_query": None,
        "planner": getattr(executor.agent, "planner", None),
        "verbose_tools": verbose_tools,
        "camera_debug": camera_debug,
    }

    def update_xyt():
        if executor.agent.robot is not None and hasattr(executor.agent.robot, "get_base_pose"):
            context["xyt_for_query"] = executor.agent.robot.get_base_pose()

    discord_bot = None
    unified_input_queue: queue.Queue[str] | None = None
    if discord and not os.environ.get("DISCORD_TOKEN"):
        print(
            colored(
                "Warning: Discord bridge is enabled but DISCORD_TOKEN is not in the environment. "
                "Discord bot will not start. Export DISCORD_TOKEN or use --no-discord.",
                "yellow",
            )
        )
    if discord and os.environ.get("DISCORD_TOKEN"):
        from emet.llms.discord_bot import EmetDiscordBot

        unified_input_queue = queue.Queue()

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
            agent_input_queue=unified_input_queue,
        )
        context["discord_bot"] = discord_bot
        executor.discord_bot = discord_bot
        executor.agent.discord_bot = discord_bot
        bot_thread = threading.Thread(target=discord_bot.run, daemon=True)
        bot_thread.start()
        print(colored("Discord bot started (DISCORD_TOKEN). Messages will be handled.", "green"))

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
            if device == "cuda":
                from emet.utils.vram_debug import cuda_pre_llm_memory_notice

                pre = cuda_pre_llm_memory_notice(device=device)
                if pre:
                    print(colored(pre, "yellow"), flush=True)
                if "9b" in llm.lower():
                    try:
                        from emet.utils.vram_debug import torch_cuda_alloc_reserved_gib

                        a, _ = torch_cuda_alloc_reserved_gib(0)
                        if a is not None and a > 10.0:
                            print(
                                colored(
                                    f"Heavy VRAM use (~{a:.1f} GiB torch) before chat LLM; "
                                    "`--llm qwen35-9B` may CUDA-OOM on a single 24GB GPU with SigLIP/detector. "
                                    "Prefer `--llm qwen35-4B` (default) or free GPU memory.",
                                    "yellow",
                                ),
                                flush=True,
                            )
                    except Exception:
                        pass
            print(
                colored(
                    "Loading LLM (HF hub progress bars off; Discord gateway logs at WARNING). …",
                    "cyan",
                )
            )
            prompt_builder = AgentPromptBuilder(tools=tools, name=agent_name, context=context)
            print_vram_snapshot("before_agent_llm_load")
            llm_client = get_llm_client(llm, prompt=prompt_builder, device=device, parameters=parameters)
            print_vram_snapshot("after_agent_llm_load")
            if hasattr(llm_client, "max_tokens"):
                llm_client.max_tokens = max_tokens
            from emet.llms.openai_client import OpenaiClient

            if isinstance(llm_client, OpenaiClient):
                openai_tools_param = [t.schema() for t in tools]
        except Exception as e:
            logger.warning("LLM failed to load (%s):", llm, e)
            print(colored("Agent mode requires an LLM; it failed to load.", "red"))
            print(colored("Fix the LLM (e.g. --llm, device) or run with --no-llm for letter commands only.", "yellow"))
            if use_llm and device == "cuda" and "out of memory" in str(e).lower():
                try:
                    from emet.utils.vram_debug import (
                        cuda_oom_followup_hint,
                        format_cuda_torch_state_line,
                    )

                    post = format_cuda_torch_state_line(label="after LLM load failure", device_index=0)
                    if post:
                        print(colored(post, "yellow"), flush=True)
                    print(colored(cuda_oom_followup_hint(llm_key=llm), "yellow"), flush=True)
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                except Exception:
                    pass
            robot_client.stop()
            chat_log.close()
            return

    if use_llm and llm_client is not None and defer_eqa_vllm:
        vm = executor.agent.get_voxel_map()
        if getattr(vm, "_eqa_pending", None) is not None:
            if isinstance(llm_client, AbstractVLLMClient):
                vm.bind_shared_vllm_from_agent(llm_client)
                print_vram_snapshot(
                    "after_bind_shared_vllm_from_agent",
                    extra="DynaMem caption/EQA uses the same VL object as --llm",
                )
            else:
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                vm.materialize_local_eqa_vllm()
                logger.info(
                    "EQA: loaded DynaMem VLM from yaml (agent --llm is not a shareable VL client). "
                    "To reuse one Qwen3-VL for chat+captions: --llm qwen3-vl-eqa with --eqa --share-memory-vllm."
                )
                print_vram_snapshot(
                    "after_materialize_local_eqa_vllm",
                    extra="second local VL vs agent text LLM; see log line above",
                )

    if use_llm and llm_client is not None:
        print_embodied_model_report(
            llm_key=llm,
            llm_client=llm_client,
            device=device,
            max_tokens=max_tokens,
            executor=executor,
            vl_include_camera=vl_include_camera,
            openai_tool_schemas=openai_tools_param is not None,
        )
        print(colored(f"LLM enabled ({llm}). Say what you want the robot to do.", "green"))
    else:
        print(colored("Enter mode [E=explore / M=pick+place / Q=question / P=send picture / QUIT]:", "green"))
    if verbose_tools:
        print(
            colored(
                "Tool debug: raw tool_calls JSON, full tool strings, executor command list, "
                "[Tool results] text sent back to the LLM, and send_image array stats. "
                "Unset EMET_AGENT_TOOL_DEBUG or omit --debug-tools to disable. "
                "For which-model tracing: EMET_AGENT_MODEL_DEBUG=1 or ``emet run agent --debug-models``. "
                "VRAM snapshots: EMET_VRAM_DEBUG=1 or ``--debug-vram`` (also enabled with --debug-models).",
                "yellow",
            )
        )
    if camera_debug:
        print(
            colored(
                "Camera debug: head-frame stats on describe_scene, send_image, and Discord PNG encode. "
                "Unset EMET_AGENT_CAMERA_DEBUG or omit --debug-camera. "
                "If Discord PNG colors look swapped but stats show valid pixels: set EMET_DISCORD_IMAGES_BGR=1 only "
                "for raw OpenCV BGR matrices (JPEG via from_jpg is already RGB).",
                "yellow",
            )
        )

    # Print system prompt once at startup when debug is on
    if debug_llm and prompt_builder is not None:
        print(colored("=" * 60, "yellow"))
        print(colored("[DEBUG] System prompt (printed once):", "yellow"))
        print(str(prompt_builder))
        print(colored("=" * 60, "yellow"))

    chat_log.log("system", str(prompt_builder) if prompt_builder else "(no LLM)")

    def _send_to_discord(text: str) -> None:
        if discord_bot is None or not hasattr(discord_bot, "push_task_to_all_channels"):
            return
        # Mirror outbound on the terminal here so logs stay ordered with stdout (Discord queue is async).
        stripped = (text or "").strip()
        if stripped and hasattr(discord_bot, "_print_discord_outbound"):
            for channel in discord_bot.allowed_channels:
                ch_name = getattr(channel, "name", "?")
                discord_bot._print_discord_outbound(ch_name, text, has_image=False)
        discord_bot.push_task_to_all_channels(message=text, skip_terminal_mirror=bool(stripped))

    # Wait for Discord to connect before sending the greeting
    if discord_bot is not None and hasattr(discord_bot, "wait_until_ready"):
        print(colored("Waiting for Discord connection...", "yellow"), end=" ", flush=True)
        if discord_bot.wait_until_ready(timeout=30.0):
            print(colored("connected.", "green"))
        else:
            print(colored("timeout (continuing without Discord).", "red"))

    # Startup greeting
    greeting = f"Hello! I'm {agent_name}. I'm online and ready to help."
    print(colored(f"{agent_name}:", "blue"), greeting, flush=True)
    _send_to_discord(greeting)
    chat_log.log("assistant", greeting)

    # Command queue for non-interactive / scripted mode
    cmd_queue: list[str] = list(commands) if commands else []
    scripted = bool(cmd_queue)

    if unified_input_queue is not None and not scripted:
        print(file=sys.stdout)
        print("-" * 60, flush=True)
        print(
            colored(
                "Ready — terminal and Discord share one input queue; type below or post in the home channel.",
                "cyan",
            ),
            flush=True,
        )
        print("-" * 60, flush=True)

        # Prompt on stderr so agent replies on stdout do not splice into the same TTY line as "You:".
        def _stdin_to_unified_queue() -> None:
            while True:
                try:
                    print(colored("You: ", "green"), end="", flush=True, file=sys.stderr)
                    line = sys.stdin.readline()
                    if not line:
                        break
                    unified_input_queue.put(line.strip())
                except (EOFError, KeyboardInterrupt):
                    break

        threading.Thread(target=_stdin_to_unified_queue, daemon=True).start()

    def _get_input(prompt_text: str) -> str | None:
        """Read next input from queue (scripted) or stdin (interactive). None = done."""
        if cmd_queue:
            text = cmd_queue.pop(0)
            print(colored(prompt_text, "green") + text)
            return text
        if scripted:
            return None  # queue exhausted
        if unified_input_queue is not None:
            return unified_input_queue.get()
        try:
            return input(colored(prompt_text, "green")).strip()
        except (EOFError, KeyboardInterrupt):
            return None

    ok = True
    # When Discord + terminal share the session, prompts go to stderr; keep stdout lines clearly separated.
    _stdout_pad = unified_input_queue is not None and not scripted

    def _print_user_turn_separator() -> None:
        if _stdout_pad:
            print(file=sys.stdout)
        print("-" * 60, flush=True)

    while ok:
        update_xyt()

        # --- LLM path ---
        if use_llm and llm_client is not None:
            user_text = _get_input("You: ")
            if user_text is None:
                break
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
                cam_image = None
                if vl_include_camera and _round == 0 and hasattr(robot_client, "get_observation"):
                    obs = robot_client.get_observation()
                    if obs is not None and getattr(obs, "rgb", None) is not None:
                        cam_image = np.asarray(obs.rgb)
                        if verbose_tools:
                            cr = np.asarray(cam_image)
                            print(
                                colored("[vl camera debug]", "magenta"),
                                f"shape={cr.shape} dtype={cr.dtype} min={cr.min()} max={cr.max()} mean={float(cr.mean()):.4f}",
                                flush=True,
                            )
                        if camera_debug or verbose_tools:
                            from emet.agent.camera_debug import print_camera_frame_diagnostics

                            print_camera_frame_diagnostics(
                                "VL first-turn (rgb passed to chat LLM)",
                                cam_image,
                                force=True,
                            )
                print_llm_invoke_line(
                    llm_client,
                    has_tools=openai_tools_param is not None,
                    has_image=cam_image is not None,
                )
                raw_response, elapsed = _call_llm(
                    llm_client,
                    current_input,
                    openai_tools_param,
                    debug_llm,
                    image=cam_image,
                )

                if debug_llm:
                    print(colored(f"[DEBUG] Raw response ({elapsed:.2f}s):", "yellow"), raw_response[:500])

                parsed = parse_tool_calls_response(raw_response)
                tool_calls = parsed.get("tool_calls") or []
                message = parsed.get("message") or ""

                if debug_llm or verbose_tools:
                    print(colored("[DEBUG] Parsed:", "blue"), json.dumps(parsed, indent=2))

                chat_log.log("assistant", message, tool_calls=tool_calls, raw=raw_response, time_s=elapsed)

                # No tool calls — this is the final answer
                if not tool_calls:
                    if message:
                        print(colored(f"{agent_name}:", "blue"), message, flush=True)
                        _send_to_discord(message)
                    break

                # Print and relay the intermediate message (e.g. "Let me check my memory.")
                if message:
                    print(colored(f"{agent_name}:", "blue"), message, flush=True)
                    _send_to_discord(message)

                # Execute tool calls
                ok, results, has_info = _dispatch_tool_calls(
                    tool_calls,
                    tools_by_name,
                    executor,
                    chat_log=chat_log,
                    debug=debug_llm,
                    verbose_tools=verbose_tools,
                )
                if not ok:
                    break

                result_text = "\n".join(results)
                if verbose_tools and result_text.strip():
                    print(colored("[tool results combined]", "magenta"), result_text, sep="\n", flush=True)

                if has_info:
                    # Feed tool results back to LLM for summarization
                    followup = f"[Tool results]\n{result_text}\n\nSummarize these results for the user in your message. Do not call any more tools."
                    if hasattr(llm_client, "add_history"):
                        llm_client.add_history({"role": "assistant", "content": raw_response})
                        llm_client.add_history({"role": "user", "content": followup})
                    current_input = followup
                    if debug_llm or verbose_tools:
                        print(
                            colored("[→ LLM follow-up user message]", "magenta"),
                            followup,
                            sep="\n",
                            flush=True,
                        )
                    continue
                else:
                    # Action-only tools: assistant message was already printed and sent to Discord above.
                    if results and hasattr(llm_client, "add_history"):
                        llm_client.add_history({"role": "assistant", "content": raw_response})
                        llm_client.add_history({"role": "user", "content": f"[Tool results]\n{result_text}"})
                    break
            if ok:
                _print_user_turn_separator()
            continue

        # --- Manual (no-LLM) path ---
        line = _get_input("You: ")
        if line is None:
            break
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
        find_query = parse_manual_find_command(line)
        if find_query:
            ok = executor([("find", find_query)])
            continue

        if use_llm:
            print(colored("LLM did not load; use letter commands: E / M / Q / P / FIND.", "yellow"))
            continue
        ok = executor([("pickup", line), ("place", "")]) if line else True

    chat_log.log("system", "session ended")
    chat_log.close()
    print(colored(f"Chat log saved: {chat_log.path}", "green"))
    print_memory_view_help_on_quit(getattr(executor, "_last_memory_save_path", None))
    robot_client.stop()
    if shutdown_sim_subprocess is not None:
        shutdown_sim_subprocess()
