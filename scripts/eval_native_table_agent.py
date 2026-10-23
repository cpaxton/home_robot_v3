#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded LLM selection over native CHAT tools on the frozen two-object fixture.

Uses the run-agent prompt, parser and dispatcher, with an in-process actuator
facade or a live ZMQ simulator and privileged geometry. It does not launch the
interactive CLI or establish observed perception. Run as a managed job; reserve
the GPU for local CUDA inference or hardware-accelerated camera rendering.
"""
import argparse
import json
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import mujoco
from eval_native_table_contact import ContactRobot
from transformers import set_seed

from emet.agent.loop import _call_llm, _dispatch_tool_calls
from emet.agent.prompt import AgentPromptBuilder, parse_tool_calls_response
from emet.agent.tools import get_tools
from emet.controller.manipulation import physical_pick_place
from emet.controller.task.tamp import native_adapter
from emet.eval.native_table_scoring import goal_predicates, load_placement_regions
from emet.llms.openai_client import OpenaiClient
from emet.llms.qwen_client import Qwen25Client
from emet.simulation.mujoco_server_stretch import default_scene_xml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--placement-regions', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--max-turns', type=int, default=12)
    parser.add_argument('--model-size', default='3B', choices=['1.5B', '3B'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base-url', help='Existing remote OpenAI-compatible inference endpoint')
    parser.add_argument('--remote-model', help='Exact model ID served by --base-url')
    parser.add_argument('--live-sim', type=Path, help='Launch a live ZMQ simulator from this config; use a managed job')
    args = parser.parse_args()
    if bool(args.base_url) != bool(args.remote_model):
        parser.error('--base-url and --remote-model must be supplied together')
    if args.max_turns < 1:
        parser.error('--max-turns must be positive')
    try:
        regions = load_placement_regions(args.placement_regions)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(default_scene_xml_path)
    if args.live_sim:
        from emet.eval.native_table_live import live_table_robot
        os.environ['EMET_PHYSICAL_EXECUTION'] = '1'
        os.environ['EMET_PHYSICAL_AUDIT'] = str((args.output / 'actuation.jsonl').resolve())
        connection = live_table_robot(args.live_sim, args.output, model)
    else:
        connection = nullcontext(ContactRobot(model, 'object1', separation=.15, separation_axis='y'))
    with connection as robot:
        return run_episode(args, robot, model, regions=regions)


def run_episode(args, robot, model, *, regions=None):
    if regions is None:
        regions = load_placement_regions(args.placement_regions)
    goals = dict(zip(('object2', 'object1'), regions, strict=True))
    adapter = native_adapter.NativeTableAdapter(robot, model=model)
    original_sync = adapter.synchronize

    def synchronize(data):
        robot.refresh()
        original_sync(data)

    adapter.synchronize = synchronize
    context = {'robot': robot, 'manip_mode': 'physical', 'tamp_inputs': 'privileged',
               '_native_tamp_adapter': adapter}
    tools = {tool.name: tool for tool in get_tools(context)
             if tool.name in {'scene_tasks', 'plan_pick_place', 'execute_pick_place_plan'}}
    prompt = AgentPromptBuilder(list(tools.values()), context=context)
    instruction = ('Complete both goals on the table, in any feasible order. The entire object footprint '
                   'must fit its world XY region, in meters. These are privileged simulator inputs. '
                   'Discover tasks, plan with placement_region, execute only a returned plan_ref, then '
                   'read the result and rediscover before planning another action. Do not invent handles. '
                   'Do not silently drop a goal constraint. Stop after partial execution because no '
                   'held-object recovery skill is available. Return at most one tool call per turn. Goals: '
                   + json.dumps({'red cylinder': goals['object2'], 'blue cube': goals['object1']}))
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                    cwd=Path(__file__).resolve().parents[1], text=True).strip()
    protocol = {'source_sha': source, 'model': args.remote_model or f'Qwen/Qwen2.5-{args.model_size}-Instruct',
                'seed': args.seed, 'max_turns': args.max_turns, 'device': args.device,
                'inference': 'remote' if args.base_url else 'local',
                'remote_sampling': 'server defaults; seed is local only' if args.base_url else None,
                'scope': ('LLM-selected CHAT tools; live ZMQ native contact; privileged inputs' if args.live_sim
                          else 'LLM-selected CHAT tools; CPU native contact; privileged inputs; no live IPC'),
                'physics_during_inference': bool(args.live_sim),
                'sim_config': args.live_sim.read_text() if args.live_sim else None,
                'cpu_affinity': sorted(os.sched_getaffinity(0)),
                'goals': goals, 'instruction': instruction, 'system_prompt': str(prompt),
                'initial_predicates': goal_predicates(robot, goals)}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    set_seed(args.seed)
    if args.base_url:
        client = OpenaiClient(prompt, model=args.remote_model, base_url=args.base_url)
        client._openai = client._openai.with_options(timeout=120, max_retries=0)
    else:
        client = Qwen25Client(prompt, model_size=args.model_size, device=args.device, quantization=None, max_tokens=256)
    history, trace = [], []
    clock = SimpleNamespace(monotonic=lambda: float(robot.data.time), sleep=robot.step)
    old_physical, old_native = physical_pick_place.time, native_adapter.time
    terminal = 'turn_budget_exhausted'
    try:
        if not args.live_sim:
            physical_pick_place.time = native_adapter.time = clock
        for turn in range(args.max_turns):
            text = instruction + '\nPrevious tool outcomes:\n' + '\n'.join(history)
            client.reset()
            robot.refresh()
            sim_before = float(robot.data.time)
            raw, seconds = _call_llm(client, text, None, False, reset_context=True)
            robot.refresh()
            sim_after = float(robot.data.time)
            parsed = parse_tool_calls_response(raw)
            calls = parsed.get('tool_calls', [])
            row = {'turn': turn, 'response': raw, 'calls': calls, 'model_wall_s': seconds,
                   'sim_time_before_inference': sim_before, 'sim_time_after_inference': sim_after}
            trace.append(row)
            # Enforce the experiment's one-action observation boundary.
            if len(calls) != 1 or calls[0].get('name') not in tools:
                row['error'] = 'expected_one_available_tool_call'
                history.append(json.dumps({'status': 'error', 'code': row['error']}))
            else:
                _, results, _ = _dispatch_tool_calls(calls, tools, SimpleNamespace())
                row['results'] = results
                history.extend(results)
                envelope = json.loads(results[0].split('] ', 1)[1])
                if envelope['status'] == 'partial':
                    terminal = 'partial_execution_requires_recovery'
                    break
                if calls[0]['name'] == 'execute_pick_place_plan' and envelope['status'] == 'ok':
                    stable = True
                    for _ in range(40):
                        robot.step(.05)
                        stable &= all(r['satisfied'] for r in goal_predicates(robot, goals).values())
                    if stable:
                        terminal = 'goals_verified'
                        break
            (args.output / 'trace.json').write_text(json.dumps(trace, indent=2))
            print(json.dumps({'turn': turn, 'calls': calls, 'terminal': terminal}), flush=True)
    finally:
        physical_pick_place.time, native_adapter.time = old_physical, old_native
        (args.output / 'trace.json').write_text(json.dumps(trace, indent=2))
        (args.output / 'execution.json').write_text(json.dumps({'events': adapter.events, 'commands': robot.commands}, indent=2))
    inference_progress = all(r['sim_time_after_inference'] > r['sim_time_before_inference'] for r in trace)
    audit_path = args.output / 'actuation.jsonl'
    audit = [json.loads(line) for line in audit_path.read_text().splitlines()] if audit_path.exists() else []
    audit_ok = bool(audit) and not any(row.get('forbidden') for row in audit)
    result = {'passed': terminal == 'goals_verified' and (not args.live_sim or (inference_progress and audit_ok)),
              'terminal': terminal, 'physics_advanced_during_all_inference_calls': inference_progress,
              'audit_commands': len(audit), 'forbidden_commands': sum(bool(row.get('forbidden')) for row in audit),
              'final_predicates': goal_predicates(robot, goals), 'turns': len(trace)}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
