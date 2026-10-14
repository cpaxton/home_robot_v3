#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded LLM selection over native CHAT tools on the frozen two-object fixture.

Uses the run-agent prompt, parser and dispatcher, with an in-process actuator
facade and privileged simulator geometry. It does not launch the interactive CLI,
test IPC, or establish observed perception. Run as a managed job; reserve the GPU
when selecting --device cuda.
"""
import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from eval_native_table_contact import ContactRobot
from transformers import set_seed

from emet.agent.loop import _call_llm, _dispatch_tool_calls
from emet.agent.prompt import AgentPromptBuilder, parse_tool_calls_response
from emet.agent.tools import get_tools
from emet.controller.manipulation import physical_pick_place
from emet.controller.task.tamp import native_adapter
from emet.llms.qwen_client import Qwen25Client
from emet.simulation.mujoco_server_stretch import default_scene_xml_path
from emet.simulation.sim_object_placements import placements_from_mujoco_model


def goal_predicates(robot, goals):
    """Independent privileged scoring: whole XY footprint, support, and stillness."""
    scratch = mujoco.MjData(robot.model)
    mujoco.mj_copyData(scratch, robot.model, robot.data)
    mujoco.mj_kinematics(robot.model, scratch)
    mujoco.mj_collision(robot.model, scratch)
    placements = placements_from_mujoco_model(robot.model, scratch)
    table = robot.model.body('table').id
    fingertips = {robot.model.body(name).id for name in ('rubber_tip_left', 'rubber_tip_right')}
    rows = {}
    for name, region in goals.items():
        body = robot.model.body(name)
        bounds = np.asarray(placements[name]['bounds'])[:, :2]
        region = np.asarray(region)
        inside = bool(np.all(bounds[0] >= region[0] - .005) and np.all(bounds[1] <= region[1] + .005))
        touched = {int(other) for c in scratch.contact if body.id in robot.model.geom_bodyid[c.geom]
                   for other in robot.model.geom_bodyid[c.geom] if other != body.id}
        supported = table in touched and not fingertips.intersection(touched)
        dof = robot.model.jnt_dofadr[body.jntadr[0]]
        settled = bool(np.linalg.norm(scratch.qvel[dof:dof+3]) <= .01
                       and np.linalg.norm(scratch.qvel[dof+3:dof+6]) <= .1)
        rows[name] = {'in_region': inside, 'supported': supported, 'settled': settled,
                      'satisfied': inside and supported and settled, 'bounds_xy': bounds.tolist()}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--placement-regions', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--max-turns', type=int, default=12)
    parser.add_argument('--model-size', default='3B', choices=['1.5B', '3B'])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.max_turns < 1:
        parser.error('--max-turns must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    regions = json.loads(args.placement_regions.read_text())
    if len(regions) != 2:
        parser.error('Expected cylinder and box regions, in that order')
    goals = dict(zip(('object2', 'object1'), regions, strict=True))
    model = mujoco.MjModel.from_xml_path(default_scene_xml_path)
    robot = ContactRobot(model, 'object1', separation=.15, separation_axis='y')
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
    protocol = {'source_sha': source, 'model': f'Qwen/Qwen2.5-{args.model_size}-Instruct',
                'seed': args.seed, 'max_turns': args.max_turns, 'device': args.device,
                'scope': 'LLM-selected CHAT tools; CPU native contact; privileged inputs; no live IPC',
                'goals': goals, 'instruction': instruction, 'system_prompt': str(prompt),
                'initial_predicates': goal_predicates(robot, goals)}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    set_seed(args.seed)
    client = Qwen25Client(prompt, model_size=args.model_size, device=args.device, quantization=None, max_tokens=256)
    history, trace = [], []
    clock = SimpleNamespace(monotonic=lambda: float(robot.data.time), sleep=robot.step)
    old_physical, old_native = physical_pick_place.time, native_adapter.time
    terminal = 'turn_budget_exhausted'
    try:
        physical_pick_place.time = native_adapter.time = clock
        for turn in range(args.max_turns):
            text = instruction + '\nPrevious tool outcomes:\n' + '\n'.join(history)
            client.reset()
            raw, seconds = _call_llm(client, text, None, False, reset_context=True)
            parsed = parse_tool_calls_response(raw)
            calls = parsed.get('tool_calls', [])
            row = {'turn': turn, 'response': raw, 'calls': calls, 'model_wall_s': seconds}
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
    result = {'passed': terminal == 'goals_verified', 'terminal': terminal,
              'final_predicates': goal_predicates(robot, goals), 'turns': len(trace)}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
