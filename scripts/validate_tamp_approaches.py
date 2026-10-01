#!/usr/bin/env python3
"""Execute preselected GT approach candidates using the unchanged TAMP scorer.

This is assisted approach validation, not autonomous planning or new fixture
admission. The adapter supplies the existing MCTS candidate `approach_pose`
argument in memory, leaving the frozen registry and scorer unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def select_approaches(episode, audit, rank=0):
    if rank < 0:
        raise ValueError('rank must be nonnegative')
    rows = audit['results']
    if len({r['object'] for r in rows}) != len(rows):
        raise ValueError('ambiguous audit categories')
    objects = episode['fixture']['clutter']
    if len({r['cat'] for r in objects}) != len(objects):
        raise ValueError('category-only audit cannot identify duplicate objects')
    result = {}
    for obj in objects:
        matches = [r for r in rows if r['object'] == obj['cat']]
        if len(matches) != 1:
            raise ValueError(f"missing audit for {obj['body']}")
        candidates = [r for r in matches[0]['candidates']
                      if not r['contacts'] and r['ik'] and r['ik']['success']]
        if rank >= len(candidates):
            raise ValueError(f"insufficient admitted candidates for {obj['body']}")
        pose = np.asarray(candidates[rank]['xyt'], dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError('invalid approach pose')
        result[obj['body']] = pose.tolist()
    return result


def with_approaches(candidates, selected):
    """Use exact body identities and preserve all other grounding inputs."""
    result = []
    for candidate in candidates:
        body = candidate.get('object_gt_body')
        if body not in selected:
            raise ValueError(f'unselected object: {body}')
        result.append({**candidate, 'approach_pose': list(selected[body])})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--episode-id', required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    episodes = yaml.safe_load(args.registry.read_text())['episodes']
    matches = [e for e in episodes if e['id'] == args.episode_id]
    if len(matches) != 1 or matches[0].get('manip_mode') != 'latch':
        parser.error('require one explicit latch episode')
    selected = select_approaches(matches[0], json.loads(args.audit.read_text()), args.rank)
    manifest = {
        'scope': 'assisted_approach_validation; kinematic latch; not autonomous/physical acceptance',
        'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'episode_id': args.episode_id, 'rank': args.rank, 'selected_approaches': selected,
        'selection_rule': 'nth collision-clear and position-IK-reachable candidate in recorded audit order',
        'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.registry, args.audit)},
        'scoring': 'unchanged frozen fixture, object identities, goal and measured acceptance tolerances',
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    if not os.environ.get('EMET_JOB_ID'):
        parser.error('run through emet jobs with exclusive GPU and CPU-safe settings')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'approach_validation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    import emet.controller.task.tamp.task_search as task_search
    original = task_search.plan_pick_place_mcts

    def plan_with_selected_approaches(robot, *, candidates, **kwargs):
        return original(robot, candidates=with_approaches(candidates, selected), **kwargs)

    evaluator_path = Path(__file__).with_name('eval_tamp_clutter.py')
    spec = importlib.util.spec_from_file_location('tamp_approach_evaluator', evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    old_argv = sys.argv
    task_search.plan_pick_place_mcts = plan_with_selected_approaches
    try:
        sys.argv = [str(evaluator_path), '--episodes', str(args.registry), '--episode-id', args.episode_id,
                    '--output-dir', str(args.output_dir)]
        return evaluator.main()
    finally:
        task_search.plan_pick_place_mcts = original
        sys.argv = old_argv


if __name__ == '__main__':
    raise SystemExit(main())
