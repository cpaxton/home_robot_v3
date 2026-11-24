# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Ground frozen recorded RGBD without starting a simulator or moving a robot."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.task.tamp.observed_capture import capture_arrays
from emet.controller.task.tamp.transfer_grounding import select_transfer
from emet.eval.panda_open_loop import estimated_grasp_section
from emet.eval.vlm_pick_hint import grounding_profile, query_pick_hint, refine_object_hint


def load_views(sensor):
    with np.load(sensor, allow_pickle=False) as data:
        return {name: SimpleNamespace(rgb=data[rgb], depth=data[depth], camera_K=data[k], camera_pose=data[pose])
            for name, rgb, depth, k, pose in (
                ('scene', 'rgb', 'depth', 'K', 'pose'),
                ('scene_alt', 'scene_alt_rgb', 'scene_alt_depth', 'scene_alt_camera_K', 'scene_alt_camera_pose'),
                ('wrist', 'wrist_rgb', 'wrist_depth', 'wrist_K', 'wrist_pose'))}


def evaluate_decision(views, decision, *, pick_only=False):
    sample = SimpleNamespace(capture=SimpleNamespace(observation_id='frozen-replay'), observations=views)
    profile = grounding_profile(decision, infer_support_from_depth='source' not in decision)
    selected = select_transfer(sample, profile=profile, grow_object_region=True, require_destination=not pick_only)
    points, target = estimated_grasp_section(views, selected, decision.get('grasp_height_fraction', .5), grasp_point=decision.get('grasp_point'))
    source = capture_arrays(views[selected.source_camera])[2][selected.source_mask]
    return {'accepted': True, 'target_world_m': target.tolist(), 'object_points': len(points),
            'source_height_m': float(np.median(source[:, 2])), 'semantic_identity_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--endpoint')
    parser.add_argument('--decisions-from', type=Path, help='Replay recorded VLM estimates without making model calls')
    parser.add_argument('--pick-only', action='store_true')
    parser.add_argument('--infer-support-from-depth', action='store_true', help='Omit support-box prediction; infer nearby support from measured depth')
    parser.add_argument('--grasp-site', action='store_true', help='Ask the crop VLM for a grasp site, height and direction')
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--model', default='Qwen/Qwen3.5-9B')
    args = parser.parse_args()
    if (args.decisions_from is None or args.refine) and args.endpoint is None:
        parser.error('supply --endpoint or --decisions-from')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.batch/'manifest.json').read_text())
    results = []
    model_calls = 0
    for case in manifest['cases']:
        sensor = args.batch/case['id']/'sensor.npz'
        views = load_views(sensor)
        row = {'id': case['id'], 'sensor_sha256': hashlib.sha256(sensor.read_bytes()).hexdigest()}
        previous = args.batch/case['id']/'vlm-decision.json'
        if previous.exists():
            try:
                row['previous'] = evaluate_decision(views, json.loads(previous.read_text()), pick_only=args.pick_only)
            except (ValueError, ObservationUnavailable) as error:
                row['previous'] = {'accepted': False, 'reason': str(error)}
        output = args.output/case['id']
        output.mkdir()
        try:
            if args.decisions_from:
                path = args.decisions_from/case['id']/'vlm-decision.json'
                refined_path = path.parent/'refinement'/'decision.json'
                if refined_path.exists():
                    path = refined_path
                decision = json.loads(path.read_text())
                row['decision_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                (output/'vlm-decision.json').write_bytes(path.read_bytes())
            else:
                model_calls += 1
                decision = query_pick_hint(views['scene_alt'].rgb, endpoint=args.endpoint, model=args.model,
                    object_name=case['spec']['object_name'], output=output, destination_name=manifest['destination'],
                    views={name: view.rgb for name, view in views.items()},
                    infer_support_from_depth=args.infer_support_from_depth)
            if args.refine:
                model_calls += 1
                decision = refine_object_hint(views[decision.get('camera', 'scene_alt')].rgb, decision,
                    endpoint=args.endpoint, model=args.model, object_name=case['spec']['object_name'],
                    output=output/'refinement', estimate_grasp_site=args.grasp_site)
            (output/'vlm-decision.json').write_text(json.dumps(decision, indent=2)+'\n')
            row['current'] = {**evaluate_decision(views, decision, pick_only=args.pick_only), 'decision': decision}
        except (ValueError, ObservationUnavailable) as error:
            row['current'] = {'accepted': False, 'reason': str(error)}
        results.append(row)
        report = {'robot_motion': False, 'destination_required': not args.pick_only, 'physical_success_evaluated': False,
                  'source_batch': str(args.batch), 'model': args.model, 'results': results,
                  'runtime_model_calls': model_calls,
                  'source_hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                      Path(__file__), Path('src/emet/controller/task/tamp/transfer_grounding.py'),
                      Path('src/emet/eval/panda_open_loop.py'), Path('src/emet/eval/vlm_pick_hint.py'))}}
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
