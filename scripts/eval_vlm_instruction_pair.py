# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Compare VLM destination instructions on identical recorded RGB-D; no motion."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emet.controller.task.tamp.transfer_grounding import select_transfer
from emet.eval.panda_open_loop import front_interior_floor
from emet.eval.vlm_pick_hint import query_pick_hint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sensor', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--object', required=True)
    parser.add_argument('--destination', action='append', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = np.load(args.sensor, allow_pickle=False)
    scene = SimpleNamespace(rgb=data['scene_alt_rgb'], depth=data['scene_alt_depth'],
        camera_K=data['scene_alt_camera_K'], camera_pose=data['scene_alt_camera_pose'])
    extras = []
    if 'wrist_depth' in data:
        extras.append(SimpleNamespace(rgb=data['wrist_rgb'], depth=data['wrist_depth'],
            camera_K=data['wrist_K'], camera_pose=data['wrist_pose']))
    identity = hashlib.sha256(scene.rgb.tobytes()).hexdigest()
    sample = SimpleNamespace(capture=SimpleNamespace(observation_id=identity), observations={'scene_alt': scene})
    results = []
    for index, instruction in enumerate(args.destination):
        output = args.output/str(index)
        output.mkdir()
        result = {'destination_instruction': instruction, 'depth_valid': False}
        try:
            decision = query_pick_hint(scene.rgb, endpoint=args.endpoint, model=args.model,
                object_name=args.object, output=output, destination_name=instruction)
            result['decision'] = decision
            profile = {'grounding': {role: {'camera': 'scene_alt', 'box': decision[role]}
                                     for role in ('object', 'source', 'destination')}}
            selection = select_transfer(sample, profile=profile, grow_object_region=True)
            goal = front_interior_floor(scene, selection.destination_mask, extra_observations=extras)
            result.update(depth_valid=True, placement_floor_world_m=goal.tolist())
        except Exception as error:
            result['failure'] = str(error)
        results.append(result)
    report = {'model': args.model, 'image_sha256': identity, 'robot_motion': False,
              'physical_success_evaluated': False, 'results': results}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if all(result['depth_valid'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
