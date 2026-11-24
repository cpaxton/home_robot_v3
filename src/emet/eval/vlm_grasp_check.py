# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Observed-only grasp checks for the simulation transfer diagnostic."""
import base64
import io
import json
import os
import urllib.request

from PIL import Image


def query_grasp_state(views, *, endpoint, model, object_name, finger_positions, output, feedback=None):
    """Ask about current images, preserving uncertainty and raw model evidence."""
    output.mkdir(parents=True, exist_ok=True)
    prompt = (
        f'The robot attempted to pick up the {object_name}. These are CURRENT views after motion. '
        'Determine whether that object is visibly held BETWEEN the gripper fingers, '
        'rather than resting on the counter, inside the sink, or merely behind the hand. '
        'Use all views; occlusion or an unclear tiny object means unknown. '
        'Do not assume success just because the hand moved. '
        'Return JSON only: {"state":"held|not_held|unknown","reason":"visible evidence"}. '
        'This visual judgment is an estimate, not an official task score.'
    )
    prompt += ' Native finger joint positions at capture: '+json.dumps(finger_positions)+'. These are encoder readings, not a success label.'
    if feedback:
        prompt += ' '+feedback
    content = [{'type': 'text', 'text': prompt}]
    for name, rgb in views.items():
        image = Image.fromarray(rgb)
        image.save(output/f'{name}.png')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        content.extend([{'type': 'text', 'text': f'Camera {name}:'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+
                base64.b64encode(buffer.getvalue()).decode()}}])
    payload = {'model': model, 'temperature': 0, 'max_tokens': 256,
               'messages': [{'role': 'user', 'content': content}]}
    (output/'request.json').write_text(json.dumps(payload)+'\n')
    (output/'finger-feedback.json').write_text(json.dumps(finger_positions)+'\n')
    request = urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=float(os.environ.get("EMET_VLM_REQUEST_TIMEOUT_S", "90"))) as response:
        reply = json.load(response)
    (output/'response.json').write_text(json.dumps(reply, indent=2)+'\n')
    text = reply['choices'][0]['message']['content']
    start = text.find('{')
    decision, _ = json.JSONDecoder().raw_decode(text[start:] if start >= 0 else text)
    if decision.get('state') not in {'held', 'not_held', 'unknown'}:
        raise ValueError('invalid_vlm_grasp_state')
    (output/'decision.json').write_text(json.dumps(decision, indent=2)+'\n')
    return decision


def clearly_missed_grasp(decision, finger_positions):
    """Require agreement of visual miss and near-closed native fingers to retry.

    Aperture alone cannot distinguish a thin object from an empty hand; visual
    uncertainty is not a refusal. This threshold describes this Panda gripper.
    """
    import numpy as np

    positions = np.asarray(list(finger_positions.values()), float)
    return bool(decision.get('state') == 'not_held' and positions.shape == (2,)
                and np.isfinite(positions).all() and np.abs(positions).sum() < .002)


def merged_retry_receipt(previous, retry):
    """Keep failed execution evidence and account for all model calls/re-estimates."""
    return {**retry, 'execution_attempts': 1+retry.get('execution_attempts', 1),
            'prior_execution_attempts': [previous, *retry.get('prior_execution_attempts', [])],
            'runtime_vlm_calls': previous.get('runtime_vlm_calls', 0)+retry.get('runtime_vlm_calls', 0),
            'grounding_attempts': previous.get('grounding_attempts', [])+retry.get('grounding_attempts', [])}


def grip_changed_while_observing(before, after):
    """A newly empty hand invalidates a potentially stale positive image estimate."""
    import numpy as np

    old, new = np.asarray(list(before.values()), float), np.asarray(list(after.values()), float)
    return bool(old.shape == (2,) and new.shape == (2,) and np.isfinite(old).all() and np.isfinite(new).all()
                and np.abs(old).sum() >= .004 and np.abs(new).sum() < .002)
