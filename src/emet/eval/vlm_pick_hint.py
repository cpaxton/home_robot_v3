# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""One fresh image to a VLM decision; depth and native control remain downstream."""
import base64
import io
import json
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

from emet.perception.grasps.observed import GroundingHint


def query_pick_hint(rgb, *, endpoint, model, object_name, output, destination_name="sink"):
    image = Image.fromarray(rgb)
    image.save(output/'vlm-input.png')
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = image.size
    for value in range(100, 1000, 100):
        x, y = round(width*value/1000), round(height*value/1000)
        draw.line((x, 0, x, height), fill=(100, 160, 170), width=1)
        draw.line((0, y, width, y), fill=(100, 160, 170), width=1)
        draw.text((x+2, 2), str(value), fill='yellow', stroke_width=1, stroke_fill='black')
        draw.text((2, y+2), str(value), fill='yellow', stroke_width=1, stroke_fill='black')
    annotated.save(output/'vlm-grid-input.png')
    buffer = io.BytesIO()
    annotated.save(buffer, format='PNG')
    prompt = (
        f'Pick up the {object_name} on the counter and place it in {destination_name}. '
        'Object names are approximate semantic descriptions, not exact asset classes. '
        'Consider common synonyms and visually related subtypes; describe the candidate you select. '
        'Do not reject solely because its visual subtype differs. If multiple plausible targets remain, say unavailable. '
        'Use the labeled coordinate grid: x increases right, y increases down. '
        'Select generous but specific image regions, normalized to 0..1000 xyxy. '
        f'object: the target; source: nearby counter surface; destination: visible floor of {destination_name}, not rim or faucet. '
        'For the available top-down parallel-jaw pick, choose grasp_yaw_degrees (0..360): '
        'finger-closing direction clockwise from image right, not a world-frame robot angle. '
        'Return JSON only: {"object":[x0,y0,x1,y1],"source":[x0,y0,x1,y1],'
        '"destination":[x0,y0,x1,y1],"grasp_yaw_degrees":180,"reason":"brief explanation"}. '
        'If the target or required surfaces cannot be identified, return {"unavailable":"reason"}.'
    )
    payload = {'model': model, 'temperature': 0, 'max_tokens': 512,
        'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+
                base64.b64encode(buffer.getvalue()).decode()}}]}]}
    (output/'vlm-request.json').write_text(json.dumps(payload)+'\n')
    request = urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=90) as response:
        reply = json.load(response)
    (output/'vlm-response.json').write_text(json.dumps(reply, indent=2)+'\n')
    content = reply['choices'][0]['message']['content'].strip()
    if content.startswith('```'):
        content = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    decision = json.loads(content)
    if 'unavailable' in decision:
        raise ValueError('vlm_grounding_unavailable: '+str(decision['unavailable']))
    for role in ('object', 'source', 'destination'):
        GroundingHint('current-image', 'scene_alt', tuple(decision[role]), role)
    yaw = float(decision['grasp_yaw_degrees'])
    if not np.isfinite(yaw) or not 0 <= yaw <= 360:
        raise ValueError('invalid_vlm_grasp_yaw')
    (output/'vlm-decision.json').write_text(json.dumps(decision, indent=2)+'\n')
    return decision
