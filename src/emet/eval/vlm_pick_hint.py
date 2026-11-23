# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.
"""Fresh images to VLM estimates; depth and native control remain downstream."""
import base64
import io
import json
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

from emet.perception.grasps.observed import GroundingHint


def query_pick_hint(rgb, *, endpoint, model, object_name, output, destination_name="sink", feedback=None, views=None):
    available = {'scene_alt': rgb} if views is None else views
    if not available:
        raise ValueError('no_vlm_views')
    Image.fromarray(rgb).save(output/'vlm-input.png')
    prompt = (
        f'Pick up the {object_name} on the counter and place it in {destination_name}. '
        'Object names are approximate semantic descriptions, not exact asset classes. '
        'Consider common synonyms and visually related subtypes; describe the candidate you select. '
        'Do not reject solely because its visual subtype differs. If multiple plausible targets remain, say unavailable. '
        f'Available cameras: {list(available)}. Choose the best object camera; source_camera and destination_camera may independently select clearer views. '
        'Avoid mistaking the white robot for a surface or object; prefer another view if it occludes the target. '
        'Use the labeled coordinate grid: x increases right, y increases down. '
        'Select generous but specific image regions, normalized to 0..1000 xyxy, object in camera, source in source_camera, destination in destination_camera. '
        f'object: target; source: nearby counter beside it, NOT the object; destination: visible floor of {destination_name}. '
        'For the top-down parallel-jaw pick, estimate grasp_yaw_degrees (0..360): '
        'finger-closing direction clockwise from image right, not a world-frame robot angle. '
        'Estimate grasp_height_fraction (0 bottom to 1 top of the observed object). '
        'Prefer a broad stable body section, not stems or thin tips. Close across the narrow dimension of long objects. '
        'Choose approach direct for an unobstructed reach or from_viewer to route in front of the counter. '
        'Return JSON only: {"camera":"scene_alt","source_camera":"scene","destination_camera":"scene_alt","object":[x0,y0,x1,y1],'
        '"source":[x0,y0,x1,y1],"destination":[x0,y0,x1,y1],'
        '"grasp_yaw_degrees":180,"grasp_height_fraction":0.5,"approach":"from_viewer","reason":"brief explanation"}. '
        'If the target or required surfaces cannot be identified, return {"unavailable":"reason"}.'
    )
    if feedback:
        prompt += (' Previous proposal feedback: '+str(feedback)+
                   '. Reinspect the images and adapt the regions or grasp estimate. '
                   'Keep the same requested target; do not substitute another object. If unavailable, say so.')
    first = next(iter(available))
    content = [{'type': 'text', 'text': prompt+f' First image: camera {first}.'}]
    for index, (camera, pixels) in enumerate(available.items()):
        image = Image.fromarray(pixels)
        image.save(output/f'vlm-view-{camera}-raw.png')
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        width, height = image.size
        for value in range(100, 1000, 100):
            x, y = round(width*value/1000), round(height*value/1000)
            draw.line((x, 0, x, height), fill=(100, 160, 170), width=1)
            draw.line((0, y, width, y), fill=(100, 160, 170), width=1)
            draw.text((x+2, 2), str(value), fill='yellow', stroke_width=1, stroke_fill='black')
            draw.text((2, y+2), str(value), fill='yellow', stroke_width=1, stroke_fill='black')
        annotated.save(output/f'vlm-view-{camera}.png')
        if camera == 'scene_alt':
            annotated.save(output/'vlm-grid-input.png')
        buffer = io.BytesIO()
        annotated.save(buffer, format='PNG')
        if index:
            content.append({'type': 'text', 'text': f'Camera {camera}:'})
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+
            base64.b64encode(buffer.getvalue()).decode()}})
    payload = {'model': model, 'temperature': 0, 'max_tokens': 512,
               'messages': [{'role': 'user', 'content': content}]}
    (output/'vlm-request.json').write_text(json.dumps(payload)+'\n')
    request = urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=90) as response:
        reply = json.load(response)
    (output/'vlm-response.json').write_text(json.dumps(reply, indent=2)+'\n')
    text = reply['choices'][0]['message']['content'].strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    start = text.find('{')
    decision, _ = json.JSONDecoder().raw_decode(text[start:] if start >= 0 else text)
    if 'unavailable' in decision:
        raise ValueError('vlm_grounding_unavailable: '+str(decision['unavailable']))
    camera = decision.get('camera', 'scene_alt')
    if camera not in available:
        raise ValueError('invalid_vlm_camera_choice')
    for role in ('object', 'source', 'destination'):
        role_camera = decision.get(role+'_camera', camera) if role != 'object' else camera
        if role_camera not in available:
            raise ValueError('invalid_vlm_camera_choice')
        GroundingHint('current-image', role_camera, tuple(decision[role]), role)
    yaw = float(decision['grasp_yaw_degrees'])
    fraction = float(decision.get('grasp_height_fraction', .5))
    if not np.isfinite(yaw) or not 0 <= yaw <= 360:
        raise ValueError('invalid_vlm_grasp_yaw')
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('invalid_vlm_grasp_height')
    if decision.get('approach', 'from_viewer') not in {'direct', 'from_viewer'}:
        raise ValueError('invalid_vlm_approach')
    (output/'vlm-decision.json').write_text(json.dumps(decision, indent=2)+'\n')
    return decision
