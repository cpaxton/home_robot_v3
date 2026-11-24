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
        f'Pick up the {object_name} and place it in {destination_name}. '
        f'The images are different cameras: {list(available)}. '
        'Find the requested object, not the white robot. Names may describe related visual subtypes. '
        'Use the labeled grid: x increases right, y down; boxes are [x0,y0,x1,y1] normalized 0..1000. '
        'camera: clearest object view. object: tight box around the actual object. '
        'source_camera and source: view and box of flat counter BESIDE the object, excluding the object and robot. '
        'destination_camera and destination: view and box of visible sink interior floor. '
        'Each box belongs to its named camera. They must not be copies of the object box. '
        'grasp_yaw_degrees: finger-closing direction clockwise from image right; close across the narrow dimension. '
        'grasp_height_fraction: 0 bottom, 1 top; choose a stable body section. '
        'approach: direct or from_viewer. '
        'Return ONLY compact JSON with these keys: camera, object, source_camera, source, '
        'destination_camera, destination, grasp_yaw_degrees, grasp_height_fraction, approach. '
        'No explanation. If the target is absent or ambiguous return {"unavailable":"brief reason"}.'
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


def refine_object_hint(rgb, decision, *, endpoint, model, object_name, output, estimate_grasp_site=False):
    """Refine a coarse semantic region in a magnified current-image crop."""
    output.mkdir(parents=True, exist_ok=True)
    height, width = rgb.shape[:2]
    box = np.asarray(decision['object'], float)/1000
    center = (box[:2]+box[2:])/2
    half = np.maximum((box[2:]-box[:2])*1.5, .15)
    low = np.floor(np.maximum(0, center-half)*[width, height]).astype(int)
    high = np.ceil(np.minimum(1, center+half)*[width, height]).astype(int)
    crop = Image.fromarray(rgb).crop((*low, *high))
    crop.save(output/'crop-raw.png')
    crop = crop.resize((640, 480))
    draw = ImageDraw.Draw(crop)
    for value in range(100, 1000, 100):
        x, y = round(640*value/1000), round(480*value/1000)
        draw.line((x, 0, x, 480), fill=(100, 160, 170))
        draw.line((0, y, 640, y), fill=(100, 160, 170))
        draw.text((x+2, 2), str(value), fill='yellow', stroke_width=1, stroke_fill='black')
        draw.text((2, y+2), str(value), fill='yellow', stroke_width=1, stroke_fill='black')
    crop.save(output/'crop-grid.png')
    buffer = io.BytesIO()
    crop.save(buffer, format='PNG')
    prompt = (f'Locate the {object_name} in this magnified image. Ignore the white robot. '
        'Use the numbered grid: x right, y down, normalized 0..1000. '
        'Choose a solid grasp site, avoiding holes, thin tips and unstable crowns. Choose a body height below the crown and close across the narrow dimension. Return ONLY JSON: {"object":[x0,y0,x1,y1],"grasp_point":[x,y],"grasp_yaw_degrees":angle,"grasp_height_fraction":fraction}. object encloses the whole target; grasp_point is the chosen solid site in this crop. angle is closing direction clockwise from image right (0..360); fraction is height from bottom (0) to top (1). '
        'If it is absent or ambiguous return {"unavailable":"reason"}. No explanation.')
    if not estimate_grasp_site:
        prompt = (f'Locate the {object_name} in this magnified image. Ignore the white robot. '
            'Use the numbered grid: x right, y down, normalized 0..1000. '
            'Return ONLY {"object":[x0,y0,x1,y1]} tightly enclosing the target object. '
            'If it is absent or ambiguous return {"unavailable":"reason"}. No explanation.')
    payload = {'model': model, 'temperature': 0, 'max_tokens': 256,
        'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+
                base64.b64encode(buffer.getvalue()).decode()}}]}]}
    (output/'request.json').write_text(json.dumps(payload)+'\n')
    request = urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=90) as response:
        reply = json.load(response)
    (output/'response.json').write_text(json.dumps(reply, indent=2)+'\n')
    text = reply['choices'][0]['message']['content']
    refined, _ = json.JSONDecoder().raw_decode(text[text.find('{'):])
    if 'unavailable' in refined:
        raise ValueError('vlm_refinement_unavailable: '+str(refined['unavailable']))
    GroundingHint('current-crop', decision.get('camera', 'scene_alt'), tuple(refined['object']), 'object')
    coords = np.asarray(refined['object'], float).reshape(2, 2)/1000
    mapped = ((low+coords*(high-low))/[width, height]*1000).reshape(4).tolist()
    result = {**decision, 'object': mapped, 'coarse_object': decision['object']}
    if 'grasp_point' in refined:
        point = np.asarray(refined['grasp_point'], float)
        if point.shape != (2,) or not np.isfinite(point).all() or np.any((point < 0) | (point > 1000)):
            raise ValueError('invalid_vlm_grasp_point')
        result['grasp_point'] = ((low+point/1000*(high-low))/[width, height]*1000).tolist()
    if 'grasp_height_fraction' in refined:
        fraction = float(refined['grasp_height_fraction'])
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('invalid_vlm_grasp_height')
        result['grasp_height_fraction'] = fraction
    if 'grasp_yaw_degrees' in refined:
        yaw = float(refined['grasp_yaw_degrees'])
        if not np.isfinite(yaw) or not 0 <= yaw <= 360:
            raise ValueError('invalid_vlm_grasp_yaw')
        direction = np.array([np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))])*(high-low)/[640, 480]
        result['grasp_yaw_degrees'] = float(np.rad2deg(np.arctan2(direction[1], direction[0])) % 360)
    (output/'decision.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
