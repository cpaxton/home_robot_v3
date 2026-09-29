"""Scene-backed admission checks for oracle/latch clutter controls.

These certify the benchmark's explicit disk navigation model, not physical grasp
or continuous collision-free manipulation. Executed reference traces are required
separately before a resolved fixture enters the scored registry.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from emet.eval.tamp_clutter import nav_path_open_around_disks, placement_obstacle_disks


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def certificate_error(fixture, *, scene_fingerprint, execution_mode, clearance_m, start):
    saved = dict(fixture)
    expected = saved.pop('sha256', None)
    if saved.get('schema') != 1 or expected != fingerprint(saved) or not saved.get('reference_success'):
        return 'invalid_fixture_certificate'
    if saved.get('scene_fingerprint') != scene_fingerprint:
        return 'fixture_scene_mismatch'
    if saved.get('execution_mode') != execution_mode:
        return 'fixture_execution_mode_mismatch'
    if abs(saved['clearance_m'] - clearance_m) > 1e-9:
        return 'fixture_clearance_mismatch'
    if np.linalg.norm(np.asarray(start) - np.asarray(saved['robot_start_xy'])) > .05:
        return 'fixture_spawn_mismatch'
    return None


def moved_placements(placements, updates):
    result = copy.deepcopy(placements)
    for body, pos in updates.items():
        previous = np.asarray(result[body]['pos'], dtype=float)
        pos = np.asarray(pos, dtype=float)
        if result[body].get('bounds') is not None:
            result[body]['bounds'] = (np.asarray(result[body]['bounds']) + pos - previous).tolist()
        result[body]['pos'] = pos.tolist()
    return result


def fixture_geometry(placements, *, start, goal, bodies, bin_body, clearance_m):
    """Require clutter necessity and a free route after the declared relocations.

    Keep every unselected obstacle, including the destination furniture. Do not
    erase occupied start/goal cells or ignore all geometry near the destination.
    """
    if bin_body not in placements:
        return {'accepted': False, 'reason': 'missing_receptacle'}
    if not bodies or len(set(bodies)) != len(bodies) or bin_body in bodies or any(b not in placements for b in bodies):
        return {'accepted': False, 'reason': 'invalid_clutter_identities'}
    initial = placement_obstacle_disks(placements)
    # The oracle place action uses the receptacle's center plus 2 cm in Z.
    destination = np.asarray(placements[bin_body]['pos'], dtype=float) + [0, 0, .02]
    after = moved_placements(placements, dict.fromkeys(bodies, destination))
    final = placement_obstacle_disks(after)
    if goal is None:
        far = all(np.linalg.norm(np.asarray(placements[b]['pos'])[:2] - destination[:2]) > .5 for b in bodies)
        return {'accepted': bool(far), 'reason': 'ok' if far else 'cleanup_already_satisfied',
                'relocation_positions': {b: destination.tolist() for b in bodies}}
    initial_open, before_probe = nav_path_open_around_disks(
        start, goal, initial, clearance_m=clearance_m, strict_endpoints=True,
    )
    final_open, after_probe = nav_path_open_around_disks(
        start, goal, final, clearance_m=clearance_m, strict_endpoints=True,
    )
    reason = 'ok'
    if before_probe.get('invalid_endpoint'):
        reason = 'occupied_initial_endpoint'
    elif initial_open:
        reason = 'clutter_not_required'
    elif not final_open:
        reason = 'route_blocked_after_relocation'
    return {'accepted': reason == 'ok', 'reason': reason, 'before': before_probe, 'after': after_probe,
            'relocation_positions': {b: destination.tolist() for b in bodies}}


def resolve_fixture(ep, placements, categories, start, *, bin_body, candidates, clearance_m=.22):
    """Bounded deterministic construction using actual scene identities.

    Candidates are furniture body IDs selected from scene metadata, never a
    requested category assumed to exist. A failure is a coverage gap, not a
    scored planner failure. No tested MCTS policy is called here.
    """
    from emet.eval.tamp_clutter import scatter_ring_targets

    if bin_body is None or bin_body not in placements:
        return None, {'reason': 'missing_receptacle'}
    movable = sorted(b for b, meta in categories.items()
                     if b in placements and not meta.get('static') and b != bin_body
                     and meta.get('cat') and b not in candidates)
    # Caller supplies only eligible movable bodies in categories.
    if len(movable) < ep.n_objects:
        return None, {'reason': 'insufficient_movable_objects', 'available': len(movable)}
    bodies = movable[:ep.n_objects]
    object_radii = {name: radius for _, radius, name in placement_obstacle_disks(placements, max_center_z_m=float('inf'))}
    static_disks = placement_obstacle_disks(placements, skip_bodies=bodies)
    goals = [(None, None)] if ep.mode == 'cleanup' else []
    if ep.mode == 'nav_goal':
        radii = {name: r for _, r, name in static_disks}
        ordered = sorted(candidates)
        if ordered:
            offset = int(ep.seed or 0) % len(ordered)
            ordered = ordered[offset:] + ordered[:offset]
        for body in ordered:
            center = np.asarray(placements[body]['pos'])[:2]
            if np.linalg.norm(center - start) > 9:
                continue
            radius = radii.get(body, .08) + clearance_m + .15
            for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
                point = center + radius * np.array([np.cos(theta), np.sin(theta)])
                if np.linalg.norm(point - start) > max(1., ep.success_radius_m * 2):
                    goals.append((body, point))
    rng = np.random.default_rng(ep.seed or 0)
    rejected = []
    for landmark, goal in goals[:96]:
        for radius in dict.fromkeys((ep.scatter_radius_m, .45, .5, .55)):
            targets = scatter_ring_targets(start, goal, ep.n_objects, radius_m=radius, rng=rng,
                                           radius_jitter=.01, angle_jitter_rad=.01)
            # Avoid constructing overlapping proxy objects or embedding them in furniture.
            if any(np.linalg.norm(a-b) < object_radii[bodies[i]] + object_radii[bodies[j]] + .02
                   for i, a in enumerate(targets) for j, b in enumerate(targets) if i < j):
                continue
            if any(np.linalg.norm(p-xy) <= r + object_radii[body] + .02
                   for body, p in zip(bodies, targets, strict=True) for xy, r, _ in static_disks):
                continue
            updates = {}
            for body, xy in zip(bodies, targets, strict=True):
                info = placements[body]
                # Preserve the object's extent above the floor when bounds are available.
                z = ep.floor_z_m
                if info.get('bounds') is not None:
                    z += max(0., float(info['pos'][2]) - float(np.asarray(info['bounds'])[0, 2]))
                updates[body] = [float(xy[0]), float(xy[1]), z]
            proposed = moved_placements(placements, updates)
            proof = fixture_geometry(proposed, start=start, goal=goal, bodies=bodies,
                                     bin_body=bin_body, clearance_m=clearance_m)
            if not proof['accepted']:
                rejected.append({'landmark': landmark, 'radius_m': radius, 'reason': proof['reason']})
                continue
            fixture = {
                'schema': 1, 'robot_start_xy': np.asarray(start).tolist(),
                'landmark_body': landmark, 'bin_body': bin_body,
                'goal_xy': None if goal is None else goal.tolist(), 'clearance_m': clearance_m,
                'clutter': [{'body': b, 'cat': categories[b]['cat'], 'pos': updates[b],
                             'quat': placements[b].get('quat')} for b in bodies],
                'geometry': proof, 'execution_mode': ep.resolved_manip_mode(),
            }
            return fixture, {'reason': 'constructed', 'rejections': rejected}
    return None, {'reason': 'no_valid_fixture_within_budget', 'rejections': rejected}
