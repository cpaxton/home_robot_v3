# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Simulation-only one-shot depth pick diagnostic; no continuous object tracker."""
import json
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from emet.controller.task.tamp.observed_capture import capture_arrays
from emet.controller.task.tamp.transfer_grounding import select_transfer
from emet.motion.mujoco_arm_ik import solve_pose_ik
from emet.robots import get_robot_backend
from emet.robots.pandaomron.observed import synchronize_pandaomron


def top_down_target(points):
    """Center a top-down pick on the visible top, not the camera-facing side."""
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 25 or not np.isfinite(points).all():
        raise ValueError('invalid_grasp_cloud')
    target = np.median(points, axis=0)
    top = points[points[:, 2] >= points[:, 2].max()-.01]
    if len(top) >= 5:
        target[:2] = np.median(top[:, :2], axis=0)
    target[2] += .015
    return target


def front_interior_floor(observation, mask, *, extra_observations=()):
    """Choose a front interior point from locally associated measured floor depth."""
    world = capture_arrays(observation)[2]
    points = world[mask]
    if len(points) < 25:
        raise ValueError('destination_geometry_unavailable')
    low, high = points.min(0), points.max(0)
    height = np.median(points[:, 2])
    pieces = [points]
    for other in extra_observations:
        cloud = capture_arrays(other)[2]
        valid = (np.isfinite(cloud).all(axis=-1) & (np.abs(cloud[..., 2]-height) <= .008)
                 & np.all(cloud[..., :2] >= low[:2]-.15, axis=-1)
                 & np.all(cloud[..., :2] <= high[:2]+.15, axis=-1))
        pieces.append(cloud[valid])
    points = np.concatenate(pieces)
    # Multiple cameras oversample different parts of the same floor. Balance
    # spatial coverage before estimating an interior point; pixel density is not area.
    _, unique = np.unique(np.floor(points[:, :2]/.005).astype(np.int64), axis=0, return_index=True)
    points = points[unique]
    center = np.median(points[:, :2], axis=0)
    _, axes = np.linalg.eigh(np.cov(points[:, :2], rowvar=False))
    toward = observation.camera_pose[:2, 3]-center
    if np.linalg.norm(toward) < .1:
        toward = observation.camera_pose[:2, 1]
    axis = axes[:, np.argmax(np.abs(axes.T @ toward))]
    axis *= 1 if axis @ toward >= 0 else -1
    front = (points[:, :2]-center) @ axis
    lateral = (points[:, :2]-center) @ np.array([-axis[1], axis[0]])
    lateral_mid = (lateral.min()+lateral.max())/2
    lateral_half = (lateral.max()-lateral.min())*.2
    chosen = ((front >= np.quantile(front, .75)) & (lateral >= lateral_mid-lateral_half)
              & (lateral <= lateral_mid+lateral_half))
    if chosen.sum() < 10:
        raise ValueError('destination_interior_unavailable')
    return np.median(points[chosen], axis=0)


def estimated_grasp_section(observations, selection, fraction):
    """Fuse local object returns and center the body section requested by the VLM."""
    reference = observations[selection.object_camera]
    points = capture_arrays(reference)[2][selection.object_mask]
    low, high = points.min(0)-.03, points.max(0)+.03
    support = capture_arrays(observations[selection.source_camera])[2][selection.source_mask]
    support_z = np.median(support[:, 2])
    pieces = [points]
    for name, observation in observations.items():
        if name == selection.object_camera:
            continue
        cloud = capture_arrays(observation)[2]
        valid = (np.all((cloud >= low) & (cloud <= high), axis=-1)
                 & (cloud[..., 2] > support_z+.006))
        pieces.append(cloud[valid])
    points = np.concatenate(pieces)
    height = points[:, 2].min()+fraction*np.ptp(points[:, 2])
    section = points[np.abs(points[:, 2]-height) <= max(.005, .2*np.ptp(points[:, 2]))]
    if len(section) < 5:
        raise ValueError('grasp_section_unavailable')
    xy = (section[:, :2].min(0)+section[:, :2].max(0))/2
    return points, np.r_[xy, height]


def score_diagnostic(diagnostic, physics, *, full_transfer):
    """Task success is distinct from diagnostic zero-contact statistics."""
    lifted = 'lift' in diagnostic.get('events', []) and physics['lift_with_two_pads_m'] >= .04
    picked = bool(lifted and (full_transfer or diagnostic.get('completed')))
    transferred = bool(diagnostic.get('completed') and lifted and physics['upstream_success']
                       and physics['basin_contains_tongs'] and physics['final_object_speed_mps'] < .02)
    zero_contact = not any(physics[k] for k in ('unintended_contact_frames',
        'unintended_touch_frames', 'unintended_self_contact_frames'))
    return {'pick_passed': picked, 'physical_pick_passed': bool(lifted),
            'physical_transfer_passed': transferred if full_transfer else None,
            'clean_transfer_passed': transferred and zero_contact if full_transfer else None,
            'zero_contact': zero_contact,
            'task_passed': transferred if full_transfer else picked}


def run_pick(client, output, profile_path, *, model_endpoint=None, model_name=None, object_name="salt shaker",
             full_transfer=False, destination_name="sink"):
    if (client.get_emet_session() or {}).get('runtime_kind') != 'robocasa_native':
        raise ValueError('simulation_diagnostic_only')
    profile = json.loads(profile_path.read_text())
    obs = client.get_observation()
    observations = {'scene': obs}
    for name, view in (obs.auxiliary_rgbd_views or {}).items():
        observations[name] = SimpleNamespace(**view)
    observations['wrist'] = SimpleNamespace(rgb=obs.ee_rgb, depth=obs.ee_depth,
        camera_K=obs.ee_camera_K, camera_pose=obs.ee_camera_pose)
    sample = SimpleNamespace(capture=SimpleNamespace(observation_id='one-shot'), observations=observations)
    from emet.controller.manipulation.native_inputs import ObservationUnavailable

    decision = None
    attempts = []
    feedback = None
    for attempt in range(3 if model_endpoint else 1):
        attempt_output = output if attempt == 0 else output/f'vlm-recovery-{attempt}'
        attempt_output.mkdir(exist_ok=True)
        try:
            if model_endpoint:
                from emet.eval.vlm_pick_hint import query_pick_hint

                decision = query_pick_hint(observations['scene_alt'].rgb, endpoint=model_endpoint,
                    model=model_name, object_name=object_name, output=attempt_output,
                    destination_name=destination_name, feedback=feedback,
                    views={name: view.rgb for name, view in observations.items()})
                profile['grounding'] = {role: {'camera': (decision.get(role+'_camera', decision.get('camera', 'scene_alt'))
                    if role != 'object' else decision.get('camera', 'scene_alt')), 'box': decision[role]}
                                        for role in ('object', 'source', 'destination')}
                profile['visual_hint_source'] = 'runtime_vlm'
            selected = select_transfer(sample, profile=profile, grow_object_region=decision is not None)
            attempts.append({'attempt': attempt+1, 'accepted': True})
            break
        except (ValueError, ObservationUnavailable) as error:
            feedback = str(error)
            attempts.append({'attempt': attempt+1, 'accepted': False, 'reason': feedback})
    else:
        (output/'open-loop-pick.json').write_text(json.dumps({
            'completed': False, 'events': [], 'reason': feedback, 'grounding_attempts': attempts,
            'runtime_vlm_calls': len(attempts) if model_endpoint else 0,
            'vlm_used': bool(model_endpoint), 'object_cad': False,
        }, indent=2)+'\n')
        return
    points = capture_arrays(observations[selected.object_camera])[2][selected.object_mask]
    target = top_down_target(points)
    if decision:
        points, target = estimated_grasp_section(observations, selected,
            float(decision.get('grasp_height_fraction', .5)))
    yaw = np.pi
    if decision:
        image_yaw = np.deg2rad(decision['grasp_yaw_degrees'])
        closing = observations[selected.object_camera].camera_pose[:3, :3] @ np.array(
            [np.cos(image_yaw), np.sin(image_yaw), 0.])
        if np.linalg.norm(closing[:2]) < .1:
            raise ValueError('vlm_grasp_direction_not_horizontal')
        yaw = np.arctan2(closing[1], closing[0])
    rotation = np.array([[np.cos(yaw), np.sin(yaw), 0.],
                         [np.sin(yaw), -np.cos(yaw), 0.], [0., 0., -1.]])
    model = mujoco.MjModel.from_xml_path(str(output/'robot-only.xml'))
    data = mujoco.MjData(model)
    arm = get_robot_backend('pandaomron').bind_observed_arm(client, model)
    synchronize_pandaomron(obs, model, data)
    addresses = [model.joint(n).qposadr[0] for n in arm.chain.joint_names]
    if decision:
        ee = data.body(arm.chain.ee_body)
        pad_centers = np.array([data.geom(f'gripper0_right_finger{i}_pad_collision').xpos for i in (1, 2)])
        pad_local = (pad_centers.mean(axis=0)-ee.xpos) @ ee.xmat.reshape(3, 3)
        target[2] -= (rotation @ pad_local)[2]
    limits = np.array([model.joint(n).range for n in arm.chain.joint_names])
    events = []
    executed = []

    def record_event(name):
        events.append(name)
        (output/'open-loop-progress.json').write_text(json.dumps(events)+'\n')

    def move(position, orientation, *, approximate_waypoint=False, position_tolerance_m=.008):
        arm.synchronize(data)
        origin = data.body(arm.chain.ee_body).xpos.copy()
        for fraction in np.linspace(0., 1., max(2, int(np.ceil(np.linalg.norm(position-origin)/.02)))+1)[1:]:
            arm.synchronize(data)
            start = data.qpos[addresses].copy()
            waypoint = origin+(position-origin)*fraction
            result = solve_pose_ik(model, data, ee_body=arm.chain.ee_body,
                joint_names=arm.chain.joint_names, target_pos=waypoint,
                target_rotation=orientation, tol_m=.005, tol_rad=.05)
            if not result.success:
                raise RuntimeError('open_loop_ik_failed')
            goal = data.qpos[addresses].copy()
            steps = max(1, int(np.ceil(np.max(np.abs(goal-start)/np.r_[np.full(7, .1), .02]))))
            for q in np.linspace(start, goal, steps+1)[1:]:
                q = np.clip(q, limits[:, 0], limits[:, 1])
                arm.command_joints(q)
                deadline = time.monotonic()+4.
                while True:
                    time.sleep(.02)
                    arm.synchronize(data)
                    error = np.abs(data.qpos[addresses]-q)
                    body = data.body(arm.chain.ee_body)
                    task_error = float(np.linalg.norm(body.xpos-waypoint))
                    task_angle = float(np.arccos(np.clip((np.trace(
                        body.xmat.reshape(3, 3).T @ orientation)-1)/2, -1., 1.)))
                    if (np.max(error[:7]) < .015 and error[7] < .001) or (
                            task_error <= position_tolerance_m and task_angle <= .1):
                        break
                    if time.monotonic() > deadline:
                        body = data.body(arm.chain.ee_body)
                        distance = float(np.linalg.norm(body.xpos-position))
                        angle = float(np.arccos(np.clip((np.trace(
                            body.xmat.reshape(3, 3).T @ orientation)-1)/2, -1., 1.)))
                        if approximate_waypoint and distance <= .035 and angle <= .15:
                            receipt.setdefault('approximate_waypoints', []).append({
                                'target_m': position.tolist(), 'measured_m': body.xpos.tolist(),
                                'translation_error_m': distance, 'rotation_error_rad': angle})
                            executed.append(data.qpos[addresses].copy())
                            return
                        receipt['last_motion_error'] = {'joint_errors': error.tolist(),
                            'tool_position_error_m': task_error, 'tool_rotation_error_rad': task_angle,
                            'waypoint_m': waypoint.tolist(), 'measured_m': body.xpos.tolist()}
                        raise RuntimeError('open_loop_arrival_failed')
                executed.append(data.qpos[addresses].copy())

    receipt = {'mode': 'one_shot_depth_open_loop_transfer' if full_transfer else 'one_shot_depth_open_loop_pick',
        'object_cad': False, 'grasp_center_method': 'multi_view_height_section' if decision else 'visible_top_band_xy',
        'camera': selected.object_camera,
        'grasp_height_fraction': decision.get('grasp_height_fraction', .5) if decision else None,
        'approach': decision.get('approach', 'from_viewer') if decision else 'legacy_staging',
        'continuous_object_tracking': False,
        'grounding': 'runtime_vlm_regions' if decision else 'diagnostic_image_regions',
        'vlm_used': decision is not None, 'visual_hint_source': profile.get('visual_hint_source', 'diagnostic_regions'),
        'runtime_vlm_calls': len(attempts) if model_endpoint else 0,
        'grounding_attempts': attempts, 'environment_collision_checks': False,
        'target_world_m': target.tolist(), 'events': events,
        'instruction': f'Pick up {object_name} and place it in {destination_name}',
        'full_transfer_requested': full_transfer}
    try:
        client.open_gripper()
        home = data.body(arm.chain.ee_body).xpos.copy()
        current_rotation = data.body(arm.chain.ee_body).xmat.reshape(3, 3).copy()
        offset = np.array([.2, -.4, 0.])
        if decision:
            toward = observations[selected.object_camera].camera_pose[:2, 3]-target[:2]
            if np.linalg.norm(toward) < .1:
                toward = home[:2]-target[:2]
            offset = np.r_[toward/max(np.linalg.norm(toward), .1)*.3, 0.]
        route = [('pregrasp', target+[0, 0, .1], rotation), ('grasp', target, rotation)]
        if not decision or decision.get('approach', 'from_viewer') == 'from_viewer':
            route = [('departure', home+offset, current_rotation),
                     ('staging', target+[0, 0, .1]+offset, rotation), *route]
        for name, pos, rot in route:
            move(pos, rot, position_tolerance_m=.02 if name in {'departure', 'staging'} else .008)
            record_event(name)
        close = dict(profile['grasps'][0]['close_joints'])
        client.command_gripper_joints([close[name] for name in arm.gripper_control_joint_names])
        time.sleep(1.)
        record_event('close')
        arm.synchronize(data)
        receipt['gripper_after_close'] = {n: float(data.joint(n).qpos[0]) for n in arm.gripper_control_joint_names}
        move(target+[0, 0, .1], rotation)
        record_event('lift')
        receipt['gripper_after_lift'] = {n: float(data.joint(n).qpos[0]) for n in arm.gripper_control_joint_names}
        carry_start = max(0, len(executed)-1)
        time.sleep(1.)
        if full_transfer:
            landing = front_interior_floor(observations[selected.destination_camera], selected.destination_mask,
                extra_observations=[o for name, o in observations.items() if name != selected.destination_camera])
            tool_bottom = target[2]-points[:, 2].min()
            place = landing + [0., 0., tool_bottom+.005]
            counter = capture_arrays(observations[selected.source_camera])[2][selected.source_mask]
            height = float(np.median(counter[:, 2])+tool_bottom+.08)
            receipt['place_world_m'] = place.tolist()
            # Simple measured-height transport corridor, no object registration.
            for name, pos in [('carry_height', np.r_[target[:2], height]),
                    ('preplace_front', np.r_[place[:2]+[0., -.15], height]),
                    ('preplace', np.r_[place[:2], height]), ('place', place)]:
                move(pos, rotation, approximate_waypoint=name == 'preplace')
                record_event(name)
            receipt['gripper_before_release'] = {n: float(data.joint(n).qpos[0]) for n in arm.gripper_control_joint_names}
            client.open_gripper()
            record_event('release')
            # Retrace measured transport configurations rather than asking IK
            # for another posture beside the faucet after releasing the object.
            path = executed[carry_start:]
            receipt['retreat_method'] = 'reverse_measured_transport'
            receipt['retreat_configurations'] = len(path)
            for goal in reversed(path):
                goal = np.clip(goal, limits[:, 0], limits[:, 1])
                scratch = mujoco.MjData(model)
                scratch.qpos[:] = data.qpos
                scratch.qpos[addresses] = goal
                mujoco.mj_kinematics(model, scratch)
                reference = scratch.body(arm.chain.ee_body)
                arm.command_joints(goal)
                deadline = time.monotonic()+4.
                while True:
                    time.sleep(.02)
                    arm.synchronize(data)
                    error = np.abs(data.qpos[addresses]-goal)
                    body = data.body(arm.chain.ee_body)
                    distance = np.linalg.norm(body.xpos-reference.xpos)
                    angle = np.arccos(np.clip((np.trace(body.xmat.reshape(3, 3).T @
                        reference.xmat.reshape(3, 3))-1)/2, -1., 1.))
                    if (np.max(error[:7]) < .015 and error[7] < .001) or (distance <= .008 and angle <= .1):
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError('open_loop_retreat_failed')
            record_event('retreat')
            time.sleep(2.)
        receipt['completed'] = True
    except Exception as exc:
        receipt.update(completed=False, reason=str(exc))
    finally:
        (output/'open-loop-pick.json').write_text(json.dumps(receipt, indent=2)+'\n')
