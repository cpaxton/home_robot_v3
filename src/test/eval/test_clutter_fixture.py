"""Admission must distinguish a solvable clutter task from a broken fixture."""
import numpy as np

from emet.eval.clutter_fixture import fingerprint, fixture_geometry, moved_placements, resolve_fixture
from emet.eval.tamp_clutter import ClutterEpisode


def room():
    scene = {'bin': {'pos': [-2., 1., .1]}, 'table': {'pos': [2., 0., .5]}}
    for i, angle in enumerate(np.linspace(0, 2*np.pi, 8, endpoint=False)):
        scene[f'item{i}'] = {'pos': [.5*np.cos(angle), .5*np.sin(angle), .02]}
    return scene


def check(scene, **kw):
    args = {'start': [0,0], 'goal': [1.5,0], 'bodies': [f'item{i}' for i in range(8)],
            'bin_body': 'bin', 'clearance_m': .22}
    args.update(kw)
    return fixture_geometry(scene, **args)


def test_only_clutter_blocks_initial_route_and_declared_relocations_open_it():
    proof = check(room())
    assert proof['accepted'] and proof['before']['blocked'] and not proof['after']['blocked']
    assert len(proof['relocation_positions']) == 8


def test_existing_furniture_is_not_erased_near_goal():
    result = check(room(), goal=[2.,0])
    assert not result['accepted'] and result['reason'] == 'occupied_initial_endpoint'


def test_removing_clutter_cannot_fix_a_permanently_blocked_goal():
    scene = room()
    for i, a in enumerate(np.linspace(0, 2*np.pi, 12, endpoint=False)):
        scene[f'fixed{i}'] = {'pos': [4+.5*np.cos(a), .5*np.sin(a), .1]}
    result = check(scene, goal=[4.,0])
    assert not result['accepted'] and result['reason'] == 'route_blocked_after_relocation'


def test_already_clear_task_and_embedded_start_are_not_scored():
    scene = room()
    scene = moved_placements(scene, {f'item{i}': [-2,2+i*.2,.02] for i in range(8)})
    assert check(scene)['reason'] == 'clutter_not_required'
    scene['item0']['pos'] = [0,0,.02]
    assert check(scene)['reason'] == 'occupied_initial_endpoint'


def test_missing_bin_and_duplicate_objects_are_rejected():
    scene = room()
    del scene['bin']
    assert check(scene)['reason'] == 'missing_receptacle'
    assert check(room(), bodies=['item0','item0'])['reason'] == 'invalid_clutter_identities'


def test_cleanup_requires_work_and_never_moves_the_receptacle():
    assert check(room(), goal=None)['accepted']
    assert not check(room(), goal=None, bodies=['bin'])['accepted']
    scene = moved_placements(room(), {f'item{i}': [-2,1,.1] for i in range(8)})
    assert check(scene, goal=None)['reason'] == 'cleanup_already_satisfied'


def test_construction_uses_actual_scene_and_is_reproducible():
    scene = room()
    cats = {b: {'cat': 'bottle', 'static': False} for b in scene if b.startswith('item')}
    cats['table'] = {'cat': 'table', 'static': True}
    ep = ClutterEpisode('test','S1','test.yaml','stretch','nav_goal',8, seed=4, scatter_radius_m=.5)
    a, _ = resolve_fixture(ep, scene, cats, np.zeros(2), bin_body='bin', candidates=['table'])
    b, _ = resolve_fixture(ep, scene, cats, np.zeros(2), bin_body='bin', candidates=['table'])
    assert a is not None and a == b and a['landmark_body'] == 'table'
    assert a['geometry']['accepted']
    old = fingerprint(a)
    a['clutter'][0]['pos'][0] += .1
    assert fingerprint(a) != old


def test_moving_bounds_does_not_mutate_scene():
    scene = {'a': {'pos': [1,2,3], 'bounds': [[0,1,2],[2,3,4]]}}
    moved = moved_placements(scene, {'a': [2,3,4]})
    assert moved['a']['bounds'] == [[1,2,3],[3,4,5]]
    assert scene['a']['pos'] == [1,2,3]


def test_replay_rejects_changed_scene_backend_start_and_unsigned_witness():
    from emet.eval.clutter_fixture import certificate_error

    fixture = {'schema': 1, 'scene_fingerprint': 'scene', 'execution_mode': 'latch',
               'clearance_m': .22, 'robot_start_xy': [0,0], 'reference_success': True}
    fixture['sha256'] = fingerprint(fixture)
    args = {'scene_fingerprint': 'scene', 'execution_mode': 'latch', 'clearance_m': .22, 'start': [0,0]}
    assert certificate_error(fixture, **args) is None
    for key, value, reason in [('scene_fingerprint','other','fixture_scene_mismatch'),
                               ('execution_mode','sim','fixture_execution_mode_mismatch'),
                               ('clearance_m',.1,'fixture_clearance_mismatch'),
                               ('start',[1,0],'fixture_spawn_mismatch')]:
        assert certificate_error(fixture, **{**args, key: value}) == reason
    fixture['reference_success'] = False
    assert certificate_error(fixture, **args) == 'invalid_fixture_certificate'


def test_reference_executes_grounded_steps_without_mcts(monkeypatch):
    from types import SimpleNamespace

    import emet.controller.task.tamp.task_search as search
    import emet.memory.graph_eqa.sim_ground_truth_graph as gt
    from emet.controller.task.tamp.clutter_chain import plan_clear_clutter

    scene = {'item': {'pos': [0,0,.02], 'cat': 'apple'},
             'ashcan': {'pos': [2,1,.2], 'cat': 'ashcan'}}
    monkeypatch.setattr(gt, 'read_sim_object_placements', lambda session: scene)
    monkeypatch.setattr(search, 'resolve_scene_grasps', lambda *a, **kw: [])
    monkeypatch.setattr(search, 'plan_pick_place_mcts', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('MCTS used for admission')))

    def ground(*args, **kwargs):
        return search.TaskPlan(steps=[search.TaskPlanStep('place', {})], object_body='item',
                               receptacle_body='ashcan', success=True)

    def execute(robot, plan, **kwargs):
        scene['item']['pos'] = list(scene['ashcan']['pos'])
        plan.completed_ops = ['approach','grasp','place']
        return plan

    monkeypatch.setattr(search, 'plan_pick_place', ground)
    monkeypatch.setattr(search, 'execute_task_plan', execute)
    result = plan_clear_clutter(SimpleNamespace(get_emet_session=lambda: {}),
                                objects=[{'object_gt_body':'item','object_query':'apple'}],
                                mode='cleanup', bin_query='ashcan', manip_mode='sim', reference=True)
    assert result['reference_execution'] and result['task_success']
    assert result['execution_trace'][0]['completed_ops'] == ['approach','grasp','place']


def test_teleport_escape_is_not_a_clutter_solution():
    from emet.eval.clutter_fixture import score_fixture_state

    fixture = {'bin_body': 'bin', 'robot_start_xy': [0,0], 'goal_xy': [1.5,0],
               'clearance_m': .22, 'clutter': [{'body': f'item{i}'} for i in range(8)]}
    result = score_fixture_state(room(), fixture, mode='nav_goal', base_xy=[1.5,0], success_radius_m=.5)
    assert result['goal_reached'] and not result['task_success'] and not result['nav_path_open']
    cleared = moved_placements(room(), {f'item{i}': [-2,1,.02] for i in range(8)})
    result = score_fixture_state(cleared, fixture, mode='nav_goal', base_xy=[1.5,0], success_radius_m=.5)
    assert result['task_success'] and result['n_relocated'] == 8
    result = score_fixture_state(cleared, fixture, mode='nav_goal', base_xy=[0,0], success_radius_m=.5)
    assert result['nav_path_open'] and not result['task_success']
    cleared['item0']['pos'] = [0,0,.02]
    assert not score_fixture_state(cleared, fixture, mode='cleanup', base_xy=[0,0], success_radius_m=.5)['task_success']


def test_simulator_arrays_round_trip_certificate_without_hash_drift():
    import json

    import pytest

    from emet.eval.clutter_fixture import json_value
    original = {'poses': np.array([[1., 2., 3.]]), 'count': np.int64(1), 'ok': np.bool_(True)}
    saved = json.loads(json.dumps(json_value(original)))
    assert saved == {'poses': [[1., 2., 3.]], 'count': 1, 'ok': True}
    assert fingerprint(saved) == fingerprint(original)
    with pytest.raises(ValueError):
        fingerprint({'pose': np.array([np.nan])})


def test_replay_checks_height_and_orientation_including_quaternion_sign():
    from emet.eval.clutter_fixture import pose_reproduced
    pose = {'pos': [0., 0., .1], 'quat': [1., 0., 0., 0.]}
    assert pose_reproduced(pose, pose)
    assert pose_reproduced({**pose, 'quat': [-1., 0., 0., 0.]}, pose)
    assert not pose_reproduced({**pose, 'pos': [0., 0., .2]}, pose)
    assert not pose_reproduced({**pose, 'quat': [0., 0., 0., 1.]}, pose)
    assert not pose_reproduced(None, pose)


def test_certificate_task_contract_changes_when_scoring_is_relaxed():
    from dataclasses import replace

    from emet.eval.clutter_fixture import task_contract
    ep = ClutterEpisode('test', 'S1', 'test.yaml', 'stretch', 'cleanup', 3)
    assert task_contract(ep) != task_contract(replace(ep, success_radius_m=1.))
    assert task_contract(ep) != task_contract(replace(ep, mode='nav_goal'))
    assert task_contract(ep) != task_contract(replace(ep, n_objects=1))
