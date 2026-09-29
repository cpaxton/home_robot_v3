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
