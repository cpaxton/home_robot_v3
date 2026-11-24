import pytest

from emet.controller.task.tamp.manipulation_request import ManipulationRequest


def test_semantic_default_and_owned_hint_copy():
    request = ManipulationRequest.parse(' pear ', 'sink')
    assert request.object_name == 'pear' and request.hint_mode == 'prefer'
    hints = {'approach': {'direction': [0, 0, -1]}}
    request = ManipulationRequest.parse('pear', 'sink', hints=hints, hint_mode='strict')
    hints['approach']['direction'][0] = 1
    assert request.hints['approach']['direction'] == [0, 0, -1]


@pytest.mark.parametrize('options', [
    {'hints': {'grasp': {'angle': float('nan')}}},
    {'hints': {'placement': {'point': [float('inf'), 0, 0]}}},
    {'hint_mode': 'ignore'}, {'hints': {'disable_collisions': {}}},
    {'constraints': {'protected_objects': 'cube'}},
    {'constraints': {'keep_base_stationary': 'yes'}},
    {'constraints': {'allow_teleport': True}},
])
def test_invalid_options_do_not_become_motion_authorization(options):
    with pytest.raises(ValueError):
        ManipulationRequest.parse('pear', 'sink', **options)
