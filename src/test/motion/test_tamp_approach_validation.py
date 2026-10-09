"""The assisted validation adapter preserves identities and does not alter scores."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[3] / 'scripts/validate_tamp_approaches.py'
spec = importlib.util.spec_from_file_location('approach_validation', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    return ({'fixture': {'clutter': [{'body': 'exact_apple', 'cat': 'apple'}]}},
            {'results': [{'object': 'apple', 'candidates': [
                {'xyt': [0, 0, 0], 'contacts': ['wall'], 'ik': {'success': True}},
                {'xyt': [1, 0, 0], 'contacts': [], 'ik': {'success': False}},
                {'xyt': [2, 0, 0], 'contacts': [], 'ik': {'success': True}},
            ]}]})


def test_selection_and_injection_preserve_candidate_identity():
    episode, audit = inputs()
    selected = module.select_approaches(episode, audit)
    assert selected == {'exact_apple': [2, 0, 0]}
    original = [{'object_gt_body': 'exact_apple', 'receptacle_gt_body': 'exact_bin'}]
    result = module.with_approaches(original, selected)
    assert result[0]['receptacle_gt_body'] == 'exact_bin'
    assert result[0]['approach_pose'] == [2, 0, 0]
    assert 'approach_pose' not in original[0]
    with pytest.raises(ValueError, match='unselected'):
        module.with_approaches([{'object_gt_body': 'another_apple'}], selected)


def test_missing_or_ambiguous_evidence_fails_closed():
    episode, audit = inputs()
    with pytest.raises(ValueError, match='insufficient'):
        module.select_approaches(episode, audit, rank=1)
    audit['results'].append(audit['results'][0])
    with pytest.raises(ValueError, match='ambiguous'):
        module.select_approaches(episode, audit)
