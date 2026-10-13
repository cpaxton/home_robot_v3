"""A floor fixture must request independently movable scene instances."""
import importlib.util
import sys
from pathlib import Path


def test_scatter_selection_deduplicates_parts_and_excludes_receptacle_assembly():
    path = Path(__file__).resolve().parents[3] / 'scripts/eval_tamp_clutter.py'
    spec = importlib.util.spec_from_file_location('clutter_instance_test_runner', path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    try:
        spec.loader.exec_module(runner)
        placements = {name: {'pos': [0, 0, 0]} for name in
                      ('part_a', 'part_b', 'object_c', 'bin_base', 'bin_lid')}
        metadata = {'objects': {
            'assembly': {'category': 'object', 'is_static': False,
                         'name_map': {'bodies': {'part_a': {}, 'part_b': {}}}},
            'independent': {'category': 'object', 'is_static': False,
                            'name_map': {'bodies': {'object_c': {}}}},
            'receptacle': {'category': 'object', 'is_static': False,
                           'name_map': {'bodies': {'bin_base': {}, 'bin_lid': {}}}},
        }}
        categories = runner._placement_category_map(placements, metadata)
        selected = runner._pickable_bodies(placements, 8, cat_map=categories,
                                            exclude_bodies={'bin_base'})
        assert selected == ['object_c', 'part_a']
        assert len({categories[b]['instance'] for b in selected}) == len(selected)
        # Keep independent instances of the same category; never dedup by label.
        assert categories['part_a']['cat'] == categories['object_c']['cat']
    finally:
        sys.modules.pop(spec.name, None)
