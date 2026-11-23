"""Clearance updates preserve validated immutable geometry and invalidate bounds."""

from dataclasses import replace

import numpy as np
import pytest

from emet.motion.observed_solids import WorldSolid
from emet.motion.placement_geometry import box_corners


@pytest.mark.parametrize('kind, dimensions', [('box', (.02, .03, .04)), ('sphere', (.02,)),
                                            ('cylinder', (.02, .04)), ('convex', ())])
def test_offset_only_update_matches_full_validation_with_cached_bounds(kind, dimensions):
    kwargs = {'vertices': box_corners([[-.02]*3, [.02]*3]), 'material_calibration_id': 'shape'} if kind == 'convex' else {}
    solid = WorldSolid(kind, dimensions, np.eye(4), .001, **kwargs)
    original = solid.bounds.copy()
    updated = solid.with_offset(-.003)
    expected = replace(solid, offset_m=-.003)
    np.testing.assert_allclose(updated.bounds, expected.bounds)
    np.testing.assert_allclose(updated.lower_distance([[0., 0., 0.], [.1, .1, .1]]),
                               expected.lower_distance([[0., 0., 0.], [.1, .1, .1]]))
    assert solid.offset_m == .001
    np.testing.assert_array_equal(solid.bounds, original)
    assert not updated.world_from_local.flags.writeable


@pytest.mark.parametrize('offset', [np.nan, np.inf, -np.inf])
def test_offset_only_update_rejects_nonfinite_clearance(offset):
    with pytest.raises(ValueError, match='invalid_observed_solid'):
        WorldSolid('sphere', (.02,), np.eye(4)).with_offset(offset)
