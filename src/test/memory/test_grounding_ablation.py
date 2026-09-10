# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from build_grounding_dataset import object_geom_ids
from grounding_ablation import BoxAblation, expand_box


def test_expansion_clips_and_rejects_malformed_boxes():
    assert expand_box([100, 200, 500, 600], 0.25) == [0, 100, 600, 700]
    for box in (None, [0, 0, 0, 5], [True, 0, 50, 50], [0, 0, 1001, 20]):
        with pytest.raises(ValueError):
            expand_box(box, 0.25)


def test_verifier_rejection_stands_and_does_not_include_original_reason():
    client = Mock(side_effect=['{"verified":true,"box":[100,100,500,500],"reason":"SECRET"}', '{"verified":false}'])
    ablation = BoxAblation(client, np.zeros((80, 60, 3), dtype=np.uint8), "cup", "verify")
    assert '"verified": false' in ablation(["locate"])
    assert "SECRET" not in client.call_args.args[0][0]
    assert len(client.call_args.args[0]) == 4


def test_object_mask_includes_descendants_not_neighbor():
    model = Mock(nbody=5, body_parentid=np.array([0, 0, 1, 2, 0]), geom_bodyid=np.array([1, 2, 3, 4]))
    assert object_geom_ids(model, 1).tolist() == [0, 1, 2]
