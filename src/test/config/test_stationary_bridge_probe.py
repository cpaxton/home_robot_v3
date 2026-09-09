# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

import numpy as np


def test_receive_only_decoder_handles_deployed_camera_aliases(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    from probe_stationary_bridge import decode_cameras

    head = np.full((4, 5, 3), 90, dtype=np.uint8)
    wrist = np.zeros_like(head)
    images = decode_cameras({"head_cam_left/image": head, "head_cam_right/image": head, "ee_cam/image": wrist})
    assert set(images) == {"head_left", "head_right", "wrist"}
    assert images["head_left"].max() == 90
    assert images["wrist"].max() == 0


def test_missing_cameras_are_not_filled_with_synthetic_images(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    from probe_stationary_bridge import decode_cameras

    assert decode_cameras({}) == {}
