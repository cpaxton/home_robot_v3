# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Optional H.264 NAL publish/decode helpers for ZMQ port 4405."""

from __future__ import annotations

from typing import Any

import numpy as np

import emet.utils.compression as compression


def decode_h264_message(msg: dict[str, Any]) -> np.ndarray | None:
    """Decode ``h264_nal`` bytes from a ZMQ H.264 side message to RGB uint8."""
    nal = msg.get("h264_nal")
    if nal is None:
        return None
    try:
        return compression.from_h264(nal)
    except Exception:
        return None
