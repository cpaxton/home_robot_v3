# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Append raw instance detections to JSONL for offline fusion tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class CalibrationFrameWriter:
    def __init__(self, path: str | Path | None) -> None:
        self._path = Path(path).expanduser() if path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.touch()

    def append(
        self,
        *,
        step: int,
        detections: list[dict[str, Any]],
        navigation_origin_xyt: list[float] | None = None,
    ) -> None:
        if self._path is None:
            return
        row: dict[str, Any] = {"step": int(step), "detections": detections}
        if navigation_origin_xyt is not None:
            row["navigation_origin_xyt"] = [float(x) for x in navigation_origin_xyt]
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def detections_to_json_rows(dets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """JSON-serializable detection dicts for calibration export."""
    out: list[dict[str, Any]] = []
    for d in dets:
        row: dict[str, Any] = {
            "label": str(d.get("label_short", d.get("label", "object"))),
            "xyz": [float(x) for x in np.asarray(d["xyz"], dtype=np.float64).reshape(3)],
        }
        if d.get("bbox_xyxy") is not None:
            row["bbox_xyxy"] = [int(x) for x in d["bbox_xyxy"]]
        if d.get("bounds_3d") is not None:
            row["bounds_3d"] = d["bounds_3d"]
        if d.get("embedding") is not None:
            emb = np.asarray(d["embedding"], dtype=np.float32).reshape(-1)
            row["embedding"] = [float(x) for x in emb]
        out.append(row)
    return out
