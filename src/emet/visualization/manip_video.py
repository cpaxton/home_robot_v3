# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Third-person manip smoke recording: overlays + MP4 encode."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


def overlay_manip_frame(
    rgb: np.ndarray,
    *,
    action: str = "",
    goal: str = "",
    detail: str = "",
    title: str = "emet manip",
) -> np.ndarray:
    """Draw status banners on an RGB frame (returns a copy)."""
    import cv2

    out = np.asarray(rgb, dtype=np.uint8).copy()
    if out.ndim != 3 or out.shape[2] < 3:
        raise ValueError(f"expected HxWx3 rgb, got {getattr(out, 'shape', None)}")
    h, w = int(out.shape[0]), int(out.shape[1])
    # Top banner
    cv2.rectangle(out, (0, 0), (w, 56), (20, 20, 20), thickness=-1)
    cv2.putText(
        out,
        str(title)[:80],
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    act = f"action: {action}" if action else "action: —"
    cv2.putText(out, act[:90], (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 120), 1, cv2.LINE_AA)
    # Bottom banner
    cv2.rectangle(out, (0, h - 48), (w, h), (20, 20, 20), thickness=-1)
    if goal:
        cv2.putText(
            out,
            f"goal: {goal}"[:100],
            (12, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 180, 255),
            1,
            cv2.LINE_AA,
        )
    if detail:
        cv2.putText(
            out,
            str(detail)[:110],
            (12, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    return out


def _grab_rgb(robot: Any) -> np.ndarray | None:
    get_obs = getattr(robot, "get_observation", None)
    if not callable(get_obs):
        return None
    try:
        obs = get_obs()
    except Exception:
        return None
    if obs is None:
        return None
    tp = getattr(obs, "third_person_image", None)
    if tp is not None:
        arr = np.asarray(tp)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3]
    rgb = getattr(obs, "rgb", None)
    if rgb is not None:
        arr = np.asarray(rgb)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3]
    return None


class ManipVideoRecorder:
    """Background sampler: third-person (or head) RGB + action/goal overlays → MP4."""

    def __init__(
        self,
        robot: Any,
        out_path: Path | str,
        *,
        fps: float = 12.0,
        title: str = "emet manip",
    ) -> None:
        self.robot = robot
        self.out_path = Path(out_path)
        self.fps = max(1.0, float(fps))
        self.title = str(title)
        self.action = ""
        self.goal = ""
        self.detail = ""
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_status(self, action: str = "", *, goal: str | None = None, detail: str | None = None) -> None:
        with self._lock:
            if action:
                self.action = str(action)
            if goal is not None:
                self.goal = str(goal)
            if detail is not None:
                self.detail = str(detail)

    def capture_once(self) -> bool:
        """Grab one overlaid frame (returns False if no RGB yet)."""
        rgb = _grab_rgb(self.robot)
        if rgb is None:
            return False
        with self._lock:
            action, goal, detail = self.action, self.goal, self.detail
        frame = overlay_manip_frame(rgb, action=action, goal=goal, detail=detail, title=self.title)
        with self._lock:
            self._frames.append(frame)
        return True

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()

        def _loop() -> None:
            period = 1.0 / self.fps
            while not self._stop.is_set():
                t0 = time.time()
                self.capture_once()
                dt = time.time() - t0
                time.sleep(max(0.0, period - dt))

        self._thread = threading.Thread(target=_loop, name="manip-video", daemon=True)
        self._thread.start()
        # Prime a first frame quickly.
        self.capture_once()

    def stop(self) -> Path | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            frames = list(self._frames)
        if not frames:
            return None
        from emet.eval.episode_video import write_rgb_sequence_mp4

        return write_rgb_sequence_mp4(frames, self.out_path, fps=self.fps)
