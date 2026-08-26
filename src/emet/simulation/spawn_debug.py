# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

import emet.utils.logger as log

logger = log.Logger(__name__)


def spawn_debug_enabled() -> bool:
    """True when ``EMET_MOLMOSPACES_SPAWN_DEBUG`` is set or ``--debug-molmospaces-spawn`` turned it on.

    When true, :func:`find_molmospaces_freejoint_xyz` logs a downsampled ASCII occupancy map and
    writes a PNG: use ``EMET_MOLMOSPACES_SPAWN_DEBUG_MAP_PNG`` for an explicit path, or leave it unset
    to default to ``./molmospaces_spawn_topdown.png`` in the process cwd. Set
    ``EMET_MOLMOSPACES_SPAWN_DEBUG_MAP_PNG=0`` to skip the default PNG (ASCII still logs).

    ASCII legend (see ``topdown_map_key`` log line): ``'.'`` free, ``'#'`` blocked, ``'='`` collision
    clip, ``'*'`` first 72 occupancy-priority samples, ``'o'`` base before autoplace, ``'@'`` chosen
    spawn (robot base XY).
    """
    v = os.environ.get("EMET_MOLMOSPACES_SPAWN_DEBUG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def spawn_dbg(msg: str) -> None:
    if spawn_debug_enabled():
        logger.info(f"[molmospaces_spawn] {msg}")


def _spawn_debug_downsample_occ(occ: np.ndarray, max_h: int, max_w: int) -> tuple[np.ndarray, int, int]:
    """Return ``(nh, nw)`` uint8 grid ``1`` = free, ``0`` = blocked, plus strides ``sh, sw``."""
    H, W = int(occ.shape[0]), int(occ.shape[1])
    sh = max(1, (H + max_h - 1) // max_h)
    sw = max(1, (W + max_w - 1) // max_w)
    nh = (H + sh - 1) // sh
    nw = (W + sw - 1) // sw
    out = np.zeros((nh, nw), dtype=np.uint8)
    for i in range(nh):
        r0, r1 = i * sh, min(H, (i + 1) * sh)
        for j in range(nw):
            c0, c1 = j * sw, min(W, (j + 1) * sw)
            patch = occ[r0:r1, c0:c1]
            if patch.size == 0:
                continue
            out[i, j] = 1 if float(np.mean(patch.astype(np.float64))) > 0.5 else 0
    return out, sh, sw


def _spawn_debug_ascii_topdown(
    ithor_map: Any,
    *,
    clip_probe: tuple[float, float, float, float] | None,
    occ_priority_xy: list[tuple[float, float]],
    placed: tuple[float, float, float] | None,
    initial_xy: tuple[float, float] | None,
) -> list[str]:
    """Build printable ASCII lines (legend + map). ``ithor_map.occupancy`` is ``True`` = free."""
    from emet.simulation.spawn_molmospaces import _world_xy_to_occ_cell

    max_h, max_w = 52, 88
    occ = np.asarray(ithor_map.occupancy, dtype=bool)
    H, W = occ.shape
    small, sh, sw = _spawn_debug_downsample_occ(occ, max_h, max_w)
    nh, nw = small.shape
    ch: list[list[str]] = [["#" if small[i, j] == 0 else "." for j in range(nw)] for i in range(nh)]

    def world_to_small(x: float, y: float) -> tuple[int, int] | None:
        rc = _world_xy_to_occ_cell(ithor_map, x, y)
        if rc is None:
            return None
        r0, c0 = rc
        return (r0 // sh, c0 // sw)

    if clip_probe is not None:
        x0, x1, y0, y1 = clip_probe
        step = max(0.18, min(0.45, 0.02 * min(x1 - x0, y1 - y0)))
        xs = np.arange(x0, x1 + 1e-9, step, dtype=np.float64)
        ys = np.arange(y0, y1 + 1e-9, step, dtype=np.float64)
        for x in xs:
            for y in (y0, y1):
                p = world_to_small(float(x), float(y))
                if p:
                    br, bc = p
                    ch[br][bc] = "="
        for y in ys:
            for x in (x0, x1):
                p = world_to_small(float(x), float(y))
                if p:
                    br, bc = p
                    ch[br][bc] = "="

    for px, py in occ_priority_xy[:72]:
        p = world_to_small(px, py)
        if p:
            br, bc = p
            if ch[br][bc] in (".", "="):
                ch[br][bc] = "*"

    if initial_xy is not None:
        p = world_to_small(initial_xy[0], initial_xy[1])
        if p:
            br, bc = p
            if ch[br][bc] != "@":
                ch[br][bc] = "o"

    if placed is not None:
        p = world_to_small(placed[0], placed[1])
        if p:
            br, bc = p
            ch[br][bc] = "@"

    lines = [
        "topdown_map_key: '.'=occupancy_free '#'=blocked '='=collision_clip_rect "
        "'*'=occ_priority_xy[:72]_on_free 'o'=base_xy_before_autoplace '@'=chosen_spawn_base_xy",
        "topdown_map: occupancy downsampled "
        f"(orig {H}x{W} stride {sh}x{sw} -> {nh}x{nw}; same symbol key as topdown_map_key line above)",
    ]
    for i in range(nh):
        lines.append("topdown_map: " + "".join(ch[i][j] for j in range(nw)))
    return lines


def _spawn_debug_write_occupancy_png(
    ithor_map: Any,
    path: str,
    *,
    clip_probe: tuple[float, float, float, float] | None,
    occ_priority_xy: list[tuple[float, float]],
    placed: tuple[float, float, float] | None,
    initial_xy: tuple[float, float] | None,
) -> None:
    from PIL import Image, ImageDraw

    from emet.simulation.spawn_molmospaces import _world_xy_to_occ_cell

    occ = np.asarray(ithor_map.occupancy, dtype=np.uint8)
    rgb = np.stack([255 - occ * 255] * 3, axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    H, W = occ.shape

    def wc(x: float, y: float) -> tuple[int, int] | None:
        rc = _world_xy_to_occ_cell(ithor_map, x, y)
        if rc is None:
            return None
        row, col = int(rc[0]), int(rc[1])
        return (col, row)  # PIL (x=col, y=row)

    if clip_probe is not None:
        x0, x1, y0, y1 = clip_probe
        poly = []
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)):
            p = wc(x, y)
            if p:
                poly.append(p)
        if len(poly) >= 2:
            draw.line(poly, fill=(255, 80, 80), width=2)
    for px, py in occ_priority_xy[:200]:
        p = wc(px, py)
        if p:
            draw.rectangle((p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1), fill=(80, 200, 255))
    if initial_xy is not None:
        p = wc(initial_xy[0], initial_xy[1])
        if p:
            draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), outline=(255, 200, 0), width=2)
    if placed is not None:
        p = wc(placed[0], placed[1])
        if p:
            draw.ellipse((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), outline=(50, 255, 80), width=2)
    outp = Path(path).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(outp))


def _spawn_debug_emit_topdown(
    ithor_map: Any | None,
    *,
    clip_probe: tuple[float, float, float, float] | None,
    occ_priority_xy: list[tuple[float, float]],
    placed: tuple[float, float, float] | None,
    how: str,
    initial_xy: tuple[float, float] | None,
) -> None:
    if not spawn_debug_enabled():
        return
    spawn_dbg(
        f"spawn_debug_summary: how={how!r} placed={placed!r} initial_base_xy={initial_xy!r} "
        f"n_occ_priority={len(occ_priority_xy)} clip_probe={'set' if clip_probe is not None else 'none'}"
    )
    if ithor_map is None:
        spawn_dbg("topdown_map: (no iTHOR occupancy — enable EMET_MOLMOSPACES_OCC_MAP or use iTHOR scene)")
        return
    try:
        for line in _spawn_debug_ascii_topdown(
            ithor_map,
            clip_probe=clip_probe,
            occ_priority_xy=occ_priority_xy,
            placed=placed,
            initial_xy=initial_xy,
        ):
            spawn_dbg(line)
    except Exception as e:
        spawn_dbg(f"topdown_map_ascii: failed ({e!r})")

    png_raw = os.environ.get("EMET_MOLMOSPACES_SPAWN_DEBUG_MAP_PNG", "")
    png_path = png_raw.strip()
    if png_path.lower() in ("0", "false", "no", "none", "off"):
        png_path = ""
    elif not png_path:
        png_path = str(Path.cwd() / "molmospaces_spawn_topdown.png")
    if png_path:
        try:
            _spawn_debug_write_occupancy_png(
                ithor_map,
                png_path,
                clip_probe=clip_probe,
                occ_priority_xy=occ_priority_xy,
                placed=placed,
                initial_xy=initial_xy,
            )
            spawn_dbg(f"topdown_map_png: wrote {png_path!r}")
        except Exception as e:
            spawn_dbg(f"topdown_map_png: failed ({e!r})")
