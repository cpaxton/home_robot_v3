# Copyright (c) Chris Paxton 2026
"""Reusable top-down panels for frontier selection (obstacles / explored / frontier / pick).

Used live when ``explore_frontier`` chooses a goal, and offline for demos or bundle
replays. Each panel is titled with the iteration so a sequence reads as a flipbook.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from emet.visualization.map_snapshot import (
    downsample_topdown_rgb_max_side,
    explored_crop_indices,
    upscale_topdown_rgb_min_side,
    world_xy_to_grid_ij,
)

# Palette (RGB) — keep saturated so a star and frontier read on Discord / paper crops.
COLOR_UNKNOWN = (248, 248, 248)
COLOR_FREE_UNEXPLORED = (220, 220, 228)
COLOR_EXPLORED = (50, 160, 80)
COLOR_OBSTACLE = (200, 55, 55)
COLOR_FRONTIER = (40, 110, 230)
COLOR_ROBOT = (255, 220, 40)
COLOR_STAR = (255, 40, 200)
COLOR_WAYPOINT = (40, 40, 55)
COLOR_WAYPOINT_FG = (255, 255, 255)
COLOR_WAYPOINT_FILL_ALPHA = 0.55  # keep map visible under the badge
COLOR_TITLE_BG = (24, 24, 32)
COLOR_TITLE_FG = (245, 245, 250)


@dataclass(frozen=True)
class FrontierPickStep:
    """One explore iteration to render."""

    iteration: int
    obstacles: np.ndarray
    explored: np.ndarray
    frontier: np.ndarray | None = None
    robot_xy: tuple[float, float] | None = None
    chosen_xy: tuple[float, float] | None = None
    # Prior + current picks in order (1-based labels). Current is usually last.
    waypoints: tuple[tuple[float, float], ...] = ()
    title: str | None = None
    subtitle: str | None = None


def _normalize_waypoints(
    waypoints: Sequence[tuple[float, float] | np.ndarray] | None,
    chosen_xy: tuple[float, float] | np.ndarray | None,
) -> list[tuple[float, float]]:
    """Ordered pick history. Ensures ``chosen_xy`` is present as the latest entry."""
    out: list[tuple[float, float]] = []
    for raw in waypoints or ():
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size >= 2:
            out.append((float(arr[0]), float(arr[1])))
    if chosen_xy is not None:
        arr = np.asarray(chosen_xy, dtype=np.float64).reshape(-1)
        cur = (float(arr[0]), float(arr[1]))
        if not out or float(np.hypot(out[-1][0] - cur[0], out[-1][1] - cur[1])) > 0.05:
            out.append(cur)
    return out


def _expand_bbox_for_xy(
    bbox: tuple[int, int, int, int] | None,
    xy: tuple[float, float] | np.ndarray,
    *,
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    shape_hw: tuple[int, int],
    pad: int,
) -> tuple[int, int, int, int]:
    h, w = shape_hw
    ci, cj = world_xy_to_grid_ij(xy, grid_origin_xy, grid_resolution, shape_hw)
    if bbox is None:
        return (
            max(0, ci - pad),
            min(h, ci + pad + 1),
            max(0, cj - pad),
            min(w, cj + pad + 1),
        )
    i0, i1, j0, j1 = bbox
    return (
        max(0, min(i0, ci - pad)),
        min(h, max(i1, ci + pad + 1)),
        max(0, min(j0, cj - pad)),
        min(w, max(j1, cj + pad + 1)),
    )


def _draw_numbered_waypoints(
    rgb: np.ndarray,
    waypoints: Sequence[tuple[float, float]],
    *,
    grid_origin_xy: np.ndarray,
    grid_resolution: float,
    full_shape_hw: tuple[int, int],
    crop_offset_ij: tuple[int, int],
    pre_scale_hw: tuple[int, int],
) -> np.ndarray:
    """Paint 1…N badges at waypoint world positions (after crop/upscale)."""
    if not waypoints:
        return rgb
    from PIL import Image, ImageDraw, ImageFont

    h_pre, w_pre = int(pre_scale_hw[0]), int(pre_scale_hw[1])
    h, w = rgb.shape[0], rgb.shape[1]
    if h_pre <= 0 or w_pre <= 0:
        return rgb
    scale_i = h / float(h_pre)
    scale_j = w / float(w_pre)
    i_off, j_off = int(crop_offset_ij[0]), int(crop_offset_ij[1])
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for n, xy in enumerate(waypoints, start=1):
        ri, rj = world_xy_to_grid_ij(xy, grid_origin_xy, grid_resolution, full_shape_hw)
        pi = int(round((ri - i_off + 0.5) * scale_i))
        pj = int(round((rj - j_off + 0.5) * scale_j))
        if not (0 <= pi < h and 0 <= pj < w):
            continue
        label = str(n)
        # Compact badges — large enough to read, small enough not to bury the map.
        r = max(5, int(round(3.5 * min(scale_i, scale_j))))
        x0, y0 = max(0, pj - r), max(0, pi - r)
        x1, y1 = min(w - 1, pj + r), min(h - 1, pi + r)
        if x1 <= x0 or y1 <= y0:
            continue
        region = img.crop((x0, y0, x1 + 1, y1 + 1)).convert("RGBA")
        overlay = Image.new("RGBA", region.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        a = int(round(255 * COLOR_WAYPOINT_FILL_ALPHA))
        # Ellipse in local crop coords (pj/pi relative to x0/y0).
        lx, ly = pj - x0, pi - y0
        od.ellipse(
            [lx - r, ly - r, lx + r, ly + r],
            fill=(*COLOR_WAYPOINT, a),
            outline=(*(COLOR_STAR if n == len(waypoints) else COLOR_FRONTIER), 220),
            width=1,
        )
        blended = Image.alpha_composite(region, overlay).convert("RGB")
        img.paste(blended, (x0, y0))
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = draw.textsize(label, font=font)  # type: ignore[attr-defined]
        draw.text(
            (pj - tw // 2, pi - th // 2 - 1),
            label,
            fill=COLOR_WAYPOINT_FG,
            font=font,
        )
    return np.asarray(img, dtype=np.uint8)


def frontier_mask_from_explored(
    explored: Any,
    obstacles: Any | None = None,
) -> np.ndarray:
    """Classic frontier: unexplored free cells that touch explored.

    Does not need a planner — suitable for demos and offline panels. Live Habitat
    may pass ``get_outside_frontier`` instead when available.
    """
    exp = np.asarray(explored, dtype=bool)
    obs = np.zeros_like(exp, dtype=bool) if obstacles is None else np.asarray(obstacles, dtype=bool)
    if exp.shape != obs.shape:
        raise ValueError(f"explored {exp.shape} != obstacles {obs.shape}")
    h, w = exp.shape
    touch = np.zeros_like(exp, dtype=bool)
    touch[1:, :] |= exp[:-1, :]
    touch[:-1, :] |= exp[1:, :]
    touch[:, 1:] |= exp[:, :-1]
    touch[:, :-1] |= exp[:, 1:]
    return touch & ~exp & ~obs


def _draw_star(rgb: np.ndarray, ri: int, rj: int, *, radius: int = 7) -> None:
    """Five-point star centered on grid cell (distinct from the yellow robot disk)."""
    h, w = rgb.shape[0], rgb.shape[1]
    r = max(4, int(radius))
    # Rasterize by testing each pixel against a 5-point star polygon in local coords.
    # Outer radius r, inner radius ~0.4 r; angle offset so a tip points up (-row).
    tips = []
    for k in range(5):
        ang = -np.pi / 2 + k * 2 * np.pi / 5
        tips.append((r * np.cos(ang), r * np.sin(ang)))
        ang_i = ang + np.pi / 5
        tips.append((0.4 * r * np.cos(ang_i), 0.4 * r * np.sin(ang_i)))
    # Point-in-polygon (ray cast) for the local window.
    for di in range(-r - 1, r + 2):
        for dj in range(-r - 1, r + 2):
            # Local coords: x=dj (col), y=-di so tip points toward decreasing row.
            x, y = float(dj), float(-di)
            inside = False
            n = len(tips)
            for a in range(n):
                x1, y1 = tips[a]
                x2, y2 = tips[(a + 1) % n]
                if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
                    inside = not inside
            if not inside:
                continue
            ii, jj = ri + di, rj + dj
            if 0 <= ii < h and 0 <= jj < w:
                rgb[ii, jj] = np.uint8(COLOR_STAR)


def _draw_robot_marker(rgb: np.ndarray, ri: int, rj: int, *, radius: int = 3) -> None:
    h, w = rgb.shape[0], rgb.shape[1]
    r = max(2, int(radius))
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di * di + dj * dj > r * r:
                continue
            ii, jj = ri + di, rj + dj
            if 0 <= ii < h and 0 <= jj < w:
                rgb[ii, jj] = np.uint8(COLOR_ROBOT)


def _title_banner(width: int, title: str, *, height: int = 36) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    banner = Image.new("RGB", (max(1, width), height), COLOR_TITLE_BG)
    draw = ImageDraw.Draw(banner)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text = title.strip() or "frontier pick"
    # Pillow >=10 uses textbbox; fall back for older.
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)  # type: ignore[attr-defined]
    x = max(8, (width - tw) // 2)
    y = max(2, (height - th) // 2)
    draw.text((x, y), text, fill=COLOR_TITLE_FG, font=font)
    return np.asarray(banner, dtype=np.uint8)


def render_frontier_pick_rgb(
    obstacles: Any,
    explored: Any,
    *,
    frontier: Any | None = None,
    robot_xy: tuple[float, float] | np.ndarray | None = None,
    chosen_xy: tuple[float, float] | np.ndarray | None = None,
    waypoints: Sequence[tuple[float, float] | np.ndarray] | None = None,
    grid_origin_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
    grid_resolution: float = 0.1,
    title: str = "frontier pick",
    margin_cells: int = 12,
    max_side: int = 640,
    min_side: int = 320,
    star_radius_cells: int = 4,
    show_legend: bool = True,
) -> np.ndarray:
    """Render obstacles / explored / frontier / numbered pick history / current star.

    ``waypoints`` is the ordered pick history (1, 2, 3…). The current goal is also
    marked with a magenta star; if ``chosen_xy`` is set it is appended when missing.
    """
    obs = np.asarray(obstacles, dtype=bool)
    exp = np.asarray(explored, dtype=bool)
    if obs.shape != exp.shape:
        raise ValueError(f"obstacles {obs.shape} != explored {exp.shape}")
    fr = frontier_mask_from_explored(exp, obs) if frontier is None else np.asarray(frontier, dtype=bool)
    if fr.shape != exp.shape:
        raise ValueError(f"frontier {fr.shape} != explored {exp.shape}")

    go = np.asarray(grid_origin_xy, dtype=np.float64).reshape(-1)[:2]
    res = float(grid_resolution) if float(grid_resolution) > 0 else 0.1
    h, w = exp.shape
    picks = _normalize_waypoints(waypoints, chosen_xy)

    # Include frontier + markers in the crop so distant numbered picks stay in frame.
    crop_mask = exp | fr
    bbox = explored_crop_indices(
        crop_mask,
        robot_xy,
        go,
        res,
        (h, w),
        margin_cells=margin_cells,
        robot_radius_cells=max(4, star_radius_cells),
    )
    pad = max(margin_cells, star_radius_cells + 4)
    for xy in picks:
        bbox = _expand_bbox_for_xy(
            bbox,
            xy,
            grid_origin_xy=go,
            grid_resolution=res,
            shape_hw=(h, w),
            pad=pad,
        )
    if chosen_xy is not None and not picks:
        bbox = _expand_bbox_for_xy(
            bbox,
            chosen_xy,
            grid_origin_xy=go,
            grid_resolution=res,
            shape_hw=(h, w),
            pad=pad,
        )
    if bbox is None:
        i0, i1, j0, j1 = 0, h, 0, w
    else:
        i0, i1, j0, j1 = bbox

    exp_c = exp[i0:i1, j0:j1]
    obs_c = obs[i0:i1, j0:j1]
    fr_c = fr[i0:i1, j0:j1]
    rgb = np.full((exp_c.shape[0], exp_c.shape[1], 3), COLOR_UNKNOWN, dtype=np.uint8)
    rgb[~obs_c & ~exp_c] = COLOR_FREE_UNEXPLORED
    free = exp_c & ~obs_c
    rgb[free] = COLOR_EXPLORED
    rgb[obs_c] = COLOR_OBSTACLE
    # Frontier on top of unknown / free edge, but never paint over obstacles.
    rgb[fr_c & ~obs_c] = COLOR_FRONTIER

    if robot_xy is not None:
        ri, rj = world_xy_to_grid_ij(robot_xy, go, res, (h, w))
        _draw_robot_marker(rgb, ri - i0, rj - j0)

    if picks:
        ci, cj = world_xy_to_grid_ij(picks[-1], go, res, (h, w))
        _draw_star(rgb, ci - i0, cj - j0, radius=star_radius_cells)
    elif chosen_xy is not None:
        ci, cj = world_xy_to_grid_ij(chosen_xy, go, res, (h, w))
        _draw_star(rgb, ci - i0, cj - j0, radius=star_radius_cells)

    pre_h, pre_w = rgb.shape[0], rgb.shape[1]
    rgb = upscale_topdown_rgb_min_side(rgb, min_side)
    rgb = downsample_topdown_rgb_max_side(rgb, max_side)
    rgb = _draw_numbered_waypoints(
        rgb,
        picks,
        grid_origin_xy=go,
        grid_resolution=res,
        full_shape_hw=(h, w),
        crop_offset_ij=(i0, j0),
        pre_scale_hw=(pre_h, pre_w),
    )
    banner = _title_banner(rgb.shape[1], title)
    if banner.shape[1] != rgb.shape[1]:
        from PIL import Image

        banner = np.asarray(
            Image.fromarray(banner).resize((rgb.shape[1], banner.shape[0]), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    parts = [banner, rgb]
    if show_legend:
        parts.append(legend_rgb(width=rgb.shape[1]))
    return np.concatenate(parts, axis=0)


def render_frontier_pick_step(
    step: FrontierPickStep,
    *,
    grid_origin_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
    grid_resolution: float = 0.1,
    max_side: int = 640,
) -> np.ndarray:
    title = step.title or f"iteration {step.iteration}"
    if step.subtitle:
        title = f"{title} — {step.subtitle}"
    return render_frontier_pick_rgb(
        step.obstacles,
        step.explored,
        frontier=step.frontier,
        robot_xy=step.robot_xy,
        chosen_xy=step.chosen_xy,
        waypoints=step.waypoints,
        grid_origin_xy=grid_origin_xy,
        grid_resolution=grid_resolution,
        title=title,
        max_side=max_side,
    )


def save_frontier_pick_rgb(rgb: np.ndarray, path: str | Path) -> Path:
    from PIL import Image

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(out)
    return out


def write_frontier_pick_steps(
    steps: Sequence[FrontierPickStep],
    out_dir: str | Path,
    *,
    grid_origin_xy: np.ndarray | tuple[float, float] = (0.0, 0.0),
    grid_resolution: float = 0.1,
    max_side: int = 640,
    prefix: str = "iter",
) -> list[Path]:
    """Write ``{prefix}_{iteration:02d}.png`` for each step; returns paths in order."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for step in steps:
        rgb = render_frontier_pick_step(
            step,
            grid_origin_xy=grid_origin_xy,
            grid_resolution=grid_resolution,
            max_side=max_side,
        )
        path = out / f"{prefix}_{int(step.iteration):02d}.png"
        save_frontier_pick_rgb(rgb, path)
        paths.append(path)
    return paths


def make_long_motion_demo_steps(
    *,
    size: int = 120,
    n_iters: int = 4,
    grid_resolution: float = 0.1,
) -> tuple[list[FrontierPickStep], np.ndarray]:
    """Synthetic hallway: star marks a distant frontier cell (long-leg commit).

    Explored already covers a long corridor strip; the robot sits near one end while
    the pick is the frontier cell at the far end — several meters ahead — instead of
    the nearest blue cell underfoot.
    """
    h = w = int(size)
    go = np.array([h / 2.0, w / 2.0], dtype=np.float64)
    res = float(grid_resolution)

    def cell_to_xy(i: int, j: int) -> tuple[float, float]:
        return (float(i) - go[0]) * res, (float(j) - go[1]) * res

    # Thin walls around a free corridor + side room (unknown stays light, not solid red).
    free = np.zeros((h, w), dtype=bool)
    free[20:100, 55:65] = True
    free[40:70, 65:95] = True
    obstacles = np.zeros((h, w), dtype=bool)
    # Wall ring: cells that neighbor free but are not free.
    touch = np.zeros_like(free)
    touch[1:, :] |= free[:-1, :]
    touch[:-1, :] |= free[1:, :]
    touch[:, 1:] |= free[:, :-1]
    touch[:, :-1] |= free[:, 1:]
    obstacles[touch & ~free] = True

    explored_ends = [55, 70, 85, 95]
    robot_rows = [28, 40, 55, 70]
    steps: list[FrontierPickStep] = []
    history: list[tuple[float, float]] = []

    for it in range(min(n_iters, len(explored_ends))):
        explored = np.zeros((h, w), dtype=bool)
        end = explored_ends[it]
        explored[24:end, 56:64] = free[24:end, 56:64]
        if it >= 2:
            explored[48:62, 64:72] = free[48:62, 64:72]
        if it >= 3:
            explored[48:62, 72:88] = free[48:62, 72:88]

        frontier = frontier_mask_from_explored(explored, obstacles)
        ri, rj = robot_rows[it], 60
        robot_xy = cell_to_xy(ri, rj)

        cells = np.argwhere(frontier)
        if len(cells) == 0:
            chosen_xy = robot_xy
            dist_m = 0.0
        else:
            # Long-leg *ahead*: among frontier cells down-corridor (or otherwise
            # farther from the robot), never the nearest underfoot cell.
            ahead = cells[cells[:, 0] > ri + 1]
            pool = ahead if len(ahead) else cells
            d = np.hypot(pool[:, 0].astype(float) - ri, pool[:, 1].astype(float) - rj)
            best = pool[int(np.argmax(d))]
            ti, tj = int(best[0]), int(best[1])
            chosen_xy = cell_to_xy(ti, tj)
            dist_m = float(np.hypot(chosen_xy[0] - robot_xy[0], chosen_xy[1] - robot_xy[1]))

        history.append(chosen_xy)
        steps.append(
            FrontierPickStep(
                iteration=it,
                obstacles=obstacles.copy(),
                explored=explored,
                frontier=frontier,
                robot_xy=robot_xy,
                chosen_xy=chosen_xy,
                waypoints=tuple(history),
                title=f"iteration {it}",
                subtitle=f"pick {dist_m:.1f} m ahead ({len(history)} waypoints)",
            )
        )
    return steps, go


def legend_rgb(*, width: int = 320, height: int = 28) -> np.ndarray:
    """Compact color key for Discord / paper captions."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (max(1, width), height), COLOR_TITLE_BG)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    swatches = [
        (COLOR_OBSTACLE, "wall"),
        (COLOR_EXPLORED, "explored"),
        (COLOR_FRONTIER, "frontier"),
        (COLOR_ROBOT, "robot"),
        (COLOR_WAYPOINT, "1..N"),
        (COLOR_STAR, "pick"),
    ]
    x = 4
    for color, label in swatches:
        draw.rectangle([x, 7, x + 8, height - 7], fill=color)
        draw.text((x + 11, 8), label, fill=COLOR_TITLE_FG, font=font)
        x += 48 + max(0, len(label) - 3) * 4
        if x > width - 36:
            break
    return np.asarray(img, dtype=np.uint8)
