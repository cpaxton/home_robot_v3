# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from emet.visualization.map_snapshot import explored_crop_indices, world_xy_to_grid_ij

DEFAULT_EXPLORED_CROP_MARGIN_CELLS = 16
DEFAULT_ROBOT_RADIUS_CELLS = 5
# Max cells on the longest crop edge before downsampling (Discord RGB maps use 640).
DEFAULT_NAVGRID_MAX_SIDE = 320
DEFAULT_LEGEND_WRAP = 76

_GLYPHS = "0123456789abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class NavGridSnapshot:
    """Pure-numpy 2D obstacle / explored booleans plus grid metadata."""

    obstacles: np.ndarray
    explored: np.ndarray
    grid_resolution_m: float
    grid_origin_xy: tuple[float, float]

    def __post_init__(self) -> None:
        obs = np.asarray(self.obstacles, dtype=bool)
        exp = np.asarray(self.explored, dtype=bool)
        if obs.shape != exp.shape:
            raise ValueError(f"obstacles shape {obs.shape} != explored shape {exp.shape}")
        object.__setattr__(self, "obstacles", obs)
        object.__setattr__(self, "explored", exp)


@dataclass(frozen=True)
class NavOverlay:
    """Semantic object footprint for glyph overlay on the nav grid."""

    kind: Literal["instance", "graph_point", "graph_extent"]
    key: str
    xy_min: tuple[float, float]
    xy_max: tuple[float, float]
    confidence: float
    labels: tuple[str, ...] = ()
    caption: str | None = None


def navgrid_ascii_enabled() -> bool:
    return os.environ.get("EMET_NAVGRID_ASCII", "").strip().lower() in ("1", "true", "yes", "on")


def navgrid_max_side() -> int:
    """Longest edge (cells) kept before max-pool; override with ``EMET_NAVGRID_MAX_SIDE``."""
    raw = os.environ.get("EMET_NAVGRID_MAX_SIDE", "").strip()
    if raw:
        try:
            return max(8, int(raw))
        except ValueError:
            pass
    return DEFAULT_NAVGRID_MAX_SIDE


def _downsample_maxpool(
    obstacles: np.ndarray,
    explored: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Uniform stride max-pool (Discord ``downsample_topdown_rgb_max_side`` semantics)."""
    h, w = obstacles.shape
    if h == 0 or w == 0:
        return obstacles, explored, 1
    m = max(h, w)
    step = 1 if m <= max_side or max_side <= 0 else int(np.ceil(m / max_side))
    nh = max(1, int(np.ceil(h / step)))
    nw = max(1, int(np.ceil(w / step)))
    obs_out = np.zeros((nh, nw), dtype=bool)
    exp_out = np.zeros((nh, nw), dtype=bool)
    for i in range(nh):
        r0, r1 = i * step, min(h, (i + 1) * step)
        for j in range(nw):
            c0, c1 = j * step, min(w, (j + 1) * step)
            obs_out[i, j] = bool(np.any(obstacles[r0:r1, c0:c1]))
            exp_out[i, j] = bool(np.any(explored[r0:r1, c0:c1]))
    return obs_out, exp_out, step


def _grid_to_world_center(gr: float, gc: float, origin: tuple[float, float], resolution: float) -> tuple[float, float]:
    wx = (gc - origin[0]) * resolution
    wy = (gr - origin[1]) * resolution
    return wx, wy


def _assign_glyphs(overlays: tuple[NavOverlay, ...]) -> dict[str, str]:
    ordered = sorted(overlays, key=lambda o: (o.kind, o.key))
    mapping: dict[str, str] = {}
    for idx, ov in enumerate(ordered):
        mapping[ov.key] = _GLYPHS[idx % len(_GLYPHS)]
    return mapping


def _overlay_hits_cell(
    ov: NavOverlay,
    gr: int,
    gc: int,
    r0: int,
    c0: int,
    sh: int,
    sw: int,
    origin: tuple[float, float],
    resolution: float,
) -> bool:
    wx0, wy0 = _grid_to_world_center(r0 + gr * sh, c0 + gc * sw, origin, resolution)
    wx1, wy1 = _grid_to_world_center(r0 + (gr + 1) * sh - 1, c0 + (gc + 1) * sw - 1, origin, resolution)
    cell_xmin, cell_xmax = min(wx0, wx1), max(wx0, wx1)
    cell_ymin, cell_ymax = min(wy0, wy1), max(wy0, wy1)
    ox0, oy0 = ov.xy_min
    ox1, oy1 = ov.xy_max
    return not (cell_xmax < ox0 or cell_xmin > ox1 or cell_ymax < oy0 or cell_ymin > oy1)


def render_navgrid_ascii(
    snapshot: NavGridSnapshot,
    overlays: tuple[NavOverlay, ...] = (),
    robot_xy: tuple[float, float] | None = None,
    max_side: int | None = None,
) -> str:
    """Render a cropped, downsampled ASCII nav grid with optional semantic glyphs."""
    obstacles = snapshot.obstacles
    explored = snapshot.explored
    origin = snapshot.grid_origin_xy
    resolution = snapshot.grid_resolution_m

    if obstacles.size == 0:
        return "navgrid: empty obstacle/explored arrays"

    side_limit = max_side if max_side is not None else navgrid_max_side()

    origin_arr = np.asarray(origin, dtype=np.float64)
    h, w = obstacles.shape
    robot_grid: tuple[int, int] | None = None
    if robot_xy is not None:
        robot_grid = world_xy_to_grid_ij(robot_xy, origin_arr, resolution, (h, w))

    bbox = explored_crop_indices(
        explored,
        robot_xy,
        origin_arr,
        resolution,
        (h, w),
        margin_cells=DEFAULT_EXPLORED_CROP_MARGIN_CELLS,
        robot_radius_cells=DEFAULT_ROBOT_RADIUS_CELLS,
    )
    if bbox is None:
        return "navgrid: no explored cells in map"
    r0, r1, c0, c1 = bbox
    crop_obs = obstacles[r0:r1, c0:c1]
    crop_exp = explored[r0:r1, c0:c1]
    if crop_obs.size == 0 or not np.any(crop_exp):
        return "navgrid: no explored cells in map"

    ds_obs, ds_exp, step = _downsample_maxpool(crop_obs, crop_exp, side_limit)
    nh, nw = ds_obs.shape
    glyph_map = _assign_glyphs(overlays)

    chars: list[list[str]] = []
    used_glyphs: dict[str, NavOverlay] = {}
    for gr in range(nh):
        row: list[str] = []
        for gc in range(nw):
            if ds_obs[gr, gc]:
                row.append("#")
                continue
            ch = " "
            if ds_exp[gr, gc]:
                ch = "."
            if robot_grid is not None:
                rr, cc = robot_grid
                block_r0, block_r1 = r0 + gr * step, r0 + min(crop_obs.shape[0], (gr + 1) * step)
                block_c0, block_c1 = c0 + gc * step, c0 + min(crop_obs.shape[1], (gc + 1) * step)
                if block_r0 <= rr < block_r1 and block_c0 <= cc < block_c1:
                    if ch == ".":
                        ch = "@"
            if ch in (".", " "):
                best: NavOverlay | None = None
                best_conf = -1.0
                for ov in overlays:
                    if not _overlay_hits_cell(ov, gr, gc, r0, c0, step, step, origin, resolution):
                        continue
                    if ov.confidence > best_conf:
                        best_conf = ov.confidence
                        best = ov
                if best is not None:
                    ch = glyph_map[best.key]
                    used_glyphs[ch] = best
            row.append(ch)
        chars.append(row)

    m_per_char = resolution * step
    lines: list[str] = [
        f"navgrid: explored crop grid[{r0}:{r1},{c0}:{c1}] downsample {nh}x{nw} "
        f"(~{m_per_char:.2f}m/char stride={step} max_side={side_limit}; "
        f"resolution={resolution:.3f}m; origin=({origin[0]:.1f},{origin[1]:.1f}))",
        "navgrid_key: '#'=obstacle '.'=explored_free ' '=unknown '@'=robot 0-9a-z=semantic",
    ]
    for row in chars:
        lines.append("".join(row))

    if used_glyphs:
        lines.append("Legend:")
        for ch in sorted(used_glyphs, key=lambda c: _GLYPHS.index(c) if c in _GLYPHS else 999):
            ov = used_glyphs[ch]
            lbl = ", ".join(ov.labels) if ov.labels else "(no labels)"
            cap = f' caption="{ov.caption[:60]}"' if ov.caption else ""
            lines.append(f'  {ch}  key={ov.key} kind={ov.kind} conf={ov.confidence:.3f} labels="{lbl}"{cap}')
    elif overlays:
        lines.append("Legend: (overlays present but none overlapped explored cells)")

    return "\n".join(lines)


def snapshot_from_numpy_2d(
    obstacles: np.ndarray,
    explored: np.ndarray,
    *,
    grid_resolution_m: float,
    grid_origin_xy: tuple[float, float],
) -> NavGridSnapshot:
    return NavGridSnapshot(
        obstacles=np.asarray(obstacles, dtype=bool),
        explored=np.asarray(explored, dtype=bool),
        grid_resolution_m=float(grid_resolution_m),
        grid_origin_xy=(float(grid_origin_xy[0]), float(grid_origin_xy[1])),
    )


def snapshot_from_sparse_voxel_map(voxel_map: Any) -> NavGridSnapshot:
    """Build snapshot from Dynamem ``SparseVoxelMap`` (torch ``get_2d_map``)."""
    obstacles, explored = voxel_map.get_2d_map()
    obs_np = obstacles.detach().cpu().numpy().astype(bool)
    exp_np = explored.detach().cpu().numpy().astype(bool)
    grid = voxel_map.grid
    origin = grid.grid_origin[:2].detach().cpu().numpy()
    return NavGridSnapshot(
        obstacles=obs_np,
        explored=exp_np,
        grid_resolution_m=float(grid.resolution),
        grid_origin_xy=(float(origin[0]), float(origin[1])),
    )


def overlays_from_instances(instances: Any, *, detection_model: Any = None) -> tuple[NavOverlay, ...]:
    """Map ``Instance`` objects (or instance-memory list) to overlays."""
    from emet.memory.graph_eqa.instance_observations import label_for_detection_category

    out: list[NavOverlay] = []
    for inst in instances or []:
        bounds = getattr(inst, "bounds", None)
        if bounds is None:
            continue
        b = np.asarray(bounds, dtype=np.float64)
        if b.shape != (3, 2):
            continue
        cat_id = getattr(inst, "category_id", None)
        label = None
        if detection_model is not None and cat_id is not None:
            try:
                label = label_for_detection_category(detection_model, int(cat_id))
            except Exception:
                label = None
        labels: list[str] = []
        if label:
            labels.append(str(label))
        elif cat_id is not None:
            labels.append(f"cat{cat_id}")
        else:
            labels.append(f"inst{getattr(inst, 'global_id', len(out))}")
        score = float(getattr(inst, "score", 1.0))
        gid = getattr(inst, "global_id", len(out))
        out.append(
            NavOverlay(
                kind="instance",
                key=f"inst:{gid}",
                xy_min=(float(b[0, 0]), float(b[1, 0])),
                xy_max=(float(b[0, 1]), float(b[1, 1])),
                confidence=score,
                labels=tuple(labels),
            )
        )
    return tuple(out)


def overlays_from_graph_eqa(memory: Any) -> tuple[NavOverlay, ...]:
    """Map ``GraphEQAMemory`` nodes to overlays (Dynagraph merges included)."""
    if memory is None:
        return ()
    out: list[NavOverlay] = []
    for n in memory.get_nodes():
        xyz = np.asarray(n.xyz, dtype=np.float64).ravel()
        if xyz.size < 3:
            continue
        ext = getattr(n, "extent_half", None)
        if ext is not None and len(ext) >= 2:
            eh = np.asarray(ext, dtype=np.float64).ravel()
            xy_min = (float(xyz[0] - eh[0]), float(xyz[1] - eh[1]))
            xy_max = (float(xyz[0] + eh[0]), float(xyz[1] + eh[1]))
            kind: Literal["instance", "graph_point", "graph_extent"] = "graph_extent"
        else:
            xy_min = (float(xyz[0]) - 0.15, float(xyz[1]) - 0.15)
            xy_max = (float(xyz[0]) + 0.15, float(xyz[1]) + 0.15)
            kind = "graph_point"
        labels = tuple(str(x) for x in (n.labels or [])[:4])
        conf = float(getattr(n, "support_count", 1.0))
        desc = getattr(n, "description", None)
        out.append(
            NavOverlay(
                kind=kind,
                key=f"node:{n.node_id}",
                xy_min=xy_min,
                xy_max=xy_max,
                confidence=conf,
                labels=labels,
                caption=str(desc) if desc else None,
            )
        )
    return tuple(out)


def build_navgrid_from_voxel_map(
    voxel_map: Any,
    *,
    graph_memory: Any = None,
    robot_xy: tuple[float, float] | None = None,
) -> str:
    """Convenience: adapters + render for live Dynamem / Dynagraph voxel maps."""
    snap = snapshot_from_sparse_voxel_map(voxel_map)
    overlays: list[NavOverlay] = []
    try:
        instances = voxel_map.get_instances()
    except Exception:
        instances = []
    det = getattr(voxel_map, "detection", None)
    overlays.extend(overlays_from_instances(instances, detection_model=det))
    overlays.extend(overlays_from_graph_eqa(graph_memory))
    return render_navgrid_ascii(snap, tuple(overlays), robot_xy=robot_xy)


def maybe_print_navgrid_ascii(text: str) -> None:
    """Print nav grid to stderr when ``EMET_NAVGRID_ASCII=1``."""
    if not navgrid_ascii_enabled() or not text.strip():
        return
    print(text, file=sys.stderr, flush=True)


def navgrid_context_allowed(context: str) -> bool:
    """When ``EMET_NAVGRID_CONTEXTS`` is set, only those hook labels may print."""
    raw = os.environ.get("EMET_NAVGRID_CONTEXTS", "").strip()
    if not raw:
        return True
    allowed = {c.strip() for c in raw.split(",") if c.strip()}
    return not context or context in allowed
