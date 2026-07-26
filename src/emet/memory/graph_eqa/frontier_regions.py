# Copyright (c) Chris Paxton 2026

"""Region-level utility for frontier exploration.

Frontier cells are already clustered into connected components when graph frontier
nodes are built (``cluster_frontier_mask``), but selection used to discard that and
walk to the nearest cell. On HM-EQA holdout q104/q105 the robot spawned in a yard and
creeped 1-2 m at a time, covering 16 m of path inside a 4.6 m box while the house
interior stayed unexplored.

Utility here trades expected area gain against travel cost, so a large unexplored
region several meters away outranks a sliver underfoot. Escape mode raises the floor
on travel distance once the answer verifier keeps reporting the target is not visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# A region this far away is worth half as much per unit of gain.
DEFAULT_DISTANCE_DISCOUNT_M = 4.0
# Keyword affinity is worth this many sqrt(m^2) of raw area.
DEFAULT_KEYWORD_BONUS = 2.0
# Goals within this radius of a recent goal are strongly demoted.
DEFAULT_RECENT_RADIUS_M = 1.25
DEFAULT_RECENT_MULTIPLIER = 0.2
# Each recorded nav failure halves a region's utility.
DEFAULT_NAV_FAILURE_DECAY = 0.5
# Regions closer than the escape floor are demoted by this factor.
DEFAULT_ESCAPE_MULTIPLIER = 0.05


@dataclass(frozen=True)
class FrontierRegion:
    """One connected unexplored component, as a navigation candidate."""

    region_id: str
    xy: tuple[float, float]
    cell_count: int = 0
    keyword_score: float = 0.0
    nav_failures: int = 0
    last_seen: int = 0

    def area_m2(self, grid_resolution_m: float) -> float:
        return float(max(0, int(self.cell_count))) * float(grid_resolution_m) ** 2


def region_from_node(node: object) -> FrontierRegion:
    """Build a region from a frontier ``GraphNode`` (missing fields degrade to 0)."""
    xyz = getattr(node, "xyz", (0.0, 0.0, 0.0))
    return FrontierRegion(
        region_id=str(getattr(node, "description", "") or getattr(node, "node_id", "")),
        xy=(float(xyz[0]), float(xyz[1])),
        cell_count=int(getattr(node, "frontier_cell_count", 0) or 0),
        keyword_score=float(getattr(node, "frontier_keyword_score", 0.0) or 0.0),
        nav_failures=int(getattr(node, "nav_failures", 0) or 0),
        last_seen=int(getattr(node, "last_seen", 0) or 0),
    )


def _planar_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def frontier_region_utility(
    region: FrontierRegion,
    robot_xy: tuple[float, float],
    *,
    grid_resolution_m: float = 0.1,
    recent: list[tuple[float, float]] | None = None,
    min_travel_m: float = 0.0,
    distance_discount_m: float = DEFAULT_DISTANCE_DISCOUNT_M,
    keyword_bonus: float = DEFAULT_KEYWORD_BONUS,
    recent_radius_m: float = DEFAULT_RECENT_RADIUS_M,
) -> float:
    """Expected gain per unit of travel. Higher is better; never negative.

    ``min_travel_m`` is the escape floor: when the verifier keeps saying the target is
    not visible here, regions inside that radius are demoted so the robot commits to
    leaving the area instead of re-scanning it.
    """
    dist = _planar_dist(region.xy, robot_xy)
    gain = math.sqrt(region.area_m2(grid_resolution_m))
    gain += float(keyword_bonus) * max(0.0, float(region.keyword_score))
    if gain <= 0.0:
        # No area/keyword signal (e.g. MuJoCo nodes): fall back to pure proximity.
        gain = 1.0
    utility = gain / (1.0 + dist / max(1e-6, float(distance_discount_m)))
    utility *= DEFAULT_NAV_FAILURE_DECAY ** max(0, int(region.nav_failures))
    if recent:
        nearest_recent = min(_planar_dist(region.xy, r) for r in recent)
        if nearest_recent < float(recent_radius_m):
            utility *= DEFAULT_RECENT_MULTIPLIER
    if dist < float(min_travel_m):
        utility *= DEFAULT_ESCAPE_MULTIPLIER
    return float(utility)


def rank_frontier_regions(
    regions: list[FrontierRegion],
    robot_xy: tuple[float, float],
    **kwargs: object,
) -> list[FrontierRegion]:
    """Regions best-first by :func:`frontier_region_utility` (ties: nearer first)."""
    return sorted(
        regions,
        key=lambda r: (
            -frontier_region_utility(r, robot_xy, **kwargs),  # type: ignore[arg-type]
            _planar_dist(r.xy, robot_xy),
            r.region_id,
        ),
    )
