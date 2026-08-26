# Close-look 2D map

Occupancy on the voxel XY grid means “we have been near this cell.” That is not
enough for a 4 cm jar on a table. The close-look map answers a different
question: **did a camera actually look at this XY, on-axis, from close range?**

Implementation: [`src/emet/mapping/close_map.py`](../src/emet/mapping/close_map.py)
(`CloseDistanceMap`). Tests: `src/test/mapping/test_close_map.py`.

## What is stored

The grid matches the voxel occupancy map (same origin, resolution, size). Each
cell keeps:

- minimum camera-to-surface range that hit it
- whether that hit was inside the optical-axis cone (`aim_deg`, default 25°)

A query neighborhood is **resolved** when at least one cell in the radius is an
aimed hit with range ≤ `r_close_m` (default 0.55 m).

RGB-D updates stamp the grid from `SparseVoxelMap.add_obs` /
DynaMem `process_rgbd` via `update_close_map_from_view`. Reset follows
`reset_cache()`.

## Stay vs escape

`decide_close_look` is the shared policy:

| Situation | Result |
|-----------|--------|
| No map yet | neither stay nor escape |
| Neighborhood resolved | done |
| Nav blocked to this XY | escape (unreachable) |
| Approaches exhausted | escape (agentic default 4, CHAT/TAMP default 2) |
| Otherwise | stay for another approach |

Agentic find (`investigate` / `navigate_to_obs`) stays on the place card until
resolved or escape. CHAT `face_toward` appends a one-line hint so the robot
does not orbit furniture forever.

## Env vars

See [environment_variables.md](environment_variables.md) (`EMET_CLOSE_MAP_*`).

Do not confuse this with `EMET_EQA_AGENTIC_CLOSE_LOOK`, which only asks the VLM
whether the *question* needs a close look (clock/count/detail). This map is
geometric.

## Layout

```
emet.mapping.close_map     CloseDistanceMap, decide_close_look, hints
emet.mapping.voxel         stamps the grid on each RGB-D view
graph_eqa agentic loop     stay/escape on investigate
emet.agent.tools           CHAT face_toward hint
```
