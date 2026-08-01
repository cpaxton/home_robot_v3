# Unified EMET configuration (`configs/emet/default.yaml`)

All major apps (`emet run agent`, `emet run dynagraph`, `emet run dynamem`, `emet stream`, `emet capture`) load the same **nested YAML** via:

```bash
emet run agent --config configs/emet/default.yaml
emet run dynagraph --config configs/emet/default.yaml   # --robot optional
emet run dynamem -S
```

Default path: [`configs/emet/default.yaml`](../configs/emet/default.yaml). Override with **`EMET_CONFIG=/path/to.yaml`** or **`--config PATH`** / **`-C`**.

**Config path precedence** (shared by agent / dynamem / dynagraph / stream / capture):

1. Explicit **`--config`** / **`-C`** (or deprecated `--agent-config` / `--dynav-config`)
2. Connection-profile **`config`** field (`emet connect save … --config …`) for `--connection NAME`, or the **active** profile when `--connection` is omitted — see [cli.md](cli.md) (`emet connect`)
3. **`EMET_CONFIG`** / packaged default

Storing a chat-agent YAML (e.g. `configs/agent_innate_mars.yaml`) on the active profile therefore also becomes the default for `emet run dynamem` / `stream` until you pass `--config` or clear that field. Prefer a Mars-only workstation for that setup, or keep agent YAML off the active profile when switching robots.

Legacy basenames (`dynav_config.yaml`, `--agent-config`, `--dynav-config`) still work; they map to the unified loader with deprecation warnings.

---

## Schema (nested sections)

| Section | Purpose |
|---------|---------|
| `mapping` | DynaMem / Dynagraph voxel, depth, EQA, motion planner (was flat `dynav_config.yaml`) |
| `agent` | Chat agent: `llm`, `eqa`, `discord`, `share_memory_vllm`, … — consumed by **`emet run agent`** when the matching CLI flag is omitted |
| `sim` / `sim_config` | MuJoCo / Robocasa / MolmoSpaces launch (see [Simulation configs](sim_configs.md)) |
| `embodied_agent` | Open-vocab scene graph + GraphEQA memory overlays |
| `rerun` | Live Rerun viewer options |
| `eval` | Episode diagnostics exports (maps, RGB frames, MP4) for Habitat/OVMM/SQA3D bundles — see [evaluation.md](evaluation.md) |
| `robots.<id>` | Per-robot overlays merged when robot is resolved |
| `robot` | Optional fixed robot id (CLI `--robot` wins when set) |
| `connection` | Named profile in `~/.stretch/connection.json` |

Packaged defaults compose via `defaults:`:

```yaml
defaults:
  - mapping: package://emet/config/mapping/default.yaml
  - agent: package://emet/config/agent/default.yaml
  - rerun: package://emet/config/agents/default_rerun.yaml
  - eval: package://emet/config/eval/default.yaml
```

Example **`eval:`** overrides (maps + videos):

```yaml
eval:
  export_map_video: true
  export_video: true
  map_video_stride: 5
  export_video_substeps: true   # Habitat: one RGB frame per sim.step during nav
  video_motion_paced: true      # repeat frames by motion delta (m / rad)
  video_meters_per_frame: 0.25
  video_radians_per_frame: 0.1745329252  # 10 deg
  video_crossfade_teleport_m: 1.5
```

Example **`mapping.eqa:`** overrides for Habitat HM-EQA (runner applies these via `setdefault` when unset in `dynav_config.yaml`):

```yaml
mapping:
  eqa:
    habitat_perfect_nav: true
    habitat_explore_frontiers: true
    image_nav_min_approach_m: 0.35
```

The Habitat wrapper loads [`src/emet/config/dynav_config.yaml`](../src/emet/config/dynav_config.yaml) today; equivalent flat `eqa:` keys there override runner defaults. See [habitat/usage.md](habitat/usage.md#navigation-habitat-only). Robocasa / ZMQ GraphEQA ignores these unless you set them explicitly.

User presets use `extends:`:

```yaml
extends: configs/emet/default.yaml
robot: innate_mars
mapping:
  eqa_vl:
    model_size: "4B"
```

---

## Robot-specific overlays

Innate Mars depth (DA3 / auto) lives under `robots.innate_mars` in the default config — no separate 170-line YAML copy:

```yaml
robots:
  innate_mars:
    mapping:
      depth_source: auto
      da3_stereo: true
      local_radius: 0.85
      # Optional post-filters (default off): speckle on inferred depth; DBSCAN on voxel PCD (any depth source)
    zmq:
      allow_missing_depth: true
```

When the runtime robot id is `innate_mars`, these keys deep-merge into `mapping` and `zmq`. See [dynav_config.md](dynav_config.md#depth--voxel-post-filters-da3-hardware-opt-in) for opt-in speckle / DBSCAN tuning.

---

## Robot resolution (when `--robot` is omitted)

1. Explicit **`--robot`** on CLI
2. Top-level **`robot:`** in config
3. **ZMQ discovery** on **localhost** (running sim publishes `emet_robot_id`; beats saved hardware connection profiles)
4. **`connection:`** profile or active `~/.stretch/connection.json` entry (remote / non-localhost hosts)
5. **ZMQ discovery** on remote hosts (when connection has no robot)
6. Fallback **`stretch`**

Same order for **`emet run dynagraph`** and **`emet run agent`** (except agent skips ZMQ discovery when **`--start-sim`** spawns the sim first).

Host resolution: explicit **`--robot-ip`** → connection profile host → `127.0.0.1`.

---

## CLI overrides (`--set` / `-O`)

Any nested key without a new Click flag:

```bash
emet run agent --set mapping.depth_source=sensor
emet run dynagraph -O mapping.dynagraph_merge_xy_m=0.3 -O agent.eqa=true
```

Precedence (low → high): `defaults:` files → main config → `robots.*` overlay → **`--set`**.

---

## Legacy flat YAML

Files like [`dynav_config.yaml`](../src/emet/config/dynav_config.yaml) (flat dynav keys) auto-wrap under `mapping:` when loaded. Existing scripts using `get_parameters("dynav_config.yaml")` keep working.

[`dynav_innate_mars.yaml`](../src/emet/config/dynav_innate_mars.yaml) is now a thin `extends:` alias; Mars tuning is in `robots.innate_mars`.

---

## Smoke validation

After config-loader or overlay changes, run:

```bash
uv run emet test src/test/config/ -q
uv run python -c "
from emet.core.parameters import get_parameters
p = get_parameters('dynav_innate_mars.yaml')
assert str(p.get('depth_source')).lower() == 'auto'
assert (p.get('graph_object_fusion') or {}).get('bounds_3d_iou_merge_min') == 0.40
print('config_smoke OK')
"
```

Full cross-track tier 0 (config + eval + backends): [cross_track_smoke.md](experiments/cross_track_smoke.md#tier-0--focused-unit-tests-15-min).

---

## Related docs

- [Agent run](AGENT_RUN.md) — `emet run agent` flags
- [Dynav / mapping keys](dynav_config.md) — section-by-section `mapping` reference (legacy doc name; content describes `mapping.*`)
- [Simulation configs](sim_configs.md) — `sim:` / `sim_config:`
- [Evaluation runbook](evaluation.md) — `eval:` keys, env vars, Habitat bundle layout
- [Habitat usage](habitat/usage.md) — HM-EQA CLI, navigation `eqa:` keys
