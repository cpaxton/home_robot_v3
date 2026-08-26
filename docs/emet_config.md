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
| `eval` | Episode diagnostics exports (maps, RGB frames, MP4) for Habitat/OVMM/SQA3D bundles — see [evaluation.md](evaluation.md). Named profile: `EMET_EVAL_OUTPUT_PROFILE=lean`. |
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

Example **`eval:`** overrides (maps + videos). For iterative HM-EQA slices prefer
``profile: lean`` (or ``EMET_EVAL_OUTPUT_PROFILE=lean``) instead of listing every flag:

```yaml
eval:
  profile: lean
```

Paper/H2H dumps stay ``full`` (default). Per-flag env still wins over the profile.

Example **`mapping.eqa:`** overrides for Habitat HM-EQA (runner applies these via `setdefault` when unset in `dynav_config.yaml`):

```yaml
mapping:
  eqa:
    habitat_perfect_nav: true
    habitat_explore_frontiers: true
    image_nav_min_approach_m: 0.35
    # Answer VLM contract (also top-level eqa: in dynav_config.yaml):
    # answer_format: json   # default when prompt_variant is hmeqa/mcq; else labeled
    # merged_memory: true   # default on; paper HM-EQA dynagraph row pins false
    # agentic_decision_policy: grounded_v2
    # attempt_ledger_mode: agent        # off | shadow | agent
    # action_progress_mode: shadow      # off | shadow | enforce; grounded_v2 only
```

### EQA answer prompt (`eqa` + `eqa_vl`)

| Key | Default | Notes |
|-----|---------|--------|
| `eqa.prompt_variant` | (unset) / harness `hmeqa` | `hmeqa` / `mcq` → JSON answers + prefill `{"reasoning":` |
| `eqa.center_zoom` | **off** | Blind center-crop zoom on `read N` / clock questions when no detector bbox. Did not convert HM-EQA clocks; keep off until a better crop. Env: `EMET_EQA_CENTER_ZOOM`. |
| `eqa.answer_format` | `json` if hmeqa/mcq else `labeled` | Override with `EMET_EQA_ANSWER_FORMAT`. JSON object keys: `reasoning`, `answer`, `confidence`, `action`, `confidence_reasoning`; labeled `Reasoning:/Answer:/…` scrape remains as fallback. |
| `eqa.merged_memory` | **on** | Fold CONFIRMED_MEMORY into `SCENE_GRAPH` lines. HM-EQA paper row pins `false` in [`configs/benchmarks/dynagraph.yaml`](../configs/benchmarks/dynagraph.yaml). Env: `EMET_EQA_MERGED_MEMORY`. |
| `eqa.attempt_ledger` | **off** | Opt-in action-outcome ledger on `GraphEQAMemory`. Accepts `true` / `false` or a dict: `enabled`, `max_records` (default `512`), `persist_absent_claims` (default `false`). When on, nav/tool/manip/verify outcomes append `AttemptRecord` rows alongside per-node `nav_attempts` / `nav_failures`. Env: `EMET_EQA_ATTEMPT_LEDGER`, `EMET_ATTEMPT_LEDGER_MAX`, `EMET_ATTEMPT_LEDGER_PERSIST_ABSENT`. Reference: [attempt_ledger.md](attempt_ledger.md); plan: [plans/2026-08-08_embodied_agent_planning.md](plans/2026-08-08_embodied_agent_planning.md). |
| `eqa.action_progress_mode` | **off** | Static-world semantic retry policy: `shadow` records counterfactual decisions; `enforce` temporarily suppresses an unchanged equivalent action. Non-`off` modes require `eqa.agentic_decision_policy=grounded_v2`. Env: `EMET_EQA_ACTION_PROGRESS_MODE`. Reference: [attempt_ledger.md](attempt_ledger.md#static-world-action-progress-policy). |
| `eqa_vl.eqa_prompt_max_tokens` | `2500` | Approximate text-token budget (char/4) for HISTORY + memory + SCENE_GRAPH. Truncation: oldest HISTORY → memory tail → edges → lowest-ranked node labels. `0` disables. Env: `EMET_EQA_PROMPT_MAX_TOKENS`. |
| `eqa_vl.eqa_max_history` | `4` | Cap on prior iterations; each entry is a one-line **outcome** (`Iter: answer=… conf=… action=… salvage=… \| …`), not a raw VLM replay. |
| `eqa_vl.include_image_descriptions` | `false` | Legacy label dump; when on, labels already on SCENE_GRAPH Image tags are omitted. |

The Habitat wrapper loads [`src/emet/config/dynav_config.yaml`](../src/emet/config/dynav_config.yaml) today; equivalent flat `eqa:` keys there override runner defaults. See [habitat/usage.md](habitat/usage.md#navigation-habitat-only) and [evaluation.md](evaluation.md#hm-eqa-answer-prompt-json--budget). Robocasa / ZMQ GraphEQA ignores Habitat-only keys unless you set them explicitly.

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
