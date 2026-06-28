# ZMQ observation tools (`capture` / `stream`)

Live and one-shot clients for the standard emet ZMQ observation port (**4401+** `port-offset`). Both CLI commands are **profile shortcuts** into the same Python runner (`emet.app.zmq_obs`).

| Command | Profile | Typical use |
|---------|---------|-------------|
| `emet capture` | `capture` | Smoke test: save montage + `metadata.json`; optional one mapping step |
| `emet stream` | `stream` | Live Rerun viewer; continuous mapping until Ctrl+C or `--max-steps` |

```bash
emet capture --help
emet stream --help
```

Related: [cli.md](cli.md) (flag tables), [rerun.md](rerun.md) (world frame / viewer), [experiments/innate_mars.md](experiments/innate_mars.md) (hardware matrix), [known_issues.md](known_issues.md) (open bugs).

---

## Architecture

```
emet capture  ──► capture_main()  ──┐
                                    ├──► run_zmq_obs(ZmqObsRun)
emet stream   ──► stream_main()   ──┘
         │
         ├─ zmq_cli_resolve        (--ip / --connection → host, robot)
         ├─ artifact save          (capture always; stream if --out-dir)
         ├─ stream_agent_factory   (--backend → controller + ZMQ client)
         └─ zmq_mapping_session    (agent.update() loop, Rerun)
```

| Module | Role |
|--------|------|
| [`stream_config.py`](../src/emet/config/stream_config.py) | ``stream:`` YAML block (`verbose`, `da3_log_level`, status interval) |
| [`zmq_obs.py`](../src/emet/app/zmq_obs.py) | CLI entrypoints, artifact I/O, `run_zmq_obs` |
| [`zmq_mapping_session.py`](../src/emet/app/zmq_mapping_session.py) | Shared mapping loop (`run_mapping_session`) |
| [`stream_agent_factory.py`](../src/emet/app/stream_agent_factory.py) | Backend agents (`STREAM_BACKENDS`) |
| [`zmq_cli_resolve.py`](../src/emet/app/zmq_cli_resolve.py) | Connection profile → host/robot |
| [`capture.py`](../src/emet/app/capture.py) / [`stream.py`](../src/emet/app/stream.py) | Thin re-exports of `zmq_obs` commands |

`emet preview-cameras --source zmq` uses the same ZMQ port for a **montage only** (no Rerun session, no mapping). Use it when you only need to check camera orientation.

---

## Profile defaults

| Behavior | **capture** | **stream** |
|----------|-------------|------------|
| Save montage + per-camera JPEGs + `metadata.json` | Always → `runs/capture/<robot>_<timestamp>/` (override `--out-dir`) | Only with `--out-dir` |
| Mapping without `--backend` | Save artifacts and exit | localhost → cameras-only Rerun; **remote → `dynamem`** |
| Mapping with `--backend` | **One** `agent.update()` | Loop at `--hz` until Ctrl+C or `--max-steps` |
| Rerun after mapping | `--rerun-hold-s` (default **30s**); `--no-rerun` to skip | Continuous while session runs |
| Graph / sim flags | — | `--cameras-only`, `--no-sensor-perception`, `--compare-to-gt`, … |

**Equivalence:** `emet capture --backend X` ≈ `emet stream --backend X --max-steps 1` plus artifact folder. `emet stream --out-dir …` adds capture-style save before streaming.

---

## `--backend` values

Same names as paper evals and `emet run dynamem` / `emet run dynagraph` controllers (without rotate/explore/nav):

| `--backend` | Stack |
|-------------|--------|
| `dynamem` | Full DynaMem voxel semantic map |
| `voxel_only` | Voxel + depth only (no SigLIP/YoloE/VLM; DA3 still runs when depth missing) |
| `graph_eqa` | Voxel + GraphEQA graph |
| `dynagraph` | Voxel + merged graph + VLM |
| `ground_truth` | Sim GT graph (`emet_session`; sim only) |
| `svm` | Instance memory |
| `scene_graph` | Voxel + open-vocab scene graph |

---

## Connection and depth

**Host / robot:** `--ip` / `--robot-ip`, or `--connection NAME` from `emet connect save …` (see [cli.md](cli.md) § connect). Default robot: `stretch`.

**Dynav YAML:** `--dynav-config` (default `dynav_config.yaml`). For **innate_mars** on a **non-localhost** host, the runner auto-selects `dynav_innate_mars.yaml` (DA3 stereo; hardware ZMQ has no depth). Override anytime with `--dynav-config`.

**Stream terminal defaults** (in dynav YAML, see `agents/default_stream.yaml`):

```yaml
stream:
  verbose: false
  status_interval_s: 5.0
  da3_log_level: WARN
  log_every_step_when_max_steps_le: 20
  rerun_hold_s: 30.0
```

Precedence: CLI `--verbose` / `--rerun-hold-s` → YAML `stream:` → env (`EMET_STREAM_VERBOSE`, `DA3_LOG_LEVEL`).

**RGB-only bridge:** `innate_mars` hardware sets `allow_missing_depth` automatically on stream; pass `--allow-missing-depth` explicitly if needed.

---

## Examples

### Sim (localhost)

```bash
# One frame on disk
emet capture

# Live cameras + MJCF in Rerun
emet stream --cameras-only

# Growing voxel map
emet stream --backend voxel_only --max-steps 10 --headless
```

### Hardware (saved profile)

```bash
emet connect save herman --user jetson1 --robot innate_mars --name herman
emet mars start --connection herman

# Smoke: images + metadata, no GPU map
emet capture --connection herman

# One voxel step, no viewer
emet capture --connection herman --backend voxel_only --no-rerun

# Live map (remote defaults to dynamem if --backend omitted)
emet stream --connection herman --backend voxel_only --max-steps 3 --headless
```

### Rerun remote viewing

```bash
emet stream --connection herman --backend dynamem --rerun-bind
# Viewer on workstation: http://<host>:9090?url=ws://<host>:9877
```

---

## Artifact layout (`capture` / `--out-dir`)

```
runs/capture/<robot>_<timestamp>/
  montage.png
  rgb_<cam>.jpg
  rgb_right_<cam>.jpg   # when present
  metadata.json         # joints, poses, GPS/compass, optional "map" stats
```

---

## What these commands do *not* do

- **No base or head motion** — unlike `emet run dynamem` / `emet run dynagraph` (rotate, explore, nav). For stationary mapping on hardware, prefer `capture` / `stream` with `--max-steps`.
- **Not a full eval harness** — for paper benchmarks use `scripts/eval_*.py` and [experiments/README.md](experiments/README.md).
- **DA3 debug without voxels** — use `emet debug-da3-depth`.

### Terminal noise (`dynamem` on hardware)

On **innate_mars** without ZMQ depth, every mapping step runs **DA3** stereo inference (the `[INFO ] Processed Images Done…` lines). That is normal for `--backend dynamem` and `--backend voxel_only`; full `dynamem` also loads OwlV2 and other models.

By default, `emet stream` sets **`DA3_LOG_LEVEL=WARN`** and prints map status **every 5 seconds** (not every step). For smoke tests with `--max-steps ≤ 20`, status prints each step.

| Quieter | Verbose |
|---------|---------|
| `emet stream … --backend voxel_only` (no SigLIP/VLM) | `emet stream … --verbose` |
| default (no flag) | `DA3_LOG_LEVEL=INFO emet stream …` |
| `--max-steps 3` (short run, per-step status) | |

Explored cell count can fluctuate slightly step-to-step (map filtering / pose change) — watch the trend, not every tick.

For **`dynagraph` / `graph_eqa`**, periodic status includes a graph breakdown, e.g. `graph 11 obj / 12 vp / 1 fr (24 total)` — **object** nodes are what matter for search/EQA; **viewpoint** (`vp`) and **frontier** (`fr`) nodes are auxiliary. See [known_issues.md](known_issues.md#dynagraph-graph-node-explosion-on-stationary-hardware-stream).
