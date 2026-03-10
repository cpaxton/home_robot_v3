# Refactor log: 2026-03-08 — Rename and skeleton

## Summary

- Renamed `stretch` → `emet` across the codebase
- Added `emet/robots/` and `emet/simulators/` skeleton with base classes and stubs
- Documented deleted-file analysis from the rename commit

---

## 1. Rename (stretch → emet)

### Strategy

- Replaced all `from stretch.` / `import stretch` with `emet` in Python
- Updated `pyproject.toml`, `setup.py`, package_data, config paths
- Updated scripts and docs (`src/stretch` → `src/emet`)
- Left unchanged: `stretch_ros2_bridge`, `stretch_mujoco`, `stretch.urdf`, `stretch.xml`, class names like `StretchCameras`, ROS topics

### Files changed

- `pyproject.toml`: `name = "emet"`, `package-data` → `emet`
- `src/setup.py`: `emet/version.py`, `name="emet"`, `package_data={"emet": [...]}`
- `src/stretch_ros2_bridge/setup.py`: `install_requires` → `"emet"`
- `src/emet/utils/config.py`: `emet.__path__` for CONFIG_ROOT
- `src/emet/utils/assets.py`: `importlib.resources.files("emet")`
- Docker scripts, run scripts, docs: `src/stretch` → `src/emet`

---

## 2. Skeleton (robots + simulators)

### Strategy

- Add `RobotSpec` and `RobotBackend` in `emet/robots/base.py`
- Add `BaseSimulatorServer` in `emet/simulators/base.py`
- Implement Stretch backend; add stubs for Mobile ALOHA, Galaxea R1, Innate Mars, YOR
- Implement MuJoCo simulator; add stubs for Behavior 1k, Molmo Spaces
- Wire `MujocoZmqServer` to `get_robot_spec()` returning `StretchBackend().get_spec()`

### Layout

```
emet/
├── robots/
│   ├── base.py          # RobotSpec, RobotBackend
│   ├── stretch/         # StretchBackend (real)
│   ├── mobile_aloha/    # stub
│   ├── galaxea_r1/      # stub
│   ├── innate_mars/     # stub
│   └── yor/             # stub
└── simulators/
    ├── base.py          # BaseSimulatorServer
    ├── mujoco/          # MujocoZmqServer (real, re-exports from simulation/)
    ├── behavior1k/      # stub
    └── molmo_spaces/    # stub
```

### Deferred

- Moving Stretch code into `emet/robots/stretch/`
- Moving MuJoCo code into `emet/simulators/mujoco/`

---

## 3. Deleted-file analysis (rename commit dfdaca2)

### Issue

During the rename, some files were dropped or moved to wrong paths.

### Wrong path mappings

| Source | Destination | Problem |
|--------|-------------|---------|
| `app/debug/__init__.py` | `config/__init__.py` | Wrong target |
| `app/dex_teleop/__init__.py` | `motion/utils/__init__.py` | Wrong target |

### Files deleted without replacement

| File | Status |
|------|--------|
| `agent/debug/arm_zmq_client.py` | Lost — debug client |
| `app/debug/base_velocity.py` | Restored |
| `app/debug/camera_info.py` | Restored |
| `config/__init__.py` | Overwritten by wrong content |
| `config/urdf/stretch.urdf` | Restored |
| `mapping/instance/*` | Restored |
| `motion/utils/__init__.py` | Overwritten by wrong content |
| `perception/captioners/example.jpg` | Lost |

### Likely cause

Bulk rename used incorrect path mappings. Some directories were not fully moved or were mapped to wrong targets.

### Restore from git

```bash
git show dfdaca2^:src/stretch/agent/debug/arm_zmq_client.py  # restore
git show dfdaca2^:src/stretch/perception/captioners/example.jpg  # restore
```

---

## 4. Restore (2026-03-08 follow-up)

- Restored `agent/debug/arm_zmq_client.py` with `emet` imports; added `start_immediately=False` for blocking_spin
- Restored `perception/captioners/example.jpg` (Git LFS pointer; run `git lfs pull` for blob)
- Updated captioners to use `Path(__file__).parent / "example.jpg"` so they work from any cwd
- Verified: `mapping.instance`, `SparseVoxelMap`, `SparseVoxelMapDynamem`, `run_dynamem`, `mapping` app all import and run

---

## Next steps

1. ~~Restore `arm_zmq_client.py` and `example.jpg`~~ (done)
2. Optionally move Stretch code into `emet/robots/stretch/`
3. Optionally move MuJoCo code into `emet/simulators/mujoco/`
4. Add `--robot` and `--simulator` CLI args to app entry points
