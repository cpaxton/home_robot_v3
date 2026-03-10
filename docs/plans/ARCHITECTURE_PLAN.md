# EMET Architecture Plan: Multi-Robot, Multi-Simulator Refactor

**EMET** = Embodied Multi-robot Environment Toolkit (rename from stretch-centric to generic).

## Summary

| Change | Approach |
|-------|----------|
| Rename | `stretch` → `emet` everywhere |
| Robots | Stretch (migrate) + stubs: Mobile ALOHA, Galaxea R1, Innate Mars, YOR |
| Simulators | MuJoCo (migrate) + stubs: Behavior 1k, Molmo Spaces |
| Extensibility | `RobotBackend` + `BaseSimulatorServer` abstractions; registry for discovery |

## Goals

1. **Rename** `src/stretch` → `src/emet` everywhere
2. **Robot abstraction** — support Stretch, Mobile ALOHA, Galaxea R1, Innate Mars, YOR
3. **Simulator abstraction** — support MuJoCo (current), Behavior 1k, Molmo Spaces
4. **Extensible design** — easy to add new robots and simulators via stubs/plugins

---

## Proposed Directory Layout

```
src/
├── emet/                          # Main package (renamed from stretch)
│   ├── __init__.py
│   ├── version.py
│   │
│   ├── core/                      # Shared interfaces (robot-agnostic)
│   │   ├── interfaces.py         # Observations, Actions, Pose
│   │   ├── robot.py              # AbstractRobotClient (unchanged interface)
│   │   ├── server.py             # BaseZmqServer (unchanged)
│   │   ├── client.py
│   │   └── comms.py
│   │
│   ├── robots/                    # Robot implementations (NEW)
│   │   ├── __init__.py
│   │   ├── base.py               # RobotSpec, RobotModel base
│   │   ├── stretch/              # Stretch 3 (current code, moved)
│   │   │   ├── __init__.py
│   │   │   ├── client.py         # StretchZmqClient (from HomeRobotZmqClient)
│   │   │   ├── kinematics.py
│   │   │   ├── motion/
│   │   │   └── ...
│   │   ├── mobile_aloha/         # STUB
│   │   │   ├── __init__.py
│   │   │   └── stub.py
│   │   ├── galaxea_r1/            # STUB
│   │   │   ├── __init__.py
│   │   │   └── stub.py
│   │   ├── innate_mars/           # STUB
│   │   │   ├── __init__.py
│   │   │   └── stub.py
│   │   └── yor/                   # STUB (YOR-robot/YOR)
│   │       ├── __init__.py
│   │       └── stub.py
│   │
│   ├── simulators/                # Simulator backends (NEW)
│   │   ├── __init__.py
│   │   ├── base.py               # BaseSimulatorServer (abstract)
│   │   ├── mujoco/                # Current MuJoCo (moved from simulation/)
│   │   │   ├── __init__.py
│   │   │   ├── server.py         # MujocoZmqServer
│   │   │   ├── stretch_mujoco/   # Stretch-specific MuJoCo
│   │   │   └── ...
│   │   ├── behavior1k/           # STUB
│   │   │   ├── __init__.py
│   │   │   └── stub.py
│   │   └── molmo_spaces/          # STUB
│   │       ├── __init__.py
│   │       └── stub.py
│   │
│   ├── perception/                # (unchanged structure, imports updated)
│   ├── mapping/
│   ├── agent/
│   ├── visualization/
│   ├── utils/
│   ├── llms/
│   ├── audio/
│   ├── config/
│   └── app/
│
├── emet_ros2_bridge/              # Renamed from stretch_ros2_bridge (Stretch-specific)
└── test/
```

---

## Key Abstractions

### 1. Robot Interface (`emet.robots.base`)

```python
# RobotSpec: declarative config for a robot (DOF, cameras, URDF path, etc.)
@dataclass
class RobotSpec:
    name: str
    dof: int
    joint_names: List[str]
    camera_names: List[str]
    urdf_path: Optional[str]
    footprint: Footprint

# Base class for robot-specific logic
class RobotBackend(ABC):
    @abstractmethod
    def get_spec(self) -> RobotSpec: ...

    @abstractmethod
    def create_client(self, robot_ip: str, **kwargs) -> AbstractRobotClient: ...

    @abstractmethod
    def create_model(self, **kwargs) -> RobotModel: ...
```

### 2. Simulator Interface (`emet.simulators.base`)

```python
class BaseSimulatorServer(ABC):
    """Server that publishes observations and receives actions (ZMQ or other)."""

    @abstractmethod
    def get_robot_spec(self) -> RobotSpec:
        """Which robot does this simulator emulate?"""
        ...

    @abstractmethod
    def get_full_observation_message(self) -> Dict[str, Any]: ...

    @abstractmethod
    def get_state_message(self) -> Dict[str, Any]: ...

    @abstractmethod
    def get_servo_message(self) -> Dict[str, Any]: ...

    @abstractmethod
    def handle_action(self, action: Dict[str, Any]): ...

    @abstractmethod
    def is_running(self) -> bool: ...
```

### 3. Registry Pattern (optional, for discovery)

```python
# emet/robots/__init__.py
ROBOT_REGISTRY = {
    "stretch": "emet.robots.stretch",
    "mobile_aloha": "emet.robots.mobile_aloha",
    "galaxea_r1": "emet.robots.galaxea_r1",
    "innate_mars": "emet.robots.innate_mars",
    "yor": "emet.robots.yor",
}

# emet/simulators/__init__.py
SIMULATOR_REGISTRY = {
    "mujoco": "emet.simulators.mujoco",
    "behavior1k": "emet.simulators.behavior1k",
    "molmo_spaces": "emet.simulators.molmo_spaces",
}
```

---

## Stub Contents

Each stub provides:

1. **`__init__.py`** — exports `RobotBackend` or `SimulatorServer` class (or placeholder)
2. **`stub.py`** or **`__init__.py`** — `NotImplementedError` / `TODO` with link to upstream

Example stub (`emet/robots/yor/__init__.py`):

```python
"""YOR Robot — https://github.com/YOR-robot/YOR"""

from emet.robots.base import RobotBackend, RobotSpec

class YORBackend(RobotBackend):
    """Stub for YOR robot integration."""

    def get_spec(self) -> RobotSpec:
        raise NotImplementedError(
            "YOR integration is a stub. See https://github.com/YOR-robot/YOR"
        )

    def create_client(self, robot_ip: str, **kwargs):
        raise NotImplementedError("YOR client not yet implemented")

    def create_model(self, **kwargs):
        raise NotImplementedError("YOR model not yet implemented")
```

---

## Migration Strategy

### Phase 1: Rename (mechanical)
- `stretch` → `emet` in all imports, package names, pyproject.toml, setup.py
- `stretch_ai` → `emet` (or keep `emet` as internal name, `stretch_ai` as PyPI name for compatibility — discuss)
- Update all `from emet.X` → `from emet.X`

### Phase 2: Extract robot layer
- Move Stretch-specific code under `emet/robots/stretch/`
- Keep `AbstractRobotClient`, `BaseZmqServer` in `emet/core/`
- `HomeRobotZmqClient` → `StretchZmqClient` in `emet/robots/stretch/client.py`

### Phase 3: Extract simulator layer
- Move `simulation/` → `simulators/mujoco/`
- Add `BaseSimulatorServer` in `emet/simulators/base.py`
- `MujocoZmqServer` extends it (already does via `BaseZmqServer`)

### Phase 4: Add stubs
- Create `emet/robots/{mobile_aloha,galaxea_r1,innate_mars,yor}/` with stubs
- Create `emet/simulators/{behavior1k,molmo_spaces}/` with stubs

### Phase 5: App entry points
- `python -m emet.app.run_dynamem --robot stretch --simulator mujoco ...`
- Or keep current entry points but resolve robot/sim via config

---

## Backward Compatibility

- **Entry points**: Keep `python -m emet.app.run_dynamem` as alias → `emet.app.run_dynamem` (or deprecate)
- **Config files**: `stretch/config/` → `emet/config/` (paths in YAML may need update)
- **Assets**: `stretch/assets/` → `emet/assets/` (or `emet/robots/stretch/assets/`)

---

## Open Questions

1. **Package name**: Keep `stretch_ai` on PyPI for existing users, or rename to `emet`?
2. **Behavior 1k / Molmo Spaces**: Are these GitHub repos public? Need to confirm APIs before stubbing.
3. **YOR**: Minimal info at https://github.com/YOR-robot/YOR — stub with link only.
4. **ROS2 bridge**: Rename to `emet_ros2_bridge` or keep `stretch_ros2_bridge` (Stretch-specific)?

---

## Implementation Order

1. ~~Create `docs/plans/ARCHITECTURE_PLAN.md` (this file)~~
2. ~~Create `src/emet/` skeleton with `core/`, `robots/base.py`, `simulators/base.py`~~
3. ~~Add robot stubs: `stretch` (real), `mobile_aloha`, `galaxea_r1`, `innate_mars`, `yor`~~
4. ~~Add simulator stubs: `mujoco` (real), `behavior1k`, `molmo_spaces`~~
5. ~~Mechanical rename: `stretch` → `emet`~~
6. Move Stretch code into `emet/robots/stretch/` (optional, incremental)
7. Move MuJoCo code into `emet/simulators/mujoco/` (optional, incremental)
8. Update all imports and tests
9. Update pyproject.toml, setup.py, install.sh, docs
