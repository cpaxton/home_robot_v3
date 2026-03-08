# Robot Model Assets

This directory contains consolidated robot model assets used across the codebase.

## Structure

- **robot/** - Stretch robot models for MuJoCo simulation
  - `stretch.xml` - Main robot MuJoCo model
  - `docking_station.xml` - Docking station model
  - `scene.xml` - Default scene (robot + docking station)
  - `assets/` - Mesh files (obj, stl) referenced by the XML models

## Usage

Use `stretch.utils.assets` to get asset paths programmatically:

```python
from stretch.utils.assets import get_mujoco_models_path, get_robot_assets_path

models_path = get_mujoco_models_path()  # Path to robot/ (stretch.xml, etc.)
scene_xml = get_mujoco_models_path() / "scene.xml"
```

## Note on stretch_urdf

URDF models and meshes for visualization/kinematics come from the external
`hello-robot-stretch-urdf` package. The assets here are MuJoCo-specific
and may differ in format (e.g., obj vs stl) from the URDF package.
