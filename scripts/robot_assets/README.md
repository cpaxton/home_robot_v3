# Robot asset pipeline (STEP → STL → MJCF)

Reusable tooling to vendor a new robot into `src/emet/assets/robot/<name>/`:

1. **`step_to_stl.py`** — convert STEP CAD files into STL meshes (needs `cadquery`/OCP;
   the main emet venv does not install it). Meshes are normalized to meters and
   recentered on their bounding-box centroid so MJCF authors can place them by
   frame alone.
2. **`urdf_to_mjcf.py`** — turn an arm URDF into an MJCF body fragment, carrying
   over the kinematic chain (joint frames/axes/limits), inertials, and visuals.
   Used when the vendor already ships a URDF (e.g. `lerobot-vulcan`'s
   `Arm.urdf` for Sourccey).
3. **`assemble_sourccey.py`** — full-robot example: base/lift/dome/arms/cameras
   assembled into one `sourccey.xml` with planar base joints + actuators.
4. **`serve_preview.py`** — render a `Scene`-style preview (top/front/side) of a
   generated MJCF so you can eyeball the assembly before committing.

## One-time env (cadquery is only needed for the STEP step)

```bash
uv venv /tmp/robot_assets_venv --python 3.10
uv pip install --python /tmp/robot_assets_venv/bin/python cadquery
export ROBOT_ASSETS_PY=/tmp/robot_assets_venv/bin/python
```

## Pipeline for a new robot

```bash
# 1. Vendor the source CAD + (optional) URDF under a scratch dir.
# 2. Convert STEP -> STL.
$ROBOT_ASSETS_PY scripts/robot_assets/step_to_stl.py \
    /path/to/CAD/Robot/Arms/Base.step \
    --out /tmp/meshes/arm_base.stl --scale 0.001

# 3. If a URDF exists, build the arm fragment + preview.
$ROBOT_ASSETS_PY scripts/robot_assets/urdf_to_mjcf.py \
    /path/to/Arm.urdf --mesh-map /tmp/mesh_map.json \
    --out /tmp/arm_frag.xml

# 4. Assemble the full robot MJCF (edit the per-robot assembler).
uv run python scripts/robot_assets/assemble_sourccey.py

# 5. Render a preview.
uv run python scripts/robot_assets/serve_preview.py \
    --mjcf src/emet/assets/robot/sourccey/sourccey.xml \
    --out /tmp/sourccey_preview.png
```

## Mesh-map file (`--mesh-map`)

JSON mapping URDF mesh basename → final STL path (or name), plus an optional
`"offset_mm"` used to compensate the STEP-part centroid:

```json
{
  "Arm-Base-V3-v1.stl": {"stl": "arm_base.stl", "offset_mm": [0, 0, 0]},
  "Feetech-Servo-Motor-v1.stl": {"box_mm": [40, 20, 36.5], "color": "servo_dark"}
}
```

Entries with `box_mm` render as a `box` geom instead of a mesh (e.g. an
off-the-shelf servo body with no STEP export).

## License

Converted meshes are derived from upstream CAD. Preserve the upstream license
(NOTICE) next to the vendored MJCF assets, see
`src/emet/assets/robot/sourccey/NOTICE.md`.
