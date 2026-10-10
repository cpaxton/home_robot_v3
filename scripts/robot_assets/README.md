# Robot asset pipeline (STEP → STL → MJCF)

Reusable tooling to vendor a new robot into `src/emet/assets/robot/<name>/`:

1. **`step_to_stl.py`** — convert STEP CAD files into STL meshes (needs `cadquery`/OCP;
   the main emet venv does not install it). Meshes are normalized to meters and
   recentered on their bounding-box centroid so MJCF authors can place them by
   frame alone.
2. **`urdf_to_mjcf.py`** — turn an arm URDF into an MJCF body fragment, carrying
   over the kinematic chain (joint frames/axes/limits), inertials, and visuals.
   This is for standalone arm imports. Sourccey Mk.V instead uses the complete
   official MuJoCo assembly (see below). `--mass-scale` scales link inertials when
   justified by measurements.
   `--recenter-joint shoulder_pan` wraps the fragment so the shoulder pivot sits at
   `arm_root` (required by `assemble_sourccey.py`). Meshes marked `"aligned": true`
   in the mesh-map are placed at the body origin with identity rotation
   (see `align_urdf_meshes.py`).
3. **`assemble_sourccey.py`** — adapt the pinned official full-body Mk.V MJCF
   to Emet planar base joints, hardware arm limits, and actuator names.
   **`sync_sourccey.py`** refreshes the snapshot from clean official simulator and
   hardware checkouts, cross-checks their URDFs and records source hashes.
4. **`align_urdf_meshes.py`** — bake URDF-joint alignment into arm-link STL meshes
   so consecutive links connect. A vendor URDF's visual origins are tuned for its
   own STL frames; STEP-derived meshes have different frames, which leaves gaps
   between links. This rotates each mesh's long axis to the link's entry→exit joint
   direction and centers it on the joint midpoint. Run it when links look
   disconnected and you cannot obtain the vendor's original meshes.
5. **`render_cameras.py`** — render every camera's RGB + depth with a table and
   objects in front, so you can eyeball camera extrinsics/intrinsics before
   trusting perception. Writes PNGs + raw depth `.npy`.
6. **`serve_preview.py`** — render a `Scene`-style preview (top/front/side) of a
   generated MJCF so you can eyeball the assembly before committing.

## Sourccey Mk.V

```bash
python scripts/robot_assets/sync_sourccey.py \
    --simulation /path/to/sourccey-simulation-mujoco \
    --hardware /path/to/sourccey-hardware
uv run python scripts/robot_assets/assemble_sourccey.py
```

No CAD conversion is needed. The original MJCF, full-body URDF and referenced
meshes are retained in `src/emet/assets/robot/sourccey/upstream/`, with the MIT
license and a revision/hash manifest. Generated `sourccey.xml` is the planar
Emet adaptation; the native model's mecanum traction needs upstream's Python
simulation loop. See [`docs/robots/sourccey.md`](../../docs/robots/sourccey.md)
for conventions, provisional dynamics/calibration and hardware arrival checks.

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

# 3. If a standalone arm URDF exists, build a fragment:
uv run python scripts/robot_assets/urdf_to_mjcf.py /path/to/arm.urdf \
    --mesh-map /path/to/mesh_map.json --out /tmp/arm_frag.xml

# 3b. If the STEP mesh frames don't match the URDF visual origins (links look
#     disconnected), bake the URDF-joint alignment into the meshes first:
uv run python scripts/robot_assets/align_urdf_meshes.py \
    --in-dir /tmp/raw_meshes_mm --out-dir /tmp/aligned_meshes \
    --links arm_shoulder=+1,arm_bicep_l=-1,arm_forearm=+1,arm_wrist=+1

# 4. Assemble the full robot MJCF (edit the per-robot assembler).
uv run python scripts/robot_assets/assemble_sourccey.py

# 5. Render a preview.
uv run python scripts/robot_assets/serve_preview.py \
    --mjcf src/emet/assets/robot/sourccey/sourccey.xml \
    --out /tmp/sourccey_preview.png

# 6. Verify cameras: RGB + depth with a table + objects in front.
uv run python scripts/robot_assets/render_cameras.py \
    --mjcf src/emet/assets/robot/sourccey/sourccey.xml \
    --out-dir /tmp/sourccey_cams
```

## Mesh-map file (`--mesh-map`)

JSON mapping URDF mesh basename → final STL path (or name), plus an optional
`"offset_mm"` used to compensate the STEP-part centroid:

```json
{
  "Arm-Base-V3-v1.stl": {"stl": "arm_base.stl", "offset_mm": [0, 0, 0]},
  "Arm-Forearm-v1.stl": {"stl": "arm_forearm.stl", "aligned": true},
  "Feetech-Servo-Motor-v1.stl": {"box_mm": [40, 20, 36.5]}
}
```

Entries with `box_mm` (off-the-shelf servo bodies with no STEP export) keep their
kinematic joints but emit **no visual geom** — the URDF fixed-joint placement can
scatter them far from the visible arm, so they'd render as floating boxes. Entries
with `"aligned": true` are meshes pre-rotated by `align_urdf_meshes.py` (placed at
the body origin with identity rotation).

## Collision & spawn (important for Robocasa/MolmoSpaces)

Robots that must be spawn-placed in Robocasa kitchens should ship **visual-only**
geoms (`contype="0" conaffinity="0"` on every robot geom), matching `innate_mars`.
The planar autoplace probes each candidate (x, y, yaw) with `mj_collision` over the
whole scene; a robot with dense mesh collision geoms makes that search slow
(hundreds of candidates × 900 geoms). With visual-only geoms the hint is accepted
instantly and spawn is O(1). Spawn safety is enforced by `planar_spawn_clip_guard_body_names`
+ footprint margin; motion-planning collision is delegated to external planners.
Sourccey initially shipped mesh collision geoms and Robocasa autoplace took >90 s —
converting to visual-only made it 0.0 s like innate_mars / xlerobot.

## License

Converted meshes are derived from upstream CAD. Preserve the upstream license
(NOTICE) next to the vendored MJCF assets, see
`src/emet/assets/robot/sourccey/NOTICE.md`.
