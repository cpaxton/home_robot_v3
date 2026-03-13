# Commit messages for feature/molmo-spaces

Suggested commits, ordered so each is a logical unit. Omit `__pycache__/` from all commits.

---

## 1. controller: make StretchZmqClient.stop() safe when __init__ fails partway

**Files:** `src/emet/controller/zmq_client.py`

- Initialize `_servo_thread` and `_rerun_thread` in `__init__` (they were referenced in `stop()` but not set there), so `stop()` never touches undefined attributes.
- In `stop()`, use `getattr(self, attr, None)` for threads and for sockets/context so we never assume they exist. If `__init__` failed before creating sockets or context, skip closing them and only join threads that exist.
- Close sockets and `context.term()` inside try/except so transient errors during teardown don’t raise.
- Docstring: clarify that `stop()` is safe to call even if `__init__` failed partway.

---

## 2. mapping/instance: restore Instance and InstanceView from home_robot_v2, extend InstanceMemory stub

**Files:** `src/emet/mapping/instance/instance.py`

- **InstanceView**: Replace minimal stub with full dataclass from home_robot_v2: required fields `bbox`, `bounds`, `timestep`; optional `text_description`, `cropped_image`, `embedding`, `mask`, `image_instance_id`, `visual_feat`, `global_instance_id`, `category_id`, `score`, `point_cloud`, `point_cloud_rgb`, `point_cloud_features`, `cam_to_world`, `pose`. Add `__repr__`, `get_pose()`, `object_coverage` (cached_property), `show()` / `_show_folder()`, and `get_image()` with proper null handling and [H,W,3] uint8 output.
- **Instance**: Replace minimal stub with dataclass: `name`, `global_id`, `category_id`, point clouds, `bounds`, `instance_views`, `score`, `score_aggregation_method`. Add `__repr__`, `id` property, `get_category_id()`, `get_image_embedding()` (with aggregation and normalize), `get_best_view()` (area/update_time, dummy view when no views), `get_instance_id()`, `get_center()`, `get_median()`, `get_closest_point()`, `show_best_view()`, and `add_instance_view()` (merge point clouds and scores). Introduce `_dummy_instance_view()` for no-view case.
- **InstanceMemory**: Add `__len__` (total instances across envs). `process_instances_for_env` accepts `**kwargs`. `global_box_compression_and_nms` returns `[]`. `pop_global_instance` takes optional `skip_reindex` and returns the popped `Instance` or None. Docstrings updated; stub behavior (no-ops) unchanged so controller/stretch imports still work without full instance_map.
- Use `Optional[Tensor]` / `List` etc. for compatibility; add `get_bounds` from `emet.utils.point_cloud_torch` for merged point-cloud bounds.

---

## 3. motion: Stretch URDF fallback and Dynamem joint_fake generation

**Files:** `src/emet/motion/constants.py`

- Prefer config `urdf/stretch.urdf` when present; when missing, discover `stretch_urdf` package and pick dex_wrist URDF (RE1V0/RE2V0) or first available RE1V0 URDF.
- Add `_generate_dynamem_stretch_urdf()`: parse package URDF, resolve mesh paths relative to package dir, inject `joint_fake` / `fake_link_x` snippet so `joint_mast` parent is `fake_link_x` (OK-Robot/Dynamem manipulation stack), write result to config path.
- Export `get_stretch_urdf_path()` and set `MANIP_STRETCH_URDF = get_stretch_urdf_path()` so kinematics and other callers get a valid URDF with or without pre-installed config file.

---

## 4. motion/kinematics: treat MANIP_STRETCH_URDF as absolute when applicable

**Files:** `src/emet/motion/kinematics.py`

- When `urdf_path` is not given, `manip_urdf` may now be an absolute path (from `get_stretch_urdf_path()`). Avoid joining with `root` when `manip_urdf` is already absolute so `HelloStretchKinematics.manip_mode_urdf_path` is correct.

---

## 5. robots: add Footprint class for base/footprint without motion dependency

**Files:** `src/emet/robots/footprint.py` (new)

- Add `Footprint(length, width, length_offset, width_offset)` with `get_box()` (3D box for visuals), `get_mask(resolution)` (boolean numpy mask), and `get_rotated_mask(resolution, angle_radians)` (scipy.ndimage.rotate, order=0). Pure numpy/scipy; no torch or pinocchio so `emet.robots.base` and backends can use it without pulling in `emet.motion` (e.g. for `emet serve mujoco --robot rby1`).

---

## 6. install: text-based install menu for sub-assets (emet install menu)

**Files:** `src/emet/install_ui.py` (new)

- Add `run_install_menu()`: status checks for submodules (SAM-2), sim (robosuite + robocasa), kitchen assets (textures/fixtures), and MolmoSpaces wrapper (.venv-molmospaces / emet-molmospaces). Menu options to install each, “Run all”, or “Sync extras” (uv sync: sim, dynamem, dev). Uses `AssetStatus` dataclass and box-drawn UI; invokes git submodule, install_simulation.sh, download_robocasa_assets.py, and MolmoSpaces venv creation as appropriate. For use by `emet install menu`.

---

## 7. scripts: Robocasa placeholder models for missing fixture paths

**Files:** `scripts/ensure_robocasa_placeholders.py` (new)

- Script that reads Robocasa fixture_registry YAMLs, collects referenced `fixtures/...` paths, and for each path missing a `model.xml` on disk writes a minimal MuJoCo placeholder (worldbody with sites and a small box). Allows `emet serve mujoco --use-robocasa` to run when the full fixture pack is not extracted or some assets are missing. Requires robocasa and pyyaml; intended to be run from project env (`uv run python scripts/ensure_robocasa_placeholders.py`).

---

## 8. scripts: MolmoSpaces wrapper venv install script

**Files:** `scripts/install_molmospaces.sh` (new)

- Bash script (run from repo root) that creates `.venv-molmospaces`, sets `MLSPACES_ASSETS_DIR` default, and installs emet (no-deps) then either `packages/emet_molmospaces` (editable) if present or molmo-spaces + mujoco>=3.4 + numpy>=2.2. Uses `uv pip` when available. Idempotent: if venv exists, only adds wrapper if dir exists and script not yet installed.

---

## 9. docs: MolmoSpaces integration testing plan

**Files:** `docs/plans/2025-03-10_molmospaces_testing.md` (new)

- Add testing plan for MolmoSpaces: core CLI tests (no wrapper), wrapper package tests (mocked molmo_spaces), graceful failure when wrapper missing, install steps, manual CLI checks (list-robots, list-scenes, install-scene, serve viewer/headless/rerun), optional integration test with wrapper, and regression (rest of CLI and sim). Includes commands and expected outcomes table.

---

## 10. packages: add emet_molmospaces wrapper package

**Files:** `packages/emet_molmospaces/` (all tracked files; exclude `**/__pycache__`)

- Add optional wrapper package providing `emet-molmospaces` console script: list-scenes, install-scene, serve (viewer/headless/rerun) using molmo_spaces and mujoco. Depends on emet (config), molmo-spaces, mujoco>=3.4, numpy>=2.2. Core emet discovers the wrapper via subprocess; this package is installed into .venv-molmospaces by install_molmospaces.sh or `emet install menu`. Include CLI, runner, tests (mocked API), README, and pyproject.toml.

---

## Quick reference (one-liners)

If you prefer single-line subject lines:

1. `controller: make StretchZmqClient.stop() safe when __init__ fails partway`
2. `mapping/instance: restore Instance/InstanceView from v2, extend InstanceMemory stub`
3. `motion: Stretch URDF fallback and Dynamem joint_fake generation`
4. `motion/kinematics: treat MANIP_STRETCH_URDF as absolute when applicable`
5. `robots: add Footprint class (no motion/pinocchio dependency)`
6. `install: add text-based install menu for sub-assets (emet install menu)`
7. `scripts: add ensure_robocasa_placeholders.py for missing fixture paths`
8. `scripts: add install_molmospaces.sh for wrapper venv`
9. `docs: add MolmoSpaces integration testing plan`
10. `packages: add emet_molmospaces wrapper package`
