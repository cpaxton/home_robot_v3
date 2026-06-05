# ZMQ `emet_session` metadata

Every full-observation, state, and (where applicable) servo message from Emet’s MuJoCo ZMQ servers includes a static dict under the key `emet_session`. Clients should treat unknown keys as forward-compatible and must not assume the block is present (older servers).

## Required shape (schema version 1)

| Key | Type | Meaning |
|-----|------|--------|
| `schema_version` | `int` | Must be `1` for this document. |
| `runtime_kind` | `str` | e.g. `robosuite_sim`, `stretch_mujoco_sim`; hardware bridges should use a distinct value (e.g. `hardware_stretch`). |
| `is_simulation` | `bool` | Same meaning as the top-level `is_simulation` flag on observations. |
| `emet_robot_id` | `str` | Duplicate of top-level `emet_robot_id` for a single merge point. |
| `capabilities` | `dict` | Declared features, e.g. `teleport_base` (bool), `depth`, `num_cameras`, `dof`. |
| `environment` | `dict` | Optional scene identity; see below. |

Optional: `mjcf_model_name`, `scene_source_basename` (short debugging strings).

### `sim_object_placements` (simulation only)

When the MuJoCo server knows object poses at load time, it may include:

```json
"sim_object_placements": {
  "apple_main": {"cat": "apple", "pos": [0.1, -0.5, 0.9], "quat": [1, 0, 0, 0]},
  "object2": {"cat": "red cylinder", "pos": [0.08, -0.55, 0.6], "quat": [1, 0, 0, 0]}
}
```

- **Robocasa**: from the scene wizard (`object_placements_info`).
- **Default table scene**: fixed table + red cylinder + blue cube from `scene_environment.xml` (Stretch `stretch_mujoco_sim` and Robosuite `robosuite_sim`, e.g. rby1).
- **MolmoSpaces merged MJCF**: body scan at server start (non-robot bodies with geoms; capped count on iTHOR-scale scenes).

Clients use this for Dynagraph **`--ground-truth`** mode and dev alignment checks (`mujoco_align`). Keys starting with `_emet_` are internal (e.g. spawn hints) and are not graph objects.

## `environment` kinds

- `molmospaces`: `scene`, `split`, `index` identify the dataset row used to build the merged MJCF.
- `robocasa`: `task`, `style`, `layout` when the Robocasa wizard was used.
- `default_table` / `stretch_default_scene`: generic packaged scenes.

## Hardware / third-party publishers

If you implement a bridge that speaks the same ZMQ observation protocol:

1. Set `is_simulation` to `false` and pick a `runtime_kind` that is not `*_sim`.
2. Populate `capabilities` honestly (e.g. `teleport_base: false` unless you implement teleport semantics).
3. Include `schema_version: 1` and `emet_robot_id` consistent with top-level `emet_robot_id`.

## Client API

- `GenericZmqClient.get_emet_session()` and `StretchZmqClient.get_emet_session()` return a shallow copy of the latest block, or `None`.
