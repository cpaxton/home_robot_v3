"""Named proprioceptive prefix shared by observation producers and consumers.

Phase, task-pose and preview fields follow this prefix and are never risk inputs.
Widths depend on the robot; previous actions must not include phase columns.
"""

# Canonical one-hot ordering of ManipulationControl phases. The ``phase``
# history field is a seven-column one-hot in exactly this order; consumers must
# not index it with bare integers.
PHASE_NAMES = ("reach", "pregrasp", "close", "carry", "place", "release", "disturbance")

# Grip command channels are only active while the object is already closed and
# held; the executor owns the close transition itself, not the learned actor.
GRIP_PHASES = ("carry", "place", "disturbance")
GRIP_PHASE_INDICES = tuple(PHASE_NAMES.index(name) for name in GRIP_PHASES)

# task_pose_preview_v1 emits a fixed seven-row preview: one past sample, the
# present and five future samples. Consumers flatten those rows into context.
PREVIEW_ROWS = 7

# Row semantics within that preview: row zero is the past reference sample,
# row one the present, and row four the four-steps-ahead future sample. The
# gain feedback and base navigation law read exactly these two rows.
CURRENT_PREVIEW_ROW = 1
FUTURE_PREVIEW_ROW = 4

# Reference-twist column layout. The nine-column motion vector is base linear
# x/y, base yaw, hand linear x/y/z and hand angular x/y/z. The differential
# base only uses forward (x) and yaw as announced feedforward; lateral is not
# directly commanded.
MOTION_BASE_FORWARD = 0
MOTION_BASE_YAW = 2
MOTION_HAND_LINEAR = 3
MOTION_HAND_ANGULAR = 6

# SensorEncoder's base-error normalization scale in [m, m, rad]. The base
# navigation feedback law maps its declared hold region through the same units.
BASE_ERROR_SCALE = (0.1, 0.1, 0.5)


def proprioceptive_layout(profile):
    widths = (
        ("positions", profile.arm_size),
        ("velocities", profile.arm_size),
        ("base_pose", 3),
        ("base_command", 2),
        ("target_error", profile.arm_size),
        ("reference_error", profile.arm_size),
        ("grip_position", 1),
        ("grip_error", 1),
        ("grip_reference", 1),
        ("previous_action", profile.action_size),
    )
    offset, fields = 0, {}
    for name, width in widths:
        fields[name] = slice(offset, offset + width)
        offset += width
    return fields, offset


def encode_proprioception(profile, **values):
    """Assemble named blocks, rejecting missing fields and incorrect widths."""
    import numpy as np

    fields, width = proprioceptive_layout(profile)
    if values.keys() != fields.keys():
        raise ValueError("proprioceptive observation fields differ from declared layout")
    token = np.empty(width, dtype=np.float32)
    for name, columns in fields.items():
        value = np.asarray(values[name]).reshape(-1)
        if len(value) != columns.stop - columns.start:
            raise ValueError(f"incorrect proprioceptive field width: {name}")
        token[columns] = value
    return token


def history_layout(profile, *, retention_context=False):
    """Name the established single-hand history layout without changing encoding.

    The target token block is two ten-column tokens plus two activity masks;
    these widths are the PoseTargets.actor_tokens(max_hands=1) contract.
    """
    prefix, offset = proprioceptive_layout(profile)
    fields = {"proprio/" + name: columns for name, columns in prefix.items()}
    widths = (
        ("phase", 7),
        ("base_pose", 3),
        ("hand_pose", 9),
        ("base_goal", 3),
        ("base_mask", 3),
        ("hand_goal", 9),
        ("hand_mask", 1),
        ("base_velocity", 6),
        ("target_tokens", 20),
        ("target_mask", 2),
    )
    if retention_context:
        widths += (("retain_object_command", 1), ("gripper_velocity", 1))
    for name, width in widths:
        fields[name] = slice(offset, offset + width)
        offset += width
    return fields, offset


def preview_layout():
    """Named columns of task_pose_preview_v1; see tracking.preview_contract."""
    fields, offset = {}, 0
    for name, width in (
        ("time_offset", 1),
        ("base_error", 3),
        ("base_mask", 3),
        ("hand_error", 9),
        ("hand_mask", 1),
        ("grip_error", 1),
    ):
        fields[name] = slice(offset, offset + width)
        offset += width
    return fields, offset
