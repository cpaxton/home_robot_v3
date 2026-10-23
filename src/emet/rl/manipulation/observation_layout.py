"""Named proprioceptive prefix shared by observation producers and consumers.

Phase, task-pose and preview fields follow this prefix and are never risk inputs.
Widths depend on the robot; previous actions must not include phase columns.
"""


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
