"""Procedural room dressing; semantic fixtures stay outside the travel corridor."""

import xml.etree.ElementTree as ET

PALETTES = {
    "kitchen": (".81 .90 .87 1", ".16 .43 .39 1"),
    "living": (".91 .84 .75 1", ".65 .32 .18 1"),
    "study": (".80 .86 .94 1", ".22 .36 .59 1"),
    "dining": (".90 .83 .88 1", ".48 .23 .39 1"),
}


def dress_rooms(world, scene):
    """Return MJCF decorations inside the back-wall furniture footprint.

    Decorations have no task-object IDs and cannot enter the oracle inventory.
    They are geometric visual cues, not room-name overlays or answer annotations.
    """
    serial = 0

    def box(pos, size, color, **kw):
        nonlocal serial
        serial += 1
        return ET.SubElement(
            world,
            "geom",
            name=f"decor_{serial}",
            type="box",
            pos=" ".join(map(str, pos)),
            size=" ".join(map(str, size)),
            rgba=color,
            **kw,
        )

    for room in scene["rooms"]:
        x = (room["bounds"][0] + room["bounds"][2]) / 2
        kind = room["id"]
        pale, accent = PALETTES.get(kind, (".88 .88 .86 1", ".3 .4 .4 1"))
        box([x, -1.88, 1.12], [1.84, 0.025, 1.1], pale)
        box([x, -1.84, 0.12], [1.84, 0.025, 0.10], accent)
        box([x, 0, 0.003], [1.88, 1.88, 0.003], ".82 .77 .67 1")
        # Three pendant-style geometric lights and local ambient illumination.
        ET.SubElement(
            world,
            "light",
            pos=f"{x} -.4 3.3",
            dir="0 0 -1",
            directional="false",
            diffuse=".4 .4 .4",
            ambient=".12 .12 .12",
            castshadow="false",
        )
        if kind == "kitchen":
            # Refrigerator, tiled backsplash, wall cabinets, counter and cooker.
            box([x - 1.42, -1.57, 0.86], [0.28, 0.24, 0.86], ".89 .91 .91 1")
            for z in [0.45, 1.24]:
                box([x - 1.18, -1.31, z], [0.018, 0.025, 0.15], ".22 .27 .28 1")
            box([x - 1.42, -1.30, 0.98], [0.27, 0.01, 0.015], ".35 .4 .4 1")
            for dx in range(-4, 5):
                for z in [0.80, 0.98, 1.16]:
                    box([x + dx * 0.19, -1.82, z], [0.087, 0.015, 0.08], ".92 .97 .95 1")
            for dx in [-0.62, 0, 0.62]:
                box([x + dx, -1.69, 1.60], [0.28, 0.15, 0.26], accent)
                box([x + dx + 0.18, -1.53, 1.55], [0.015, 0.015, 0.08], ".8 .8 .7 1")
            box([x + 1.39, -1.57, 0.42], [0.28, 0.24, 0.42], ".23 .27 .29 1")
            box([x + 1.39, -1.30, 0.40], [0.22, 0.015, 0.20], ".05 .08 .10 1")
            for dx in [-0.13, 0.13]:
                box([x + 1.39 + dx, -1.57, 0.85], [0.075, 0.08, 0.009], ".04 .05 .05 1")
        elif kind == "living":
            # Upholstered sofa and a framed landscape, behind the task table.
            box([x, -1.63, 0.48], [0.92, 0.17, 0.18], accent)
            box([x, -1.79, 0.76], [0.92, 0.075, 0.36], accent)
            for dx in [-0.98, 0.98]:
                box([x + dx, -1.64, 0.58], [0.10, 0.20, 0.27], accent)
            for dx in [-0.60, 0, 0.60]:
                box([x + dx, -1.59, 0.69], [0.28, 0.12, 0.05], ".83 .57 .36 1")
            box([x, -1.80, 1.55], [0.62, 0.025, 0.30], ".21 .18 .14 1")
            box([x, -1.76, 1.57], [0.56, 0.02, 0.24], ".56 .73 .79 1")
            box([x, -1.73, 1.43], [0.55, 0.02, 0.10], ".31 .48 .32 1")
        elif kind == "study":
            # Broad bookcase with contrasting spines, plus a desk monitor.
            for dx in [-1.20, 1.20]:
                box([x + dx, -1.68, 0.97], [0.07, 0.15, 0.92], ".31 .23 .17 1")
            for z in [0.30, 0.72, 1.14, 1.56, 1.88]:
                box([x, -1.68, z], [1.27, 0.15, 0.035], ".31 .23 .17 1")
            colors = [".65 .3 .22 1", ".26 .44 .60 1", ".70 .60 .31 1", ".35 .55 .4 1"]
            for row, z in enumerate([0.51, 0.93, 1.35, 1.73]):
                for j in range(12):
                    box(
                        [x - 1.05 + j * 0.19, -1.63, z],
                        [0.066, 0.095, 0.13 if row == 3 else 0.17],
                        colors[(j + row) % 4],
                    )
        elif kind == "dining":
            # Two recognizable high-backed chairs and a pendant over the table.
            for dx in [-0.62, 0.62]:
                box([x + dx, -1.67, 0.72], [0.22, 0.07, 0.48], accent)
                box([x + dx, -1.54, 0.40], [0.23, 0.19, 0.045], accent)
                for leg in [-0.17, 0.17]:
                    box([x + dx + leg, -1.64, 0.2], [0.025, 0.025, 0.2], ".25 .19 .16 1")
            box([x, -1.53, 1.86], [0.035, 0.035, 0.25], ".22 .20 .18 1")
            box([x, -1.53, 1.60], [0.32, 0.17, 0.07], ".92 .75 .43 1")
