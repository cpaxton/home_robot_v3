# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Helpers for memory save/load UX (print messages, view instructions).

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from termcolor import colored

from emet.utils.logger import Logger

if TYPE_CHECKING:
    from emet.memory.format import MemoryState

logger = Logger("memory")


def print_memory_state(state: "MemoryState") -> None:
    """Print a readable summary of a loaded MemoryState (backend, point cloud, frames, scene graph)."""
    from emet.memory.format import MemoryState

    assert isinstance(state, MemoryState)
    print("\n" + "=" * 60)
    print("MEMORY STATE SUMMARY")
    print("=" * 60)
    if state.manifest:
        m = state.manifest
        print(f"  Backend:     {m.backend}")
        print(f"  Created:     {m.created_at or 'N/A'}")
        print(f"  Version:     {m.version}")
    print()
    if state.point_cloud is not None:
        pc = state.point_cloud
        n = pc.xyz.shape[0]
        print(f"  Point cloud: {n} points")
        if pc.xyz.size > 0:
            print(
                f"    XYZ range: x=[{pc.xyz[:, 0].min():.3f}, {pc.xyz[:, 0].max():.3f}] "
                f"y=[{pc.xyz[:, 1].min():.3f}, {pc.xyz[:, 1].max():.3f}] "
                f"z=[{pc.xyz[:, 2].min():.3f}, {pc.xyz[:, 2].max():.3f}]"
            )
        if pc.rgb is not None:
            print(f"    RGB: present ({pc.rgb.shape})")
    else:
        print("  Point cloud: (none)")
    print(f"  Frames:      {len(state.frames)}")
    if state.graph is not None:
        g = state.graph
        print(f"  Scene graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
        for i, node in enumerate(g.nodes[:10]):
            labels = ", ".join(node.labels) if node.labels else "(no labels)"
            print(f"    Node {node.node_id}: xyz={node.xyz} labels=[{labels}]")
        if len(g.nodes) > 10:
            print(f"    ... and {len(g.nodes) - 10} more nodes")
        if g.edges:
            print("  Relationships:")
            for e in g.edges[:15]:
                print(f"    {e.id1} --{e.relation}--> {e.id2}")
            if len(g.edges) > 15:
                print(f"    ... and {len(g.edges) - 15} more edges")
    else:
        print("  Scene graph: (none)")
    if state.obstacles_2d is not None:
        print(f"  2D obstacles: grid shape {state.obstacles_2d.shape}")
    if state.explored_2d is not None:
        print(f"  2D explored:  grid shape {state.explored_2d.shape}")
    if state.text_descriptions:
        print(f"  Text descriptions: {len(state.text_descriptions)} items")
    print("=" * 60 + "\n")


def print_memory_from_path(path: str) -> None:
    """Load a saved memory directory and print its summary. Fails if path is not a memory directory."""
    from emet.memory.format import is_memory_directory, load_memory

    if not is_memory_directory(path):
        raise SystemExit(
            f"Not a memory directory: {path}\n"
            "Expected a directory with manifest.json (e.g. from emet run dynamem, create-and-print-memory)."
        )
    state = load_memory(path)
    print_memory_state(state)


def print_memory_saved_help(path: str) -> None:
    """Print a clear message that memory was saved and how to view it."""
    path_abs = os.path.abspath(path)
    sep = "=" * 60
    logger.alert("Memory saved.")
    print(colored(sep, "green"))
    print(colored("  Path: ", "white") + colored(path_abs, "cyan"))
    print()
    print(colored("  To view in Rerun:", "yellow"))
    print(colored(f"    emet show-memory {path_abs}", "white"))
    print()
    print(colored("  Or with 2D maps:", "yellow"))
    print(colored(f"    python -m emet.app.read_map -i {path_abs}", "white"))
    print(colored(sep, "green") + "\n")


def print_memory_view_help_on_quit(path: str | None) -> None:
    """Print a short reminder on quit showing how to view memory (if path is set)."""
    if not path:
        return
    path_abs = os.path.abspath(path)
    print()
    print(
        colored("To view this memory: ", "yellow")
        + colored(f"emet show-memory {path_abs}", "cyan")
    )
