# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# CLI: print a human-readable scene graph from a saved memory directory
# (common format with graph.json from GraphEQABackend).

import click

from emet.memory.backend import get_memory_backend
from emet.memory.format import is_memory_directory
from emet.memory.graph_eqa import GraphEQAMemory, format_scene_graph_pretty


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--max-nodes", type=int, default=None, help="Truncate node listing for huge graphs")
def main(path: str, max_nodes: int | None) -> None:
    """Load memory from PATH and print a formatted scene graph to stdout."""
    if not is_memory_directory(path):
        raise click.ClickException(f"Not a valid memory directory: {path}")

    mem = GraphEQAMemory(defer_llm_clients=True)
    backend = get_memory_backend("graph_eqa", graph_memory=mem)
    backend.load(path)
    click.echo(format_scene_graph_pretty(mem, max_nodes=max_nodes))


if __name__ == "__main__":
    main()
