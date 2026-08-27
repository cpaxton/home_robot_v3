# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI entry for LazyGraph (DynaMem find + Qwen commit on arrival). See docs/lazy_graph.md."""

from emet.app.graph_nav_cli import configure_graph_nav, main
from emet.controller.controller_lazy_graph import LazyGraphController

configure_graph_nav(LazyGraphController, product="LazyGraph")

__all__ = ["main"]

if __name__ == "__main__":
    main()
