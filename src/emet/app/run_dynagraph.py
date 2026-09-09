# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI entry for Dynagraph (DynaMem + GraphEQA graph lifecycle). See docs/dynagraph.md."""

from emet.app.graph_nav_cli import configure_graph_nav, main
from emet.app.graph_nav_gt import ensure_ground_truth_ready as _ensure_ground_truth_ready
from emet.controller.controller_dynagraph import DynagraphController

configure_graph_nav(DynagraphController, product="Dynagraph")

__all__ = ["main", "_ensure_ground_truth_ready"]

if __name__ == "__main__":
    main()
