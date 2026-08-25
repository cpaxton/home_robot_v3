# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CLI entry for LazyGraph (DynaMem find + Qwen commit on arrival). See docs/lazy_graph.md."""

from __future__ import annotations

import emet.app.run_dynagraph as _dynagraph_mod
from emet.controller.controller_lazy_graph import LazyGraphController

# Reuse dynagraph CLI; swap controller class before main is invoked.
_dynagraph_mod.DynagraphController = LazyGraphController  # type: ignore[misc,assignment]

from emet.app.run_dynagraph import main  # noqa: E402

if __name__ == "__main__":
    main()
