# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Compatibility shim — implementation: ``emet.memory.graph_eqa.agentic.types``."""

from emet.memory.graph_eqa._compat import install_shim
from emet.memory.graph_eqa.agentic import types as _impl

install_shim(globals(), _impl)
