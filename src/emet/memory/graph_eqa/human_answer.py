# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Compatibility shim — implementation: ``emet.memory.graph_eqa.eqa.human_answer``."""

from emet.memory.graph_eqa._compat import install_shim
from emet.memory.graph_eqa.eqa import human_answer as _impl

install_shim(globals(), _impl)
