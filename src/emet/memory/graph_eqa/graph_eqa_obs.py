# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Compatibility shim — implementation: ``emet.memory.graph_eqa.eqa.graph_eqa_obs``."""

from emet.memory.graph_eqa._compat import install_shim
from emet.memory.graph_eqa.eqa import graph_eqa_obs as _impl

install_shim(globals(), _impl)
