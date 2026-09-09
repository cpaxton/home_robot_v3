# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Compatibility shim — prefer ``graph_object_fusion.attach``."""

from emet.memory.graph_eqa._compat import install_shim
from emet.memory.graph_eqa.graph_object_fusion import attach as _impl

install_shim(globals(), _impl)
