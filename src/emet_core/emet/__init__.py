# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# Namespace package: emet-core provides core, motion, utils subset, config, audio.
# When emet-agent/emet-sim are installed, they extend this namespace.
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

# ---------------------------------------------------------------------------
# Compatibility: pinocchio 3.x still does `import hppfcl` but the library was
# renamed to `coal`.  Register coal as hppfcl in sys.modules so pinocchio
# finds it without installing the (version-incompatible) hpp-fcl package.
# ---------------------------------------------------------------------------
import importlib
import sys

if "hppfcl" not in sys.modules:
    try:
        _coal = importlib.import_module("coal")
        sys.modules["hppfcl"] = _coal
    except ImportError:
        pass
