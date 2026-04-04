# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
import sys

# Read by huggingface_hub / transformers when they first configure loggers (before those imports).
if os.environ.get("EMET_VERBOSE_HF", "").strip().lower() not in ("1", "true", "yes"):
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ---------------------------------------------------------------------------
# Compatibility: pinocchio 3.x still does `import hppfcl` but the library was
# renamed to `coal`.  Register coal as hppfcl in sys.modules so pinocchio
# finds it without installing the (version-incompatible) hpp-fcl package.
# ---------------------------------------------------------------------------
import importlib

if "hppfcl" not in sys.modules:
    try:
        _coal = importlib.import_module("coal")
        sys.modules["hppfcl"] = _coal
    except ImportError:
        pass

# Quiet Hugging Face Hub / httpx INFO lines (e.g. every HEAD/GET during model load).
from emet.utils.logger import suppress_hf_hub_http_logging as _suppress_hf_hub_http_logging

_suppress_hf_hub_http_logging()
