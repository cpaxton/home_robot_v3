# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""``emet capture`` — one-shot ZMQ artifact save (+ optional map step).

Re-exports ``capture_main`` from :mod:`emet.app.zmq_obs`. See ``docs/zmq_obs.md``.
"""

from emet.app.zmq_obs import capture_main as main

__all__ = ["main"]
