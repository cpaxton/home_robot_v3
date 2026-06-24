# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""``emet stream`` — live ZMQ → Rerun (+ optional mapping loop).

Re-exports ``stream_main`` from :mod:`emet.app.zmq_obs`. See ``docs/zmq_obs.md``.
"""

from emet.app.zmq_obs import stream_main as main

__all__ = ["main"]
