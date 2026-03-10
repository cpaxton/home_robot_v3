# Copyright (c) Hello Robot, Inc.
# Namespace package: emet-core provides core, motion, utils subset, config, audio.
# When emet-agent/emet-sim are installed, they extend this namespace.
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)
