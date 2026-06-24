# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Typed ``stream:`` block for ``emet capture`` / ``emet stream`` (see ``agents/default_stream.yaml``)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import draccus

from emet.config.rerun_config import _env_truthy, resolve_rerun_bool
from emet.core.parameters import Parameters
from emet.utils.logger import configure_mapping_session_logging


@dataclass
class StreamZmqObsConfig:
    """Terminal + loop defaults for ``emet.app.zmq_obs`` mapping sessions."""

    verbose: bool | None = None
    status_interval_s: float = 5.0
    da3_log_level: str = "WARN"
    log_every_step_when_max_steps_le: int = 20
    # capture profile: seconds to hold Rerun after a one-shot map step
    rerun_hold_s: float | None = None


def load_stream_config_from_parameters(parameters: Parameters | dict[str, Any] | None) -> StreamZmqObsConfig:
    """Read ``stream:`` from a loaded dynav / agent parameters dict."""
    if parameters is None:
        return StreamZmqObsConfig()
    subset = parameters.get("stream") if hasattr(parameters, "get") else None
    if subset is None or not isinstance(subset, dict):
        return StreamZmqObsConfig()
    return draccus.decode(StreamZmqObsConfig, subset)


def resolve_stream_verbose(*, cli_verbose: bool = False, yaml_value: bool | None = None) -> bool:
    """Precedence: ``--verbose`` > YAML ``stream.verbose`` > ``EMET_STREAM_VERBOSE`` > default off."""
    return resolve_rerun_bool(
        cli=cli_verbose,
        yaml_value=yaml_value,
        env_key="EMET_STREAM_VERBOSE",
        default=False,
    )


def apply_stream_logging_config(cfg: StreamZmqObsConfig, *, verbose: bool) -> None:
    """Set ``DA3_LOG_LEVEL`` before DA3 model load (upstream ``depth_anything_3`` logger)."""
    if verbose:
        os.environ["DA3_LOG_LEVEL"] = "INFO"
        return
    level = str(cfg.da3_log_level or "WARN").strip().upper()
    if level not in ("ERROR", "WARN", "INFO", "DEBUG"):
        level = "WARN"
    os.environ.setdefault("DA3_LOG_LEVEL", level)
    configure_mapping_session_logging(verbose=False)


def should_log_every_step(*, max_steps: int, cfg: StreamZmqObsConfig, verbose: bool) -> bool:
    if verbose:
        return True
    if max_steps > 0 and max_steps <= int(cfg.log_every_step_when_max_steps_le):
        return True
    return False
