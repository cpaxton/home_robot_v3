# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""``stream:`` block in dynav YAML for capture/stream CLI."""

from emet.config.stream_config import (
    StreamZmqObsConfig,
    apply_stream_logging_config,
    load_stream_config_from_parameters,
    resolve_stream_verbose,
    should_log_every_step,
)


def test_default_stream_config():
    cfg = StreamZmqObsConfig()
    assert cfg.verbose is None
    assert cfg.da3_log_level == "WARN"
    assert cfg.log_every_step_when_max_steps_le == 20


def test_load_stream_from_parameters():
    params = {
        "stream": {
            "verbose": True,
            "da3_log_level": "INFO",
            "log_every_step_when_max_steps_le": 5,
        }
    }
    cfg = load_stream_config_from_parameters(params)
    assert cfg.verbose is True
    assert cfg.da3_log_level == "INFO"
    assert cfg.log_every_step_when_max_steps_le == 5


def test_should_log_every_step_from_yaml():
    cfg = StreamZmqObsConfig(log_every_step_when_max_steps_le=10)
    assert should_log_every_step(max_steps=3, cfg=cfg, verbose=False)
    assert not should_log_every_step(max_steps=50, cfg=cfg, verbose=False)
    assert should_log_every_step(max_steps=50, cfg=cfg, verbose=True)


def test_resolve_stream_verbose_cli_wins():
    assert resolve_stream_verbose(cli_verbose=True, yaml_value=False)
