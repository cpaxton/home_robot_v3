# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bad native task configurations fail before robot connection or model startup."""
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from emet.app.run_agent import _load_native_task_config, main


@pytest.mark.parametrize('raw', ['not JSON', '{}', '{"schema_version":true}'])
def test_bad_configuration_is_clean_cli_error(tmp_path, raw):
    path = tmp_path/'task.json'
    path.write_text(raw)
    result = CliRunner().invoke(main, ['--state-only-sim', '--native-task-config', str(path)])
    assert result.exit_code == 2 and 'Invalid native task JSON' in result.output


def test_offline_cannot_enable_native_provider(tmp_path):
    path = tmp_path/'task.json'
    path.write_text('{}')
    result = CliRunner().invoke(main, ['--offline', '--native-task-config', str(path)])
    assert result.exit_code == 2 and 'requires --state-only-sim' in result.output


def test_config_example_and_overlap_rejection(tmp_path):
    path = Path(__file__).resolve().parents[3]/'configs/eval/native_stretch_transfer_debug.json'
    config = _load_native_task_config(str(path))
    assert config['robot'] == 'stretch' and config['input_mode'] == 'privileged'
    config['regions'][1]['bounds'] = config['regions'][0]['bounds']
    bad = tmp_path/'bad.json'
    bad.write_text(json.dumps(config))
    result = CliRunner().invoke(main, ['--state-only-sim', '--native-task-config', str(bad)])
    assert result.exit_code == 2 and 'Invalid native task JSON' in result.output


def test_native_config_cannot_be_silently_ignored_by_eqa_eval(tmp_path):
    config = tmp_path / 'task.json'
    config.write_text('not JSON')
    result = CliRunner().invoke(main, ['--state-only-sim', '--native-task-config',
                                      str(config), '--eqa-eval'])
    assert result.exit_code == 2
    assert '--native-task-config is incompatible with --eqa-eval' in result.output
