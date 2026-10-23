# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Invalid experiment inputs must fail before simulator/model startup."""
import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('contents', ['[]', '[{}]', '{broken', '[[[0, 0], [1, 1]], [[1, 1], [0, 0]]]'])
def test_invalid_regions_are_argparse_errors_before_model_load(monkeypatch, tmp_path, capsys, contents):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    module = importlib.import_module('eval_native_table_agent')
    regions = tmp_path / 'regions.json'
    regions.write_text(contents)
    output = tmp_path / 'output'
    monkeypatch.setattr('sys.argv', ['eval_native_table_agent', '--output', str(output),
                                    '--placement-regions', str(regions)])
    monkeypatch.setattr(module.mujoco.MjModel, 'from_xml_path', lambda *a, **kw: pytest.fail('must not load model'))
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert 'error:' in stderr
    assert 'Traceback' not in stderr
    assert not output.exists()


def test_valid_regions_preserve_requested_goals(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    module = importlib.import_module('eval_native_table_agent')
    regions = [[[-.1, -.8], [.1, -.7]], [[-.1, -.6], [.1, -.5]]]
    path = tmp_path / 'regions.json'
    path.write_text(json.dumps(regions))
    assert module.load_placement_regions(path) == regions
