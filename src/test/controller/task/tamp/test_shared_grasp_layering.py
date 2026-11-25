"""The shared grasp pipeline must remain independent of evaluation packages."""
import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest


@pytest.mark.parametrize('module', ['grounded_grasps', 'visual_grounding', 'observed_candidates',
    'observed_arm', 'live_transfer'])
def test_shared_grasp_modules_do_not_import_experiment_layers(module):
    root = Path(__file__).resolve().parents[4] / 'emet/controller/task/tamp'
    tree = ast.parse((root / f'{module}.py').read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ''
            imports.append(resolve_name('.'*node.level+name, 'emet.controller.task.tamp')
                           if node.level else name)
    assert not [name for name in imports if name.startswith(
        ('emet.eval', 'emet.benchmarks', 'emet.habitat', 'emet.paper'))]
