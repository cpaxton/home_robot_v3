import os
import shutil
import subprocess
import sys
from pathlib import Path

from emet.rl.manipulation import oracle_runner
from emet.rl.manipulation.post_training import oracle_runner_command


def test_post_training_launches_the_packaged_runner():
    assert oracle_runner_command() == [sys.executable, str(Path(oracle_runner.__file__).resolve())]
    assert "scripts" not in Path(oracle_runner_command()[1]).parts


def test_oracle_cli_works_from_an_installed_layout_without_a_repository(tmp_path):
    # A package layout with no repository root, scripts directory or editable install.
    package = tmp_path / "site/emet"
    runner = package / "rl/manipulation/oracle_runner.py"
    runner.parent.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    shutil.copyfile(oracle_runner.__file__, runner)
    wrapper = tmp_path / "compat.py"
    original_wrapper = Path(__file__).resolve().parents[4] / "scripts/eval_manipulation_tamp_oracle.py"
    shutil.copyfile(original_wrapper, wrapper)
    for entry in (runner, wrapper):
        result = subprocess.run(
            [sys.executable, str(entry), "--help"],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(package.parent)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "--native-root" in result.stdout and "--controller" in result.stdout
