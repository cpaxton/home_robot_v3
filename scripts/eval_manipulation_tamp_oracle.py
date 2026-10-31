"""Compatibility CLI for the installed manipulation oracle runner."""

import importlib.util
import runpy
from pathlib import Path

if __name__ == "__main__":
    # Avoid importing emet before the runner selects the native checkout.
    package = importlib.util.find_spec("emet")
    runner = next(
        Path(root) / "rl/manipulation/oracle_runner.py"
        for root in package.submodule_search_locations
        if (Path(root) / "rl/manipulation/oracle_runner.py").is_file()
    )
    runpy.run_path(str(runner), run_name="__main__")
