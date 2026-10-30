"""Reject generated compressed control artifacts in the Git index (including force-add)."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    files = (
        subprocess.check_output(["git", "ls-files", "-z", "paper/data/whole_body_control"], cwd=root)
        .decode()
        .split("\0")
    )
    forbidden = [p for p in files if p.endswith((".gz", ".zip", ".xz", ".bz2", ".zst", ".jsonl"))]
    if forbidden:
        raise RuntimeError("Generated control artifacts must be ignored, not committed: " + ", ".join(forbidden))
    return True


if __name__ == "__main__":
    check()
