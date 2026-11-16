"""Pack tracked simulator/test inputs, including bridge sources, for a compute host.

Generated traces, logs, weights and credentials are excluded. Robot/test binary
assets are read from the worktree (resolved LFS contents), with a SHA-256 manifest.
This does not qualify a host or replace its fresh regression/teacher gates.
"""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    ".gitattributes",
    "LICENSE",
    "src/innate_mars_bridge/innate_mars_bridge/ros/camera.py",
    "src/emet/perception/matterport_category_mappings.tsv",
    "src/test/core/test_image_timing.py",
)
TEXT = {".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".tsv", ".txt", ".xml", ".urdf", ".srdf", ".md", ".svg"}


def build_snapshot(root, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise ValueError("fresh snapshot output required")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    selected = []
    for name in tracked:
        if not name or Path(name).suffix in {".gz", ".jsonl", ".pyc"}:
            continue
        if name in REQUIRED or (
            name.startswith(("src/", "scripts/", "configs/", "docs/", "paper/data/whole_body_control/"))
            and (Path(name).suffix in TEXT or name.startswith(("src/test/", "src/emet/assets/robot/")))
        ):
            selected.append(name)
    missing = [name for name in REQUIRED if name not in selected or not (root / name).is_file()]
    if missing:
        raise ValueError("Missing required regression inputs: " + ", ".join(missing))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest = {"format": "emet-mars-training-snapshot-v1", "revision": revision, "files": []}
    output.parent.mkdir(exist_ok=True, parents=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    try:
        with tarfile.open(temporary, "x") as archive:
            for name in selected:
                path = root / name
                if not path.resolve().is_relative_to(root):
                    raise ValueError("Snapshot input escapes repository: " + name)
                before = path.stat()
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    first = stream.read(1024)
                    if first.startswith(b"version https://git-lfs.github.com/spec/v1"):
                        raise ValueError("Unresolved LFS input: " + name)
                    digest.update(first)
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                archive.add(path, arcname=name, recursive=False)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError("Snapshot input changed while packing: " + name)
                manifest["files"].append({"path": name, "bytes": before.st_size, "sha256": digest.hexdigest()})
            raw = (json.dumps(manifest, indent=2) + "\n").encode()
            entry = tarfile.TarInfo("snapshot-manifest.json")
            entry.size = len(raw)
            archive.addfile(entry, io.BytesIO(raw))
        # Refuse a racing overwrite, while publishing only a complete archive.
        import os

        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_snapshot(ROOT, args.output)
    print(f"Packed {len(manifest['files'])} checksummed inputs into {args.output}")


if __name__ == "__main__":
    main()
