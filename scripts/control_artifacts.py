"""List, verify or restore ignored control evidence from an external artifact store."""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emet.rl.artifacts import PREFIX, catalog, checked_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "verify", "restore"])
    parser.add_argument("--store", type=Path, default=ROOT / "logs/artifacts/whole_body_control")
    args = parser.parse_args()
    rows = catalog(ROOT)
    for entry in rows:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.is_relative_to(PREFIX):
            raise ValueError("Artifact path escapes declared generated-data root")
        if args.command == "list":
            print(entry["sha256"], entry["bytes"], entry["path"])
            continue
        source = checked_path(args.store / relative.relative_to(PREFIX), entry)
        if args.command == "restore":
            target = ROOT / relative
            if target.exists():
                checked_path(target, entry)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    if args.command != "list":
        print(f"{args.command}: {len(rows)} checksummed artifacts; no Git writes")


if __name__ == "__main__":
    main()
