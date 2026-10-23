# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""CPU-only registry validation. This command does not claim task qualification."""
import argparse
import json

from .registry import describe_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registry', help='Executable, versioned JSON episode registry')
    args = parser.parse_args()
    try:
        result = describe_registry(args.registry)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
