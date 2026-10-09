#!/usr/bin/env python3
"""Replay a private placement snapshot, emitting one JSON result; never actuates."""
import argparse
import contextlib
import faulthandler
import json
import sys

from emet.motion.placement_replay import replay_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    args = parser.parse_args()
    faulthandler.enable()
    with contextlib.redirect_stdout(sys.stderr):
        result = replay_snapshot(args.snapshot)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["solutions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
