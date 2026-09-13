"""Owned subprocess entry: native crashes cannot corrupt the parent evaluator."""

import argparse
from pathlib import Path

from .runner import run_fixture
from .spec import load_suite, select_episode
from .visualize import export_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--control", default="witness")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    suite = load_suite(args.suite)
    out = Path(args.out)
    metrics = run_fixture(
        suite,
        select_episode(suite, args.episode),
        out,
        render=not args.no_render,
        control=args.control,
        timeout_s=args.timeout,
    )
    export_run(out)
    return 0 if metrics["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
