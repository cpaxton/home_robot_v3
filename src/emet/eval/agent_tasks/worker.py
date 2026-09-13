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
    parser.add_argument("--agent-model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-rounds", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--skill-interface", default="atomic", choices=["atomic", "plan_execute"])
    args = parser.parse_args()
    suite = load_suite(args.suite)
    out = Path(args.out)
    if args.agent_model:
        from .agent_runner import run_local_agent

        metrics = run_local_agent(
            suite,
            select_episode(suite, args.episode),
            out,
            model=args.agent_model,
            device=args.device,
            max_rounds=args.max_rounds,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout,
            skill_interface=args.skill_interface,
        )
    else:
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
