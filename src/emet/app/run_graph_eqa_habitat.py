# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Delegate Habitat EQA episodes to the ``emet-habitat`` wrapper venv."""

from __future__ import annotations

import subprocess
import sys

import click

from emet.habitat.wrapper_config import build_habitat_wrapper_command, ensure_habitat_eqa_data_dir_env


def _delegate(argv: list[str]) -> int:
    cmd = build_habitat_wrapper_command(argv)
    if cmd is None:
        click.echo(
            "Habitat wrapper not found. From repo root run:\n"
            "  ./scripts/install_habitat.sh\n"
            "See docs/habitat_eqa.md",
            err=True,
        )
        return 1
    env = dict(__import__("os").environ)
    ensure_habitat_eqa_data_dir_env(env)
    return subprocess.call(cmd, env=env)


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
@click.option("--dataset", type=click.Choice(["hmeqa"]), default="hmeqa")
@click.option("--question-id", default=0, type=int)
@click.option("--method", type=click.Choice(["graph_eqa", "dynagraph"]), default="dynagraph")
@click.option("--mock-llm", is_flag=True, default=False)
@click.option("--max-planning-steps", default=5, type=int)
def main(
    ctx: click.Context,
    dataset: str,
    question_id: int,
    method: str,
    mock_llm: bool,
    max_planning_steps: int,
) -> None:
    """Run GraphEQA / Dynagraph on Habitat HM-EQA (requires ``.venv-habitat``)."""
    argv = [
        "run-episode",
        "--dataset",
        dataset,
        "--question-id",
        str(question_id),
        "--method",
        method,
        "--max-planning-steps",
        str(max_planning_steps),
    ]
    if mock_llm:
        argv.append("--mock-llm")
    argv.extend(ctx.args)
    sys.exit(_delegate(argv))


if __name__ == "__main__":
    main(standalone_mode=True)
