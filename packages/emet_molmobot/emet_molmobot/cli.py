# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

import sys

import click

from emet_molmobot.runner import serve_policy


@click.group()
def main() -> None:
    """MolmoBot policy bridge (optional separate venv)."""


@main.command("serve-policy")
@click.option("--hf-repo", required=True, help="HuggingFace model repo, e.g. allenai/MolmoBot-DROID")
@click.option("--action-type", default="joint_pos", show_default=True)
@click.argument("extra_args", nargs=-1)
def serve_policy_cmd(hf_repo: str, action_type: str, extra_args: tuple[str, ...]) -> None:
    """Delegate to upstream MolmoBot ``serve_molmo.py``."""
    argv = ["--hf-repo", hf_repo, "--action-type", action_type, *extra_args]
    sys.exit(serve_policy(argv))


if __name__ == "__main__":
    main()
