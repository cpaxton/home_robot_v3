# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0.

"""Command-line entry points for RoboCasa WBC execution."""

import click


@click.group("wbc-robocasa")
def wbc_robocasa_group() -> None:
    """Run privileged-target RoboCasa WBC experiments."""


@wbc_robocasa_group.command(
    context_settings={"ignore_unknown_options": True, "help_option_names": []},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def pick_place(args: tuple[str, ...]) -> None:
    """Run one dynamic PandaOmron pick-and-place episode."""
    from emet.app.wbc_robocasa_pick_place import main

    raise click.exceptions.Exit(main(list(args)))


@wbc_robocasa_group.command(
    context_settings={"ignore_unknown_options": True, "help_option_names": []},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def sweep(args: tuple[str, ...]) -> None:
    """Run the fixed-base cabinet task bridge and export training data."""
    from emet.app.wbc_robocasa_sweep import main

    raise click.exceptions.Exit(main(list(args)))
