# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Click group that lists heavy subcommands without importing MuJoCo / sim stacks.

``emet jobs`` / ``emet eval`` / ``emet --help`` must stay import-light: after a
sim job releases the GPU lock, importing ``mujoco`` in a new ``emet`` process
has SIGSEGV'd (``_functions.so`` / ``_enums.so``).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from gettext import gettext as _
from typing import Any

import click

# cmd_name -> (module, attribute, short_help)
LazySpec = tuple[str, str, str]


class LazyClickGroup(click.Group):
    """``list_commands`` / help use stored names; ``get_command`` imports on use."""

    def __init__(
        self,
        *args: Any,
        lazy_subcommands: Mapping[str, LazySpec] | None = None,
        **kwargs: Any,
    ) -> None:
        self.lazy_subcommands: dict[str, LazySpec] = dict(lazy_subcommands or {})
        super().__init__(*args, **kwargs)

    def list_commands(self, ctx: click.Context) -> list[str]:
        names = set(super().list_commands(ctx))
        names.update(self.lazy_subcommands)
        return sorted(names)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        spec = self.lazy_subcommands.get(cmd_name)
        if spec is None:
            return None
        module_name, attr, short_help = spec
        command = getattr(importlib.import_module(module_name), attr)
        if not isinstance(command, click.Command):
            raise TypeError(f"{module_name}.{attr} is not a click.Command")
        if short_help:
            command.short_help = short_help
        self.add_command(command)
        return command

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows: list[tuple[str, str]] = []
        names = self.list_commands(ctx)
        if not names:
            return
        limit = formatter.width - 6 - max(len(name) for name in names)
        for name in names:
            spec = self.lazy_subcommands.get(name)
            if spec is not None and name not in self.commands:
                help_text = spec[2]
            else:
                cmd = super().get_command(ctx, name)
                if cmd is None or cmd.hidden:
                    continue
                help_text = cmd.get_short_help_str(limit)
            if len(help_text) > limit:
                help_text = help_text[: max(0, limit - 3)].rstrip() + "..."
            rows.append((name, help_text))
        if rows:
            with formatter.section(_("Commands")):
                formatter.write_dl(rows)
