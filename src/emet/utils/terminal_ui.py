# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
"""Small helpers for readable CLI output (colors when stdout is a TTY)."""

from __future__ import annotations

import os
import sys

import click


def color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("EMET_NO_COLOR", "").strip().lower() in ("1", "true", "yes"):
        return False
    return sys.stdout.isatty()


def style(text: str, *, fg: str | None = None, bold: bool = False, dim: bool = False) -> str:
    if not color_enabled():
        return text
    return click.style(text, fg=fg, bold=bold, dim=dim)


def heading(text: str) -> None:
    print(style(text, bold=True))


def rule(char: str = "─", width: int = 52) -> None:
    print(style(char * width, dim=True))


def kv(label: str, value: str, *, tone: str | None = None) -> None:
    label_s = style(f"  {label:<12}", dim=True)
    if tone == "ok":
        value_s = style(value, fg="green")
    elif tone == "warn":
        value_s = style(value, fg="yellow")
    elif tone == "err":
        value_s = style(value, fg="red")
    else:
        value_s = value
    print(f"{label_s} {value_s}")


def bullet(title: str, detail: str) -> None:
    print(f"  {style(title, bold=True)}  {style(detail, dim=True)}")


def note(text: str) -> None:
    print(style(f"  {text}", dim=True))
