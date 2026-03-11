# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Helpers for memory save/load UX (print messages, view instructions).

import os

from termcolor import colored

from emet.utils.logger import Logger

logger = Logger("memory")


def print_memory_saved_help(path: str) -> None:
    """Print a clear message that memory was saved and how to view it."""
    path_abs = os.path.abspath(path)
    sep = "=" * 60
    logger.alert("Memory saved.")
    print(colored(sep, "green"))
    print(colored("  Path: ", "white") + colored(path_abs, "cyan"))
    print()
    print(colored("  To view in Rerun:", "yellow"))
    print(colored(f"    emet show-memory {path_abs}", "white"))
    print()
    print(colored("  Or with 2D maps:", "yellow"))
    print(colored(f"    python -m emet.app.read_map -i {path_abs}", "white"))
    print(colored(sep, "green") + "\n")


def print_memory_view_help_on_quit(path: str | None) -> None:
    """Print a short reminder on quit showing how to view memory (if path is set)."""
    if not path:
        return
    path_abs = os.path.abspath(path)
    print()
    print(
        colored("To view this memory: ", "yellow")
        + colored(f"emet show-memory {path_abs}", "cyan")
    )
