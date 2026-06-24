# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Central logger for emet: red errors, yellow warnings, optional name prefix.

Use this instead of print() for errors and warnings so output is colored and
errors/warnings go to stderr (safe for piping stdout).

  from emet.utils.logger import Logger
  log = Logger(__name__)
  log.error("Something failed")
  log.warning("Optional feature unavailable")

  # Or use module-level functions (no name prefix):
  from emet.utils.logger import error, warning
  error("Failed to load config")
  warning("Using default port")
"""

import logging
import os
import sys
from typing import TextIO

from termcolor import colored


class Logger:
    """Logger with colored error (red) and warning (yellow); errors/warnings go to stderr."""

    def __init__(
        self,
        name: str | None = None,
        hide_info: bool = False,
        hide_debug: bool = True,
        *,
        stderr_for_errors: bool = True,
    ) -> None:
        self.name = name
        self._hide_info = hide_info
        self._hide_debug = hide_debug
        self._stderr_for_errors = stderr_for_errors

    def hide_info(self) -> None:
        self._hide_info = True

    def hide_debug(self) -> None:
        self._hide_debug = True

    def show_info(self) -> None:
        self._hide_info = False

    def show_debug(self) -> None:
        self._hide_debug = False

    def _flatten(self, args: tuple) -> str:
        """Flatten a tuple of arguments into a string joined by spaces.

        Args:
            args (tuple): Tuple of arguments to flatten.

        Returns:
            str: Flattened string.
        """
        text = " ".join([str(arg) for arg in args])
        if self.name is not None:
            text = f"[{self.name}] {text}"
        return text

    def _out(self, text: str, color: str, stream: TextIO) -> None:
        print(colored(text, color), file=stream)

    def error(self, *args) -> None:
        text = self._flatten(args)
        stream = sys.stderr if self._stderr_for_errors else sys.stdout
        self._out(text, "red", stream)

    def info(self, *args) -> None:
        if not self._hide_info:
            text = self._flatten(args)
            self._out(text, "white", sys.stdout)

    def debug(self, *args) -> None:
        if not self._hide_debug:
            text = self._flatten(args)
            self._out(text, "white", sys.stdout)

    def warning(self, *args) -> None:
        text = self._flatten(args)
        stream = sys.stderr if self._stderr_for_errors else sys.stdout
        self._out(text, "yellow", stream)

    def warn(self, *args) -> None:
        """Alias for warning()."""
        self.warning(*args)

    def alert(self, *args) -> None:
        text = self._flatten(args)
        self._out(text, "green", sys.stdout)


_default_logger = Logger(None)


def get_logger(name: str, **kwargs: bool) -> Logger:
    """Return a Logger with the given name prefix (e.g. __name__)."""
    return Logger(name, **kwargs)


def error(*args) -> None:
    _default_logger.error(*args)


def info(*args) -> None:
    _default_logger.info(*args)


def warning(*args) -> None:
    _default_logger.warning(*args)


def warn(*args) -> None:
    """Alias for warning()."""
    _default_logger.warning(*args)


def alert(*args) -> None:
    _default_logger.alert(*args)


def debug(*args) -> None:
    _default_logger.debug(*args)


def suppress_hf_hub_http_logging() -> None:
    """Silence INFO-level HTTP chatter from Hugging Face Hub / httpx when loading models.

    Does nothing if ``EMET_VERBOSE_HF`` is set to 1/true/yes (useful for debugging Hub traffic).
    ``emet`` package ``__init__`` also sets ``HF_HUB_VERBOSITY`` / ``TRANSFORMERS_VERBOSITY`` early;
    call this again before Hub traffic if another library reset log levels.
    """
    val = os.environ.get("EMET_VERBOSE_HF", "").strip().lower()
    if val in ("1", "true", "yes"):
        return
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    for name in (
        "httpx",
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
        "huggingface_hub",
        "transformers",
        "urllib3",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    if "huggingface_hub" in sys.modules:
        try:
            from huggingface_hub.utils import logging as hub_logging

            hub_logging.set_verbosity_error()
        except Exception:
            pass
    if "transformers" in sys.modules:
        try:
            from transformers.utils import logging as tf_logging

            tf_logging.set_verbosity_error()
        except Exception:
            pass


def configure_mapping_session_logging(*, verbose: bool = False) -> None:
    """Reduce DA3 / Hugging Face terminal noise during ``emet stream`` mapping loops.

    Honors ``EMET_STREAM_VERBOSE=1`` and ``--verbose`` on ``emet stream``. DA3 uses
  ``DA3_LOG_LEVEL`` (default INFO in upstream); we set WARN unless verbose.
    """
    if verbose or os.environ.get("EMET_STREAM_VERBOSE", "").strip().lower() in ("1", "true", "yes"):
        return
    os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
    suppress_hf_hub_http_logging()
