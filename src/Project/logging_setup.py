"""
Project-wide logging configuration.

Importing this module configures the root logger once (idempotent) with a
single stream handler that writes `INFO`+ to stderr in a compact format:

    HH:MM:SS LEVEL  module: message

Override the level with the `NMM_LOG_LEVEL` env var (`DEBUG`, `WARNING`, …).
"""

from __future__ import annotations

import logging
import os
import sys


_CONFIGURED = False


def _configure_once() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.environ.get("NMM_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if not any(getattr(h, "_nmm_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-5s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        handler._nmm_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for `name` (usually `__name__`)."""
    _configure_once()
    return logging.getLogger(name)
