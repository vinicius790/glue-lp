"""Thin stdlib logging helpers for glue_lp scripts and optional API."""
from __future__ import annotations

import logging
import sys
from typing import Any


_CONFIGURED = False


def setup_logging(level: int | str = logging.INFO, *, name: str = "glue_lp") -> logging.Logger:
    """Idempotent root logger for the package (stderr, simple format)."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root = logging.getLogger("glue_lp")
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))
        root.propagate = False
        _CONFIGURED = True
    return logger


def log_kv(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Structured-enough one-line event: event key=val ..."""
    parts = [event] + [f"{k}={v!r}" for k, v in fields.items()]
    logger.info(" ".join(parts))
