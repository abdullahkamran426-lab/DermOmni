"""Shared environment-variable parsing helpers.

Single source of truth for int/float env parsing used by
``voice_of_the_doctor.py`` and ``voice_of_the_patient.py``.
"""
from __future__ import annotations

import os


def read_int_env(name: str, default: int, minimum: int = 1) -> int:
    """Read an int env var, clamped to ``minimum``; fall back to ``default``."""
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def read_float_env(name: str, default: float, minimum: float = 1.0) -> float:
    """Read a float env var, clamped to ``minimum``; fall back to ``default``."""
    try:
        return max(minimum, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default
