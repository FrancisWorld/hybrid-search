"""Durability helpers."""

from __future__ import annotations

import os
from pathlib import Path


def fsync_dir(directory: Path) -> None:
    """Flush a directory so file creations, renames and truncations in it survive power loss."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
