"""Tiny argv helper shared by the example scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .paths import OUTPUTS_DIR


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--write",
        action="store_true",
        help=f"also write a markdown snapshot under {OUTPUTS_DIR}",
    )
    return p


def maybe_write(args: argparse.Namespace, filename: str, body: str) -> Path | None:
    if not getattr(args, "write", False):
        return None
    from .report import write_text

    path = OUTPUTS_DIR / filename
    write_text(path, body)
    return path
