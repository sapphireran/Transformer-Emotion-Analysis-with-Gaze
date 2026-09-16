"""Shared CLI bits for the personal examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--write",
        action="store_true",
        help="Write a markdown snapshot under examples/outputs/",
    )
    return p


def maybe_write(args: argparse.Namespace, name: str, content: str) -> Path | None:
    if not getattr(args, "write", False):
        return None
    from zuco_lab.reports import write_text

    path = OUTPUT_DIR / name
    write_text(path, content)
    print(f"wrote {path.relative_to(ROOT)}")
    return path
