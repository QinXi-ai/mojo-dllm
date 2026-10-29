"""Colour and number formatting shared by every command."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TextIO

_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33"}


@dataclass(frozen=True)
class Style:
    enabled: bool

    @staticmethod
    def for_stream(stream: TextIO, env: Mapping[str, str] | None = None) -> Style:
        env = os.environ if env is None else env
        return Style(stream.isatty() and "NO_COLOR" not in env)

    def paint(self, name: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\x1b[{_CODES[name]}m{text}\x1b[0m"

    def bold(self, text: str) -> str:
        return self.paint("bold", text)

    def dim(self, text: str) -> str:
        return self.paint("dim", text)


def human_size(n: int) -> str:
    """Decimal units, like the Hugging Face website: 4925226272 -> '4.9 GB'."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000:
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} TB"
