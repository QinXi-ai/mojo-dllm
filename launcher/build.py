"""The compiled binary: where it lives, when it is stale, how to build it."""

from __future__ import annotations

import collections
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from launcher.errors import LauncherError

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "build" / "mojo-dllm"


def project_env() -> dict[str, str]:
    """The environment for uv calls into the project's .venv.

    `uv run --script` exports VIRTUAL_ENV for its own cache environment, and uv
    warns when that differs from the project's .venv.
    """
    return {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


def is_stale(binary_mtime: float | None, source_mtimes: Iterable[float]) -> bool:
    if binary_mtime is None:
        return True
    return any(m > binary_mtime for m in source_mtimes)


def last_lines(text: str, n: int) -> str:
    return "\n".join(text.splitlines()[-n:])


def exec_binary(argv: list[str]) -> None:
    """Replace this process with the binary; a missing or unrunnable one is a launcher error."""
    try:
        os.execv(argv[0], argv)
    except OSError as e:
        raise LauncherError(f"cannot start {argv[0]}: {e.strerror}. Run ./mojo-dllm doctor to check the build.") from e


def source_mtimes() -> list[float]:
    return [p.stat().st_mtime for p in (ROOT / "src").rglob("*.mojo")]


def binary_mtime() -> float | None:
    return BINARY.stat().st_mtime if BINARY.is_file() else None


def needs_build() -> bool:
    return is_stale(binary_mtime(), source_mtimes())


def ensure_built() -> bool:
    """Build when the binary is missing or stale. Returns whether it built."""
    if not needs_build():
        return False
    print("Building mojo-dllm (about a minute)...", file=sys.stderr, flush=True)
    tail: collections.deque[str] = collections.deque(maxlen=20)
    proc = subprocess.Popen(
        ["bash", "scripts/build.sh"],
        cwd=ROOT,
        env=project_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stderr.write(line)
        tail.append(line.rstrip("\n"))
    if proc.wait() != 0:
        raise LauncherError(
            "the build failed (bash scripts/build.sh); last 20 lines:\n"
            + "\n".join(tail)
            + "\nRun ./mojo-dllm doctor to check the toolchain."
        )
    return True
