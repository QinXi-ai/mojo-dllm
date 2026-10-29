"""Which cached model files have passed their sha256 check.

The Hugging Face cache cannot say whether a file was ever verified: an
interrupted run, or another tool, can leave one there. After a check passes,
the file's size and mtime are recorded here, and a file whose record is
missing or stale is hashed again before it is used.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path


def default_state_path(env: Mapping[str, str], home: Path) -> Path:
    base = Path(env["XDG_CACHE_HOME"]) if env.get("XDG_CACHE_HOME") else home / ".cache"
    return base / "mojo-dllm" / "verified.json"


def default_store() -> VerifiedStore:
    return VerifiedStore(default_state_path(os.environ, Path.home()))


class VerifiedStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> dict[str, dict[str, object]]:
        # An unreadable file is the same as no records: every file is hashed again.
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _key(blob: Path) -> str:
        return str(blob.resolve())

    def is_verified(self, blob: Path) -> bool:
        try:
            stat = blob.stat()
        except OSError:
            return False
        record = self._load().get(self._key(blob))
        return (
            isinstance(record, dict)
            and record.get("size") == stat.st_size
            and record.get("mtime_ns") == stat.st_mtime_ns
        )

    def record(self, blob: Path, sha256: str) -> None:
        stat = blob.stat()
        records = self._load()
        records[self._key(blob)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".verified-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(records, f, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
