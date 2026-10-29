from __future__ import annotations

import json
import os
from pathlib import Path

from launcher.verified import VerifiedStore, default_state_path

HOME = Path("/home/u")
SHA = "ab" * 32


def _blob(tmp_path: Path, content: bytes = b"12345") -> Path:
    blob = tmp_path / "blob"
    blob.write_bytes(content)
    return blob


def test_state_path_follows_xdg_cache_home() -> None:
    assert default_state_path({"XDG_CACHE_HOME": "/x"}, HOME) == Path("/x/mojo-dllm/verified.json")
    assert default_state_path({}, HOME) == HOME / ".cache/mojo-dllm/verified.json"


def test_a_file_with_no_record_is_unverified(tmp_path: Path) -> None:
    assert not VerifiedStore(tmp_path / "state" / "verified.json").is_verified(_blob(tmp_path))


def test_a_recorded_file_is_verified_across_store_instances(tmp_path: Path) -> None:
    state, blob = tmp_path / "state" / "verified.json", _blob(tmp_path)
    VerifiedStore(state).record(blob, SHA)
    assert VerifiedStore(state).is_verified(blob)


def test_the_record_holds_size_mtime_and_sha256_under_the_resolved_path(tmp_path: Path) -> None:
    state, blob = tmp_path / "verified.json", _blob(tmp_path)
    link = tmp_path / "snap.gguf"
    link.symlink_to(blob)
    VerifiedStore(state).record(link, SHA)
    data = json.loads(state.read_text())
    assert data == {str(blob.resolve()): {"size": 5, "mtime_ns": blob.stat().st_mtime_ns, "sha256": SHA}}
    assert VerifiedStore(state).is_verified(blob)


def test_a_changed_size_or_mtime_unverifies_the_file(tmp_path: Path) -> None:
    state, blob = tmp_path / "verified.json", _blob(tmp_path)
    store = VerifiedStore(state)
    store.record(blob, SHA)
    os.utime(blob, ns=(1, blob.stat().st_mtime_ns + 1_000_000))
    assert not store.is_verified(blob)
    store.record(blob, SHA)
    blob.write_bytes(b"1234567")
    assert not store.is_verified(blob)


def test_records_for_other_files_survive_a_new_record(tmp_path: Path) -> None:
    state = tmp_path / "verified.json"
    a, b = _blob(tmp_path), tmp_path / "other"
    b.write_bytes(b"zz")
    VerifiedStore(state).record(a, SHA)
    VerifiedStore(state).record(b, SHA)
    assert VerifiedStore(state).is_verified(a) and VerifiedStore(state).is_verified(b)


def test_an_unreadable_state_file_means_unverified_and_is_replaced(tmp_path: Path) -> None:
    state, blob = tmp_path / "verified.json", _blob(tmp_path)
    state.write_text("{not json")
    assert not VerifiedStore(state).is_verified(blob)
    VerifiedStore(state).record(blob, SHA)
    assert VerifiedStore(state).is_verified(blob)


def test_the_write_is_atomic_and_leaves_no_temp_file(tmp_path: Path, monkeypatch) -> None:
    state, blob = tmp_path / "s" / "verified.json", _blob(tmp_path)
    replaced: list[tuple[str, str]] = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: (replaced.append((str(src), str(dst))), real(src, dst))[1])
    VerifiedStore(state).record(blob, SHA)
    assert len(replaced) == 1 and replaced[0][1] == str(state) and replaced[0][0] != str(state)
    assert [p.name for p in state.parent.iterdir()] == ["verified.json"]


def test_a_vanished_file_is_unverified_not_an_error(tmp_path: Path) -> None:
    assert not VerifiedStore(tmp_path / "verified.json").is_verified(tmp_path / "gone")
