from __future__ import annotations

import os
from pathlib import Path

from launcher.hub import Hub, hf_cache_dir

HOME = Path("/home/u")


def test_cache_dir_precedence() -> None:
    assert hf_cache_dir({"HF_HUB_CACHE": "/a", "HF_HOME": "/b"}, HOME) == Path("/a")
    assert hf_cache_dir({"HF_HOME": "/b"}, HOME) == Path("/b/hub")
    assert hf_cache_dir({"XDG_CACHE_HOME": "/x"}, HOME) == Path("/x/huggingface/hub")
    assert hf_cache_dir({}, HOME) == HOME / ".cache/huggingface/hub"


def _snapshot(cache: Path, repo: str, rev: str, name: str, mtime: float) -> Path:
    owner, _, model = repo.partition("/")
    snap = cache / f"models--{owner}--{model}" / "snapshots" / rev
    blob = cache / f"models--{owner}--{model}" / "blobs" / f"blob-{rev}"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"12345")
    snap.mkdir(parents=True, exist_ok=True)
    link = snap / name
    link.symlink_to(blob)
    os.utime(blob, (mtime, mtime))
    return link


def test_cache_lookup_reads_the_huggingface_layout(tmp_path: Path) -> None:
    link = _snapshot(tmp_path, "o/r", "rev1", "m.Q4_K_M.gguf", 10)
    hub = Hub(tmp_path)
    assert hub.cached_path("o/r", "m.Q4_K_M.gguf") == link
    assert hub.cached_path("o/r", "other.gguf") is None
    assert hub.cached_path("o/none", "m.Q4_K_M.gguf") is None


def test_offline_listing_is_what_the_cache_holds(tmp_path: Path) -> None:
    _snapshot(tmp_path, "o/r", "rev1", "m.Q4_K_M.gguf", 10)
    _snapshot(tmp_path, "o/r", "rev2", "m.Q8_0.gguf", 20)
    assert Hub(tmp_path, offline=True).list_files("o/r") == {
        "m.Q4_K_M.gguf": 5,
        "m.Q8_0.gguf": 5,
    }


def test_the_newest_revision_wins(tmp_path: Path) -> None:
    _snapshot(tmp_path, "o/r", "old", "m.gguf", 10)
    new = _snapshot(tmp_path, "o/r", "new", "m.gguf", 20)
    assert Hub(tmp_path).cached_path("o/r", "m.gguf") == new


def test_every_file_the_offline_listing_names_is_in_the_cache(tmp_path: Path) -> None:
    # plan_model picks from this listing, so an offline plan always has a cached file.
    _snapshot(tmp_path, "o/r", "rev1", "m.Q4_K_M.gguf", 10)
    _snapshot(tmp_path, "o/r", "rev2", "m.Q8_0.gguf", 20)
    _snapshot(tmp_path, "o/r", "rev2", "m.Q2_K.gguf", 20)
    hub = Hub(tmp_path, offline=True)
    listed = hub.list_files("o/r")
    assert listed and all(hub.cached_path("o/r", name) is not None for name in listed)
