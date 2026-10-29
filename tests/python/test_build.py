from __future__ import annotations

from launcher.build import is_stale


def test_missing_binary_is_stale() -> None:
    assert is_stale(None, [1.0])


def test_newer_source_makes_it_stale() -> None:
    assert is_stale(100.0, [50.0, 100.5])


def test_equal_or_older_sources_are_fresh() -> None:
    assert not is_stale(100.0, [50.0, 100.0])
    assert not is_stale(100.0, [])
