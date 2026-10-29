from __future__ import annotations

from launcher.build import is_stale, last_lines


def test_missing_binary_is_stale() -> None:
    assert is_stale(None, [1.0])


def test_newer_source_makes_it_stale() -> None:
    assert is_stale(100.0, [50.0, 100.5])


def test_equal_or_older_sources_are_fresh() -> None:
    assert not is_stale(100.0, [50.0, 100.0])
    assert not is_stale(100.0, [])


def test_failure_report_keeps_the_last_20_lines() -> None:
    text = "\n".join(f"line {i}" for i in range(50))
    tail = last_lines(text, 20)
    assert tail.splitlines()[0] == "line 30" and tail.splitlines()[-1] == "line 49"
    assert len(tail.splitlines()) == 20
