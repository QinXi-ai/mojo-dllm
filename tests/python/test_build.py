from __future__ import annotations

import pytest

from launcher import build
from launcher.build import is_stale
from launcher.errors import LauncherError


def test_missing_binary_is_stale() -> None:
    assert is_stale(None, [1.0])


def test_newer_source_makes_it_stale() -> None:
    assert is_stale(100.0, [50.0, 100.5])


def test_equal_or_older_sources_are_fresh() -> None:
    assert not is_stale(100.0, [50.0, 100.0])
    assert not is_stale(100.0, [])


class FakeBuild:
    """Stands in for the Popen of scripts/build.sh."""

    def __init__(self, lines: list[str], code: int) -> None:
        self.stdout = iter(f"{line}\n" for line in lines)
        self._code = code

    def wait(self) -> int:
        return self._code


def _fake_popen(monkeypatch: pytest.MonkeyPatch, lines: list[str], code: int) -> list[list[str]]:
    calls: list[list[str]] = []

    def popen(cmd: list[str], **kwargs: object) -> FakeBuild:
        calls.append(cmd)
        return FakeBuild(lines, code)

    monkeypatch.setattr(build, "needs_build", lambda: True)
    monkeypatch.setattr(build.subprocess, "Popen", popen)
    return calls


def test_a_failed_build_raises_with_the_last_20_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_popen(monkeypatch, [f"line {i}" for i in range(50)], 2)
    with pytest.raises(LauncherError) as e:
        build.ensure_built()
    msg = str(e.value)
    assert "the build failed" in msg and "./mojo-dllm doctor" in msg
    assert "line 30" in msg.splitlines()[1] and "line 49" in msg and "line 29\n" not in msg + "\n"


def test_a_good_build_reports_that_it_built(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_popen(monkeypatch, ["ok"], 0)
    assert build.ensure_built() is True
    assert calls == [["bash", "scripts/build.sh"]]


def test_a_current_binary_is_not_rebuilt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_popen(monkeypatch, [], 0)
    monkeypatch.setattr(build, "needs_build", lambda: False)
    assert build.ensure_built() is False and calls == []
