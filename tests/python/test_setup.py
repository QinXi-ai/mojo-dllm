from __future__ import annotations

import io
from pathlib import Path

import pytest

from conftest import FakeHub, make_probes
from launcher.errors import UsageError
from launcher.ui import Style
from launcher.verified import VerifiedStore
from launcher.wizard import SetupArgs, SetupEnv, parse_setup_args, run_setup

REPO = "mradermacher/LLaDA-8B-Instruct-GGUF"
FILE = "LLaDA-8B-Instruct.Q4_K_M.gguf"


class Env:
    """Builds a SetupEnv with recording fakes."""

    def __init__(self, hub: FakeHub, store: VerifiedStore, *, tty: bool = False, answer: str = "", **probe_overrides: object) -> None:
        self.calls: list[str] = []
        self.questions: list[str] = []
        self.out, self.err = io.StringIO(), io.StringIO()

        def ask(question: str) -> str:
            self.questions.append(question)
            return answer

        self.env = SetupEnv(
            probes=make_probes(**probe_overrides),
            hub=hub,
            store=store,
            is_tty=tty,
            ask=ask,
            sync=lambda: self.calls.append("sync"),
            build=self._build,
            smoke=self._smoke,
            style=Style(False),
            out=self.out,
            err=self.err,
        )

    def _build(self) -> bool:
        self.calls.append("build")
        return True

    def _smoke(self, model: Path, device: str) -> float:
        self.calls.append(f"smoke:{device}")
        return 3.25

    def run(self, argv: list[str]) -> int:
        return run_setup(parse_setup_args(argv), self.env)


def test_without_a_tty_or_yes_the_model_is_an_outstanding_step(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store)
    assert e.run([]) == 3
    text = e.out.getvalue()
    assert hub.downloads == [] and e.questions == []
    assert e.calls == ["sync", "build"]
    assert "Setup complete." not in text and "Try it:" not in text
    assert "  ./mojo-dllm pull" in text
    assert "not downloaded (no terminal to ask on; pass --yes to download)" in text
    for step in ("[1/4] Toolchain", "[2/4] Build", "[3/4] Model", "[4/4] Check"):
        assert step in text


def test_outstanding_steps_are_printed_before_the_end_summary(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store)
    e.run([])
    text = e.out.getvalue()
    assert text.rstrip().splitlines()[-1] == "  ./mojo-dllm pull"
    assert "1 step outstanding" in text


def test_yes_downloads_and_finishes(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store)
    assert e.run(["--yes"]) == 0
    text = e.out.getvalue()
    assert hub.downloads == [(REPO, FILE)] and e.questions == []
    assert "sha256 verified" in text and "Setup complete." in text
    assert text.index("Setup complete.") < text.index("Try it:")
    assert './mojo-dllm run "Explain speculative decoding in two sentences."' in text
    assert "./mojo-dllm doctor" in text
    assert e.calls == ["sync", "build", "smoke:gpu"]
    assert "3.2 s" in text or "3.3 s" in text


def test_smoke_test_uses_cpu_without_a_gpu(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store, gpu_text=None)
    assert e.run(["--yes"]) == 0 and e.calls[-1] == "smoke:cpu"
    assert "device: cpu (no NVIDIA GPU found)" in e.out.getvalue()


def test_tty_asks_and_empty_answer_means_yes(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store, tty=True, answer="")
    assert e.run([]) == 0
    assert len(e.questions) == 1
    assert e.questions[0].startswith(f"Download {REPO}/{FILE} (4.9 GB) to ")
    assert e.questions[0].endswith("? [Y/n]")
    assert str(hub.cache_dir) in e.questions[0]
    assert hub.downloads == [(REPO, FILE)]


def test_tty_no_leaves_the_step_outstanding(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store, tty=True, answer="n")
    assert e.run([]) == 3 and hub.downloads == []


def test_no_model_skips_silently_and_completes(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store, tty=True)
    assert e.run(["--no-model"]) == 0
    assert e.questions == [] and hub.downloads == []
    assert "Setup complete." in e.out.getvalue()
    assert e.calls == ["sync", "build"]


def test_a_cached_model_is_not_asked_about_and_setup_is_idempotent(
    hub: FakeHub, store: VerifiedStore, tmp_path: Path
) -> None:
    cached = tmp_path / "c.gguf"
    cached.write_bytes(hub.content)
    hub.cached[(REPO, FILE)] = cached
    e = Env(hub, store, tty=True)
    assert e.run([]) == 0
    assert e.questions == [] and hub.downloads == []
    assert "already downloaded" in e.out.getvalue()


def test_setup_checks_an_unverified_cached_model_once(hub: FakeHub, store: VerifiedStore, tmp_path: Path) -> None:
    cached = tmp_path / "c.gguf"
    cached.write_bytes(hub.content)
    hub.cached[(REPO, FILE)] = cached
    assert Env(hub, store).run([]) == 0
    assert Env(hub, store).run([]) == 0
    assert hub.sha_requests == 1 and store.is_verified(cached)


def test_setup_refuses_a_cached_model_that_fails_its_check(hub: FakeHub, store: VerifiedStore, tmp_path: Path) -> None:
    cached = tmp_path / "c.gguf"
    cached.write_bytes(b"truncated")
    hub.cached[(REPO, FILE)] = cached
    e = Env(hub, store)
    assert e.run([]) == 1
    assert "does not match" in e.err.getvalue() and not cached.exists()
    assert "smoke" not in " ".join(e.calls)


def test_a_failing_preflight_stops_before_anything_runs(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store, glibc="2.31")
    assert e.run(["--yes"]) == 1
    assert e.calls == [] and hub.downloads == []
    assert "✗" in e.out.getvalue() + e.err.getvalue()
    assert "error:" in e.err.getvalue()


def test_model_flag_selects_the_model(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store)
    e.run(["--yes", "--model", "a/b:Q8_0"])
    assert hub.downloads == [("a/b", "LLaDA-8B-Instruct.Q8_0.gguf")]


def test_outstanding_pull_names_a_non_default_model(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store)
    e.run(["--model", "a/b:Q8_0"])
    assert "  ./mojo-dllm pull a/b:Q8_0" in e.out.getvalue()


def test_usage_errors() -> None:
    for argv in (["--bogus"], ["--model"], ["extra"]):
        with pytest.raises(UsageError):
            parse_setup_args(argv)
    assert parse_setup_args(["--yes", "--no-model", "--model", "x/y"]) == SetupArgs(
        model="x/y", yes=True, no_model=True
    )
