from __future__ import annotations

import io
from pathlib import Path

import pytest

from launcher import cli
from launcher.ui import Style

COMMANDS = ["setup", "pull", "run", "doctor", "help"]
ENV_VARS = ["HF_TOKEN", "HF_HOME", "HF_HUB_CACHE", "MOJO_DLLM_GPU_ARCH"]


def test_help_lists_every_command_and_the_environment() -> None:
    text = cli.help_text(Style(False))
    for name in COMMANDS:
        assert f"  {name} " in text, name
    for var in ENV_VARS:
        assert var in text, var
    assert "Quick start" in text and "Examples" in text
    assert "./mojo-dllm run" in text and "./mojo-dllm doctor" in text and "./mojo-dllm setup" in text
    assert "inspect" in text and "tokenize" in text


def test_help_has_no_colour_when_disabled_and_some_when_enabled() -> None:
    assert "\x1b[" not in cli.help_text(Style(False))
    assert "\x1b[" in cli.help_text(Style(True))


def test_style_follows_tty_and_no_color() -> None:
    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    assert Style.for_stream(Tty(), {}).enabled
    assert not Style.for_stream(Tty(), {"NO_COLOR": "1"}).enabled
    assert not Style.for_stream(io.StringIO(), {}).enabled


@pytest.mark.parametrize("argv", [[], ["help"], ["-h"], ["--help"]])
def test_help_forms_exit_zero(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(argv) == 0
    assert "Quick start" in capsys.readouterr().out


def test_launcher_commands_answer_help_flags(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--help"]) == 0
    assert "Quick start" in capsys.readouterr().out


def test_run_without_a_prompt_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and './mojo-dllm run "' in err


def test_pull_nonsense_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["pull", "nonsense"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "owner/repo[:QUANT]" in err


def test_setup_usage_error_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["setup", "--bogus"]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_other_words_go_to_the_binary_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(cli, "ensure_built", lambda *a, **k: seen.append(("built", [])))
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: seen.append((path, argv)))
    cli.main(["inspect", "m.gguf", "--x"])
    assert seen[0][0] == "built"
    path, argv = seen[1]
    assert path == str(cli.BINARY) and argv == [str(cli.BINARY), "inspect", "m.gguf", "--x"]
