from __future__ import annotations

from pathlib import Path

import pytest

from launcher.errors import LauncherError
from launcher.runcmd import BOOLEAN_FLAGS, RunArgs, parse_run_args, run_argv

ROOT = Path(__file__).resolve().parents[2]


def test_prompt_model_and_device_are_extracted() -> None:
    a = parse_run_args(["Say hi", "-m", "a/b:Q8_0", "--device", "cpu"])
    assert a == RunArgs(prompt="Say hi", model="a/b:Q8_0", device="cpu", offline=False, passthrough=[])


def test_defaults() -> None:
    a = parse_run_args(["hi"])
    assert (a.model, a.device, a.offline) == ("llada-8b", "auto", False)


def test_long_model_flag_and_offline() -> None:
    a = parse_run_args(["--model", "x/y", "--offline", "hi"])
    assert (a.model, a.offline, a.prompt) == ("x/y", True, "hi")


def test_other_flags_pass_through_in_order() -> None:
    a = parse_run_args(
        ["--max-tokens", "64", "--verbose", "Explain", "--steps", "32", "--seed", "7", "--json"]
    )
    assert a.prompt == "Explain"
    assert a.passthrough == ["--max-tokens", "64", "--verbose", "--steps", "32", "--seed", "7", "--json"]


def test_a_flag_value_is_never_the_prompt() -> None:
    with pytest.raises(LauncherError) as e:
        parse_run_args(["--max-tokens", "64"])
    assert './mojo-dllm run "Explain speculative decoding in two sentences."' in str(e.value)


def test_unknown_flags_take_a_value() -> None:
    a = parse_run_args(["--future-flag", "v", "hi"])
    assert a.prompt == "hi" and a.passthrough == ["--future-flag", "v"]


def test_a_second_bare_word_asks_for_quotes() -> None:
    with pytest.raises(LauncherError, match="quote the prompt"):
        parse_run_args(["Explain", "diffusion"])


def test_a_flag_without_its_value_is_an_error() -> None:
    with pytest.raises(LauncherError, match="--steps needs a value"):
        parse_run_args(["hi", "--steps"])
    with pytest.raises(LauncherError, match="-m needs a value"):
        parse_run_args(["hi", "-m"])


def test_bad_device_is_an_error() -> None:
    with pytest.raises(LauncherError, match="auto, cpu or gpu"):
        parse_run_args(["hi", "--device", "tpu"])


def test_argv_for_the_binary() -> None:
    a = parse_run_args(["Hi there", "--max-tokens", "8", "--verbose"])
    assert run_argv(Path("/r/build/mojo-dllm"), Path("/m/x.gguf"), a, "gpu") == [
        "/r/build/mojo-dllm", "run", "--model", "/m/x.gguf", "--prompt", "Hi there",
        "--device", "gpu", "--max-tokens", "8", "--verbose",
    ]


def test_boolean_flags_match_the_binary() -> None:
    src = (ROOT / "src/main.mojo").read_text()
    body = src.split("def boolean_flags()", 1)[1].split("return [", 1)[1].split("]", 1)[0]
    in_binary = {w.strip('"') for w in body.replace(",", " ").split() if w.startswith('"')}
    assert {f.removeprefix("--") for f in BOOLEAN_FLAGS} == in_binary
