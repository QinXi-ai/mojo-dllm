from __future__ import annotations

import re
import tomllib
from pathlib import Path

import httpx2
import pytest

from launcher.errors import LauncherError
from launcher.hub import _hub_errors

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "error",
    [
        httpx2.ConnectError("connection refused"),
        httpx2.ReadTimeout("timed out"),
        OSError("network is unreachable"),
    ],
    ids=["connect", "timeout", "oserror"],
)
def test_network_failures_name_the_offline_flag(error: Exception) -> None:
    with pytest.raises(LauncherError) as e:
        with _hub_errors("a/b"):
            raise error
    assert "could not reach Hugging Face for a/b" in str(e.value)
    assert "--offline" in str(e.value)


def test_other_exceptions_pass_through() -> None:
    with pytest.raises(KeyError):
        with _hub_errors("a/b"):
            raise KeyError("x")


def test_the_test_group_pins_the_hub_version_the_script_uses() -> None:
    script = (ROOT / "mojo-dllm").read_text()
    in_script = re.search(r'"(huggingface-hub==[^"]+)"', script)
    assert in_script is not None, "mojo-dllm has no huggingface-hub pin"
    group = tomllib.loads((ROOT / "pyproject.toml").read_text())["dependency-groups"]["test"]
    assert in_script.group(1) in group
