from __future__ import annotations

import pytest

from launcher.errors import LauncherError
from launcher.probes import read_mojo_pin


def test_the_pin_is_read_from_pyproject() -> None:
    assert read_mojo_pin('dependencies = [\n    "mojo==1.1.0",\n    "max-core==26.6.0",\n]') == "1.1.0"


def test_a_missing_pin_is_a_launcher_error_naming_the_file() -> None:
    with pytest.raises(LauncherError, match=r"pyproject\.toml has no mojo==X pin"):
        read_mojo_pin('dependencies = ["max-core==26.6.0"]')
