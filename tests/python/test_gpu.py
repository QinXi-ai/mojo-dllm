from __future__ import annotations

import pytest

from launcher.errors import LauncherError
from launcher.gpu import Gpu, choose_device, parse_nvidia_smi

RTX = Gpu("NVIDIA GeForce RTX 3050 6GB Laptop GPU", 8.6, 5120)
TURING = Gpu("NVIDIA GeForce GTX 1650", 7.5, 3900)


def test_one_gpu() -> None:
    text = "NVIDIA GeForce RTX 3050 6GB Laptop GPU, 8.6, 5120\n"
    assert parse_nvidia_smi(text) == [RTX]


def test_two_gpus() -> None:
    text = "NVIDIA GeForce GTX 1650, 7.5, 3900\nNVIDIA A100, 8.0, 40000\n"
    assert parse_nvidia_smi(text) == [TURING, Gpu("NVIDIA A100", 8.0, 40000)]


def test_a_name_with_a_comma_keeps_its_comma() -> None:
    assert parse_nvidia_smi("Tesla, Inc. T4, 7.5, 15000\n") == [Gpu("Tesla, Inc. T4", 7.5, 15000)]


@pytest.mark.parametrize("text", ["", "\n", "  \n\n"])
def test_empty_output_means_no_gpu(text: str) -> None:
    assert parse_nvidia_smi(text) == []


@pytest.mark.parametrize(
    "text", ["garbage", "name, [N/A], 100", "name, 8.6", "name, 8.6, lots"]
)
def test_a_malformed_line_is_an_error_not_a_missing_gpu(text: str) -> None:
    with pytest.raises(LauncherError, match="nvidia-smi"):
        parse_nvidia_smi(text)


def test_auto_picks_gpu_from_compute_8_0() -> None:
    d = choose_device("auto", [RTX])
    assert d.kind == "gpu"
    assert d.line == (
        "device: gpu (NVIDIA GeForce RTX 3050 6GB Laptop GPU, compute 8.6, 5120 MiB free)"
    )
    assert choose_device("auto", [Gpu("A100", 8.0, 1)]).kind == "gpu"


def test_auto_uses_the_first_device_that_qualifies() -> None:
    d = choose_device("auto", [TURING, RTX])
    assert d.kind == "gpu" and "RTX 3050" in d.line


def test_auto_without_a_gpu_is_cpu() -> None:
    d = choose_device("auto", [])
    assert (d.kind, d.line) == ("cpu", "device: cpu (no NVIDIA GPU found)")


def test_auto_below_8_0_is_cpu_and_says_why() -> None:
    d = choose_device("auto", [TURING])
    assert (d.kind, d.line) == (
        "cpu",
        "device: cpu (GPU compute capability 7.5 is below 8.0)",
    )


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
def test_an_explicit_device_is_passed_as_is(kind: str) -> None:
    d = choose_device(kind, [])
    assert d.kind == kind and d.line == f"device: {kind} (requested)"
