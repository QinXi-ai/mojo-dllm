"""NVIDIA GPU detection and the cpu/gpu choice."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from launcher.errors import LauncherError

# The int8 tensor-core kernels need Ampere or newer.
MIN_COMPUTE_CAP = 8.0
QUERY = "name,compute_cap,memory.free"


@dataclass(frozen=True)
class Gpu:
    name: str
    compute_cap: float
    free_mib: int


@dataclass(frozen=True)
class Device:
    kind: str
    detail: str

    @property
    def line(self) -> str:
        return f"device: {self.kind} ({self.detail})"


def parse_nvidia_smi(text: str) -> list[Gpu]:
    """Parse `nvidia-smi --query-gpu=name,compute_cap,memory.free --format=csv,noheader,nounits`."""
    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        # rsplit: a GPU name may contain a comma.
        parts = [p.strip() for p in line.rsplit(",", 2)]
        try:
            if len(parts) != 3 or not parts[0]:
                raise ValueError
            gpus.append(Gpu(parts[0], float(parts[1]), int(parts[2])))
        except ValueError:
            raise LauncherError(
                f"nvidia-smi printed a line the launcher cannot read: {line!r}\n"
                "Run nvidia-smi to check the driver, or pass --device cpu to skip GPU detection."
            ) from None
    return gpus


def choose_device(requested: str, gpus: Sequence[Gpu]) -> Device:
    if requested != "auto":
        return Device(requested, "requested")
    if not gpus:
        return Device("cpu", "no NVIDIA GPU found")
    for gpu in gpus:
        if gpu.compute_cap >= MIN_COMPUTE_CAP:
            return Device(
                "gpu", f"{gpu.name}, compute {gpu.compute_cap:.1f}, {gpu.free_mib} MiB free"
            )
    best = max(g.compute_cap for g in gpus)
    return Device("cpu", f"GPU compute capability {best:.1f} is below {MIN_COMPUTE_CAP:.1f}")
