"""`run` and `pull`: argument handling and the hand-off to the binary."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from launcher.build import BINARY, ensure_built
from launcher.errors import LauncherError
from launcher.gpu import QUERY, choose_device, parse_nvidia_smi
from launcher.hub import Hub, offline_requested
from launcher.models import DEFAULT_MODEL, ensure_model

# Flags of the binary that take no value. A test compares this list with
# boolean_flags() in src/main.mojo.
BOOLEAN_FLAGS = (
    "--verbose", "--full-logits", "--no-chat", "--dump-step-stats",
    "--json", "--visual", "--no-early-stop",
)
EXAMPLE = './mojo-dllm run "Explain speculative decoding in two sentences."'


@dataclass(frozen=True)
class RunArgs:
    prompt: str
    model: str
    device: str
    offline: bool
    passthrough: list[str]


def parse_run_args(argv: list[str]) -> RunArgs:
    prompt: str | None = None
    model, device, offline = DEFAULT_MODEL, "auto", False
    passthrough: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--offline":
            offline = True
        elif arg.startswith("-"):
            if arg in BOOLEAN_FLAGS:
                passthrough.append(arg)
            else:
                if i + 1 >= len(argv):
                    raise LauncherError(f"{arg} needs a value. Example: {EXAMPLE}")
                value = argv[i + 1]
                i += 1
                if arg in ("-m", "--model"):
                    model = value
                elif arg == "--device":
                    device = value
                else:
                    passthrough += [arg, value]
        elif prompt is None:
            prompt = arg
        else:
            raise LauncherError(f"unexpected argument {arg!r}: quote the prompt. Example: {EXAMPLE}")
        i += 1
    if device not in ("auto", "cpu", "gpu"):
        raise LauncherError(f"--device must be auto, cpu or gpu, not {device!r}")
    if prompt is None:
        raise LauncherError(f"run needs a prompt. Example: {EXAMPLE}")
    return RunArgs(prompt, model, device, offline, passthrough)


def run_argv(binary: Path, model: Path, args: RunArgs, device: str) -> list[str]:
    return [
        str(binary), "run", "--model", str(model), "--prompt", args.prompt,
        "--device", device, *args.passthrough,
    ]


def detect_gpus() -> list:
    if shutil.which("nvidia-smi") is None:
        return []
    done = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=60,
    )
    return parse_nvidia_smi(done.stdout) if done.returncode == 0 else []


def cmd_run(argv: list[str]) -> int:
    args = parse_run_args(argv)
    ensure_built()
    hub = Hub(offline=offline_requested(args.offline, os.environ))
    # Progress and status go to stderr so stdout stays the model's text.
    model = ensure_model(args.model, hub, sys.stderr, announce_cached=False)
    device = choose_device(args.device, detect_gpus() if args.device == "auto" else [])
    print(device.line, file=sys.stderr, flush=True)
    argv_out = run_argv(BINARY, model.resolve(), args, device.kind)
    os.execv(argv_out[0], argv_out)


def cmd_pull(argv: list[str]) -> int:
    offline = False
    refs: list[str] = []
    for arg in argv:
        if arg == "--offline":
            offline = True
        elif arg.startswith("-"):
            raise LauncherError(f"unknown option {arg} for pull. Usage: ./mojo-dllm pull [MODEL] [--offline]")
        else:
            refs.append(arg)
    if len(refs) > 1:
        raise LauncherError("pull takes one model. Usage: ./mojo-dllm pull [MODEL] [--offline]")
    hub = Hub(offline=offline_requested(offline, os.environ))
    ensure_model(refs[0] if refs else DEFAULT_MODEL, hub, sys.stdout, announce_cached=True)
    return 0
