"""Model references, choosing a file inside a repo, and getting it onto disk."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from launcher.errors import LauncherError
from launcher.ui import human_size

DEFAULT_MODEL = "llada-8b"
ALIASES = {"llada-8b": ("mradermacher/LLaDA-8B-Instruct-GGUF", "Q4_K_M")}
# llama.cpp's order when no quant is requested; the first file is the last resort.
DEFAULT_QUANTS = ("Q4_K_M", "Q8_0")

_REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(:[A-Za-z0-9._-]+)?$")
_SHARD = re.compile(r"-\d{5}-of-\d{5}")


class Hub(Protocol):
    cache_dir: Path
    offline: bool

    def list_files(self, repo: str) -> dict[str, int]: ...
    def cached_path(self, repo: str, filename: str) -> Path | None: ...
    def download(self, repo: str, filename: str) -> Path: ...
    def expected_sha256(self, repo: str, filename: str) -> str | None: ...


@dataclass(frozen=True)
class ModelRef:
    path: Path | None = None
    repo: str | None = None
    quant: str | None = None


@dataclass(frozen=True)
class ModelPlan:
    repo: str
    filename: str
    size: int | None
    cached: Path | None

    @property
    def label(self) -> str:
        return f"{self.repo}/{self.filename}"


def parse_model_ref(text: str) -> ModelRef:
    path = Path(text)
    if text and path.is_file():
        return ModelRef(path=path)
    if text.endswith(".gguf"):
        raise LauncherError(f"no such file: {text}\nCheck the path, or use llada-8b to download the default model.")
    if text in ALIASES:
        repo, quant = ALIASES[text]
        return ModelRef(repo=repo, quant=quant)
    if _REPO.match(text):
        repo, _, quant = text.partition(":")
        return ModelRef(repo=repo, quant=quant or None)
    raise LauncherError(
        f"unknown model {text!r}. Accepted: a path to a .gguf file, the alias llada-8b, "
        "or owner/repo[:QUANT]."
    )


def _has_quant(name: str, quant: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(quant)}(?![A-Za-z0-9])", name, re.I) is not None


def pick_gguf(files: Sequence[str], quant: str | None) -> str:
    ggufs = [f for f in files if f.endswith(".gguf") and not _SHARD.search(f)]
    if not ggufs:
        raise LauncherError("the repo has no .gguf file (split shards are not supported)")
    wanted = (quant,) if quant else DEFAULT_QUANTS
    for q in wanted:
        for name in ggufs:
            if _has_quant(name, q):
                return name
    if quant:
        listing = "\n".join(f"  {f}" for f in ggufs)
        raise LauncherError(f"no GGUF file matches quant {quant}\nAvailable GGUF files:\n{listing}")
    return ggufs[0]


def plan_model(ref: ModelRef, hub: Hub) -> ModelPlan:
    assert ref.repo is not None
    files = hub.list_files(ref.repo)
    if hub.offline and not files:
        raise LauncherError(
            f"{ref.repo} ({ref.quant or 'default quant'}) has no GGUF file in the local cache "
            f"({hub.cache_dir}), and offline mode is on.\n"
            f"On a connected machine run ./mojo-dllm pull {ref.repo}"
            + (f":{ref.quant}" if ref.quant else "")
            + ", then copy the cache over or run again without --offline."
        )
    filename = pick_gguf(list(files), ref.quant)
    return ModelPlan(ref.repo, filename, files.get(filename), hub.cached_path(ref.repo, filename))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_model(plan: ModelPlan, hub: Hub, out: TextIO) -> Path:
    size = f" ({human_size(plan.size)})" if plan.size is not None else ""
    print(f"Downloading {plan.label}{size}", file=out, flush=True)
    path = hub.download(plan.repo, plan.filename)
    expected = hub.expected_sha256(plan.repo, plan.filename)
    if expected is None:
        _delete(path)
        raise LauncherError(f"Hugging Face reported no sha256 for {plan.label}, so the download cannot be verified.")
    if _sha256(path) != expected:
        _delete(path)
        raise LauncherError(
            f"sha256 of {plan.label} does not match the hub's. The file was deleted.\n"
            "Run the command again to download it fresh."
        )
    print("sha256 verified", file=out, flush=True)
    return path


def _delete(path: Path) -> None:
    path.resolve().unlink(missing_ok=True)
    path.unlink(missing_ok=True)


def ensure_model(text: str, hub: Hub | None, out: TextIO, *, announce_cached: bool) -> Path:
    """Resolve a model reference to a local file, downloading it when needed."""
    ref = parse_model_ref(text)
    if ref.path is not None:
        return ref.path
    assert hub is not None
    plan = plan_model(ref, hub)
    if plan.cached is not None:
        if announce_cached:
            print(f"already downloaded\n{plan.cached}", file=out)
        return plan.cached
    if hub.offline:
        raise LauncherError(
            f"{plan.label} is not in the local cache ({hub.cache_dir}), and offline mode is on.\n"
            f"On a connected machine run ./mojo-dllm pull {text}, or run again without --offline."
        )
    path = fetch_model(plan, hub, out)
    if announce_cached:
        print(f"{path} ({human_size(path.stat().st_size)})", file=out)
    return path
