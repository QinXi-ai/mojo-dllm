from __future__ import annotations

import io
from pathlib import Path

import pytest

from launcher.errors import LauncherError
from launcher.models import (
    DEFAULT_MODEL,
    ModelRef,
    ensure_model,
    fetch_model,
    parse_model_ref,
    pick_gguf,
    plan_model,
)

REPO = "mradermacher/LLaDA-8B-Instruct-GGUF"


def test_existing_file_is_used_as_is(tmp_path: Path) -> None:
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    assert parse_model_ref(str(f)) == ModelRef(path=f)


def test_alias_names_the_llada_repo() -> None:
    assert parse_model_ref("llada-8b") == ModelRef(repo=REPO, quant="Q4_K_M")
    assert DEFAULT_MODEL == "llada-8b"


def test_owner_repo_with_and_without_quant() -> None:
    assert parse_model_ref("a/b") == ModelRef(repo="a/b", quant=None)
    assert parse_model_ref("a/b:q8_0") == ModelRef(repo="a/b", quant="q8_0")


@pytest.mark.parametrize("text", ["nonsense", "a/b/c", "a/b:", "/", "", "a b/c"])
def test_anything_else_is_an_error_that_says_what_is_accepted(text: str) -> None:
    with pytest.raises(LauncherError) as e:
        parse_model_ref(text)
    msg = str(e.value)
    assert "a path to a .gguf file" in msg and "llada-8b" in msg and "owner/repo[:QUANT]" in msg


def test_missing_gguf_path_is_not_mistaken_for_a_repo(tmp_path: Path) -> None:
    with pytest.raises(LauncherError, match="no such file"):
        parse_model_ref(str(tmp_path / "gone.gguf"))
    with pytest.raises(LauncherError, match="no such file"):
        parse_model_ref("models/gone.gguf")


FILES = ["README.md", "m.Q8_0.gguf", "m.Q2_K.gguf", "m.Q4_K_M.gguf"]


def test_default_order_is_q4_k_m_then_q8_0_then_first() -> None:
    assert pick_gguf(FILES, None) == "m.Q4_K_M.gguf"
    assert pick_gguf(["m.Q2_K.gguf", "m.Q8_0.gguf"], None) == "m.Q8_0.gguf"
    assert pick_gguf(["z.Q2_K.gguf", "a.Q3_K_S.gguf"], None) == "z.Q2_K.gguf"


def test_quant_matches_case_insensitively_as_a_whole_token() -> None:
    assert pick_gguf(FILES, "q2_k") == "m.Q2_K.gguf"
    # IQ4_XS must not satisfy a request for Q4_XS.
    with pytest.raises(LauncherError):
        pick_gguf(["m.IQ4_XS.gguf"], "Q4_XS")
    assert pick_gguf(["m-Q5_K_M.gguf"], "Q5_K_M") == "m-Q5_K_M.gguf"
    assert pick_gguf(["m_Q5_K_M.gguf"], "Q5_K_M") == "m_Q5_K_M.gguf"


def test_split_shards_are_ignored() -> None:
    files = ["big.Q4_K_M-00001-of-00003.gguf", "small.Q8_0.gguf"]
    assert pick_gguf(files, None) == "small.Q8_0.gguf"
    with pytest.raises(LauncherError):
        pick_gguf(files, "Q4_K_M")


def test_a_requested_quant_that_matches_nothing_lists_the_files() -> None:
    with pytest.raises(LauncherError) as e:
        pick_gguf(FILES, "Q9_9")
    msg = str(e.value)
    assert "Q9_9" in msg
    assert "Available GGUF files:\n" in msg
    assert "  m.Q8_0.gguf\n" in msg + "\n" and "README.md" not in msg


def test_repo_with_no_gguf_is_an_error() -> None:
    with pytest.raises(LauncherError, match="no .gguf"):
        pick_gguf(["README.md"], None)


def test_plan_reports_size_and_cache_state(hub) -> None:
    plan = plan_model(ModelRef(repo=REPO, quant="Q4_K_M"), hub)
    assert plan.filename == "LLaDA-8B-Instruct.Q4_K_M.gguf"
    assert plan.size == 4_925_226_272 and plan.cached is None
    hub.cached[(REPO, plan.filename)] = Path("/c/x.gguf")
    assert plan_model(ModelRef(repo=REPO, quant="Q4_K_M"), hub).cached == Path("/c/x.gguf")


def test_fetch_verifies_sha256_and_prints_it(hub) -> None:
    out = io.StringIO()
    plan = plan_model(ModelRef(repo=REPO, quant="Q4_K_M"), hub)
    path = fetch_model(plan, hub, out)
    assert path.read_bytes() == hub.content
    text = out.getvalue()
    assert f"Downloading {REPO}/{plan.filename} (4.9 GB)" in text
    assert "sha256 verified" in text


def test_fetch_deletes_a_file_whose_hash_does_not_match(hub) -> None:
    hub.sha256 = "0" * 64
    plan = plan_model(ModelRef(repo=REPO, quant="Q4_K_M"), hub)
    with pytest.raises(LauncherError, match="sha256"):
        fetch_model(plan, hub, io.StringIO())
    assert not (hub._tmp / plan.filename).exists()


def test_fetch_refuses_when_the_hub_reports_no_hash(hub) -> None:
    hub.sha256 = None
    plan = plan_model(ModelRef(repo=REPO, quant="Q4_K_M"), hub)
    with pytest.raises(LauncherError, match="no sha256"):
        fetch_model(plan, hub, io.StringIO())
    assert not (hub._tmp / plan.filename).exists()


def test_ensure_model_cached_prints_already_downloaded(hub, tmp_path: Path) -> None:
    cached = tmp_path / "c.gguf"
    cached.write_bytes(b"x" * 10)
    hub.cached[(REPO, "LLaDA-8B-Instruct.Q4_K_M.gguf")] = cached
    out = io.StringIO()
    path = ensure_model("llada-8b", hub, out, announce_cached=True)
    assert path == cached and hub.downloads == []
    assert f"already downloaded\n{cached}" in out.getvalue()


def test_ensure_model_downloads_when_missing(hub) -> None:
    out = io.StringIO()
    path = ensure_model("llada-8b", hub, out, announce_cached=True)
    assert hub.downloads == [(REPO, "LLaDA-8B-Instruct.Q4_K_M.gguf")]
    assert str(path) in out.getvalue() and "10 B" in out.getvalue()


def test_ensure_model_local_path_never_touches_the_hub(hub, tmp_path: Path) -> None:
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    assert ensure_model(str(f), None, io.StringIO(), announce_cached=True) == f


def test_offline_miss_says_only_the_cache_was_searched(hub) -> None:
    hub.offline = True
    hub.files = {}
    with pytest.raises(LauncherError) as e:
        plan_model(ModelRef(repo=REPO, quant="Q4_K_M"), hub)
    msg = str(e.value)
    assert "offline" in msg and str(hub.cache_dir) in msg
    assert REPO in msg and "Q4_K_M" in msg and f"./mojo-dllm pull {REPO}:Q4_K_M" in msg


def test_offline_selection_uses_the_cached_listing_only(hub, tmp_path: Path) -> None:
    hub.offline = True
    hub.files = {"x.Q8_0.gguf": 5}  # all the cache holds for this repo
    cached = tmp_path / "x.gguf"
    cached.write_bytes(b"12345")
    hub.cached[("a/b", "x.Q8_0.gguf")] = cached
    assert ensure_model("a/b", hub, io.StringIO(), announce_cached=False) == cached
    with pytest.raises(LauncherError, match="Q4_K_M"):
        ensure_model("a/b:Q4_K_M", hub, io.StringIO(), announce_cached=False)
    assert hub.downloads == []
