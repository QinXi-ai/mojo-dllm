"""A real subprocess produces the candidate and a content-bound execution receipt."""

from __future__ import annotations

import importlib
import json
import os
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
parity = importlib.import_module("parity")
candidate = importlib.import_module("dg_candidate")


def fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"tiny fixture model")
    meta = dict(prompt="fixed", prompt_tokens=[2], canvas_tokens=[3], canvas_length=1, n_vocab=2,
                prev_temp_inv=1.25, final_logit_softcapping="30", model=model.name,
                model_bytes=model.stat().st_size, model_path=str(model), model_mtime_ns=model.stat().st_mtime_ns,
                model_sha256=None, llama_cpp_commit="pin")
    case = tmp_path / "case0"
    case.mkdir()
    (case / "case.json").write_text(json.dumps(meta))
    (case / "unified.ref.f32").write_bytes(struct.pack("<2f", 1.0, 2.0))
    return case, model, meta


def reference_controls() -> dict:
    metrics = dict(rows=1, argmax_agree=1, min_cosine=1.0, mean_top5_overlap=2.0,
                   max_rel_rms=0.0, top1_max_rel_err=0.0)
    return {name: metrics.copy() for name in ("cached vs unified", "cached-sc vs unified-sc")}


@pytest.mark.parametrize(("label", "state"), [("cached vs unified", "no-sc"), ("cached-sc vs unified-sc", "sc")])
def test_local_reference_outside_its_state_bound_stops_before_candidate(tmp_path: Path, label: str, state: str) -> None:
    case, _, meta = fixture(tmp_path)
    meta["ref_spread"] = reference_controls()
    meta["ref_spread"][label]["max_rel_rms"] = 0.5
    (case / "case.json").write_text(json.dumps(meta))
    limits = dict(min_cosine=0.9, argmax_fraction=1.0, mean_top5_overlap=2.0,
                  max_rel_rms=0.5, max_top1_rel_err=0.0)
    states = {"no-sc": limits, "sc": limits}
    states[state] = dict(limits, max_rel_rms=0.1)
    bound = dict(protocol="dg-fixed-input-v1", model=meta["model"], llama_cpp_commit="pin",
                 cases={"fixed": dict(inputs=meta, states=states)})
    with pytest.raises(SystemExit, match=f"this reference does not reproduce the calibration.*{label}.*RMS"):
        parity.dg_compare(tmp_path, bound)


@pytest.mark.parametrize("label", ["cached vs unified", "cached-sc vs unified-sc"])
def test_a_reference_control_that_lacks_a_canvas_row_stops_before_candidate(tmp_path: Path, label: str) -> None:
    case, _, meta = fixture(tmp_path)
    meta["ref_spread"] = reference_controls()
    meta["ref_spread"][label]["rows"] = meta["canvas_length"] - 1
    (case / "case.json").write_text(json.dumps(meta))
    limits = dict(min_cosine=0.9, argmax_fraction=1.0, mean_top5_overlap=2.0,
                  max_rel_rms=0.5, max_top1_rel_err=0.0)
    bound = dict(protocol="dg-fixed-input-v1", model=meta["model"], llama_cpp_commit="pin",
                 cases={"fixed": dict(inputs=meta, states={"no-sc": limits, "sc": limits})})
    with pytest.raises(SystemExit, match=f"lacks a complete ref_spread for {label}"):
        parity.dg_compare(tmp_path, bound)


def runtime_command(*, sc: bool = False, code: str = "") -> list[str]:
    program = "import pathlib,struct,sys; pathlib.Path(sys.argv[1]).write_bytes(struct.pack('<2f',1.,2.)); " + code
    command = [sys.executable, "-c", program, "{out}", "{model}", "{case_json}", "{variant}"]
    if sc:
        command += ["{prev}", "{prev_temp_inv}"]
    return command


def test_candidate_passes_canvas_positions_to_logits_argv(tmp_path: Path) -> None:
    case, model, meta = fixture(tmp_path)
    meta.update(prompt_tokens=[11, 12, 13], canvas_tokens=[99, 99], canvas_length=2)
    (case / "case.json").write_text(json.dumps(meta))
    program = (
        "import pathlib,sys; "
        "assert sys.argv[1:5] == ['--tokens', '11,12,13,99,99', '--rows', '3,4']; "
        "pathlib.Path(sys.argv[5]).write_bytes(bytes(16))"
    )
    command = [sys.executable, "-c", program, "--tokens", "{prompt_csv},{canvas_csv}",
               "--rows", "{rows_csv}", "{out}", "{model}", "{variant}"]
    candidate.export_candidate(case, model, "unified", command, candidate_revision="fixture")


@pytest.mark.parametrize(("variant", "path"), [
    ("unified", "unified"), ("cached", "cached"), ("unified-sc", "unified"), ("cached-sc", "cached"),
])
def test_candidate_passes_logits_path(tmp_path: Path, variant: str, path: str) -> None:
    case, model, _ = fixture(tmp_path)
    command = runtime_command(sc=variant.endswith("-sc"), code=f"assert sys.argv[4] == {path!r}")
    command[command.index("{variant}")] = "{path}"
    candidate.export_candidate(case, model, variant, command, candidate_revision="fixture")


@pytest.mark.parametrize("variant", ["unified", "cached"])
@pytest.mark.parametrize("placeholder", ["{prev}", "{prev_temp_inv}", "{prev!s}", "{prev_temp_inv:>8}"])
def test_candidate_refuses_previous_inputs_without_sc(tmp_path: Path, variant: str, placeholder: str) -> None:
    case, model, _ = fixture(tmp_path)
    command = runtime_command() + [placeholder]
    with pytest.raises(SystemExit, match="self-conditioning"):
        candidate.export_candidate(case, model, variant, command, candidate_revision="fixture")
    assert not (case / f"{variant}.mojo.f32").exists()


@pytest.mark.parametrize("changed", ["path", "size", "mtime"])
def test_candidate_model_must_match_reference_source(tmp_path: Path, changed: str) -> None:
    case, model, meta = fixture(tmp_path)
    if changed == "path":
        other = tmp_path / "other.gguf"
        other.write_bytes(model.read_bytes())
        os.utime(other, ns=(model.stat().st_atime_ns, meta["model_mtime_ns"]))
        model = other
    elif changed == "size":
        model.write_bytes(model.read_bytes() + b"x")
        os.utime(model, ns=(model.stat().st_atime_ns, meta["model_mtime_ns"]))
    else:
        os.utime(model, ns=(model.stat().st_atime_ns, model.stat().st_mtime_ns + 1_000_000_000))
    with pytest.raises(SystemExit, match="source model"):
        candidate.export_candidate(case, model, "unified", runtime_command(), candidate_revision="fixture")


def test_candidate_rejects_unknown_variant_before_execution(tmp_path: Path) -> None:
    case, model, _ = fixture(tmp_path)
    with pytest.raises(SystemExit, match="unknown.*variant"):
        candidate.export_candidate(case, model, "unknown", runtime_command(), candidate_revision="fixture")


@pytest.mark.parametrize("missing", ["model", "out", "variant", "case_json", "prompt_csv", "canvas_csv", "prev", "prev_temp_inv"])
def test_candidate_command_must_consume_every_required_input(tmp_path: Path, missing: str) -> None:
    case, model, _ = fixture(tmp_path)
    command = runtime_command(sc=True)
    if missing in ("prompt_csv", "canvas_csv"):
        command.remove("{case_json}")
        command += ["{prompt_csv}", "{canvas_csv}"]
    command.remove("{" + missing + "}")
    with pytest.raises(SystemExit, match="input placeholders"):
        candidate.export_candidate(case, model, "unified-sc", command, candidate_revision="fixture")


def test_candidate_requires_nonempty_build_revision(tmp_path: Path) -> None:
    case, model, _ = fixture(tmp_path)
    with pytest.raises(SystemExit, match="candidate revision"):
        candidate.export_candidate(case, model, "unified", runtime_command(), candidate_revision="  ")


@pytest.mark.parametrize("fault, message", [("truncated", "must contain"), ("model", "model changed"), ("previous", "previous logits changed")])
def test_candidate_does_not_issue_receipt_after_incomplete_or_changed_execution(tmp_path: Path, fault: str, message: str) -> None:
    case, model, _ = fixture(tmp_path)
    code = {
        "truncated": "pathlib.Path(sys.argv[1]).write_bytes(struct.pack('<f',1.))",
        "model": "pathlib.Path(sys.argv[2]).write_bytes(b'changed model')",
        "previous": "pathlib.Path(sys.argv[5]).write_bytes(struct.pack('<2f',2.,1.))",
    }[fault]
    with pytest.raises(SystemExit, match=message):
        candidate.export_candidate(case, model, "unified-sc", runtime_command(sc=True, code=code), candidate_revision="fixture")
    assert not (case / "unified-sc.mojo.receipt.json").exists()


def test_candidate_nonzero_exit_cannot_issue_a_receipt_even_after_writing_output(tmp_path: Path) -> None:
    case, model, _ = fixture(tmp_path)
    with pytest.raises(SystemExit, match="exit status 7; no receipt written"):
        candidate.export_candidate(case, model, "unified", runtime_command(code="sys.exit(7)"),
                                   candidate_revision="fixture")
    assert (case / "unified.mojo.f32").is_file()
    assert not (case / "unified.mojo.receipt.json").exists()


@pytest.mark.parametrize("field, value", [("producer", "other"), ("exit_status", 7), ("argv", []), ("executable", {}), ("candidate_revision", "")])
def test_receipt_requires_completed_identified_execution(tmp_path: Path, field: str, value: object) -> None:
    case, model, meta = fixture(tmp_path)
    candidate.export_candidate(case, model, "unified", runtime_command(), candidate_revision="fixture")
    path = case / "unified.mojo.receipt.json"
    receipt = json.loads(path.read_text())
    target = receipt if field == "producer" else receipt["execution"]
    target[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(SystemExit, match="completed candidate execution"):
        parity.dg_validate_receipt(case, meta, "unified")


def test_receipt_binds_previous_temperature(tmp_path: Path) -> None:
    case, model, meta = fixture(tmp_path)
    candidate.export_candidate(case, model, "unified-sc", runtime_command(sc=True), candidate_revision="fixture")
    with pytest.raises(SystemExit, match="inputs"):
        parity.dg_validate_receipt(case, dict(meta, prev_temp_inv=1.0), "unified-sc")


def test_receipt_rejects_truncation_even_with_matching_output_identity(tmp_path: Path) -> None:
    case, model, meta = fixture(tmp_path)
    candidate.export_candidate(case, model, "unified", runtime_command(), candidate_revision="fixture")
    out = case / "unified.mojo.f32"
    out.write_bytes(struct.pack("<f", 1.0))
    path = case / "unified.mojo.receipt.json"
    receipt = json.loads(path.read_text())
    receipt["output"] = dict(file=out.name, bytes=out.stat().st_size, sha256=parity._sha256(out))
    path.write_text(json.dumps(receipt))
    with pytest.raises(SystemExit, match="output"):
        parity.dg_validate_receipt(case, meta, "unified")


def test_executed_identical_output_has_a_valid_receipt_and_replaced_output_fails(tmp_path: Path) -> None:
    case, model, meta = fixture(tmp_path)
    program = tmp_path / "tiny_runtime.py"
    program.write_text("import pathlib,struct,sys\npathlib.Path(sys.argv[1]).write_bytes(struct.pack('<2f',1.,2.))\n")
    template = [sys.executable, str(program), "{out}", "{model}", "{case_json}", "{variant}"]
    candidate.export_candidate(case, model, "unified", template, candidate_revision="tiny-fixture")
    parity.dg_validate_receipt(case, meta, "unified")
    out = case / "unified.mojo.f32"
    assert out.read_bytes() == (case / "unified.ref.f32").read_bytes()
    out.write_bytes(struct.pack("<2f", 2.0, 1.0))
    with pytest.raises(SystemExit, match="output"):
        parity.dg_validate_receipt(case, meta, "unified")


def test_a_same_named_output_without_execution_receipt_is_refused(tmp_path: Path) -> None:
    case, _, meta = fixture(tmp_path)
    (case / "unified.mojo.f32").write_bytes((case / "unified.ref.f32").read_bytes())
    with pytest.raises(SystemExit, match="receipt"):
        parity.dg_validate_receipt(case, meta, "unified")


def test_changed_case_inputs_are_refused_after_export(tmp_path: Path) -> None:
    case, model, meta = fixture(tmp_path)
    program = tmp_path / "tiny_runtime.py"
    program.write_text("import pathlib,struct,sys\npathlib.Path(sys.argv[1]).write_bytes(struct.pack('<2f',1.,2.))\n")
    candidate.export_candidate(case, model, "unified", [sys.executable, str(program), "{out}", "{model}",
                                                        "{case_json}", "{variant}"], candidate_revision="tiny-fixture")
    with pytest.raises(SystemExit, match="inputs"):
        parity.dg_validate_receipt(case, dict(meta, canvas_tokens=[4]), "unified")


def test_wrong_variant_or_replaced_shared_previous_logits_is_refused(tmp_path: Path) -> None:
    case, model, meta = fixture(tmp_path)
    program = tmp_path / "tiny_runtime.py"
    program.write_text("import pathlib,struct,sys\npathlib.Path(sys.argv[1]).write_bytes(struct.pack('<2f',1.,2.))\n")
    candidate.export_candidate(case, model, "unified-sc", [sys.executable, str(program), "{out}", "{model}",
                                                           "{case_json}", "{variant}", "{prev}", "{prev_temp_inv}"],
                               candidate_revision="tiny-fixture")
    receipt_path = case / "unified-sc.mojo.receipt.json"
    receipt = json.loads(receipt_path.read_text())
    changed = json.loads(receipt_path.read_text())
    changed["inputs"]["variant"] = "cached-sc"
    receipt_path.write_text(json.dumps(changed))
    with pytest.raises(SystemExit, match="inputs"):
        parity.dg_validate_receipt(case, meta, "unified-sc")
    receipt_path.write_text(json.dumps(receipt))
    (case / "unified.ref.f32").write_bytes(struct.pack("<2f", 2.0, 1.0))
    with pytest.raises(SystemExit, match="inputs"):
        parity.dg_validate_receipt(case, meta, "unified-sc")


def test_real_comparator_accepts_zero_spread_identical_executed_candidates(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy", reason="numpy is in the `fixtures` group, not the `test` group; CI runs this file there")
    case, model, meta = fixture(tmp_path)
    meta["ref_spread"] = reference_controls()
    (case / "case.json").write_text(json.dumps(meta))
    program = tmp_path / "tiny_runtime.py"
    program.write_text("import pathlib,struct,sys\npathlib.Path(sys.argv[1]).write_bytes(struct.pack('<2f',1.,2.))\n")
    for variant in parity.DG_VARIANTS:
        (case / f"{variant}.ref.f32").write_bytes(struct.pack("<2f", 1.0, 2.0))
        command = [sys.executable, str(program), "{out}", "{model}", "{case_json}", "{variant}"]
        if variant.endswith("-sc"):
            command += ["{prev}", "{prev_temp_inv}"]
        candidate.export_candidate(case, model, variant, command, candidate_revision="tiny-fixture")
    metrics = parity.dg_metrics(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]), 0)
    zero = dict(min_cosine=metrics["min_cosine"], argmax_fraction=1.0, mean_top5_overlap=2.0,
                max_rel_rms=0.0, max_top1_rel_err=0.0)
    bound = dict(protocol="dg-fixed-input-v1", model=meta["model"], llama_cpp_commit="pin",
                 cases={"fixed": dict(inputs=meta, states={"no-sc": zero, "sc": zero})})
    entries, violations = parity.dg_compare(tmp_path, bound)
    assert len(entries) == 4 and violations == []
    assert all(e["candidate_receipt"]["execution"]["candidate_revision"] == "tiny-fixture" for e in entries)
    assert all(e["mojo_dllm_commit"] == "tiny-fixture" for e in entries)
    receipt_path = case / "cached.mojo.receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["execution"]["candidate_revision"] = "another-build"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(SystemExit, match="different builds"):
        parity.dg_compare(tmp_path, bound)
    receipt["execution"]["candidate_revision"] = "tiny-fixture"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(SystemExit, match="case set"):
        parity.dg_compare(tmp_path, dict(bound, cases={**bound["cases"], "missing prompt": bound["cases"]["fixed"]}))
    (case / "case.json").write_text(json.dumps(dict(meta, prompt="stale prompt")))
    with pytest.raises(SystemExit, match="stale prompt"):
        parity.dg_compare(tmp_path, bound)
    (case / "case.json").write_text(json.dumps(meta))
    # A complete candidate still cannot be compared against an extra reference row.
    (case / "cached.ref.f32").write_bytes(struct.pack("<4f", 1.0, 2.0, 1.0, 2.0))
    with pytest.raises(SystemExit, match="shape"):
        parity.dg_compare(tmp_path, bound)
    (case / "cached.ref.f32").write_bytes(struct.pack("<2f", 1.0, 2.0))
    # Truncating both sides in the same way used to satisfy the shape comparison.
    (case / "unified.ref.f32").write_bytes(struct.pack("<f", 1.0))
    (case / "unified.mojo.f32").write_bytes(struct.pack("<f", 1.0))
    with pytest.raises(SystemExit, match="output"):
        parity.dg_compare(tmp_path, bound)


@pytest.mark.parametrize("state", ["no-sc", "sc"])
def test_protocol_refuses_copied_output_with_a_valid_receipt_and_nonzero_spread(tmp_path: Path, state: str) -> None:
    case, model, meta = fixture(tmp_path)
    meta["ref_spread"] = reference_controls()
    (case / "case.json").write_text(json.dumps(meta))
    for variant in parity.DG_VARIANTS:
        (case / f"{variant}.ref.f32").write_bytes(struct.pack("<2f", 1.0, 2.0))
        program = "import pathlib,shutil,sys; shutil.copyfile(pathlib.Path(sys.argv[1]).with_name(sys.argv[2]+'.ref.f32'),sys.argv[3])"
        command = [sys.executable, "-c", program, "{case_json}", "{variant}", "{out}", "{model}"]
        if variant.endswith("-sc"):
            command += ["{prev}", "{prev_temp_inv}"]
        candidate.export_candidate(case, model, variant, command, candidate_revision="copy-fixture")
    zero = dict(min_cosine=0.99, argmax_fraction=1.0, mean_top5_overlap=2.0,
                max_rel_rms=0.0, max_top1_rel_err=0.0)
    states = {"no-sc": zero, "sc": zero}
    states[state] = dict(zero, max_rel_rms=0.1)
    bound = dict(protocol="dg-fixed-input-v1", model=meta["model"], llama_cpp_commit="pin",
                 cases={"fixed": dict(inputs=meta, states=states)})
    expected = "unified-sc.mojo.f32" if state == "sc" else "unified.mojo.f32"
    with pytest.raises(SystemExit, match=f"{expected}.*byte-identical"):
        parity.dg_compare(tmp_path, bound)


def test_protocol_softcap_rejects_executed_outputs_with_all_distance_metrics_in_bound(tmp_path: Path) -> None:
    pytest.importorskip("numpy", reason="numpy is in the `fixtures` group, not the `test` group; CI runs this file there")
    case, model, meta = fixture(tmp_path)
    meta["ref_spread"] = reference_controls()
    (case / "case.json").write_text(json.dumps(meta))
    for variant in parity.DG_VARIANTS:
        (case / f"{variant}.ref.f32").write_bytes(struct.pack("<2f", 29.0, 14.0))
        command = runtime_command(sc=variant.endswith("-sc"),
                                  code="pathlib.Path(sys.argv[1]).write_bytes(struct.pack('<2f',31.,15.))")
        candidate.export_candidate(case, model, variant, command, candidate_revision="cap-fixture")
    limits = dict(min_cosine=0.9, argmax_fraction=1.0, mean_top5_overlap=2.0,
                  max_rel_rms=0.5, max_top1_rel_err=0.5)
    bound = dict(protocol="dg-fixed-input-v1", model=meta["model"], llama_cpp_commit="pin",
                 cases={"fixed": dict(inputs=meta, states={"no-sc": limits, "sc": limits})})
    entries, violations = parity.dg_compare(tmp_path, bound)
    assert len(entries) == 4
    assert all(parity.dg_violations(entry, limits) == [] for entry in entries)
    assert all(entry["max_abs_logit"] == 31.0 for entry in entries)
    assert len(violations) == 4 and all("exceeds model soft-cap 30" in message for message in violations)
