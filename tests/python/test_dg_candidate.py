"""A real subprocess produces the candidate and a content-bound execution receipt."""

from __future__ import annotations

import importlib
import json
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
    np = pytest.importorskip("numpy")
    case, model, meta = fixture(tmp_path)
    meta["ref_spread"] = {}
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
    # Truncating both sides in the same way used to satisfy the shape comparison.
    (case / "unified.ref.f32").write_bytes(struct.pack("<f", 1.0))
    (case / "unified.mojo.f32").write_bytes(struct.pack("<f", 1.0))
    with pytest.raises(SystemExit, match="output"):
        parity.dg_compare(tmp_path, bound)
