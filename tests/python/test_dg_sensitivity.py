"""The sensitivity experiment must compare like inputs and retain its controls."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
spec = importlib.util.spec_from_file_location("dg_sensitivity", ROOT / "bench/dg_sensitivity.py")
assert spec is not None and spec.loader is not None
sensitivity = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sensitivity
spec.loader.exec_module(sensitivity)


def test_suite_has_more_than_two_distinct_prompts() -> None:
    assert len(sensitivity.PROMPTS) >= 4
    assert len(set(sensitivity.PROMPTS)) == len(sensitivity.PROMPTS)


def test_original_plan_has_path_and_repeat_controls_in_both_states(tmp_path: Path) -> None:
    jobs = sensitivity.forward_plan(tmp_path, tmp_path, "12", mutant=False)
    by_name = {job.name: job for job in jobs}
    assert set(by_name) == {
        "unified", "cached", "unified-repeat", "unified-sc", "cached-sc", "unified-sc-repeat", "unified-sc-noisy"
    }
    assert by_name["unified"].previous is None
    assert by_name["unified-repeat"].previous is None
    assert by_name["cached"].cached
    for name in ("unified-sc", "cached-sc", "unified-sc-repeat"):
        assert by_name[name].previous == tmp_path / "unified.ref.f32"
    assert by_name["unified-sc-noisy"].previous == tmp_path / "prev-noisy.f32"


def test_mutant_self_conditioning_uses_original_previous_step(tmp_path: Path) -> None:
    original, mutant = tmp_path / "original", tmp_path / "mutant"
    jobs = sensitivity.forward_plan(mutant, original, "12", mutant=True)
    assert {job.name for job in jobs} == {"unified", "cached", "unified-sc", "cached-sc"}
    assert all(job.previous == original / "unified.ref.f32" for job in jobs if job.name.endswith("-sc"))
    assert all(job.output.parent == mutant for job in jobs)


def test_case_identity_includes_canvas_and_previous_temperature() -> None:
    meta = dict(prompt="same", prompt_tokens=[2, 3], canvas_tokens=[4, 5], n_vocab=7, prev_temp_inv=1.25)
    sensitivity.require_same_inputs(meta, dict(meta), "case0")
    for key, wrong in (("prompt_tokens", [2, 4]), ("canvas_tokens", [5, 4]), ("prev_temp_inv", 1.0)):
        try:
            sensitivity.require_same_inputs(meta, dict(meta, **{key: wrong}), "case0")
        except SystemExit as exc:
            assert key in str(exc) and "case0" in str(exc)
        else:
            raise AssertionError(f"{key} mismatch was accepted")


def test_reconstructed_source_is_recorded_as_a_manifest_not_git_head(tmp_path: Path) -> None:
    manifest = tmp_path / "source.json"
    source = dict(pin=sensitivity.parity.dg_pinned_commit(ROOT), mechanism="copy and overlay pinned files",
                  files=[dict(path="src/model.cpp", github_blob="source-id")])
    manifest.write_text(json.dumps(source))
    identity = sensitivity.reference_source(tmp_path, manifest)
    assert identity["manifest"] == source
    assert "reconstructed" in identity["mechanism"]
    manifest.write_text(json.dumps(dict(source, pin="0" * 40)))
    try:
        sensitivity.reference_source(tmp_path, manifest)
    except SystemExit as exc:
        assert str(manifest) in str(exc)
    else:
        raise AssertionError("a manifest for another commit was accepted")


def test_input_perturbations_do_not_set_the_implementation_bound() -> None:
    small = dict(min_cosine=0.99, argmax_agree=9, rows=10, mean_top5_overlap=4.5,
                 max_rel_rms=0.1, top1_max_rel_err=0.02)
    noisy = dict(small, min_cosine=0.1, max_rel_rms=2.0)
    records = [dict(small, state="no-sc", kind="repeat", same_inputs=True),
               dict(small, state="sc", kind="cache-path", same_inputs=True),
               dict(noisy, state="sc", kind="previous-logit-noise", same_inputs=False)]
    envelopes = sensitivity.same_input_envelopes(records)
    assert envelopes["sc"]["min_cosine"] == 0.99
    assert envelopes["sc"]["max_rel_rms"] == 0.1
    assert envelopes["no-sc"]["max_top1_rel_err"] == 0.02


def test_missing_controls_are_an_error_not_an_unbounded_state() -> None:
    try:
        sensitivity.same_input_envelopes([])
    except SystemExit as exc:
        assert "no-sc" in str(exc)
    else:
        raise AssertionError("missing non-self-conditioned controls were accepted")


def test_mutant_with_its_own_previous_logits_is_refused(tmp_path: Path) -> None:
    meta = dict(prompt="same", prompt_tokens=[2], canvas_tokens=[3], n_vocab=4, prev_temp_inv=1.25)
    previous = tmp_path / "original" / "unified.ref.f32"
    sensitivity.require_mutant_inputs(meta, dict(meta, prev_file=str(previous)), "case0", previous)
    try:
        sensitivity.require_mutant_inputs(meta, dict(meta, prev_file=str(tmp_path / "mutant" / previous.name)),
                                          "case0", previous)
    except SystemExit as exc:
        assert "original reference's previous" in str(exc)
    else:
        raise AssertionError("the mutant's own previous-step logits were accepted")


def test_documented_spread_is_rendered_from_the_measurement_record() -> None:
    import check_dg_bound

    evidence = json.loads((ROOT / sensitivity.parity.DG_SENSITIVITY_EVIDENCE).read_text(encoding="utf-8"))
    docs = (ROOT / "docs/correctness.md").read_text(encoding="utf-8")
    section = docs.split("<!-- dg-sensitivity:start -->\n", 1)[1].split("<!-- dg-sensitivity:end -->", 1)[0]
    assert section == check_dg_bound.render_sensitivity(evidence)
