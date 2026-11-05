"""Run with the fixtures and test groups: numerical edge cases in DG diagnostics."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
check = importlib.import_module("check_dg_bound")
sensitivity = importlib.import_module("dg_sensitivity")


def test_uncapping_saturated_float32_logits_is_finite_and_detectable() -> None:
    logits = np.array([[-30.0, 0.0, 30.0]], dtype=np.float32)
    with np.errstate(all="raise"):
        mutant = check.remove_softcap(logits, 30.0)
    assert mutant.dtype == np.float64
    assert np.isfinite(mutant).all()
    assert mutant[0, 0] < -30.0 and mutant[0, 2] > 30.0
    assert np.argmax(mutant) == np.argmax(logits)


def test_distribution_distance_ignores_constant_logit_offset() -> None:
    a = np.array([[3.0, 1.0, -2.0], [-1.0, 4.0, 2.0]])
    metrics = sensitivity.sampling_metrics(a + 123.0, a, temperature=0.8)
    assert metrics["max_total_variation"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["max_js_divergence"] == pytest.approx(0.0, abs=1e-14)


def test_distribution_distance_detects_changed_winner_and_stays_finite() -> None:
    a, b = np.array([[1000.0, -1000.0]]), np.array([[-1000.0, 1000.0]])
    metrics = sensitivity.sampling_metrics(a, b, temperature=0.8)
    assert metrics["max_total_variation"] == pytest.approx(1.0)
    assert metrics["max_js_divergence"] == pytest.approx(math.log(2.0))
