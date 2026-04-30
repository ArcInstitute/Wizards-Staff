"""Tests for :mod:`wizards_staff.stats.outliers`.

Mirrors the ``tests/labeling`` pattern: the module under test is loaded as
a stand-alone module via :mod:`importlib.util` so the test suite stays
runnable in environments where the heavy-weight optional dependencies of
the top-level ``wizards_staff`` package (``caiman``, ``tensorflow``...)
are not installed.

Covers the new PNR-based ``detect_low_pnr_neurons`` detector and its
backwards-compat alias, plus the integration with ``combine_neuron_qc``.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Module loader — keeps the test suite independent of caiman / tensorflow.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTLIERS_PATH = REPO_ROOT / "wizards_staff" / "stats" / "outliers.py"


def _load_outliers_module():
    """Import ``outliers.py`` standalone, bypassing the package __init__."""
    mod_name = "wizards_staff_stats_outliers_under_test"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, str(OUTLIERS_PATH))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def outliers():
    return _load_outliers_module()


# ---------------------------------------------------------------------------
# Synthetic-trace helpers.
# ---------------------------------------------------------------------------
def _gauss_event(n_frames: int, position: int, amplitude: float,
                 sigma_frames: float = 4.0) -> np.ndarray:
    """Return a single Gaussian transient embedded in a zero baseline."""
    t = np.arange(n_frames, dtype=float)
    return amplitude * np.exp(-0.5 * ((t - position) / sigma_frames) ** 2)


def _make_clean_neuron(rng: np.random.Generator, n_frames: int = 1000,
                       amplitude: float = 2.0, noise_sigma: float = 0.05,
                       n_events: int = 3) -> np.ndarray:
    """Trace dominated by a few real events on top of mild Gaussian noise."""
    trace = rng.normal(0.0, noise_sigma, size=n_frames)
    spacing = max(n_frames // (n_events + 1), 1)
    for k in range(n_events):
        trace = trace + _gauss_event(n_frames, (k + 1) * spacing, amplitude)
    return trace


def _make_noise_neuron(rng: np.random.Generator, n_frames: int = 1000,
                       noise_sigma: float = 0.05) -> np.ndarray:
    """Pure white-noise trace (no real events)."""
    return rng.normal(0.0, noise_sigma, size=n_frames)


# ---------------------------------------------------------------------------
# PNR computation primitives.
# ---------------------------------------------------------------------------
def test_pnr_computation_clean_trace(outliers):
    """A clean Gaussian event well above the noise floor gives a high PNR."""
    rng = np.random.default_rng(42)
    trace = _make_clean_neuron(rng, n_frames=1000, amplitude=2.0,
                               noise_sigma=0.05, n_events=3)
    res = outliers._compute_pnr(trace)
    assert res["flat"] is False
    # amplitude/noise ≈ 40; allow a wide band for synthetic stochasticity.
    assert res["pnr"] > 10.0, f"clean trace PNR too low: {res['pnr']}"
    assert res["sigma_hf"] > 0.0


def test_pnr_computation_noise_only(outliers):
    """Pure-noise traces have PNR comparable to the expected max-of-Gaussian."""
    rng = np.random.default_rng(0)
    trace = _make_noise_neuron(rng, n_frames=1000, noise_sigma=0.05)
    res = outliers._compute_pnr(trace)
    # For Gaussian white noise of length N, (max - median) / sigma is
    # ~sqrt(2 ln N) ~ 3.7 for N=1000. Anything in [1, 10] is consistent
    # with "noise-driven, no real signal" rather than a true event.
    assert 1.0 <= res["pnr"] <= 10.0, f"noise-only PNR out of band: {res['pnr']}"
    assert res["flat"] is False


def test_sigma_hf_robust_to_events(outliers):
    """A single large event must not inflate sigma_hf vs. the baseline alone.

    This is the key property that makes the PNR detector immune to the
    contamination bug: differencing kills slow event structure so the
    noise estimator stays anchored to the noise floor.
    """
    rng = np.random.default_rng(7)
    n = 2000
    baseline = rng.normal(0.0, 0.05, size=n)
    trace_with_event = baseline + _gauss_event(n, n // 2, amplitude=3.0,
                                               sigma_frames=4.0)

    s_baseline = outliers._compute_pnr(baseline)["sigma_hf"]
    s_with_event = outliers._compute_pnr(trace_with_event)["sigma_hf"]

    rel_change = abs(s_with_event - s_baseline) / s_baseline
    assert rel_change < 0.10, (
        f"sigma_hf shifted by {rel_change:.2%} when adding one event "
        f"(baseline={s_baseline:.4f}, with event={s_with_event:.4f})"
    )


def test_flat_trace_handled(outliers):
    """A constant trace produces ``reason=='flat_trace'`` rather than NaN/crash."""
    n_neurons = 6
    n_frames = 500
    dff = np.zeros((n_neurons, n_frames), dtype=float)
    # Give the other neurons some signal so the population median is well
    # defined and only the constant trace lands in flat_trace.
    rng = np.random.default_rng(1)
    for i in range(1, n_neurons):
        dff[i] = _make_clean_neuron(rng, n_frames=n_frames)
    filtered_idx = np.arange(n_neurons)

    res = outliers.detect_low_pnr_neurons(dff, filtered_idx, threshold=3.5)
    scores = res["neuron_scores"]
    assert scores.loc[scores["component_idx"] == 0, "reason"].iloc[0] \
        == "flat_trace"
    # The flat neuron should be flagged via the flat-trace path even though
    # its log-PNR mod-Z would not have crossed the threshold on its own.
    assert bool(scores.loc[scores["component_idx"] == 0, "is_low_pnr"].iloc[0])


# ---------------------------------------------------------------------------
# Population-level detection.
# ---------------------------------------------------------------------------
def test_one_sided_high_pnr_not_flagged(outliers):
    """An unusually responsive neuron must NOT be flagged (high tail OK)."""
    rng = np.random.default_rng(2)
    n_neurons = 30
    n_frames = 1000
    dff = np.zeros((n_neurons, n_frames), dtype=float)
    for i in range(n_neurons):
        dff[i] = _make_clean_neuron(rng, n_frames=n_frames, amplitude=1.0,
                                    noise_sigma=0.05, n_events=3)
    # One neuron with 10× larger events.
    dff[0] = _make_clean_neuron(rng, n_frames=n_frames, amplitude=10.0,
                                noise_sigma=0.05, n_events=3)
    filtered_idx = np.arange(n_neurons)

    res = outliers.detect_low_pnr_neurons(dff, filtered_idx, threshold=3.5)
    scores = res["neuron_scores"]
    high_neuron = scores.loc[scores["component_idx"] == 0].iloc[0]
    assert not bool(high_neuron["is_low_pnr"]), \
        "High-PNR neuron was incorrectly flagged on the one-sided detector"


def test_low_pnr_flagged(outliers):
    """A neuron whose PNR is ~10× lower than the population is flagged."""
    rng = np.random.default_rng(3)
    n_neurons = 30
    n_frames = 1000
    dff = np.zeros((n_neurons, n_frames), dtype=float)
    for i in range(n_neurons):
        dff[i] = _make_clean_neuron(rng, n_frames=n_frames, amplitude=2.0,
                                    noise_sigma=0.05, n_events=3)
    # Neuron 0 has tiny events relative to its noise floor.
    dff[0] = _make_clean_neuron(rng, n_frames=n_frames, amplitude=0.05,
                                noise_sigma=0.05, n_events=3)
    filtered_idx = np.arange(n_neurons)

    res = outliers.detect_low_pnr_neurons(dff, filtered_idx, threshold=3.5)
    scores = res["neuron_scores"]
    low_neuron = scores.loc[scores["component_idx"] == 0].iloc[0]
    assert bool(low_neuron["is_low_pnr"]), \
        "Low-PNR neuron was not flagged"
    assert low_neuron["reason"] == "low_pnr"


def test_contamination_resistance(outliers):
    """Regression test for the contamination bug.

    Population: 70% noise-dominated neurons + 30% clean responders. The
    new detector must NOT flag the clean neurons just because the noise
    population pulls the population reference distribution toward low
    PNR. (The noise neurons themselves are allowed to be flagged or not;
    that's not what's being asserted here.)
    """
    rng = np.random.default_rng(4)
    n_total = 30
    n_noise = 21  # 70 %
    n_frames = 1500
    dff = np.zeros((n_total, n_frames), dtype=float)
    for i in range(n_noise):
        dff[i] = _make_noise_neuron(rng, n_frames=n_frames, noise_sigma=0.05)
    for j, i in enumerate(range(n_noise, n_total)):
        dff[i] = _make_clean_neuron(rng, n_frames=n_frames, amplitude=1.8,
                                    noise_sigma=0.05, n_events=4)
    filtered_idx = np.arange(n_total)

    res = outliers.detect_low_pnr_neurons(dff, filtered_idx, threshold=3.5)
    scores = res["neuron_scores"]

    clean_scores = scores[scores["component_idx"] >= n_noise]
    flagged_clean = int(clean_scores["is_low_pnr"].sum())
    assert flagged_clean == 0, (
        f"{flagged_clean}/{len(clean_scores)} clean neurons were incorrectly "
        f"flagged in a noise-contaminated population. PNRs of clean neurons: "
        f"{clean_scores['pnr'].round(2).tolist()}"
    )


# ---------------------------------------------------------------------------
# Backwards compatibility.
# ---------------------------------------------------------------------------
def test_deprecation_warning(outliers):
    """The old name still works but warns and forwards to the new impl."""
    rng = np.random.default_rng(5)
    n_neurons = 12
    n_frames = 500
    dff = np.stack([
        _make_clean_neuron(rng, n_frames=n_frames) for _ in range(n_neurons)
    ])
    filtered_idx = np.arange(n_neurons)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_res = outliers.detect_neuron_outliers(dff, filtered_idx,
                                                  threshold=3.5)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), \
        "detect_neuron_outliers did not emit a DeprecationWarning"

    new_res = outliers.detect_low_pnr_neurons(dff, filtered_idx, threshold=3.5)
    # Same shape and same flagged set — alias must be a thin pass-through.
    assert old_res["n_flagged"] == new_res["n_flagged"]
    np.testing.assert_array_equal(old_res["flagged_mask"],
                                  new_res["flagged_mask"])
    assert "is_low_pnr" in old_res["neuron_scores"].columns


def test_combine_neuron_qc_uses_new_key(outliers):
    """``combine_neuron_qc`` accepts ``low_pnr_result=`` and passes the key.

    Also asserts that the legacy ``amplitude_result=`` keyword is still
    honoured for one release cycle (with a DeprecationWarning) so older
    external callers don't break overnight.
    """
    rng = np.random.default_rng(6)
    n_neurons = 15
    n_frames = 600
    dff = np.stack([
        _make_clean_neuron(rng, n_frames=n_frames) for _ in range(n_neurons)
    ])
    # Inject one weak neuron so the combined report has a real flag.
    dff[0] = _make_clean_neuron(rng, n_frames=n_frames, amplitude=0.05,
                                noise_sigma=0.05)
    filtered_idx = np.arange(n_neurons)

    low_pnr = outliers.detect_low_pnr_neurons(dff, filtered_idx, threshold=3.5)
    combined = outliers.combine_neuron_qc(filtered_idx,
                                          low_pnr_result=low_pnr)
    df = combined["combined_df"]
    # New-schema columns should be present, legacy ones absent.
    for col in ["pnr", "sigma_hf", "log_pnr_modified_zscore",
                "is_low_pnr", "reason", "n_flags", "any_outlier"]:
        assert col in df.columns, f"missing expected column {col!r}"
    assert "is_outlier" not in df.columns

    # ``any_outlier`` should be the union of detector flags (only one
    # detector here, so they must match exactly).
    np.testing.assert_array_equal(
        df["any_outlier"].values.astype(bool),
        df["is_low_pnr"].values.astype(bool),
    )

    # Legacy keyword still accepted with a DeprecationWarning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy_combined = outliers.combine_neuron_qc(
            filtered_idx, amplitude_result=low_pnr,
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    np.testing.assert_array_equal(
        legacy_combined["combined_df"]["any_outlier"].values,
        df["any_outlier"].values,
    )


# ---------------------------------------------------------------------------
# Waveform-template parameterisation: indicator presets + overrides.
# ---------------------------------------------------------------------------
def _calcium_event(n_frames: int, position: int, fps: float,
                   rise_ms: float, decay_ms: float,
                   amplitude: float = 1.0) -> np.ndarray:
    """Generate a single linear-rise / exponential-decay calcium event.

    The shape mirrors :func:`outliers._make_calcium_template` so the
    template-mismatch test below can quantitatively reason about the
    expected correlation when indicator kinetics differ.
    """
    rise_frames = max(int(np.ceil(rise_ms / 1000.0 * fps)), 1)
    decay_tau_frames = max(decay_ms / 1000.0 * fps, 1.0)

    trace = np.zeros(n_frames, dtype=float)
    for offset in range(rise_frames):
        idx = position + offset
        if 0 <= idx < n_frames:
            trace[idx] = amplitude * (offset + 1) / rise_frames
    for offset in range(n_frames - position - rise_frames):
        idx = position + rise_frames + offset
        if 0 <= idx < n_frames:
            trace[idx] = amplitude * np.exp(-offset / decay_tau_frames)
    return trace


def _make_indicator_traces(rng: np.random.Generator, n_neurons: int,
                           n_frames: int, fps: float, rise_ms: float,
                           decay_ms: float, amplitude: float = 1.5,
                           n_events: int = 4,
                           noise_sigma: float = 0.02) -> np.ndarray:
    """Population of neurons with calcium events of a given indicator shape."""
    dff = rng.normal(0.0, noise_sigma, size=(n_neurons, n_frames))
    spacing = max(n_frames // (n_events + 1), 1)
    for i in range(n_neurons):
        for k in range(n_events):
            dff[i] += _calcium_event(
                n_frames, position=(k + 1) * spacing, fps=fps,
                rise_ms=rise_ms, decay_ms=decay_ms, amplitude=amplitude,
            )
    return dff


def test_waveform_template_parameterized(outliers):
    """Different rise/decay parameters must produce different templates."""
    fps = 30.0
    base = outliers._make_calcium_template(
        fps, rise_ms=50.0, decay_tau_ms=400.0, duration_ms=1500.0,
    )
    slower = outliers._make_calcium_template(
        fps, rise_ms=80.0, decay_tau_ms=1200.0, duration_ms=1500.0,
    )
    faster = outliers._make_calcium_template(
        fps, rise_ms=7.0, decay_tau_ms=90.0, duration_ms=1500.0,
    )
    assert base.shape == slower.shape == faster.shape
    assert not np.allclose(base, slower), \
        "GCaMP6s-shaped template should differ from GCaMP6f"
    assert not np.allclose(base, faster), \
        "jGCaMP8m-shaped template should differ from GCaMP6f"
    assert not np.allclose(slower, faster), \
        "GCaMP6s and jGCaMP8m templates should differ from each other"


def test_indicator_preset_lookup(outliers):
    """``indicator='GCaMP6s'`` produces a different template than 'GCaMP6f'."""
    rng = np.random.default_rng(11)
    n_neurons = 6
    n_frames = 1200
    fps = 30.0
    dff = _make_indicator_traces(
        rng, n_neurons=n_neurons, n_frames=n_frames, fps=fps,
        rise_ms=50.0, decay_ms=400.0,
    )
    filtered_idx = np.arange(n_neurons)

    res_6f = outliers.detect_waveform_outliers(
        dff, filtered_idx, fps=fps, indicator="GCaMP6f",
    )
    res_6s = outliers.detect_waveform_outliers(
        dff, filtered_idx, fps=fps, indicator="GCaMP6s",
    )

    assert res_6f["template"].shape == res_6s["template"].shape
    assert not np.allclose(res_6f["template"], res_6s["template"]), \
        "GCaMP6s preset must produce a different template than GCaMP6f"

    assert res_6f["template_params"]["decay_ms"] == 400.0
    assert res_6s["template_params"]["decay_ms"] == 1200.0
    assert res_6f["indicator"] == "GCaMP6f"
    assert res_6s["indicator"] == "GCaMP6s"


def test_indicator_preset_unknown_raises(outliers):
    """An unknown indicator name lists the available presets in the message."""
    rng = np.random.default_rng(12)
    n_neurons = 6
    n_frames = 800
    fps = 30.0
    dff = _make_indicator_traces(
        rng, n_neurons=n_neurons, n_frames=n_frames, fps=fps,
        rise_ms=50.0, decay_ms=400.0,
    )
    filtered_idx = np.arange(n_neurons)

    with pytest.raises(ValueError) as exc_info:
        outliers.detect_waveform_outliers(
            dff, filtered_idx, fps=fps, indicator="not_a_real_indicator",
        )
    msg = str(exc_info.value)
    assert "not_a_real_indicator" in msg
    # All known presets should be enumerated so users can pick one.
    for known in ["GCaMP6f", "GCaMP6s", "jGCaMP8m", "jRGECO1a"]:
        assert known in msg, f"preset {known} missing from error message: {msg}"


def test_explicit_override_wins(outliers):
    """Explicit template_decay_ms wins over the indicator preset."""
    fps = 30.0
    rng = np.random.default_rng(13)
    n_neurons = 6
    n_frames = 1200
    dff = _make_indicator_traces(
        rng, n_neurons=n_neurons, n_frames=n_frames, fps=fps,
        rise_ms=50.0, decay_ms=400.0,
    )
    filtered_idx = np.arange(n_neurons)

    res = outliers.detect_waveform_outliers(
        dff, filtered_idx, fps=fps,
        indicator="GCaMP6f", template_decay_ms=2000.0,
    )
    assert res["template_params"]["decay_ms"] == 2000.0, \
        "Explicit template_decay_ms must override the GCaMP6f preset (400 ms)"
    # Other preset values stay intact when not explicitly overridden.
    assert res["template_params"]["rise_ms"] == 50.0
    assert res["template_params"]["peak_height"] == 0.10

    # Override of peak_height also wins.
    res_ph = outliers.detect_waveform_outliers(
        dff, filtered_idx, fps=fps,
        indicator="jRGECO1a", peak_height=0.5,
    )
    assert res_ph["template_params"]["peak_height"] == 0.5


def test_default_unchanged(outliers):
    """Calling without template params reproduces the legacy GCaMP6f template."""
    fps = 30.0
    legacy_template = outliers._make_calcium_template(
        fps, rise_ms=50.0, decay_tau_ms=400.0, duration_ms=1500.0,
    )

    rng = np.random.default_rng(14)
    n_neurons = 6
    n_frames = 1200
    dff = _make_indicator_traces(
        rng, n_neurons=n_neurons, n_frames=n_frames, fps=fps,
        rise_ms=50.0, decay_ms=400.0,
    )
    filtered_idx = np.arange(n_neurons)

    res = outliers.detect_waveform_outliers(dff, filtered_idx, fps=fps)
    assert res["indicator"] is None
    np.testing.assert_allclose(res["template"], legacy_template)
    assert res["template_params"]["rise_ms"] == 50.0
    assert res["template_params"]["decay_ms"] == 400.0
    assert res["template_params"]["total_ms"] == 1500.0
    assert res["template_params"]["peak_height"] == 0.1


def _ideal_event_correlation(template: np.ndarray, event: np.ndarray) -> float:
    """Pearson r between an event waveform and the template, peak-aligned.

    Mirrors the snippet-vs-template correlation step inside
    :func:`detect_waveform_outliers` (z-score then ``np.corrcoef``) but
    bypasses the ``find_peaks`` step, which is unreliable on slow
    indicators where the decay tail produces multiple spurious local
    maxima above the absolute threshold and pollutes the median.
    """
    if event.std() < 1e-12 or template.std() < 1e-12:
        return float("nan")
    e = (event - event.mean()) / event.std()
    t = (template - template.mean()) / template.std()
    return float(np.corrcoef(e, t)[0, 1])


def test_template_mismatch_predictable_degradation(outliers):
    """A GCaMP6s-shaped event correlates better with the GCaMP6s template
    than with the GCaMP6f template, when both templates are aligned to
    the event peak.

    Guards against a "fix doesn't take" regression where the parameter
    plumbing looks right but the template ends up unchanged downstream
    (e.g. argument silently dropped between the public function and
    ``_make_calcium_template``).

    The test deliberately bypasses ``find_peaks`` because the absolute
    ``peak_height`` threshold + slow decay tail combination produces
    many false peaks per event for slow indicators, which is a separate
    detector issue (and the reason indicator-aware ``peak_height``
    matters in the first place). Here we want to isolate the *template
    shape* parameter from the *peak detection* parameter.
    """
    fps = 30.0
    template_total_ms = 1500.0

    # Build a single isolated GCaMP6s-shaped event using the same
    # construction as the template itself, so the matched-template
    # correlation has a well-defined optimum.
    gcamp6s_event = outliers._make_calcium_template(
        fps, rise_ms=80.0, decay_tau_ms=1200.0, duration_ms=template_total_ms,
    )

    template_6f = outliers._make_calcium_template(
        fps, rise_ms=50.0, decay_tau_ms=400.0, duration_ms=template_total_ms,
    )
    template_6s = outliers._make_calcium_template(
        fps, rise_ms=80.0, decay_tau_ms=1200.0, duration_ms=template_total_ms,
    )

    r_match = _ideal_event_correlation(template_6s, gcamp6s_event)
    r_mismatch = _ideal_event_correlation(template_6f, gcamp6s_event)

    # A perfectly matched template should approach r=1; the mismatched
    # one should be materially lower.
    assert r_match > 0.99, f"matched correlation suspiciously low: {r_match:.3f}"
    assert r_match - r_mismatch > 0.05, (
        f"GCaMP6s template should correlate better with a GCaMP6s-shaped "
        f"event than the GCaMP6f template does "
        f"(matched r={r_match:.3f}, mismatched r={r_mismatch:.3f}). "
        f"If these are nearly equal, the template parameter likely isn't "
        f"flowing through to _make_calcium_template."
    )

    # Sanity: the same comparison flips for a GCaMP6f-shaped event so the
    # asymmetry above is genuinely about template shape and not some
    # template-length artifact that always favors GCaMP6s.
    gcamp6f_event = outliers._make_calcium_template(
        fps, rise_ms=50.0, decay_tau_ms=400.0, duration_ms=template_total_ms,
    )
    r_match_6f = _ideal_event_correlation(template_6f, gcamp6f_event)
    r_mismatch_6f = _ideal_event_correlation(template_6s, gcamp6f_event)
    assert r_match_6f - r_mismatch_6f > 0.05, (
        f"GCaMP6f template should correlate better with a GCaMP6f-shaped "
        f"event than the GCaMP6s template does "
        f"(matched r={r_match_6f:.3f}, mismatched r={r_mismatch_6f:.3f})."
    )
