"""
Outlier detection and handling for calcium imaging data.

Provides multiple methods for identifying outliers with clear guidance
on when to use each. Important: Always investigate outliers before
removing them - they may represent real biological phenomena!

Functions
---------
detect_low_pnr_neurons : Flag neurons whose peak-to-noise ratio is anomalously low
detect_waveform_outliers : Flag neurons whose transients don't match a calcium template
detect_spectral_outliers : Flag neurons with abnormal frequency-domain profiles
combine_neuron_qc : Merge results from multiple outlier detectors into a unified report
detect_outliers : Identify potential outliers using various methods
visualize_outliers : Create visualization showing outliers
handle_outliers : Apply various outlier handling strategies

Deprecated
----------
detect_neuron_outliers : alias of :func:`detect_low_pnr_neurons` retained for
    backwards compatibility with older callers; emits ``DeprecationWarning``.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

# Numerical floors used by the PNR detector. Exposed at module level so the
# tests and the validation tooling can reference the same constants without
# having to know magic numbers.
SIGMA_HF_FLOOR: float = 1e-9
PNR_LOG_EPSILON: float = 1e-3


# ---------------------------------------------------------------------------
# Calcium indicator presets for the waveform template detector.
# ---------------------------------------------------------------------------
# These are *starting points*, not authoritative values. They reflect
# approximate published kinetics for common GECIs and should be overridden
# with a per-experiment measurement whenever possible (different temperature,
# expression level, indicator titration, neuronal subtype, and acquisition
# noise floor all shift the effective rise/decay one observes in ΔF/F).
#
# References (rise time = 10–90 % rise, decay = single-exponential τ off):
#   - GCaMP6f / GCaMP6m / GCaMP6s: Chen et al., Nature 2013
#     ("Ultrasensitive fluorescent proteins for imaging neuronal activity").
#   - GCaMP7f: Dana et al., Nat. Methods 2019
#     ("High-performance calcium sensors for imaging activity in neuronal
#     populations and microcompartments").
#   - jGCaMP8f / jGCaMP8m / jGCaMP8s: Zhang et al., Nature 2023
#     ("Fast and sensitive GCaMP calcium indicators for imaging neural
#     populations").
#   - jRGECO1a / jRCaMP1a: Dana et al., eLife 2016
#     ("Sensitive red protein calcium indicators for imaging neural
#     activity").
#   - GCaMP3: Tian et al., Nat. Methods 2009
#     ("Imaging neural activity in worms, flies and mice with improved
#     GCaMP calcium indicators").
#
# ``peak_height`` is the absolute ΔF/F threshold passed to ``find_peaks``.
# Red indicators (jRGECO/jRCaMP) get a lower default because their typical
# excursion magnitude is roughly half that of green indicators, so the
# 0.10 GCaMP-tuned threshold would silently miss real events.
INDICATOR_PRESETS: Dict[str, Dict[str, float]] = {
    "GCaMP6f":  {"rise_ms": 50.0,  "decay_ms": 400.0,  "peak_height": 0.10},
    "GCaMP6s":  {"rise_ms": 80.0,  "decay_ms": 1200.0, "peak_height": 0.10},
    "GCaMP6m":  {"rise_ms": 60.0,  "decay_ms": 700.0,  "peak_height": 0.10},
    "GCaMP7f":  {"rise_ms": 30.0,  "decay_ms": 250.0,  "peak_height": 0.10},
    "jGCaMP8m": {"rise_ms": 7.0,   "decay_ms": 90.0,   "peak_height": 0.10},
    "jGCaMP8s": {"rise_ms": 15.0,  "decay_ms": 250.0,  "peak_height": 0.10},
    "jGCaMP8f": {"rise_ms": 4.0,   "decay_ms": 60.0,   "peak_height": 0.10},
    "jRGECO1a": {"rise_ms": 70.0,  "decay_ms": 600.0,  "peak_height": 0.05},
    "jRCaMP1a": {"rise_ms": 80.0,  "decay_ms": 1000.0, "peak_height": 0.05},
    "GCaMP3":   {"rise_ms": 200.0, "decay_ms": 700.0,  "peak_height": 0.10},
}

# Defaults applied when neither ``indicator`` nor an explicit override is
# provided. Match the legacy GCaMP6f-like values so behavior is unchanged
# for existing callers.
_DEFAULT_TEMPLATE_RISE_MS: float = 50.0
_DEFAULT_TEMPLATE_DECAY_MS: float = 400.0
_DEFAULT_TEMPLATE_TOTAL_MS: float = 1500.0
_DEFAULT_PEAK_HEIGHT: float = 0.1


def _compute_pnr(trace: np.ndarray) -> Dict[str, float]:
    """Compute the peak-to-noise ratio for a single ΔF/F trace.

    The noise estimate ``sigma_hf`` is the MAD of successive frame-to-frame
    differences scaled to a Gaussian standard deviation. Differencing
    suppresses low-frequency event structure, so the noise estimate stays
    robust even when the trace is dominated by genuine calcium transients
    (a property the older max/mean/std-based detector did not have).

    Args:
        trace: 1-D ΔF/F trace for a single neuron.

    Returns:
        Dict with keys:
            ``pnr``: ``(max(trace) - median(trace)) / sigma_hf``, or 0.0
                when the trace is degenerate (constant / near-constant).
            ``sigma_hf``: Robust high-frequency noise estimate.
            ``flat``: True iff ``sigma_hf < SIGMA_HF_FLOOR``, in which case
                ``pnr`` is forced to 0 and the neuron should be reported
                with ``reason="flat_trace"`` rather than ``"low_pnr"``.
    """
    arr = np.asarray(trace, dtype=float)
    if arr.size < 2:
        return {"pnr": 0.0, "sigma_hf": 0.0, "flat": True}

    diffs = np.diff(arr)
    mad_diff = np.median(np.abs(diffs))
    sigma_hf = 1.4826 * mad_diff / np.sqrt(2.0)

    if not np.isfinite(sigma_hf) or sigma_hf < SIGMA_HF_FLOOR:
        return {"pnr": 0.0, "sigma_hf": float(sigma_hf), "flat": True}

    peak = float(np.max(arr) - np.median(arr))
    pnr = peak / sigma_hf
    if not np.isfinite(pnr):
        return {"pnr": 0.0, "sigma_hf": float(sigma_hf), "flat": True}
    return {"pnr": float(pnr), "sigma_hf": float(sigma_hf), "flat": False}


def detect_low_pnr_neurons(
    dff_dat: np.ndarray,
    filtered_idx: np.ndarray,
    threshold: float = 3.5,
) -> Dict:
    """Flag neurons whose peak-to-noise ratio is anomalously low.

    For each neuron the per-trace peak-to-noise ratio (PNR) is computed
    via :func:`_compute_pnr`. The population's PNR distribution is then
    transformed to ``log(pnr + epsilon)`` and a modified Z-score on MAD
    is applied **one-sided** to the low tail only:

    .. code-block:: text

        log_pnr  = log(pnr + epsilon)
        mod_z    = 0.6745 * (log_pnr - median(log_pnr)) / MAD(log_pnr)
        flagged  = mod_z < -threshold

    The log transform is required because PNR is right-heavy-tailed; on a
    linear scale the MAD is dominated by the bulk and the long upper tail
    pushes thresholds into nonsense territory. The detector is one-sided
    because "this neuron is unusually responsive" is not a quality-control
    failure mode — if anything, it is the opposite. Replacing the older
    two-sided ``max``/``mean``/``std`` detector eliminates a contamination
    bug in which datasets with many noise-dominated neurons would shift
    the population reference distribution upward and cause clean,
    high-activity neurons to be flagged as outliers.

    Args:
        dff_dat: Full ΔF/F₀ matrix of shape ``(n_components, n_frames)``.
        filtered_idx: Row indices into ``dff_dat`` that passed spatial
            filtering. The detector operates on ``dff_dat[filtered_idx]``.
        threshold: Modified Z-score cutoff applied to ``log(pnr)``. A
            neuron is flagged when its log-PNR mod-Z is below
            ``-threshold``. Defaults to 3.5 (the same family value the
            other QC detectors use).

    Returns:
        Dict with the standard QC schema used by the other detectors,
        plus PNR-specific columns on ``neuron_scores``:

        ``flagged_mask``: bool array, length ``len(filtered_idx)``.
        ``flagged_idx``: original component indices that were flagged.
        ``clean_idx``: original component indices that were not flagged.
        ``n_flagged``: int.
        ``pct_flagged``: float.
        ``neuron_scores``: DataFrame with one row per neuron and columns
            ``component_idx``, ``pnr``, ``sigma_hf``,
            ``log_pnr_modified_zscore``, ``is_low_pnr``, ``reason``.
            ``reason`` is ``"ok"``, ``"low_pnr"``, or ``"flat_trace"``.
        ``threshold``: float (echo of the input).
        ``summary``: human-readable summary string.
    """
    traces = dff_dat[filtered_idx, :]
    n_neurons = traces.shape[0]

    if n_neurons < 4:
        empty_scores = pd.DataFrame(
            {
                "component_idx": np.asarray(filtered_idx),
                "pnr": np.zeros(n_neurons),
                "sigma_hf": np.zeros(n_neurons),
                "log_pnr_modified_zscore": np.zeros(n_neurons),
                "is_low_pnr": np.zeros(n_neurons, dtype=bool),
                "reason": ["ok"] * n_neurons,
            }
        )
        return {
            "flagged_mask": np.zeros(n_neurons, dtype=bool),
            "flagged_idx": np.array([], dtype=int),
            "clean_idx": np.asarray(filtered_idx),
            "n_flagged": 0,
            "pct_flagged": 0.0,
            "neuron_scores": empty_scores,
            "threshold": threshold,
            "summary": "Too few neurons for low-PNR outlier detection.",
        }

    pnr = np.zeros(n_neurons, dtype=float)
    sigma_hf = np.zeros(n_neurons, dtype=float)
    flat = np.zeros(n_neurons, dtype=bool)
    for i, trace in enumerate(traces):
        result = _compute_pnr(trace)
        pnr[i] = result["pnr"]
        sigma_hf[i] = result["sigma_hf"]
        flat[i] = bool(result["flat"])

    log_pnr = np.log(pnr + PNR_LOG_EPSILON)
    median = float(np.median(log_pnr))
    mad = float(np.median(np.abs(log_pnr - median)))
    if mad < 1e-12:
        mad = 1e-12
    mod_z = 0.6745 * (log_pnr - median) / mad

    is_low_pnr = mod_z < -threshold
    flagged = is_low_pnr | flat

    reasons = np.array(["ok"] * n_neurons, dtype=object)
    reasons[is_low_pnr] = "low_pnr"
    # ``flat`` overrides ``low_pnr`` in the reason column because it is the
    # more specific diagnosis: a flat trace cannot meaningfully be said to
    # have a "low" PNR — it simply has no signal at all.
    reasons[flat] = "flat_trace"

    scores_df = pd.DataFrame(
        {
            "component_idx": np.asarray(filtered_idx),
            "pnr": pnr,
            "sigma_hf": sigma_hf,
            "log_pnr_modified_zscore": mod_z,
            "is_low_pnr": flagged,
            "reason": reasons,
        }
    )

    flagged_idx = np.asarray(filtered_idx)[flagged]
    clean_idx = np.asarray(filtered_idx)[~flagged]
    n_flagged = int(flagged.sum())
    pct_flagged = (n_flagged / n_neurons) * 100 if n_neurons else 0.0

    summary = (
        f"Low-PNR outlier detection (threshold={threshold}, one-sided low tail): "
        f"flagged {n_flagged}/{n_neurons} neurons ({pct_flagged:.1f}%)."
    )

    return {
        "flagged_mask": flagged,
        "flagged_idx": flagged_idx,
        "clean_idx": clean_idx,
        "n_flagged": n_flagged,
        "pct_flagged": pct_flagged,
        "neuron_scores": scores_df,
        "threshold": threshold,
        "summary": summary,
    }


def detect_neuron_outliers(
    dff_dat: np.ndarray,
    filtered_idx: np.ndarray,
    threshold: float = 3.5,
) -> Dict:
    """Deprecated alias for :func:`detect_low_pnr_neurons`.

    The previous max/mean/std-based detector was renamed (and replaced)
    because it conflated signal amplitude with noise amplitude and was
    contaminated by noise-dominated neurons in mixed populations. New
    callers should use :func:`detect_low_pnr_neurons` directly.

    .. deprecated:: 0.2
       Use :func:`detect_low_pnr_neurons` instead.
    """
    warnings.warn(
        "detect_neuron_outliers is deprecated and will be removed in a future "
        "release; use detect_low_pnr_neurons instead. The detector's semantic "
        "has changed from a two-sided max/mean/std modified Z-score to a "
        "one-sided low-tail test on log(PNR). See CHANGELOG for migration "
        "details.",
        DeprecationWarning,
        stacklevel=2,
    )
    return detect_low_pnr_neurons(
        dff_dat=dff_dat, filtered_idx=filtered_idx, threshold=threshold
    )


def _make_calcium_template(
    fps: float,
    rise_ms: float = 50.0,
    decay_tau_ms: float = 400.0,
    duration_ms: float = 1500.0,
) -> np.ndarray:
    """Build a unit-normalised synthetic GCaMP-like transient template.

    Parameters
    ----------
    fps : float
        Frame rate of the recording (frames per second).
    rise_ms : float
        Rise time of the template in milliseconds.
    decay_tau_ms : float
        Exponential decay time-constant in milliseconds.
    duration_ms : float
        Total template length in milliseconds.

    Returns
    -------
    np.ndarray
        1-D template normalised to unit L2 norm.
    """
    n_frames = max(int(np.ceil(duration_ms / 1000.0 * fps)), 4)
    rise_frames = max(int(np.ceil(rise_ms / 1000.0 * fps)), 1)
    decay_tau_frames = max(decay_tau_ms / 1000.0 * fps, 1.0)

    template = np.zeros(n_frames)
    template[:rise_frames] = np.linspace(0, 1, rise_frames)
    t_decay = np.arange(n_frames - rise_frames)
    template[rise_frames:] = np.exp(-t_decay / decay_tau_frames)

    norm = np.linalg.norm(template)
    if norm > 0:
        template /= norm
    return template


def _resolve_template_params(
    indicator: Optional[str],
    template_rise_ms: Optional[float],
    template_decay_ms: Optional[float],
    template_total_ms: Optional[float],
    peak_height: Optional[float],
) -> Dict[str, float]:
    """Combine ``indicator`` preset values with explicit overrides.

    Explicit (non-``None``) ``template_rise_ms`` / ``template_decay_ms`` /
    ``peak_height`` values always win over the preset, so callers can
    pick a preset and tweak a single parameter. ``template_total_ms`` is
    not part of the presets — it falls back to the module default unless
    explicitly overridden.

    Args:
        indicator: Name of a calcium indicator in :data:`INDICATOR_PRESETS`,
            or ``None`` for the legacy GCaMP6f-like defaults.
        template_rise_ms: Override for the template's 0→peak rise time
            in milliseconds.
        template_decay_ms: Override for the template's exponential decay
            time-constant in milliseconds.
        template_total_ms: Override for the template's total length in
            milliseconds.
        peak_height: Override for the absolute ΔF/F threshold passed to
            :func:`scipy.signal.find_peaks`.

    Returns:
        Dict with keys ``rise_ms``, ``decay_ms``, ``total_ms``,
        ``peak_height`` — fully resolved numeric values ready for use.

    Raises:
        ValueError: If ``indicator`` is set to a name not present in
            :data:`INDICATOR_PRESETS`. The message lists every available
            preset so the user can correct the typo without grepping.
    """
    base: Dict[str, float] = {
        "rise_ms": _DEFAULT_TEMPLATE_RISE_MS,
        "decay_ms": _DEFAULT_TEMPLATE_DECAY_MS,
        "total_ms": _DEFAULT_TEMPLATE_TOTAL_MS,
        "peak_height": _DEFAULT_PEAK_HEIGHT,
    }
    if indicator is not None:
        if indicator not in INDICATOR_PRESETS:
            available = ", ".join(sorted(INDICATOR_PRESETS.keys()))
            raise ValueError(
                f"Unknown indicator preset: {indicator!r}. "
                f"Available presets: {available}. "
                f"Pass indicator=None and supply template_rise_ms / "
                f"template_decay_ms / peak_height directly to use a "
                f"custom indicator."
            )
        preset = INDICATOR_PRESETS[indicator]
        base["rise_ms"] = float(preset["rise_ms"])
        base["decay_ms"] = float(preset["decay_ms"])
        base["peak_height"] = float(preset["peak_height"])

    if template_rise_ms is not None:
        base["rise_ms"] = float(template_rise_ms)
    if template_decay_ms is not None:
        base["decay_ms"] = float(template_decay_ms)
    if template_total_ms is not None:
        base["total_ms"] = float(template_total_ms)
    if peak_height is not None:
        base["peak_height"] = float(peak_height)

    return base


def detect_waveform_outliers(
    dff_dat: np.ndarray,
    filtered_idx: np.ndarray,
    fps: float = 30.0,
    threshold: float = 3.5,
    peak_height: Optional[float] = None,
    min_events: int = 3,
    indicator: Optional[str] = None,
    template_rise_ms: Optional[float] = None,
    template_decay_ms: Optional[float] = None,
    template_total_ms: Optional[float] = None,
) -> Dict:
    """Flag neurons whose transient waveforms don't resemble calcium events.

    For every neuron a canonical calcium template (linear rise, single
    exponential decay) is correlated with each detected transient.
    Per-neuron summaries — **median template correlation** and
    **fraction of low-correlation events** — are then scored across the
    population with a modified Z-score on MAD (the same family used by
    the other QC detectors).

    The template's kinetics are indicator-specific. The defaults
    (``rise_ms=50``, ``decay_ms=400``, ``peak_height=0.10``) approximate
    GCaMP6f and were chosen for backwards compatibility. Running the
    detector with a GCaMP6f-shaped template on data acquired with a
    different indicator (slower GCaMP6s, faster jGCaMP8, smaller-ΔF/F
    red indicators, etc.) causes silent miscalibration: real calcium
    events get flagged as shape outliers because they don't match the
    template, and the absolute ``peak_height`` threshold may be
    inappropriate for the indicator's typical excursion magnitude.

    Either pass ``indicator=<name>`` to load a published-kinetics preset
    from :data:`INDICATOR_PRESETS`, or supply ``template_rise_ms`` /
    ``template_decay_ms`` / ``peak_height`` directly. Explicit values
    override preset values, so ``indicator="GCaMP6s",
    template_decay_ms=2000`` is valid and uses 2000 ms decay.

    Presets are starting points only — they reflect typical published
    kinetics under one set of acquisition conditions. Verify against
    your own measurements when accuracy matters; see the
    :data:`INDICATOR_PRESETS` module-level docstring for citations.

    Args:
        dff_dat: Full ΔF/F₀ matrix, shape ``(n_components, n_frames)``.
        filtered_idx: Indices into ``dff_dat`` rows that passed spatial
            filtering.
        fps: Recording frame rate in Hz.
        threshold: Modified Z-score cutoff for flagging (default 3.5).
        peak_height: Minimum ΔF/F₀ peak height passed to
            :func:`scipy.signal.find_peaks`. ``None`` (default) resolves
            to the preset value when ``indicator`` is set, otherwise to
            ``0.1`` (GCaMP-like).
        min_events: Neurons with fewer detected events are excluded from
            scoring (not flagged, not counted).
        indicator: Name of a calcium indicator in
            :data:`INDICATOR_PRESETS`. When set, the preset's
            ``rise_ms`` / ``decay_ms`` / ``peak_height`` are used as the
            base template kinetics. Explicit ``template_rise_ms``,
            ``template_decay_ms``, ``peak_height`` arguments override
            the preset. Unknown names raise ``ValueError`` listing the
            available presets. ``None`` (default) preserves the legacy
            GCaMP6f-like defaults for backwards compatibility.
        template_rise_ms: Override for the template's 0→peak rise time
            in milliseconds. ``None`` (default) uses the
            ``indicator`` preset, falling back to ``50.0``.
        template_decay_ms: Override for the template's exponential
            decay time-constant in milliseconds. ``None`` (default)
            uses the ``indicator`` preset, falling back to ``400.0``.
        template_total_ms: Override for the template's total length in
            milliseconds. Not part of the presets — defaults to
            ``1500.0`` regardless of ``indicator`` unless explicitly
            set. The total length should comfortably exceed several
            decay time-constants so the tail of the template captures
            the event's true shape.

    Returns:
        Dict with the same schema as :func:`detect_low_pnr_neurons`,
        plus waveform-specific entries:
            ``flagged_mask``, ``flagged_idx``, ``clean_idx``,
            ``n_flagged``, ``pct_flagged``, ``neuron_scores``,
            ``summary``, ``template`` (1-D ndarray, unit-normalised),
            ``indicator`` (echo of the input or ``None``),
            ``template_params`` (resolved ``rise_ms`` / ``decay_ms`` /
            ``total_ms`` / ``peak_height``).

    Raises:
        ValueError: When ``indicator`` is not a key of
            :data:`INDICATOR_PRESETS`.
    """
    params = _resolve_template_params(
        indicator=indicator,
        template_rise_ms=template_rise_ms,
        template_decay_ms=template_decay_ms,
        template_total_ms=template_total_ms,
        peak_height=peak_height,
    )
    resolved_peak_height = params["peak_height"]

    traces = dff_dat[filtered_idx, :]
    n_neurons = traces.shape[0]

    if n_neurons < 4:
        result = _empty_waveform_result(filtered_idx, n_neurons)
        result["indicator"] = indicator
        result["template_params"] = params
        return result

    if indicator is not None:
        logger.info(
            "Waveform detector using indicator preset %r "
            "(rise_ms=%.1f, decay_ms=%.1f, peak_height=%.3f).",
            indicator, params["rise_ms"], params["decay_ms"],
            resolved_peak_height,
        )

    template = _make_calcium_template(
        fps,
        rise_ms=params["rise_ms"],
        decay_tau_ms=params["decay_ms"],
        duration_ms=params["total_ms"],
    )
    half_len = len(template) // 2

    median_corrs = np.full(n_neurons, np.nan)
    frac_low = np.full(n_neurons, np.nan)
    n_events = np.zeros(n_neurons, dtype=int)

    for i, trace in enumerate(traces):
        peaks, _ = find_peaks(trace, height=resolved_peak_height, distance=half_len)
        n_events[i] = len(peaks)
        if len(peaks) < min_events:
            continue

        corrs = []
        for pk in peaks:
            start = pk - half_len
            end = start + len(template)
            if start < 0 or end > len(trace):
                continue
            snippet = trace[start:end].copy()
            std = snippet.std()
            if std < 1e-10:
                continue
            snippet = (snippet - snippet.mean()) / std
            t_norm = (template - template.mean()) / template.std()
            r = np.corrcoef(snippet, t_norm)[0, 1]
            corrs.append(r)

        if len(corrs) >= min_events:
            corrs_arr = np.array(corrs)
            median_corrs[i] = np.median(corrs_arr)
            frac_low[i] = np.mean(corrs_arr < 0.3)

    scorable = ~np.isnan(median_corrs)
    flagged = np.zeros(n_neurons, dtype=bool)
    score_cols: Dict[str, np.ndarray] = {}

    for name, values in [("median_template_corr", median_corrs),
                          ("frac_low_corr_events", frac_low)]:
        mz = np.full(n_neurons, np.nan)
        if scorable.sum() >= 4:
            v = values[scorable]
            median = np.median(v)
            mad = np.median(np.abs(v - median))
            if mad == 0:
                mad = 1e-10
            mz[scorable] = 0.6745 * (values[scorable] - median) / mad
            if name == "median_template_corr":
                flagged[scorable] |= mz[scorable] < -threshold
            else:
                flagged[scorable] |= mz[scorable] > threshold
        score_cols[f"{name}_mzscore"] = mz

    scores_df = pd.DataFrame({
        "component_idx": np.asarray(filtered_idx),
        "n_events": n_events,
        "median_template_corr": median_corrs,
        "frac_low_corr_events": frac_low,
        **score_cols,
        "is_waveform_outlier": flagged,
    })

    flagged_idx = np.asarray(filtered_idx)[flagged]
    clean_idx = np.asarray(filtered_idx)[~flagged]
    n_flagged = int(flagged.sum())
    n_scored = int(scorable.sum())
    pct_flagged = (n_flagged / n_scored * 100) if n_scored else 0.0

    summary = (
        f"Waveform template outlier detection (threshold={threshold}): "
        f"flagged {n_flagged}/{n_scored} scored neurons ({pct_flagged:.1f}%). "
        f"{n_neurons - n_scored} neurons had too few events to score."
    )

    return {
        "flagged_mask": flagged,
        "flagged_idx": flagged_idx,
        "clean_idx": clean_idx,
        "n_flagged": n_flagged,
        "pct_flagged": pct_flagged,
        "neuron_scores": scores_df,
        "threshold": threshold,
        "summary": summary,
        "template": template,
        "indicator": indicator,
        "template_params": params,
        "peak_height": resolved_peak_height,
    }


def _empty_waveform_result(filtered_idx, n_neurons):
    return {
        "flagged_mask": np.zeros(n_neurons, dtype=bool),
        "flagged_idx": np.array([], dtype=int),
        "clean_idx": np.asarray(filtered_idx),
        "n_flagged": 0,
        "pct_flagged": 0.0,
        "neuron_scores": pd.DataFrame(),
        "summary": "Too few neurons for waveform outlier detection.",
        "template": np.array([]),
    }


def detect_spectral_outliers(
    dff_dat: np.ndarray,
    filtered_idx: np.ndarray,
    fps: float = 30.0,
    threshold: float = 3.5,
    bio_band: tuple = (0.1, 2.0),
    high_band_floor: float = 5.0,
    drift_band_ceil: float = 0.05,
) -> Dict:
    """Flag neurons with abnormal frequency-domain profiles.

    For each neuron, the power spectral density is computed via FFT and
    partitioned into three bands:

    * **biological** (default 0.1–2 Hz): where GCaMP transients live.
    * **high-frequency** (>5 Hz): too fast for calcium indicators.
    * **drift** (<0.05 Hz): photobleaching / slow motion artefacts.

    The fraction of total power in each band is computed per neuron,
    then scored across the population with modified Z-scores.

    Parameters
    ----------
    dff_dat : np.ndarray
        Full ΔF/F₀ matrix, shape ``(n_components, n_frames)``.
    filtered_idx : np.ndarray
        Indices into ``dff_dat`` rows that passed spatial filtering.
    fps : float
        Recording frame rate in Hz.
    threshold : float
        Modified Z-score cutoff (default 3.5).
    bio_band : tuple of float
        ``(low_hz, high_hz)`` defining the expected biological signal band.
    high_band_floor : float
        Lower edge of the "too-fast" frequency band in Hz.
    drift_band_ceil : float
        Upper edge of the "drift" band in Hz.

    Returns
    -------
    dict
        Same schema as ``detect_low_pnr_neurons``:
        ``flagged_mask``, ``flagged_idx``, ``clean_idx``,
        ``n_flagged``, ``pct_flagged``, ``neuron_scores``,
        ``summary``.
    """
    traces = dff_dat[filtered_idx, :]
    n_neurons, n_frames = traces.shape

    if n_neurons < 4 or n_frames < 8:
        return {
            "flagged_mask": np.zeros(n_neurons, dtype=bool),
            "flagged_idx": np.array([], dtype=int),
            "clean_idx": np.asarray(filtered_idx),
            "n_flagged": 0,
            "pct_flagged": 0.0,
            "neuron_scores": pd.DataFrame(),
            "summary": "Too few neurons or frames for spectral outlier detection.",
        }

    freqs = np.fft.rfftfreq(n_frames, d=1.0 / fps)
    bio_mask = (freqs >= bio_band[0]) & (freqs <= bio_band[1])
    high_mask = freqs > high_band_floor
    drift_mask = (freqs > 0) & (freqs < drift_band_ceil)

    frac_bio = np.zeros(n_neurons)
    frac_high = np.zeros(n_neurons)
    frac_drift = np.zeros(n_neurons)

    for i, trace in enumerate(traces):
        power = np.abs(np.fft.rfft(trace)) ** 2
        total = power[1:].sum()
        if total < 1e-20:
            continue
        frac_bio[i] = power[bio_mask].sum() / total
        frac_high[i] = power[high_mask].sum() / total
        frac_drift[i] = power[drift_mask].sum() / total

    flagged = np.zeros(n_neurons, dtype=bool)
    score_cols: Dict[str, np.ndarray] = {}

    for name, values, flag_direction in [
        ("frac_biological_power", frac_bio, "low"),
        ("frac_highfreq_power", frac_high, "high"),
        ("frac_drift_power", frac_drift, "high"),
    ]:
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:
            mad = 1e-10
        mz = 0.6745 * (values - median) / mad
        score_cols[f"{name}_mzscore"] = mz
        if flag_direction == "low":
            flagged |= mz < -threshold
        else:
            flagged |= mz > threshold

    scores_df = pd.DataFrame({
        "component_idx": np.asarray(filtered_idx),
        "frac_biological_power": frac_bio,
        "frac_highfreq_power": frac_high,
        "frac_drift_power": frac_drift,
        **score_cols,
        "is_spectral_outlier": flagged,
    })

    flagged_idx = np.asarray(filtered_idx)[flagged]
    clean_idx = np.asarray(filtered_idx)[~flagged]
    n_flagged = int(flagged.sum())
    pct_flagged = (n_flagged / n_neurons * 100) if n_neurons else 0.0

    summary = (
        f"Spectral outlier detection (threshold={threshold}): "
        f"flagged {n_flagged}/{n_neurons} neurons ({pct_flagged:.1f}%). "
        f"Bio band={bio_band} Hz, high>{high_band_floor} Hz, drift<{drift_band_ceil} Hz."
    )

    return {
        "flagged_mask": flagged,
        "flagged_idx": flagged_idx,
        "clean_idx": clean_idx,
        "n_flagged": n_flagged,
        "pct_flagged": pct_flagged,
        "neuron_scores": scores_df,
        "threshold": threshold,
        "summary": summary,
        "bio_band": bio_band,
        "high_band_floor": high_band_floor,
        "drift_band_ceil": drift_band_ceil,
    }


def combine_neuron_qc(
    filtered_idx: np.ndarray,
    low_pnr_result: Optional[Dict] = None,
    waveform_result: Optional[Dict] = None,
    spectral_result: Optional[Dict] = None,
    min_flags: int = 1,
    amplitude_result: Optional[Dict] = None,
) -> Dict:
    """Merge results from multiple outlier detectors into a unified QC report.

    Args:
        filtered_idx: Component indices (shared across all detectors).
        low_pnr_result: Return value of :func:`detect_low_pnr_neurons`.
        waveform_result: Return value of :func:`detect_waveform_outliers`.
        spectral_result: Return value of :func:`detect_spectral_outliers`.
        min_flags: A neuron is marked ``any_outlier=True`` when it is
            flagged by at least this many detectors (default 1 = union).
        amplitude_result: Deprecated alias for ``low_pnr_result``. Accepted
            so older callers (pre-PNR rewrite) keep working; emits a
            ``DeprecationWarning`` and is forwarded as ``low_pnr_result``.
            Ignored when both arguments are supplied.

    Returns:
        Dict with:
            ``combined_df``: DataFrame with one row per neuron and columns
                from every supplied detector plus ``n_flags`` and
                ``any_outlier``.
            ``summary``: Human-readable summary string.
    """
    if amplitude_result is not None:
        warnings.warn(
            "combine_neuron_qc(amplitude_result=...) is deprecated; "
            "pass low_pnr_result=... instead. The amplitude detector was "
            "replaced by detect_low_pnr_neurons.",
            DeprecationWarning,
            stacklevel=2,
        )
        if low_pnr_result is None:
            low_pnr_result = amplitude_result

    n_neurons = len(filtered_idx)
    combined = pd.DataFrame({"component_idx": np.asarray(filtered_idx)})
    flag_arrays: List[np.ndarray] = []

    if low_pnr_result and not low_pnr_result["neuron_scores"].empty:
        amp = low_pnr_result["neuron_scores"]
        # New PNR-based schema (preferred). The legacy max/mean/std columns
        # are kept in the loop so an older external caller that hand-builds
        # an "amplitude_result" dict in the old shape still merges cleanly.
        for col in [
            "pnr", "sigma_hf", "log_pnr_modified_zscore",
            "is_low_pnr", "reason",
            "max_dff", "mean_dff", "std_dff",
            "max_dff_mzscore", "mean_dff_mzscore", "std_dff_mzscore",
            "is_outlier",
        ]:
            if col in amp.columns:
                combined[col] = amp[col].values
        flag_arrays.append(low_pnr_result["flagged_mask"].astype(int))

    if waveform_result and not waveform_result["neuron_scores"].empty:
        wf = waveform_result["neuron_scores"]
        for col in ["median_template_corr", "frac_low_corr_events",
                     "median_template_corr_mzscore",
                     "frac_low_corr_events_mzscore",
                     "is_waveform_outlier"]:
            if col in wf.columns:
                combined[col] = wf[col].values
        flag_arrays.append(waveform_result["flagged_mask"].astype(int))

    if spectral_result and not spectral_result["neuron_scores"].empty:
        sp = spectral_result["neuron_scores"]
        for col in ["frac_biological_power", "frac_highfreq_power",
                     "frac_drift_power",
                     "frac_biological_power_mzscore",
                     "frac_highfreq_power_mzscore",
                     "frac_drift_power_mzscore",
                     "is_spectral_outlier"]:
            if col in sp.columns:
                combined[col] = sp[col].values
        flag_arrays.append(spectral_result["flagged_mask"].astype(int))

    if flag_arrays:
        n_flags = np.sum(flag_arrays, axis=0)
    else:
        n_flags = np.zeros(n_neurons, dtype=int)

    combined["n_flags"] = n_flags
    combined["any_outlier"] = n_flags >= min_flags

    n_any = int((n_flags >= min_flags).sum())
    pct = (n_any / n_neurons * 100) if n_neurons else 0.0
    methods_used = sum([
        low_pnr_result is not None,
        waveform_result is not None,
        spectral_result is not None,
    ])
    summary = (
        f"Combined QC ({methods_used} detectors, min_flags={min_flags}): "
        f"{n_any}/{n_neurons} neurons flagged ({pct:.1f}%)."
    )

    return {"combined_df": combined, "summary": summary}


def detect_outliers(
    data: pd.DataFrame,
    metric_col: str,
    method: str = "iqr",
    threshold: Optional[float] = None,
    group_col: Optional[str] = None
) -> Dict:
    """
    Detect potential outliers in the data.
    
    IMPORTANT: Outliers in biological data may represent real phenomena!
    Always investigate outliers before deciding how to handle them.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to analyze.
    metric_col : str
        Column name containing the metric to check for outliers.
    method : str
        Detection method to use:
        
        - "iqr": Interquartile range method (1.5*IQR rule).
          Best for: Most calcium imaging data. Robust to non-normality.
          
        - "zscore": Z-score > threshold.
          Best for: Normally distributed data.
          
        - "mad": Median Absolute Deviation.
          Best for: Data with non-normal distributions or heavy tails.
          
        - "isolation_forest": ML-based detection.
          Best for: Multivariate outliers (not yet implemented).
          
    threshold : float, optional
        Method-specific threshold:
        - For "iqr": Multiplier for IQR (default 1.5)
        - For "zscore": Z-score cutoff (default 3.0)
        - For "mad": Number of MADs from median (default 3.5)
        
    group_col : str, optional
        Column for group-wise outlier detection. If provided, outliers
        are detected within each group separately.
    
    Returns
    -------
    dict
        Dictionary containing:
        - outlier_mask: Boolean mask of outliers (True = outlier)
        - outlier_indices: DataFrame indices of outlier rows
        - outlier_values: The outlier values
        - n_outliers: Number of outliers detected
        - pct_outliers: Percentage of data that are outliers
        - bounds: Lower and upper bounds used (for IQR/MAD)
        - summary: Description of outliers found
        - recommendation: Whether to remove or investigate
    
    Examples
    --------
    >>> # Detect outliers in firing rate data
    >>> result = detect_outliers(
    ...     data=frpm_df,
    ...     metric_col="mean_Firing Rate Per Min",
    ...     method="iqr"
    ... )
    >>> print(result["summary"])
    >>> print(f"Found {result['n_outliers']} outliers")
    
    >>> # Detect outliers within each treatment group
    >>> result = detect_outliers(
    ...     data=frpm_df,
    ...     metric_col="mean_Firing Rate Per Min",
    ...     group_col="Treatment"
    ... )
    """
    df = data.copy()
    values = df[metric_col].values
    
    # Set default thresholds
    default_thresholds = {
        "iqr": 1.5,
        "zscore": 3.0,
        "mad": 3.5,
        "isolation_forest": 0.1,
    }
    if threshold is None:
        threshold = default_thresholds.get(method, 1.5)
    
    # Detect outliers based on method
    if group_col is not None:
        # Detect within groups
        outlier_mask = np.zeros(len(df), dtype=bool)
        bounds = {}
        
        for group_name in df[group_col].unique():
            group_mask = df[group_col] == group_name
            group_values = values[group_mask]
            group_result = _detect_outliers_single(
                group_values, method, threshold
            )
            # Map back to original indices
            outlier_mask[group_mask] = group_result["outlier_mask"]
            bounds[group_name] = group_result.get("bounds")
    else:
        result = _detect_outliers_single(values, method, threshold)
        outlier_mask = result["outlier_mask"]
        bounds = result.get("bounds")
    
    # Extract outlier information
    outlier_indices = df.index[outlier_mask].tolist()
    outlier_values = values[outlier_mask]
    n_outliers = int(outlier_mask.sum())
    pct_outliers = (n_outliers / len(df)) * 100
    
    # Build summary
    if n_outliers == 0:
        summary = f"No outliers detected using {method} method (threshold: {threshold})."
        recommendation = "No action needed."
    elif n_outliers < 5:
        summary = (
            f"Detected {n_outliers} outlier(s) ({pct_outliers:.1f}% of data) "
            f"using {method} method. Values: {outlier_values.tolist()}"
        )
        recommendation = (
            "Review these cases individually to determine if they represent "
            "real biological variation or technical artifacts."
        )
    else:
        summary = (
            f"Detected {n_outliers} outliers ({pct_outliers:.1f}% of data) "
            f"using {method} method. Range: {outlier_values.min():.2f} to {outlier_values.max():.2f}"
        )
        if pct_outliers > 10:
            recommendation = (
                "High outlier rate suggests possible data quality issues or "
                "non-normal distribution. Consider using robust statistics."
            )
        else:
            recommendation = (
                "Consider flagging outliers for sensitivity analysis rather "
                "than removing them outright."
            )
    
    return {
        "outlier_mask": outlier_mask,
        "outlier_indices": outlier_indices,
        "outlier_values": outlier_values,
        "n_outliers": n_outliers,
        "pct_outliers": pct_outliers,
        "bounds": bounds,
        "method": method,
        "threshold": threshold,
        "summary": summary,
        "recommendation": recommendation,
    }


def _detect_outliers_single(
    values: np.ndarray,
    method: str,
    threshold: float
) -> Dict:
    """Detect outliers for a single array of values."""
    values = np.asarray(values)
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    
    if len(valid_values) < 4:
        return {
            "outlier_mask": np.zeros(len(values), dtype=bool),
            "bounds": None,
        }
    
    if method == "iqr":
        q1 = np.percentile(valid_values, 25)
        q3 = np.percentile(valid_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        bounds = {"lower": lower_bound, "upper": upper_bound, "q1": q1, "q3": q3}
        
    elif method == "zscore":
        mean = np.mean(valid_values)
        std = np.std(valid_values, ddof=1)
        if std == 0:
            outlier_mask = np.zeros(len(values), dtype=bool)
            bounds = None
        else:
            z_scores = (values - mean) / std
            outlier_mask = np.abs(z_scores) > threshold
            bounds = {
                "lower": mean - threshold * std,
                "upper": mean + threshold * std,
                "mean": mean,
                "std": std,
            }
    
    elif method == "mad":
        median = np.median(valid_values)
        mad = np.median(np.abs(valid_values - median))
        if mad == 0:
            # Use a small value to avoid division by zero
            mad = 1e-10
        modified_z = 0.6745 * (values - median) / mad
        outlier_mask = np.abs(modified_z) > threshold
        bounds = {
            "lower": median - threshold * mad / 0.6745,
            "upper": median + threshold * mad / 0.6745,
            "median": median,
            "mad": mad,
        }
    
    elif method == "isolation_forest":
        try:
            from sklearn.ensemble import IsolationForest
            
            # Reshape for sklearn
            X = valid_values.reshape(-1, 1)
            clf = IsolationForest(contamination=threshold, random_state=42)
            predictions = clf.fit_predict(X)
            
            # Map back to original array
            outlier_mask = np.zeros(len(values), dtype=bool)
            outlier_mask[valid_mask] = predictions == -1
            bounds = None
            
        except ImportError:
            raise ImportError(
                "Isolation Forest requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )
    
    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Valid options: 'iqr', 'zscore', 'mad', 'isolation_forest'"
        )
    
    return {"outlier_mask": outlier_mask, "bounds": bounds}


def visualize_outliers(
    data: pd.DataFrame,
    metric_col: str,
    outlier_result: Dict,
    group_col: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Create visualization showing data distribution and outliers.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    metric_col : str
        Column name containing the metric.
    outlier_result : dict
        Result dictionary from detect_outliers().
    group_col : str, optional
        Column for grouping the visualization.
    figsize : tuple
        Figure size (width, height) in inches.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    
    Examples
    --------
    >>> outliers = detect_outliers(df, "firing_rate")
    >>> fig = visualize_outliers(df, "firing_rate", outliers)
    >>> plt.show()
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    values = data[metric_col].values
    outlier_mask = outlier_result["outlier_mask"]
    
    # Left plot: Box plot with outliers highlighted
    ax1 = axes[0]
    if group_col is not None:
        groups = data[group_col].unique()
        positions = []
        labels = []
        for i, group in enumerate(groups):
            group_mask = data[group_col] == group
            group_values = values[group_mask]
            positions.append(i + 1)
            labels.append(str(group))
            
            # Box plot
            bp = ax1.boxplot([group_values], positions=[i + 1], widths=0.6)
            
            # Highlight outliers
            group_outliers = group_values[outlier_mask[group_mask]]
            ax1.scatter(
                [i + 1] * len(group_outliers), group_outliers,
                c='red', marker='x', s=100, zorder=5, label='Outlier' if i == 0 else None
            )
        
        ax1.set_xticks(positions)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
    else:
        ax1.boxplot([values], widths=0.6)
        outlier_values = values[outlier_mask]
        ax1.scatter(
            [1] * len(outlier_values), outlier_values,
            c='red', marker='x', s=100, zorder=5, label='Outlier'
        )
    
    ax1.set_ylabel(metric_col)
    ax1.set_title("Box Plot with Outliers")
    if outlier_result["n_outliers"] > 0:
        ax1.legend()
    
    # Right plot: Histogram with bounds
    ax2 = axes[1]
    ax2.hist(values[~np.isnan(values)], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(np.median(values[~np.isnan(values)]), color='green', linestyle='--', 
                label='Median', linewidth=2)
    
    # Show bounds if available
    bounds = outlier_result.get("bounds")
    if bounds is not None and isinstance(bounds, dict):
        if "lower" in bounds:
            ax2.axvline(bounds["lower"], color='red', linestyle=':', 
                       label=f'Lower bound ({bounds["lower"]:.2f})', linewidth=2)
        if "upper" in bounds:
            ax2.axvline(bounds["upper"], color='red', linestyle=':', 
                       label=f'Upper bound ({bounds["upper"]:.2f})', linewidth=2)
    
    ax2.set_xlabel(metric_col)
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution with Outlier Bounds")
    ax2.legend(fontsize=8)
    
    # Add summary text
    method = outlier_result.get("method", "unknown")
    n_outliers = outlier_result.get("n_outliers", 0)
    pct = outlier_result.get("pct_outliers", 0)
    fig.suptitle(
        f"Outlier Detection: {n_outliers} outliers ({pct:.1f}%) using {method} method",
        fontsize=12, fontweight='bold'
    )
    
    plt.tight_layout()
    return fig


def handle_outliers(
    data: pd.DataFrame,
    outlier_result: Dict,
    method: str = "flag"
) -> pd.DataFrame:
    """
    Handle outliers according to specified method.
    
    ALWAYS returns a copy of the data - never modifies the original.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    outlier_result : dict
        Result dictionary from detect_outliers().
    method : str
        How to handle outliers:
        
        - "flag": Add a column marking outliers (recommended for transparency)
        - "remove": Remove outlier rows
        - "winsorize": Replace outliers with boundary values
        - "impute_median": Replace outliers with group median
        - "impute_mean": Replace outliers with group mean
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame (always a copy, original is unchanged).
    
    Notes
    -----
    Recommendations by use case:
    
    - **Exploratory analysis**: Use "flag" to see impact of outliers
    - **Robust analysis**: Use "winsorize" to reduce outlier influence
    - **Final analysis**: Consider running with both "flag" and "remove"
      to assess sensitivity
    
    Examples
    --------
    >>> outliers = detect_outliers(df, "firing_rate")
    >>> 
    >>> # Option 1: Flag outliers (recommended)
    >>> df_flagged = handle_outliers(df, outliers, method="flag")
    >>> 
    >>> # Option 2: Remove outliers
    >>> df_clean = handle_outliers(df, outliers, method="remove")
    >>> 
    >>> # Option 3: Winsorize (cap at boundaries)
    >>> df_winsorized = handle_outliers(df, outliers, method="winsorize")
    """
    df = data.copy()
    outlier_mask = outlier_result["outlier_mask"]
    
    if method == "flag":
        df["is_outlier"] = outlier_mask
        df["outlier_method"] = outlier_result.get("method", "unknown")
        
    elif method == "remove":
        df = df[~outlier_mask].reset_index(drop=True)
        
    elif method == "winsorize":
        bounds = outlier_result.get("bounds")
        if bounds is None:
            raise ValueError("Winsorization requires bounds from outlier detection")
        
        # Get the metric column (assume it's stored or infer from bounds)
        # Find numeric columns that might have been analyzed
        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col].values
            if isinstance(bounds, dict) and "lower" in bounds and "upper" in bounds:
                df.loc[values < bounds["lower"], col] = bounds["lower"]
                df.loc[values > bounds["upper"], col] = bounds["upper"]
                break
    
    elif method in ["impute_median", "impute_mean"]:
        agg_func = "median" if method == "impute_median" else "mean"
        
        # Find the metric column
        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col].values
            non_outlier_values = values[~outlier_mask]
            
            if len(non_outlier_values) > 0:
                if agg_func == "median":
                    replacement = np.median(non_outlier_values)
                else:
                    replacement = np.mean(non_outlier_values)
                
                df.loc[outlier_mask, col] = replacement
            break
    
    else:
        valid_methods = ["flag", "remove", "winsorize", "impute_median", "impute_mean"]
        raise ValueError(f"Unknown method: '{method}'. Valid options: {valid_methods}")
    
    return df

