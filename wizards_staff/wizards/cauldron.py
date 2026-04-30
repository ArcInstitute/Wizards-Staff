# import
## batteries
import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union
from functools import partial
from multiprocessing import get_context
# joblib is imported locally in run_all when threads > 1
## 3rd party
import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm.notebook import tqdm
## package
from wizards_staff.wizards.spellbook import (
    calc_rise_tm, calc_fwhm_spikes, calc_frpm, calc_fall_tm, 
    calc_peak_amplitude, calc_peak_to_peak, calc_mask_shape_metrics, convert_f_to_cs
)
from wizards_staff.plotting import (
    plot_kmeans_heatmap, plot_cluster_activity, plot_spatial_activity_map,
    plot_dff_activity, plot_neuron_outliers, plot_waveform_qc, plot_spectral_qc,
    plot_sample_mean_dff_with_events, plot_neuron_dff_traces_with_events,
)
from wizards_staff.pwc import run_pwc
from wizards_staff.metadata import append_metadata_to_dfs
from wizards_staff.wizards.familiars import spatial_filtering
from wizards_staff.wizards.shard import Shard
from wizards_staff.stats.outliers import (
    detect_low_pnr_neurons, detect_waveform_outliers,
    detect_spectral_outliers, combine_neuron_qc,
)
from wizards_staff.reporting import generate_run_report

# Suppress RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Module-level constants
_LABEL_DISAGREEMENT_MODES: Tuple[str, ...] = ("drop", "keep", "majority")


def _resolve_label_drops(
    sample_id: str,
    labels_corpus: Optional[Union[str, Path]],
    on_disagreement: str,
    logger: logging.Logger,
) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], str], int, int, int]:
    """
    Resolve a labels corpus into the per-shard set of events to drop.

    The corpus schema is the one written by
    :class:`wizards_staff.labeling.event_labeler.EventLabeler` — at minimum
    it must carry ``sample_id``, ``roi_id``, ``event_idx``, ``label``,
    and ``labeler_id`` columns. Rows for *other* samples are ignored.
    Rows whose ``label`` is anything other than ``"True"`` or ``"False"``
    (notably ``"Unsure"`` or empty) are treated as not-labeled and never
    cause a drop.

    For each (roi_id, event_idx) the per-labeler votes are tallied:

      * all True / no False  -> keep
      * all False / no True  -> drop
      * mixed True+False     -> inter-rater disagreement, resolved by
                                 ``on_disagreement``:
            ``"drop"``     -> drop (precautionary, default)
            ``"keep"``     -> keep
            ``"majority"`` -> majority of {True, False} votes; ties drop.

    The function never raises on a missing or malformed corpus — it logs a
    warning and returns the empty drop-set so the caller can proceed
    without label-based filtering.

    Args:
        sample_id: The sample name to filter the corpus to.
        labels_corpus: Path to the corpus CSV, or ``None`` to skip.
        on_disagreement: One of ``"drop"`` / ``"keep"`` / ``"majority"``.
        logger: Logger used to surface warnings about a missing /
            malformed corpus and the disagreement count.

    Returns:
        Tuple ``(drops, drop_reasons, n_disagreements, n_events_seen,
        n_unsure_only)``. ``drops`` is the set of
        ``(roi_id, event_idx)`` keys to drop. ``drop_reasons`` maps
        each dropped key to either ``"human_label_false"`` (consensus
        False vote with no overriding True) or
        ``"human_label_disagreement_drop"`` (mixed True+False votes
        that ``on_disagreement`` resolved to drop). ``n_disagreements``
        is the number of events with mixed True+False votes that the
        resolution policy had to break. ``n_events_seen`` is the number
        of distinct events in the corpus for this shard (any label,
        including Unsure / empty). ``n_unsure_only`` is the number of
        events whose only votes were Unsure / empty (treated as
        not-labeled).
    """
    drops: Set[Tuple[int, int]] = set()
    drop_reasons: Dict[Tuple[int, int], str] = {}
    if labels_corpus is None:
        return drops, drop_reasons, 0, 0, 0
    if on_disagreement not in _LABEL_DISAGREEMENT_MODES:
        raise ValueError(
            f"on_disagreement must be one of {_LABEL_DISAGREEMENT_MODES}; "
            f"got {on_disagreement!r}"
        )

    path = Path(labels_corpus)
    if not path.exists():
        logger.warning(
            f"{sample_id}: labels_corpus {str(path)!r} does not exist; "
            f"proceeding without label-based filtering."
        )
        return drops, drop_reasons, 0, 0, 0

    try:
        df = pd.read_csv(
            path,
            dtype={
                "label": str,
                "sample_id": str,
                "labeler_id": str,
            },
        )
    except pd.errors.EmptyDataError:
        return drops, drop_reasons, 0, 0, 0
    except Exception as exc:
        logger.warning(
            f"{sample_id}: failed to load labels_corpus {str(path)!r}: "
            f"{exc}; proceeding without label-based filtering."
        )
        return drops, drop_reasons, 0, 0, 0

    if df.empty:
        return drops, drop_reasons, 0, 0, 0
    required = {"sample_id", "roi_id", "event_idx", "label"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(
            f"{sample_id}: labels_corpus {str(path)!r} is missing required "
            f"columns {sorted(missing)}; proceeding without label-based "
            f"filtering."
        )
        return drops, drop_reasons, 0, 0, 0

    sample_df = df[df["sample_id"].astype(str) == str(sample_id)]
    if sample_df.empty:
        return drops, drop_reasons, 0, 0, 0

    n_disagreements = 0
    n_events_seen = 0
    n_unsure_only = 0
    for (roi_id_raw, event_idx_raw), group in sample_df.groupby(
        ["roi_id", "event_idx"], sort=False
    ):
        try:
            roi_id = int(roi_id_raw)
            event_idx = int(event_idx_raw)
        except (TypeError, ValueError):
            continue
        n_events_seen += 1

        labels = [str(l) for l in group["label"].tolist()]
        true_votes = sum(1 for l in labels if l == "True")
        false_votes = sum(1 for l in labels if l == "False")

        if true_votes == 0 and false_votes == 0:
            n_unsure_only += 1
            continue
        is_disagreement = false_votes > 0 and true_votes > 0
        if true_votes == 0:
            decision = "drop"
        elif false_votes == 0:
            decision = "keep"
        else:
            n_disagreements += 1
            if on_disagreement == "drop":
                decision = "drop"
            elif on_disagreement == "keep":
                decision = "keep"
            else:  # majority
                if false_votes > true_votes:
                    decision = "drop"
                elif true_votes > false_votes:
                    decision = "keep"
                else:
                    decision = "drop"

        if decision == "drop":
            drops.add((roi_id, event_idx))
            # Distinguish consensus-False drops from disagreement-driven
            # drops so the ledger can surface why each event was rejected.
            drop_reasons[(roi_id, event_idx)] = (
                "human_label_disagreement_drop"
                if is_disagreement
                else "human_label_false"
            )

    return drops, drop_reasons, n_disagreements, n_events_seen, n_unsure_only


# functions
def _apply_event_filters(
    shard: Shard,
    *,
    filter_events: bool,
    min_event_amplitude: Optional[float],
    max_event_amplitude: Optional[float],
    min_event_fwhm: Optional[int],
    max_event_fwhm: Optional[int],
    labels_corpus: Optional[Union[str, Path]] = None,
    on_disagreement: str = "drop",
    log_summary: bool = True,
) -> dict:
    """
    Derive every filtered per-event metric on ``shard`` from the raw event
    lists, using a single shared per-(sample, neuron) keep-mask so that
    every metric describes the same surviving event set.

    Reads from (raw, never mutated):
        ``_raw_peak_amplitude_data``, ``_raw_fwhm_data``,
        ``_raw_rise_time_data``, ``_raw_fall_time_data``,
        ``_raw_peak_to_peak_data``, ``_raw_frpm_data``.

    Writes to (replaced wholesale, idempotent):
        ``_peak_amplitude_data``, ``_max_peak_amplitude_data``,
        ``_fwhm_data``, ``_rise_time_data``, ``_fall_time_data``,
        ``_peak_to_peak_data``, ``_frpm_data``.

    The keep-mask is the *intersection* of three layers, applied in order:

      1. **NaN/Inf scrub (always on)** — deconvolution artefacts that
         can't legitimately participate in aggregations are rejected
         regardless of ``filter_events``.
      2. **Amplitude / FWHM bounds (``filter_events=True``)** — events
         outside the user-supplied numeric bounds are rejected.
      3. **Human labels (``labels_corpus`` provided)** — events labeled
         ``"False"`` in the labels corpus (CSV produced by
         :class:`wizards_staff.labeling.event_labeler.EventLabeler`) are
         rejected. Labels are an *additional* rejection layer only:
         a label of ``"True"`` cannot recover an event that layers 1 or
         2 already dropped. Inter-rater disagreement is resolved by
         ``on_disagreement`` (``"drop"`` / ``"keep"`` / ``"majority"``).

    Because every walker in :mod:`wizards_staff.wizards.spellbook`
    shares the same canonical crossing walk (see ``calc_rise_tm``
    docstring), the i-th raw event for each neuron corresponds across
    rise / fall / peak-amp / peak-to-peak / FWHM, so a single positional
    mask can be applied to all of them.

    Inter-spike intervals (``peak_to_peak_data``) are recomputed from the
    *surviving* peak positions, not by masking the raw intervals — if
    event 5 is dropped the interval between event 4 and event 6 becomes
    a single gap, not two.

    Firing rate per minute (``frpm_data``) is recomputed from the
    surviving event count and the per-shard recording length:
    ``FRPM = n_kept_events * 60 * frate / n_frames``. This redefines FRPM
    as *events per minute* (one per detected calcium transient); earlier
    versions returned *frames-above-threshold per minute*, which double-
    counted multi-frame above-threshold runs and was inconsistent with
    every other per-event metric.

    Returns a summary dict with drop counts that is also surfaced via
    ``shard._logger.info`` when ``log_summary=True`` and the filter is
    active.

    Per-event drop ledger
    ---------------------
    Alongside the aggregate counts, this function populates
    ``shard._event_drop_log`` with one dict per rejected event:

      ``sample_id``, ``neuron_idx``, ``event_idx``, ``peak_amplitude``
      (may be NaN/Inf), ``fwhm_frames`` (may be NaN/Inf),
      ``drop_reason``.

    ``drop_reason`` is one of ``"nan_inf"``, ``"amplitude_below_min"``,
    ``"amplitude_above_max"``, ``"fwhm_below_min"``,
    ``"fwhm_above_max"``, ``"human_label_false"``, or
    ``"human_label_disagreement_drop"``. When an event would be dropped
    for multiple reasons the FIRST reason in the order
    ``nan_inf > amplitude_below_min > amplitude_above_max >
    fwhm_below_min > fwhm_above_max > human_label_*`` is recorded —
    so an event whose amplitude is both NaN and below the lower bound
    is attributed to ``"nan_inf"``, and an event labeled ``"False"``
    that *also* fails an amplitude bound is attributed to the bound
    (the label layer never gets to "claim" an event it didn't
    independently cause to drop). The ledger is replaced wholesale on
    every call so it always reflects the current filter configuration,
    never accumulating history across ``Orb.refilter_events`` calls.
    """
    apply_amp_filter = filter_events and (
        min_event_amplitude is not None or max_event_amplitude is not None
    )
    apply_fwhm_filter = filter_events and (
        min_event_fwhm is not None or max_event_fwhm is not None
    )

    # ── Resolve the human-label drop set (layer 3) ────────────────────
    # Loaded once per shard up front so the per-ROI loop below can
    # apply it positionally without re-reading the corpus. ``drops`` is
    # a set of ``(roi_id, event_idx)`` keys; events outside the set
    # are unaffected by this layer (labeled True, Unsure, or unlabeled
    # all map to keep=True for the label layer specifically).
    (
        label_drops,
        label_drop_reasons,
        label_disagreements,
        label_events_seen,
        label_unsure_only,
    ) = _resolve_label_drops(
        sample_id=shard.sample_name,
        labels_corpus=labels_corpus,
        on_disagreement=on_disagreement,
        logger=shard._logger,
    )
    apply_label_filter = labels_corpus is not None
    dropped_by_labels = 0

    # ── Per-event drop ledger (audit trail) ──────────────────────────
    # Replaced wholesale on every call so the ledger always reflects the
    # CURRENT filter configuration (matching the contract of
    # ``Orb.refilter_events``: re-derived, never appended). Each row
    # records (sample, neuron, event_idx, peak_amplitude, fwhm_frames,
    # drop_reason) for one rejected event. The drop_reason ordering is
    # ``nan_inf > amplitude_below_min > amplitude_above_max >
    # fwhm_below_min > fwhm_above_max > human_label_*`` — first reason
    # checked wins, so an event dropped for multiple reasons is
    # attributed to the earliest-checked one (see the function
    # docstring).
    shard._event_drop_log = []

    # ── Build per-(sample, neuron) keep-masks ─────────────────────────
    # Index masks by (Sample, Neuron) so we can apply them uniformly to
    # every raw list regardless of insertion order. Each mask is a numpy
    # boolean array of length equal to the raw event count for that
    # neuron. Lengths come from the raw amplitude list which mirrors the
    # canonical walk; the FWHM list is asserted to have the same length
    # per neuron (and a defensive log fires if not).
    masks: dict = {}
    fwhm_lookup: dict = {}
    for raw_row in shard._raw_fwhm_data:
        fwhm_lookup[(raw_row['Sample'], raw_row['Neuron'])] = raw_row

    dropped_by_amplitude_nan = 0
    dropped_by_amplitude_bounds = 0
    dropped_by_fwhm_nan = 0
    dropped_by_fwhm_bounds = 0
    total_events_before = 0
    propagation_mismatches = 0

    for raw_row in shard._raw_peak_amplitude_data:
        key = (raw_row['Sample'], raw_row['Neuron'])
        amps = np.asarray(raw_row.get('Peak Amplitudes', []), dtype=float)
        n_events = amps.size
        total_events_before += n_events

        amp_finite = np.isfinite(amps)
        amp_keep = amp_finite.copy()
        if apply_amp_filter:
            if min_event_amplitude is not None:
                amp_keep &= amps >= min_event_amplitude
            if max_event_amplitude is not None:
                amp_keep &= amps <= max_event_amplitude
        dropped_by_amplitude_nan += int((~amp_finite).sum())
        # Subtract NaN drops to avoid double-counting in the bounds bucket.
        dropped_by_amplitude_bounds += int(((~amp_keep) & amp_finite).sum())

        # ── Layer 3: human labels ────────────────────────────────────
        # Build the per-ROI label-keep mask positionally from the
        # corpus drop-set. Labels can only DROP events, never recover
        # them — see the docstring's "additional rejection layer only"
        # contract. The combined mask is composed below once the FWHM
        # mask for this ROI is known.
        _label_keep_for_roi = np.ones(n_events, dtype=bool)
        _label_keep_active = False
        if apply_label_filter and n_events > 0:
            try:
                roi_id_int = int(raw_row['Neuron'])
            except (TypeError, ValueError):
                roi_id_int = None
            if roi_id_int is not None:
                _label_keep_active = True
                for ev_idx in range(n_events):
                    if (roi_id_int, ev_idx) in label_drops:
                        _label_keep_for_roi[ev_idx] = False

        # FWHM values for this neuron. Pipeline invariant: ``_run_all``
        # populates ``_raw_fwhm_data`` alongside ``_raw_peak_amplitude_data``
        # using the shared canonical crossing walk in
        # :mod:`wizards_staff.wizards.spellbook`, so a row must exist
        # for every (sample, neuron) and its length must equal the
        # amplitude row's length. Violating either condition is a bug
        # in the calling pipeline, not a degraded data condition we can
        # paper over — silently defaulting to "keep everything" would
        # let NaN/Inf FWHM events leak into downstream metrics. Fail
        # loudly so the bug surfaces instead.
        fwhm_row = fwhm_lookup.get(key)
        if fwhm_row is None:
            raise RuntimeError(
                f"FWHM data inconsistent for neuron {raw_row['Neuron']} "
                f"in sample {shard.sample_name}: row missing from "
                f"_raw_fwhm_data. This indicates a violated pipeline "
                f"invariant — _raw_fwhm_data should always be present "
                f"and length-matched to _raw_peak_amplitude_data after "
                f"_run_all populates them. Please report this as a bug."
            )
        fwhm_values = np.asarray(fwhm_row.get('FWHM Values', []), dtype=float)
        if fwhm_values.size != n_events:
            raise RuntimeError(
                f"FWHM data inconsistent for neuron {raw_row['Neuron']} "
                f"in sample {shard.sample_name}: length mismatch with "
                f"{n_events} amplitude events vs {fwhm_values.size} "
                f"FWHM values. This indicates a violated pipeline "
                f"invariant — _raw_fwhm_data should always be present "
                f"and length-matched to _raw_peak_amplitude_data after "
                f"_run_all populates them. Please report this as a bug."
            )
        fwhm_finite = np.isfinite(fwhm_values)
        fwhm_keep = fwhm_finite.copy()
        if apply_fwhm_filter:
            if min_event_fwhm is not None:
                fwhm_keep &= fwhm_values >= min_event_fwhm
            if max_event_fwhm is not None:
                fwhm_keep &= fwhm_values <= max_event_fwhm
        dropped_by_fwhm_nan += int((~fwhm_finite).sum())
        dropped_by_fwhm_bounds += int(((~fwhm_keep) & fwhm_finite).sum())

        # Compose the three layers: NaN/Inf scrub + amplitude bounds +
        # FWHM bounds + human labels. Count label drops as the
        # additional events removed *only* by the label layer relative
        # to the amplitude+FWHM intersection — i.e. attribution stays
        # honest when an event would have been dropped by bounds anyway.
        bounds_keep = amp_keep & fwhm_keep
        if _label_keep_active:
            label_only_drops = int((bounds_keep & (~_label_keep_for_roi)).sum())
            dropped_by_labels += label_only_drops
        final_keep = bounds_keep & _label_keep_for_roi
        masks[key] = final_keep

        # ── Per-event drop ledger build ──────────────────────────────
        # Walk every event positionally and, for those rejected by the
        # final keep-mask, attribute the rejection to the FIRST reason
        # in the configured order (see the function docstring). The
        # mask is the source of truth for whether an event is dropped;
        # this loop only assigns a reason. The FWHM length-equality
        # invariant is enforced above, so NaN/Inf attribution is
        # symmetric across amplitude and FWHM.
        ledger_neuron_idx = int(raw_row['Neuron'])
        ledger_label_eligible = (
            apply_label_filter and roi_id_int is not None
        )
        for ev_idx in range(n_events):
            if final_keep[ev_idx]:
                continue
            amp_val = float(amps[ev_idx])
            fwhm_val = float(fwhm_values[ev_idx])
            amp_is_nan = not np.isfinite(amp_val)
            fwhm_is_nan = not np.isfinite(fwhm_val)
            if amp_is_nan or fwhm_is_nan:
                reason = "nan_inf"
            elif (
                apply_amp_filter
                and min_event_amplitude is not None
                and amp_val < min_event_amplitude
            ):
                reason = "amplitude_below_min"
            elif (
                apply_amp_filter
                and max_event_amplitude is not None
                and amp_val > max_event_amplitude
            ):
                reason = "amplitude_above_max"
            elif (
                apply_fwhm_filter
                and min_event_fwhm is not None
                and fwhm_val < min_event_fwhm
            ):
                reason = "fwhm_below_min"
            elif (
                apply_fwhm_filter
                and max_event_fwhm is not None
                and fwhm_val > max_event_fwhm
            ):
                reason = "fwhm_above_max"
            elif (
                ledger_label_eligible
                and (roi_id_int, ev_idx) in label_drops
            ):
                reason = label_drop_reasons.get(
                    (roi_id_int, ev_idx), "human_label_false"
                )
            else:
                # Defensive: the mask says drop, but no rule in the
                # configured ordering claims it. Shouldn't happen given
                # the invariants enforced above; skip rather than emit
                # a mystery row.
                continue
            shard._event_drop_log.append({
                "sample_id": shard.sample_name,
                "neuron_idx": ledger_neuron_idx,
                "event_idx": int(ev_idx),
                "peak_amplitude": amp_val,
                "fwhm_frames": fwhm_val,
                "drop_reason": reason,
            })

    total_events_after = int(sum(int(m.sum()) for m in masks.values()))

    def _mask_for(row) -> np.ndarray:
        key = (row['Sample'], row['Neuron'])
        mask = masks.get(key)
        if mask is None:
            return np.ones(0, dtype=bool)
        return mask

    def _coerce_list(arr) -> list:
        return arr.tolist() if hasattr(arr, 'tolist') else list(arr)

    # ── Peak amplitudes (and per-neuron max/std) ──────────────────────
    new_peak_rows = []
    new_max_peak_rows = []
    for raw_row in shard._raw_peak_amplitude_data:
        amps = np.asarray(raw_row.get('Peak Amplitudes', []), dtype=float)
        positions = np.asarray(raw_row.get('Peak Positions', []))
        keep = _mask_for(raw_row)
        if keep.size != amps.size:
            keep = np.isfinite(amps)
        kept_amps = amps[keep] if amps.size else amps
        kept_positions = (
            positions[keep] if positions.size == amps.size else positions
        )

        new_peak_rows.append({
            'Sample': raw_row['Sample'],
            'Neuron': raw_row['Neuron'],
            'Peak Amplitudes': _coerce_list(kept_amps),
            'Peak Positions': _coerce_list(kept_positions),
            'is_outlier': raw_row.get('is_outlier', False),
        })

        if kept_amps.size:
            max_peak = float(np.nanmax(kept_amps))
            peak_std = float(np.nanstd(kept_amps))
        else:
            max_peak = float('nan')
            peak_std = float('nan')
        new_max_peak_rows.append({
            'Sample': raw_row['Sample'],
            'Neuron': raw_row['Neuron'],
            'Max Peak df/f': max_peak,
            'Peak Amplitude Std': peak_std,
            'N Peaks': int(kept_amps.size),
            'is_outlier': raw_row.get('is_outlier', False),
        })

    # ── FWHM events ──────────────────────────────────────────────────
    new_fwhm_rows = []
    for raw_row in shard._raw_fwhm_data:
        values = np.asarray(raw_row.get('FWHM Values', []), dtype=float)
        back = np.asarray(raw_row.get('FWHM Backward Positions', []))
        fwd = np.asarray(raw_row.get('FWHM Forward Positions', []))
        counts = np.asarray(raw_row.get('Spike Counts', []))
        keep = _mask_for(raw_row)
        if keep.size != values.size:
            keep = np.isfinite(values)

        def _trim(arr):
            if arr.size == values.size:
                return arr[keep]
            return arr

        new_fwhm_rows.append({
            'Sample': raw_row['Sample'],
            'Neuron': raw_row['Neuron'],
            'FWHM Backward Positions': _coerce_list(_trim(back)),
            'FWHM Forward Positions': _coerce_list(_trim(fwd)),
            'FWHM Values': _coerce_list(values[keep]) if values.size else [],
            'Spike Counts': _coerce_list(_trim(counts)),
            'is_outlier': raw_row.get('is_outlier', False),
        })

    # ── Rise / Fall times ─────────────────────────────────────────────
    new_rise_rows = []
    for raw_row in shard._raw_rise_time_data:
        times = np.asarray(raw_row.get('Rise Times', []), dtype=float)
        positions = np.asarray(raw_row.get('Rise Positions', []))
        keep = _mask_for(raw_row)
        if keep.size != times.size:
            keep = np.isfinite(times)
        new_rise_rows.append({
            'Sample': raw_row['Sample'],
            'Neuron': raw_row['Neuron'],
            'Rise Times': _coerce_list(times[keep]) if times.size else [],
            'Rise Positions': _coerce_list(
                positions[keep] if positions.size == times.size else positions
            ),
            'is_outlier': raw_row.get('is_outlier', False),
        })

    new_fall_rows = []
    for raw_row in shard._raw_fall_time_data:
        times = np.asarray(raw_row.get('Fall Times', []), dtype=float)
        positions = np.asarray(raw_row.get('Fall Positions', []))
        keep = _mask_for(raw_row)
        if keep.size != times.size:
            keep = np.isfinite(times)
        new_fall_rows.append({
            'Sample': raw_row['Sample'],
            'Neuron': raw_row['Neuron'],
            'Fall Times': _coerce_list(times[keep]) if times.size else [],
            'Fall Positions': _coerce_list(
                positions[keep] if positions.size == times.size else positions
            ),
            'is_outlier': raw_row.get('is_outlier', False),
        })

    # ── Peak-to-peak (recomputed from surviving peak positions) ───────
    # Cannot mask raw intervals positionally: dropping event 5 collapses
    # (4→5) + (5→6) into a single (4→6) gap.
    new_peak_to_peak_rows = []
    for raw_row in shard._raw_peak_to_peak_data:
        positions = np.asarray(raw_row.get('Peak Positions', []))
        keep = _mask_for(raw_row)
        if keep.size == positions.size and positions.size:
            kept_positions = positions[keep]
        else:
            kept_positions = positions
        if kept_positions.size >= 2:
            kept_arr = np.asarray(kept_positions)
            intervals = np.diff(kept_arr).astype(float).tolist()
        else:
            intervals = []
        new_peak_to_peak_rows.append({
            'Sample': raw_row['Sample'],
            'Neuron': raw_row['Neuron'],
            'Inter-Spike Intervals': intervals,
            'is_outlier': raw_row.get('is_outlier', False),
        })

    # ── Firing rate per minute (events / min from surviving events) ──
    new_frpm_rows = []
    for raw_row in shard._raw_frpm_data:
        keep = _mask_for(raw_row)
        n_kept = int(keep.sum()) if keep.size else 0
        n_frames = int(raw_row.get('N Frames', 0) or 0)
        frate = float(raw_row.get('Frate', 0) or 0)
        if n_frames > 0 and frate > 0:
            firing_rate = n_kept * 60.0 * frate / n_frames
        else:
            firing_rate = float('nan')
        new_frpm_rows.append({
            'Sample': raw_row['Sample'],
            'Neuron': raw_row['Neuron'],
            'Neuron Index': raw_row.get('Neuron Index', raw_row['Neuron']),
            'Firing Rate Per Min': firing_rate,
            'N Events': n_kept,
            'N Frames': n_frames,
            'Frate': frate,
            'is_outlier': raw_row.get('is_outlier', False),
        })

    # Replace (don't append) — _apply_event_filters is idempotent.
    shard._peak_amplitude_data = new_peak_rows
    shard._max_peak_amplitude_data = new_max_peak_rows
    shard._fwhm_data = new_fwhm_rows
    shard._rise_time_data = new_rise_rows
    shard._fall_time_data = new_fall_rows
    shard._peak_to_peak_data = new_peak_to_peak_rows
    shard._frpm_data = new_frpm_rows

    dropped_by_amplitude = dropped_by_amplitude_nan + dropped_by_amplitude_bounds
    dropped_by_fwhm = dropped_by_fwhm_nan + dropped_by_fwhm_bounds
    n_events_dropped = total_events_before - total_events_after
    naive_intersection_dropped = (
        total_events_before - max(total_events_before - dropped_by_amplitude, 0)
        if False else None
    )

    if log_summary:
        # Always log when *any* event was dropped, even if user bounds
        # are off — the unconditional NaN/Inf scrub still fires.
        if (
            apply_amp_filter
            or apply_fwhm_filter
            or apply_label_filter
            or dropped_by_amplitude_nan
            or dropped_by_fwhm_nan
        ):
            parts = []
            if apply_amp_filter or dropped_by_amplitude_nan:
                parts.append(
                    f'amplitude filter dropped '
                    f'{dropped_by_amplitude}/{total_events_before} peaks '
                    f'(bounds=[{min_event_amplitude}, {max_event_amplitude}], '
                    f'NaN/Inf={dropped_by_amplitude_nan})'
                )
            if apply_fwhm_filter or dropped_by_fwhm_nan:
                parts.append(
                    f'FWHM filter dropped '
                    f'{dropped_by_fwhm}/{total_events_before} events '
                    f'(bounds=[{min_event_fwhm}, {max_event_fwhm}] frames, '
                    f'NaN/Inf={dropped_by_fwhm_nan})'
                )
            if apply_label_filter:
                parts.append(
                    f'human labels dropped '
                    f'{dropped_by_labels}/{total_events_before} events '
                    f'(corpus={str(labels_corpus)!r}, '
                    f'on_disagreement={on_disagreement!r}, '
                    f'unsure_only={label_unsure_only})'
                )
            pct = (
                100.0 * total_events_after / total_events_before
                if total_events_before else 0.0
            )
            parts.append(
                f'combined keep-mask: {total_events_after}/{total_events_before} '
                f'events ({pct:.1f}%) propagated to '
                f'rise/fall/peak-to-peak/FRPM'
            )
            shard._logger.info(
                f'{shard.sample_name}: per-event filtering -> ' + '; '.join(parts)
            )
        if apply_label_filter and label_disagreements:
            shard._logger.info(
                f'{shard.sample_name}: {label_disagreements} events have '
                f'inter-rater disagreement; treating as '
                f'{on_disagreement!r}.'
            )
        if propagation_mismatches:
            shard._logger.warning(
                f'{shard.sample_name}: {propagation_mismatches} neuron(s) had '
                f'amplitude/FWHM length mismatches; mask propagation may be '
                f'incomplete for those neurons.'
            )

    return {
        'apply_amp_filter': apply_amp_filter,
        'apply_fwhm_filter': apply_fwhm_filter,
        'apply_label_filter': apply_label_filter,
        'dropped_by_amplitude': dropped_by_amplitude,
        'dropped_by_amplitude_nan': dropped_by_amplitude_nan,
        'dropped_by_amplitude_bounds': dropped_by_amplitude_bounds,
        'dropped_by_fwhm': dropped_by_fwhm,
        'dropped_by_fwhm_nan': dropped_by_fwhm_nan,
        'dropped_by_fwhm_bounds': dropped_by_fwhm_bounds,
        'dropped_by_labels': dropped_by_labels,
        'label_disagreements': label_disagreements,
        'label_events_seen': label_events_seen,
        'label_unsure_only': label_unsure_only,
        'total_events_before': total_events_before,
        'total_events_after': total_events_after,
        'n_events_dropped': n_events_dropped,
        'propagation_mismatches': propagation_mismatches,
        # Legacy keys preserved for tests/callers that still read them.
        'total_peaks_before': total_events_before,
        'total_fwhm_before': total_events_before,
    }


def run_all(orb: "Orb", 
            frate: int=None, 
            zscore_threshold: int=3, 
            percentage_threshold: float=0.2,
            p_th: float=75, 
            min_clusters: int=2, 
            max_clusters: int=10, 
            random_seed: int=1111111, 
            group_name: str=None, 
            poly: bool=False, 
            size_threshold: int=20000, 
            show_plots: bool=False, 
            save_files: bool=False, 
            output_dir: str='wizards_staff_outputs', 
            threads: int=1, 
            debug: bool=False, 
            outlier_threshold: float=3.5,
            outlier_methods: tuple=("low_pnr", "waveform", "spectral"),
            remove_outlier: bool=False,
            show_outlier: bool=False,
            filter_events: bool=False,
            min_event_amplitude: Optional[float]=0.05,
            max_event_amplitude: Optional[float]=10.0,
            min_event_fwhm: Optional[int]=2,
            max_event_fwhm: Optional[int]=None,
            indicator: Optional[str]=None,
            template_rise_ms: Optional[float]=None,
            template_decay_ms: Optional[float]=None,
            template_total_ms: Optional[float]=None,
            peak_height: Optional[float]=None,
            labels_corpus: Optional[Union[str, Path]]=None,
            on_disagreement: str="drop",
            generate_report: bool=True,
            **kwargs
            ) -> None:
    """
    Process the results folder, computes metrics, and stores them in DataFrames.

    Args:
        results_folder: Path to the results folder.
        metadata_path: Path to the metadata CSV file.
        frate: Frames per second of the imaging session. If None (default), reads from 
               the 'Frate' column in metadata for each sample, supporting per-sample frame rates.
        zscore_threshold: Z-score threshold for spike detection.
        percentage_threshold: Percentage threshold for FWHM calculation.
        p_th: Percentile threshold for image processing.
        min_clusters: The minimum number of clusters to try.
        max_clusters: The maximum number of clusters to try.
        random_seed: The seed for random number generation in K-means.
        group_name: Name of the group to which the data belongs. Required for PWC analysis.
        poly: Flag to control whether to perform polynomial fitting during PWC analysis.
        size_threshold: Size threshold for filtering out noise events.
        show_plots: Flag to control whether plots are displayed.
        save_files: Flag to control whether files are saved.
        output_dir: Directory where output files will be saved.
        threads: Number of threads to use for processing.
        outlier_threshold: Modified Z-score threshold for neuron outlier
            detection. For the ``low_pnr`` detector, neurons whose
            ``log(PNR)`` modified Z-score is below ``-threshold`` are
            flagged (one-sided low tail). For ``waveform`` and
            ``spectral`` the same threshold magnitude is applied per
            their respective sign conventions. Set to 0 to disable.
            Default 3.5.
        outlier_methods: Which outlier detectors to run. A tuple containing any
            combination of "low_pnr", "waveform", and "spectral".
            Default is all three. The legacy key "amplitude" is accepted
            as a deprecated alias for "low_pnr" (emits a
            DeprecationWarning). Examples:
            - ("low_pnr",)                — only low-PNR / dead-neuron rejection
            - ("waveform",)               — only template matching
            - ("spectral",)               — only frequency-domain analysis
            - ("low_pnr", "waveform")     — low_pnr + waveform, skip spectral
        remove_outlier: If True, exclude detected outlier neurons from downstream
            metric DataFrames (FWHM, FRPM, rise time, fall time, peak amplitude,
            peak-to-peak) and from per-shard non-QC plots. Also emits the
            ``plot_sample_mean_dff_with_events`` and
            ``plot_neuron_dff_traces_with_events`` event plots per sample.
            Default False (current behavior preserved).
        show_outlier: If True (and ``remove_outlier=True``), additionally emit
            "with outliers" copies of every affected plot under
            ``output_dir/with_outliers/`` and expose ``orb.*_data_with_outliers``
            DataFrames containing the full (un-cleaned) results — useful for a
            visual before/after QC comparison. Has no effect when
            ``remove_outlier=False`` (a warning is logged).
        filter_events: Master switch for per-event filtering. When ``False``
            (default) the ``min_event_*`` / ``max_event_*`` bounds below are
            ignored and every detected event is stored — this preserves the
            legacy behavior so downstream analyses remain bit-identical. When
            ``True`` the bounds are applied in-place after peak / FWHM
            detection and only surviving events end up in ``fwhm_data``,
            ``peak_amplitude_data``, ``max_peak_amplitude_data`` and
            ``well_peak_amplitude_data``. Independent of ``remove_outlier``:
            the two flags can be used in any combination.
        min_event_amplitude: Per-event filter applied after peak detection.
            Drops events whose peak ΔF/F amplitude is below this value.
            Default ``0.05`` rejects sub-noise and negative "peaks" that are
            numerically impossible for real calcium transients. Set to None
            to disable the lower bound. Only takes effect when
            ``filter_events=True``.
        max_event_amplitude: Per-event filter applied after peak detection.
            Drops events whose peak ΔF/F amplitude exceeds this value.
            Default ``10.0`` rejects deconvolution numerical artifacts (real
            calcium transients almost never exceed ~2-3 ΔF/F). Set to None
            to disable. Only takes effect when ``filter_events=True``.
        min_event_fwhm: Per-event filter applied after FWHM calculation.
            Drops events whose FWHM (in frames) is below this value. Default
            ``2`` rejects 1-frame "spikes" that are implausibly fast for real
            calcium dynamics. Set to None to disable the lower bound. Only
            takes effect when ``filter_events=True``.
        max_event_fwhm: Per-event filter applied after FWHM calculation.
            Drops events whose FWHM (in frames) exceeds this value. Default
            ``None`` (no upper bound). Useful to set to ~200 at 30 fps to
            reject plateau / drift features that span seconds. Frame-rate
            dependent; adjust when using different frame rates. Only takes
            effect when ``filter_events=True``.
        indicator: Calcium indicator preset name forwarded to
            :func:`wizards_staff.stats.outliers.detect_waveform_outliers`.
            When set (e.g. ``"GCaMP6s"``, ``"jGCaMP8m"``, ``"jRGECO1a"``)
            the waveform detector loads published-kinetics defaults from
            ``INDICATOR_PRESETS`` instead of the legacy GCaMP6f-like
            template. Default ``None`` preserves the existing behavior
            (50 ms rise / 400 ms decay / 0.10 peak height). Required
            when working with anything other than GCaMP6f-like green
            indicators — running the GCaMP6f template against e.g.
            GCaMP6s data silently flags real events as shape outliers.
            Unknown names raise ``ValueError``.
        template_rise_ms: Override for the waveform template's 0→peak
            rise time (ms). Wins over the ``indicator`` preset and
            lets users tweak a single parameter without abandoning the
            preset. Default ``None`` (use preset / GCaMP6f-like 50 ms).
        template_decay_ms: Override for the waveform template's
            single-exponential decay time-constant (ms). Default
            ``None`` (use preset / GCaMP6f-like 400 ms).
        template_total_ms: Override for the waveform template's total
            length (ms). Default ``None`` (1500 ms regardless of
            indicator). Should comfortably exceed several decay
            time-constants.
        peak_height: Override for the absolute ΔF/F threshold passed to
            the underlying ``find_peaks`` call. Default ``None`` (use
            preset / GCaMP6f-like 0.10). Red indicators have smaller
            typical excursions, hence the preset value of 0.05 for
            jRGECO1a / jRCaMP1a.
        labels_corpus: Path to a CSV labels corpus produced by
            :class:`wizards_staff.labeling.event_labeler.EventLabeler`.
            When provided, events labeled ``"False"`` for the matching
            ``(sample_id, roi_id, event_idx)`` are dropped from every
            per-event metric (third filter layer, intersected with the
            NaN/Inf scrub and amplitude/FWHM bounds). ``"Unsure"`` and
            unlabeled events pass through. Labels can ONLY drop events;
            a ``"True"`` label cannot recover an event already rejected
            by a prior layer. A missing file is logged as a warning and
            ignored. Default ``None`` (no label-based filtering).
        on_disagreement: How to resolve events with conflicting labels
            from multiple labelers in the corpus. One of ``"drop"``
            (default, precautionary), ``"keep"``, or ``"majority"``
            (ties drop). Has no effect when ``labels_corpus`` is None.
        generate_report: If True (default) a markdown run summary is printed
            to stdout at the end of processing — parameters, per-sample
            neuron/event counts, outlier impact, and describe() tables for
            each per-event metric. When ``save_files=True`` the report is
            also written as ``run_report.md`` in ``output_dir`` and
            histograms for each metric are saved under
            ``output_dir/run_report/``. Set to False to silence for batch or
            CI runs.
        kwargs: Additional keyword arguments that will be passed to run_pwc
    """
    # Check if the output directory exists
    if save_files:
        orb._logger.info(f'Saving output to: {output_dir}')
        # Expand the user directory if it exists in the output_dir path
        output_dir = os.path.expanduser(output_dir)
        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

    # Normalise outlier_methods to a set for fast lookups
    outlier_methods = set(m.lower() for m in outlier_methods) if outlier_methods else set()

    # Map the deprecated "amplitude" key here so the DeprecationWarning is
    # emitted once at the top-level user call, not once per shard inside
    # the (possibly parallel) worker pool.
    if "amplitude" in outlier_methods:
        warnings.warn(
            'outlier_methods=("amplitude", ...) is deprecated; use '
            '"low_pnr" instead. The amplitude detector was replaced by '
            'a one-sided log(PNR) test that does not get contaminated '
            'by noise-dominated neurons.',
            DeprecationWarning,
            stacklevel=2,
        )
        outlier_methods = (outlier_methods - {"amplitude"}) | {"low_pnr"}

    # Validate on_disagreement up front so a typo fails the run before
    # spinning up the (possibly parallel) worker pool.
    if on_disagreement not in _LABEL_DISAGREEMENT_MODES:
        raise ValueError(
            f"on_disagreement must be one of {_LABEL_DISAGREEMENT_MODES}; "
            f"got {on_disagreement!r}"
        )
    if labels_corpus is not None:
        labels_path = Path(labels_corpus)
        if not labels_path.exists():
            orb._logger.warning(
                f"labels_corpus {str(labels_path)!r} does not exist; "
                f"proceeding without label-based filtering."
            )

    # Validate the indicator name once up front so a typo fails the run
    # immediately at the user-facing entry point instead of being raised
    # repeatedly inside each parallel shard worker.
    if indicator is not None:
        from wizards_staff.stats.outliers import INDICATOR_PRESETS
        if indicator not in INDICATOR_PRESETS:
            available = ", ".join(sorted(INDICATOR_PRESETS.keys()))
            raise ValueError(
                f"Unknown indicator preset: {indicator!r}. "
                f"Available presets: {available}."
            )

    # show_outlier requires remove_outlier to make a meaningful before/after comparison.
    if show_outlier and not remove_outlier:
        orb._logger.warning(
            "show_outlier=True has no effect without remove_outlier=True; ignoring."
        )
        show_outlier = False

    # Stash the outlier flags so Orb accessors and save_results can read them.
    orb._remove_outlier = remove_outlier
    orb._show_outlier = show_outlier

    # Announce per-event filtering policy up front so users know what they're
    # opting into vs. opting out of.
    if filter_events:
        print(
            f'Per-event filtering ENABLED (filter_events=True): '
            f'amplitude in [{min_event_amplitude}, {max_event_amplitude}] '
            f'\u0394F/F, FWHM in [{min_event_fwhm}, {max_event_fwhm}] frames. '
            f'A single keep-mask per (sample, neuron) is applied uniformly '
            f'to fwhm_data / peak_amplitude_data / max_peak_amplitude_data / '
            f'well_peak_amplitude_data / rise_time_data / fall_time_data / '
            f'peak_to_peak_data / frpm_data, so every per-event metric '
            f'describes the same surviving event set.',
            flush=True,
        )
    else:
        print(
            'Per-event filtering disabled (filter_events=False). '
            'All detected events are kept; min_event_* / max_event_* '
            'bounds are ignored. NaN/Inf deconvolution artefacts are '
            'still scrubbed unconditionally and that scrub propagates to '
            'every per-event metric. Pass filter_events=True to enable '
            'amplitude/FWHM bounds.',
            flush=True,
        )

    if labels_corpus is not None:
        print(
            f'Human-label filtering ENABLED: labels_corpus='
            f'{str(labels_corpus)!r}, on_disagreement={on_disagreement!r}. '
            f'Events labeled False (and with no overriding True consensus) '
            f'will be dropped from every per-event metric.',
            flush=True,
        )

    # Process each sample (shard) in parallel
    func = partial(
        _run_all, 
        frate=frate, 
        zscore_threshold=zscore_threshold, 
        percentage_threshold=percentage_threshold,
        p_th=p_th,
        min_clusters=min_clusters,
        max_clusters=max_clusters, 
        random_seed=random_seed,
        group_name=group_name,
        poly=poly,
        size_threshold=size_threshold,
        show_plots=show_plots,
        save_files=save_files,
        output_dir=output_dir,
        outlier_threshold=outlier_threshold,
        outlier_methods=outlier_methods,
        remove_outlier=remove_outlier,
        show_outlier=show_outlier,
        filter_events=filter_events,
        min_event_amplitude=min_event_amplitude,
        max_event_amplitude=max_event_amplitude,
        min_event_fwhm=min_event_fwhm,
        max_event_fwhm=max_event_fwhm,
        indicator=indicator,
        template_rise_ms=template_rise_ms,
        template_decay_ms=template_decay_ms,
        template_total_ms=template_total_ms,
        peak_height=peak_height,
        labels_corpus=labels_corpus,
        on_disagreement=on_disagreement,
    )
    desc = 'Shattering the orb and processing each shard...'
    shards = list(orb.shatter())
    
    if debug or threads == 1:
        # Sequential processing (more memory efficient)
        for shard in tqdm(shards, desc=desc):
            try:
                updated_shard = func(shard)
                orb._shards[updated_shard.sample_name] = updated_shard
            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
    else:
        # Parallel processing using joblib for better numpy array handling
        from joblib import Parallel, delayed
        safe_func = partial(_run_all_safe, func=func)
        logging.disable(logging.CRITICAL)
        try:
            results = Parallel(n_jobs=threads, prefer="processes")(
                delayed(safe_func)(shard) for shard in tqdm(shards, desc=desc)
            )
            for updated_shard in results:
                if updated_shard is not None:
                    orb._shards[updated_shard.sample_name] = updated_shard
        finally:
            logging.disable(logging.NOTSET)
    
    # Run PWC analysis if group_name is provided.
    # Wrapped in try/except so a PWC failure (common on tiny tutorial datasets
    # with only one sample per group) doesn't abort downstream event plotting
    # or the save_results call.
    if group_name:
        print('Running pairwise correlation analysis...', flush=True)
        try:
            orb.run_pwc(
                group_name,
                poly = poly,
                p_th = p_th,
                size_threshold = size_threshold,
                show_plots = show_plots,
                **kwargs
            )
            print('Pairwise correlation analysis complete.', flush=True)
        except Exception as e:
            import traceback
            print(
                f'WARNING: Pairwise correlation analysis failed: {e}',
                file=sys.stderr, flush=True,
            )
            traceback.print_exc(file=sys.stderr)
    else:
        orb._logger.warning('Skipping PWC analysis as group_name is not provided.')

    # ── Population-mean and per-neuron event plots ───────────────────
    # These post-aggregation plots consume orb.outlier_data and orb.fwhm_data,
    # so they're emitted once after the parallel loop completes. Only run when
    # the user opted into outlier-aware analysis via remove_outlier=True
    # (which also implicitly enables show_outlier handling below).
    if remove_outlier:
        # Surface the outlier counts so users can tell whether removing
        # outliers actually changed anything (otherwise the cleaned plots are
        # byte-identical to the with-outlier plots).
        try:
            od = orb.outlier_data
            if od is None or od.empty or 'any_outlier' not in od.columns:
                print(
                    'NOTE: remove_outlier=True but no outlier flags were produced '
                    '(outlier_threshold=0 or no detectors enabled). Cleaned plots '
                    'will match the un-cleaned plots.',
                    flush=True,
                )
            else:
                per_sample = (
                    od.groupby('Sample')['any_outlier']
                      .agg(['sum', 'count'])
                      .reset_index()
                )
                total_flagged = int(per_sample['sum'].sum())
                total_neurons = int(per_sample['count'].sum())
                print(
                    f'Outlier summary (remove_outlier=True): '
                    f'{total_flagged}/{total_neurons} neurons flagged across '
                    f'{len(per_sample)} samples.',
                    flush=True,
                )
                for _, row in per_sample.iterrows():
                    print(
                        f'   {row["Sample"]}: '
                        f'{int(row["sum"])}/{int(row["count"])} flagged',
                        flush=True,
                    )
                if total_flagged == 0:
                    print(
                        '   → No neurons were flagged, so cleaned plots will '
                        'match the un-cleaned plots. Try lowering '
                        'outlier_threshold or enabling additional '
                        'outlier_methods if you expect more removal.',
                        flush=True,
                    )
        except Exception as e:
            orb._logger.warning(f'Failed to summarize outlier counts: {e}')

        print('Plotting population-mean ΔF/F traces with event bars...', flush=True)
        for sh in list(orb.shards):
            try:
                plot_sample_mean_dff_with_events(
                    orb,
                    sample_name=sh.sample_name,
                    exclude_outlier_neurons=True,
                    p_th=p_th,
                    size_threshold=size_threshold,
                    zscore_threshold=zscore_threshold,
                    percentage_threshold=percentage_threshold,
                    show_plots=show_plots,
                    save_files=save_files,
                    output_dir=output_dir,
                )
            except Exception as e:
                print(
                    f'WARNING: plot_sample_mean_dff_with_events failed for '
                    f'{sh.sample_name}: {e}',
                    file=sys.stderr, flush=True,
                )

        print('Plotting per-neuron ΔF/F traces with event bars...', flush=True)
        try:
            plot_neuron_dff_traces_with_events(
                orb,
                exclude_outlier_neurons=True,
                p_th=p_th,
                size_threshold=size_threshold,
                show_plots=show_plots,
                save_files=save_files,
                output_dir=output_dir,
            )
        except Exception as e:
            print(
                f'WARNING: plot_neuron_dff_traces_with_events failed: {e}',
                file=sys.stderr, flush=True,
            )

        if show_outlier:
            print(
                'show_outlier=True: also emitting "with outliers" event plots...',
                flush=True,
            )
            out_with = os.path.join(output_dir, "with_outliers")
            for sh in list(orb.shards):
                try:
                    plot_sample_mean_dff_with_events(
                        orb,
                        sample_name=sh.sample_name,
                        exclude_outlier_neurons=False,
                        p_th=p_th,
                        size_threshold=size_threshold,
                        zscore_threshold=zscore_threshold,
                        percentage_threshold=percentage_threshold,
                        show_plots=show_plots,
                        save_files=save_files,
                        output_dir=out_with,
                    )
                except Exception as e:
                    print(
                        f'WARNING: plot_sample_mean_dff_with_events '
                        f'(with outliers) failed for {sh.sample_name}: {e}',
                        file=sys.stderr, flush=True,
                    )
            try:
                plot_neuron_dff_traces_with_events(
                    orb,
                    exclude_outlier_neurons=False,
                    p_th=p_th,
                    size_threshold=size_threshold,
                    show_plots=show_plots,
                    save_files=save_files,
                    output_dir=out_with,
                )
            except Exception as e:
                print(
                    f'WARNING: plot_neuron_dff_traces_with_events '
                    f'(with outliers) failed: {e}',
                    file=sys.stderr, flush=True,
                )

    # ── Run-summary report ───────────────────────────────────────────
    # Build a markdown summary of parameters, per-sample counts, outlier
    # impact, and distribution statistics for every per-event metric. Always
    # prints to stdout; writes to disk only when save_files=True.
    if generate_report:
        report_params = {
            "frate": frate,
            "zscore_threshold": zscore_threshold,
            "percentage_threshold": percentage_threshold,
            "p_th": p_th,
            "size_threshold": size_threshold,
            "min_clusters": min_clusters,
            "max_clusters": max_clusters,
            "group_name": group_name,
            "threads": threads,
            "outlier_threshold": outlier_threshold,
            "outlier_methods": tuple(sorted(outlier_methods)) if outlier_methods else (),
            "remove_outlier": remove_outlier,
            "show_outlier": show_outlier,
            "filter_events": filter_events,
            "min_event_amplitude": min_event_amplitude,
            "max_event_amplitude": max_event_amplitude,
            "min_event_fwhm": min_event_fwhm,
            "max_event_fwhm": max_event_fwhm,
            "labels_corpus": str(labels_corpus) if labels_corpus is not None else None,
            "on_disagreement": on_disagreement,
        }
        try:
            generate_run_report(
                orb,
                params=report_params,
                output_dir=output_dir,
                save_files=save_files,
            )
        except Exception as e:
            print(
                f'WARNING: run report generation failed: {e}',
                file=sys.stderr, flush=True,
            )

    # Save DataFrames as CSV files if required
    if save_files:
        print(f'Saving results to {output_dir}...', flush=True)
        orb.save_results(output_dir)
        print('Results saved.', flush=True)

def _run_all_safe(shard: Shard, func) -> Shard:
    """
    Wrapper that catches per-shard exceptions during parallel processing,
    so one failed shard doesn't discard all successfully processed results.
    """
    try:
        return func(shard)
    except Exception as e:
        import traceback
        print(
            f'WARNING: Error processing shard {shard.sample_name}: {e}',
            file=sys.stderr
        )
        traceback.print_exc(file=sys.stderr)
        return None

def _run_all(shard: Shard, 
            #  os_environ: dict,
             frate: int, 
             zscore_threshold: int, 
             percentage_threshold: float, 
             p_th: float, 
             min_clusters: int, 
             max_clusters: int, 
             random_seed: int, 
             size_threshold: int,
             group_name: str = None, 
             poly: bool = False,
             show_plots: bool = True, 
             save_files: bool = True, 
             output_dir: str = 'wizard_staff_outputs',
             outlier_threshold: float = 3.5,
             outlier_methods: set = None,
             remove_outlier: bool = False,
             show_outlier: bool = False,
             filter_events: bool = False,
             min_event_amplitude: Optional[float] = None,
             max_event_amplitude: Optional[float] = None,
             min_event_fwhm: Optional[int] = None,
             max_event_fwhm: Optional[int] = None,
             indicator: Optional[str] = None,
             template_rise_ms: Optional[float] = None,
             template_decay_ms: Optional[float] = None,
             template_total_ms: Optional[float] = None,
             peak_height: Optional[float] = None,
             labels_corpus: Optional[Union[str, Path]] = None,
             on_disagreement: str = "drop",
             ) -> Shard:
    """
    Process each shard of the Orb and compute metrics.
    Args:
        See run_all function.
    Returns:
        shard: The updated shard object
    """    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    shard._logger.info(f'Processing shard: {shard.sample_name}')

    # Get frame rate from metadata if not explicitly provided
    if frate is None:
        frate = int(shard.metadata['Frate'].iloc[0])
        shard._logger.info(f'Using frame rate from metadata: {frate} fps')

    # Check for required inputs
    for key in ['dff_dat', 'minprojection', 'cnm_A']:
        if not shard.has_file(key):
            shard._logger.warning(f'Missing required input: {key}; skipping {shard.sample_name}')
            return shard

    # Apply spatial filtering to the data to remove noise
    filtered_idx = shard.spatial_filtering(
        p_th=p_th, 
        size_threshold=size_threshold,
        plot=False, 
        silence=True,
    )
    
    # ── Outlier detection ────────────────────────────────────────────
    if outlier_methods is None:
        outlier_methods = {"low_pnr", "waveform", "spectral"}

    # Backwards-compat: accept the legacy "amplitude" key for one release
    # cycle, mapping it to the new "low_pnr" detector. Emit a warning so
    # callers update their tuples.
    if "amplitude" in outlier_methods:
        warnings.warn(
            'outlier_methods=("amplitude", ...) is deprecated; use '
            '"low_pnr" instead. The amplitude detector was replaced by '
            'a one-sided log(PNR) test that does not get contaminated '
            'by noise-dominated neurons.',
            DeprecationWarning,
            stacklevel=2,
        )
        outlier_methods = (set(outlier_methods) - {"amplitude"}) | {"low_pnr"}

    # Component indices that the outlier detectors flagged (any method).
    # Populated below if outlier detection runs; otherwise stays empty so
    # downstream `is_outlier` tagging defaults to False.
    outlier_components: set = set()

    if outlier_threshold > 0 and outlier_methods:
        dff_raw = shard.get_input('dff_dat', req=True)

        low_pnr_result = None
        waveform_result = None
        spectral_result = None

        if "low_pnr" in outlier_methods:
            low_pnr_result = detect_low_pnr_neurons(
                dff_raw, filtered_idx, threshold=outlier_threshold
            )
            if low_pnr_result["n_flagged"] > 0:
                shard._logger.info(
                    f'{shard.sample_name}: {low_pnr_result["summary"]}'
                )

        if "waveform" in outlier_methods:
            waveform_result = detect_waveform_outliers(
                dff_raw, filtered_idx, fps=frate, threshold=outlier_threshold,
                indicator=indicator,
                template_rise_ms=template_rise_ms,
                template_decay_ms=template_decay_ms,
                template_total_ms=template_total_ms,
                peak_height=peak_height,
            )
            if waveform_result["n_flagged"] > 0:
                shard._logger.info(
                    f'{shard.sample_name}: {waveform_result["summary"]}'
                )

        if "spectral" in outlier_methods:
            spectral_result = detect_spectral_outliers(
                dff_raw, filtered_idx, fps=frate, threshold=outlier_threshold,
            )
            if spectral_result["n_flagged"] > 0:
                shard._logger.info(
                    f'{shard.sample_name}: {spectral_result["summary"]}'
                )

        qc = combine_neuron_qc(
            filtered_idx,
            low_pnr_result=low_pnr_result,
            waveform_result=waveform_result,
            spectral_result=spectral_result,
        )
        combined_df = qc["combined_df"]

        for _, row in combined_df.iterrows():
            entry = {
                'Sample': shard.sample_name,
                'Neuron Index': int(row['component_idx']),
            }
            for col in combined_df.columns:
                if col != 'component_idx':
                    entry[col] = row[col]
            shard._outlier_data.append(entry)

        # Build the set of component indices flagged by any enabled detector.
        if 'any_outlier' in combined_df.columns:
            outlier_components = set(
                int(combined_df.loc[i, 'component_idx'])
                for i in combined_df.index
                if bool(combined_df.loc[i, 'any_outlier'])
            )

        if low_pnr_result is not None:
            plot_neuron_outliers(
                low_pnr_result,
                sample_name=shard.sample_name,
                show_plots=show_plots,
                save_files=save_files,
                output_dir=output_dir,
            )
        if waveform_result is not None:
            plot_waveform_qc(
                waveform_result,
                dff_raw, filtered_idx, fps=frate,
                sample_name=shard.sample_name,
                show_plots=show_plots,
                save_files=save_files,
                output_dir=output_dir,
                peak_height=waveform_result.get(
                    "peak_height", 0.1,
                ),
            )
        if spectral_result is not None:
            plot_spectral_qc(
                spectral_result,
                dff_raw, filtered_idx, fps=frate,
                sample_name=shard.sample_name,
                show_plots=show_plots,
                save_files=save_files,
                output_dir=output_dir,
            )

    # Cleaned filtered_idx: spatial-filtered components minus outlier components.
    # Order is preserved relative to filtered_idx so local indices into per-neuron
    # metric dicts remain consistent for the cleaned-plot path.
    clean_idx = [int(c) for c in filtered_idx if int(c) not in outlier_components]
    # Index used for non-QC, cross-neuron / clustering plots.
    plot_idx = clean_idx if remove_outlier else filtered_idx

    # Safety guard: if outlier removal would empty out the plot index for this
    # sample (every spatially-filtered neuron got flagged), fall back to the
    # un-cleaned index so clustering / heatmap plots still produce something
    # rather than crashing on a zero-row input.
    if remove_outlier and len(plot_idx) == 0 and len(filtered_idx) > 0:
        shard._logger.warning(
            f'{shard.sample_name}: all {len(filtered_idx)} filtered neurons '
            f'flagged as outliers; falling back to filtered_idx for per-shard '
            f'cross-neuron plots so they still render.'
        )
        plot_idx = filtered_idx

    # Convert ΔF/F₀ to calcium signals and spike events
    calcium_signals, spike_events = shard.convert_f_to_cs(p=2)

    # Z-score the spike events
    zscored_spike_events = zscore(np.copy(spike_events), axis=1)

    # Filter the calcium signals and z-scored spike events based on the spatial filtering
    zscored_spike_events_filtered = zscored_spike_events[filtered_idx, :]
    calcium_signals_filtered = calcium_signals[filtered_idx, :]

    # Calculate rise time and positions:
    rise_tm, rise_tm_pos = shard.calc_rise_tm(
        calcium_signals_filtered, 
        zscored_spike_events_filtered, 
        zscore_threshold=zscore_threshold
    )

    # Calculate FWHM and related metrics
    fwhm_pos_back, fwhm_pos_fwd, fwhm, spike_counts = shard.calc_fwhm_spikes(
        calcium_signals_filtered, 
        zscored_spike_events_filtered,
        zscore_threshold=zscore_threshold, 
        percentage_threshold=percentage_threshold
    )

    # Calculate FRPM:
    _, frpm  = shard.calc_frpm(
        zscored_spike_events, filtered_idx, frate, 
        zscore_threshold=zscore_threshold
    )

    # Calculate fall time
    fall_tm, fall_tm_pos = shard.calc_fall_tm(
        calcium_signals_filtered,
        zscored_spike_events_filtered,
        zscore_threshold=zscore_threshold
    )

    # Get raw ΔF/F₀ data filtered by spatial filtering for amplitude measurement
    dff_data_raw = shard.get_input('dff_dat', req=True)
    dff_data_filtered = dff_data_raw[filtered_idx, :]

    # Calculate peak amplitude (using raw ΔF/F₀ for interpretable units)
    peak_amp, peak_pos = shard.calc_peak_amplitude(
        calcium_signals_filtered,
        zscored_spike_events_filtered,
        zscore_threshold=zscore_threshold,
        dff_data=dff_data_filtered
    )

    # Calculate peak-to-peak intervals (inter-spike intervals)
    peak_to_peak = shard.calc_peak_to_peak(
        calcium_signals_filtered,
        zscored_spike_events_filtered,
        zscore_threshold=zscore_threshold
    )

    # ── Per-event filtering (deferred) ──────────────────────────────
    # Raw events for *every* per-event metric are stored unconditionally
    # below; per-event filtering is applied afterwards by
    # ``_apply_event_filters``. This keeps a faithful pre-filter copy on
    # each shard so ``orb.refilter_events(...)`` can re-derive every
    # filtered view (peak/FWHM/rise/fall/peak-to-peak/FRPM) with new
    # bounds without re-running spatial filtering, outlier detection, or
    # the per-event metric calculations. ``_apply_event_filters`` is the
    # single source of truth for the keep-mask and applies it
    # consistently to each metric so downstream analyses join across
    # metrics on the *same* event set.

    # Helper: map a local index (into filtered_idx) to its underlying component
    # index, then check if that component was flagged as an outlier.
    def _is_outlier(loc_idx: int) -> bool:
        try:
            comp = int(filtered_idx[loc_idx])
        except (IndexError, TypeError):
            return False
        return comp in outlier_components

    # Stash the recording-level scalars _apply_event_filters needs to
    # recompute filtered FRPM (events/minute) from the keep-mask without
    # re-loading the z-score matrix.
    shard._recording_n_frames = int(zscored_spike_events.shape[1])
    shard._recording_frate = int(frate)

    # Raw rise-time rows. Filtered _rise_time_data is derived from these
    # by _apply_event_filters below.
    for neuron_idx, rise_times in rise_tm.items():
        shard._raw_rise_time_data.append({
            'Sample': shard.sample_name,
            'Neuron': neuron_idx,
            'Rise Times': rise_times,
            'Rise Positions': rise_tm_pos[neuron_idx],
            'is_outlier': _is_outlier(neuron_idx),
        })

    # Raw FWHM rows.
    for neuron_idx, fwhm_values in fwhm.items():
        shard._raw_fwhm_data.append({
            'Sample': shard.sample_name,
            'Neuron': neuron_idx,
            'FWHM Backward Positions': fwhm_pos_back[neuron_idx],
            'FWHM Forward Positions': fwhm_pos_fwd[neuron_idx],
            'FWHM Values': fwhm_values,
            'Spike Counts': spike_counts[neuron_idx],
            'is_outlier': _is_outlier(neuron_idx),
        })

    # Raw fall-time rows.
    for neuron_idx, fall_times in fall_tm.items():
        shard._raw_fall_time_data.append({
            'Sample': shard.sample_name,
            'Neuron': neuron_idx,
            'Fall Times': fall_times,
            'Fall Positions': fall_tm_pos[neuron_idx],
            'is_outlier': _is_outlier(neuron_idx),
        })

    # Raw peak-amplitude rows.
    for neuron_idx, amplitudes in peak_amp.items():
        shard._raw_peak_amplitude_data.append({
            'Sample': shard.sample_name,
            'Neuron': neuron_idx,
            'Peak Amplitudes': amplitudes,
            'Peak Positions': peak_pos[neuron_idx],
            'is_outlier': _is_outlier(neuron_idx),
        })

    # Raw peak-to-peak rows. We persist both the raw inter-spike intervals
    # (mirroring the filtered schema) and the underlying peak positions —
    # _apply_event_filters needs the positions to *recompute* intervals
    # between surviving peaks, since masking raw intervals positionally
    # would treat (4->5) and (5->6) as separate gaps when event 5 is
    # dropped.
    for neuron_idx, intervals in peak_to_peak.items():
        shard._raw_peak_to_peak_data.append({
            'Sample': shard.sample_name,
            'Neuron': neuron_idx,
            'Peak Positions': peak_pos[neuron_idx],
            'Inter-Spike Intervals': intervals,
            'is_outlier': _is_outlier(neuron_idx),
        })

    # Raw FRPM rows. We store the per-neuron raw event count (same as
    # ``len(rise_tm[neuron])``), the recording length, and the frame
    # rate, so _apply_event_filters can derive *events per minute* from
    # the keep-mask. Note: this redefines FRPM relative to legacy
    # versions, where ``calc_frpm`` returned frames-above-threshold per
    # minute. The new definition counts each detected calcium transient
    # once and matches the rise/fall/peak event sets — see
    # _apply_event_filters and CHANGELOG.md for details.
    n_frames = int(zscored_spike_events.shape[1])
    raw_frpm_value = (
        lambda n_events: float(n_events * 60.0 * frate / n_frames)
        if n_frames > 0 else float('nan')
    )
    for neuron_idx, rise_times in rise_tm.items():
        comp_idx = int(filtered_idx[neuron_idx])
        n_events = len(rise_times)
        shard._raw_frpm_data.append({
            'Sample': shard.sample_name,
            'Neuron': neuron_idx,
            'Neuron Index': comp_idx,
            'Firing Rate Per Min': raw_frpm_value(n_events),
            'N Events': n_events,
            'N Frames': n_frames,
            'Frate': frate,
            'is_outlier': comp_idx in outlier_components,
        })

    # Apply per-event filters (or just copy raw -> filtered when
    # filter_events=False) and derive every filtered metric from the
    # shared keep-mask.
    _apply_event_filters(
        shard,
        filter_events=filter_events,
        min_event_amplitude=min_event_amplitude,
        max_event_amplitude=max_event_amplitude,
        min_event_fwhm=min_event_fwhm,
        max_event_fwhm=max_event_fwhm,
        labels_corpus=labels_corpus,
        on_disagreement=on_disagreement,
    )

    # Calculate mask metrics and store them      
    if shard.has_file('mask'):
        mask_metrics = shard.calc_mask_shape_metrics()
        shard._mask_metrics_data.append({
            'Sample': shard.sample_name,
            'Roundness':  mask_metrics.get('roundness'),
            'Diameter': mask_metrics.get('diameter'),
            'Area': mask_metrics.get('area')
        })

    # ── Cross-neuron / clustering plots ──────────────────────────────
    # These plots depend on which neurons are included (clustering, heatmaps,
    # spatial maps), so we re-route them through ``plot_idx`` (cleaned when
    # ``remove_outlier=True``). When ``show_outlier=True`` we additionally emit
    # a "with outliers" copy under ``output_dir/with_outliers/``.
    #
    # Each call is wrapped so one failed plot (e.g. clustering on too-few neurons)
    # doesn't abort the whole shard's processing.
    def _safe_plot(name, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            shard._logger.warning(
                f'{shard.sample_name}: {name} failed: {e}'
            )
            return None

    def _emit_shard_plots(idx, out_dir):
        if len(idx) == 0:
            shard._logger.warning(
                f'{shard.sample_name}: empty index for cross-neuron plots '
                f'(out_dir={out_dir}); skipping per-shard plotting.'
            )
            return None, None

        _safe_plot(
            'plot_dff_activity', plot_dff_activity,
            shard.get_input('dff_dat', req=True),
            idx, frate, shard.sample_name,
            sz_per_neuron=0.5,
            show_plots=show_plots,
            save_files=save_files,
            output_dir=out_dir,
        )

        kmeans_res = _safe_plot(
            'plot_kmeans_heatmap', plot_kmeans_heatmap,
            dff_dat=shard.get_input('dff_dat', req=True),
            filtered_idx=idx,
            sample_name=shard.sample_name,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            random_seed=random_seed,
            show_plots=show_plots,
            save_files=save_files,
            output_dir=out_dir,
        )
        sscore, nclust = kmeans_res if kmeans_res is not None else (None, None)

        _safe_plot(
            'plot_cluster_activity', plot_cluster_activity,
            dff_dat=shard.get_input('dff_dat', req=True),
            filtered_idx=idx,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            random_seed=random_seed,
            norm=False,
            show_plots=show_plots,
            save_files=save_files,
            sample_name=shard.sample_name,
            output_dir=out_dir,
        )

        _safe_plot(
            'plot_spatial_activity_map', plot_spatial_activity_map,
            shard.get_input('minprojection', req=True),
            shard.get_input('cnm_A', req=True),
            idx,
            shard.sample_name,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            random_seed=random_seed,
            show_plots=show_plots,
            save_files=save_files,
            dff_dat=shard.get_input('dff_dat'),
            output_dir=out_dir,
        )

        _safe_plot(
            'plot_spatial_activity_map (clustering)', plot_spatial_activity_map,
            shard.get_input('minprojection', req=True),
            shard.get_input('cnm_A', req=True),
            idx,
            shard.sample_name,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            random_seed=random_seed,
            clustering=True,
            dff_dat=shard.get_input('dff_dat', req=True),
            show_plots=show_plots,
            save_files=save_files,
            output_dir=out_dir,
        )

        return sscore, nclust

    # Primary (cleaned when remove_outlier=True, otherwise full) plots.
    # Silhouette score is recorded from this primary pass only.
    silhouette_score, num_clusters = _emit_shard_plots(plot_idx, output_dir)

    # Optional "with outliers" mirror tree.
    if show_outlier:
        _emit_shard_plots(filtered_idx, os.path.join(output_dir, "with_outliers"))

    # Append silhouette score to the list (may be None if k-means plot failed
    # or was skipped due to an empty plot_idx — record it anyway for traceability).
    shard._silhouette_scores_data.append({
        'Sample': shard.sample_name,
        'Silhouette Score': silhouette_score,
        'Number of Clusters': num_clusters
    })

    return shard