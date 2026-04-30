# Changelog

All notable changes to Wizards-Staff are recorded here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Fixed (correctness fix — pre-publication behavior change)

- **Per-event filtering (Stage 4) now consistently applies to
  `rise_time_data`, `fall_time_data`, `peak_to_peak_data`, and
  `frpm_data`, not only to peak/FWHM data.** Previously these metrics
  reflected raw z-score crossings even when `filter_events=True`,
  producing inconsistent event sets across metrics on the same shard:
  any cross-metric analysis (e.g. "how does rise time scale with
  amplitude") was implicitly joining mismatched event lists. The
  unconditional NaN/Inf scrub had the same scope problem — even with
  `filter_events=False`, NaN/Inf events were dropped from peak/FWHM
  but remained in rise/fall/peak-to-peak/FRPM.

  `wizards_staff.wizards.cauldron._apply_event_filters` now builds a
  single keep-mask per `(sample, neuron)` from the raw amplitude /
  FWHM lists and applies it uniformly to every per-event metric.
  `peak_to_peak_data` is recomputed from the *surviving* peak
  positions (so dropping event 5 produces a single 4→6 gap, not two
  separate gaps). `frpm_data` is recomputed from the surviving event
  count and the recording length (events per minute).

  Two supporting changes were required to make a single positional
  mask sound:

  - `calc_fwhm_spikes` and `calc_fall_tm` in
    `wizards_staff/wizards/spellbook.py` now share the canonical
    crossing walk used by `calc_rise_tm` / `calc_peak_amplitude` /
    `calc_peak_to_peak`. Previously `calc_fwhm_spikes` used `>`
    instead of `>=` for the threshold and advanced past the FWHM
    forward window, while `calc_fall_tm` advanced past the fall
    window. Both could silently drop events that the other walkers
    caught, breaking the i-th-event correspondence relied on by the
    keep-mask. As a side effect, both functions now report the same
    number of events per neuron as the other walkers — datasets with
    closely-spaced spikes (where one transient's fall window contains
    the next transient's rise) will report more FWHM / fall-time
    events than before.

  - `Orb.refilter_events` now refreshes `rise_time_data`,
    `fall_time_data`, `peak_to_peak_data`, and `frpm_data` accessors
    (and their orb-level caches) in addition to the peak / FWHM /
    `max_peak_amplitude_data` accessors it already refreshed. The
    bootstrap path for legacy pickled orbs was extended to populate
    the new `_raw_*_data` slots when missing.

  **Behavior change semantics.** `frpm_data["Firing Rate Per Min"]`
  has been redefined as *events per minute* (one entry per detected
  calcium transient that survives filtering). Earlier versions
  returned *frames-above-threshold per minute*, which double-counted
  multi-frame above-threshold runs. Datasets re-analyzed after this
  change will show different `rise_time_data`, `fall_time_data`,
  `peak_to_peak_data`, and `frpm_data` distributions. We have not
  gone live yet, but this affects pre-publication results computed
  with `filter_events=True` or on data containing deconvolution
  NaN/Inf artefacts.

  - New shard attributes: `_raw_rise_time_data`, `_raw_fall_time_data`,
    `_raw_peak_to_peak_data`, `_raw_frpm_data`, `_recording_n_frames`,
    `_recording_frate`. The filtered lists are derived from the raw
    lists via `_apply_event_filters`, which is now a pure function of
    the raw lists plus bounds (idempotent, never mutates raw).
  - New tests in `tests/wizards/test_event_filters.py` cover keep-mask
    consistency, NaN/Inf scrub propagation, amplitude/FWHM bounds
    propagation, peak-to-peak recomputation from surviving positions,
    FRPM-from-filtered-events, `Orb.refilter_events` end-to-end,
    raw-list immutability, and the spellbook walk-order invariant.

### Changed

- **FWHM data inconsistencies (missing `_raw_fwhm_data` row or length
  mismatch with `_raw_peak_amplitude_data`) now raise `RuntimeError`
  instead of silently keeping all events.** This path should not fire
  on normal data; if it does, it indicates a bug worth reporting. The
  previous fallback logged a warning and defaulted `fwhm_keep` to
  all-ones, which let NaN/Inf FWHM values leak into downstream metrics
  asymmetrically with amplitude (whose NaN/Inf scrub always fires).
- **`EventLabeler.display()` no longer renders the UI multiple times.**
  Returning the root widget caused Jupyter to auto-render the cell's
  return value on top of the explicit `IPython.display.display(root)`
  call, and the matplotlib figure created inside the Output capture
  was being re-flushed by the inline backend's end-of-cell
  `flush_figures` hook. `display()` now returns `None` (the root is
  still accessible programmatically via `labeler._widgets["root"]`)
  and the figure is detached from pyplot's global tracking via
  `plt.close(self._fig)` immediately after creation.

### Renamed / removed (pre-publication; no on-disk migration provided)

- **EventLabeler corpus column `neuron_id` → `roi_id`** (corpus
  `corpus_version` bumped 1 → 2). Calling labeled components "neurons"
  was technically inaccurate — they are ROIs (candidate components
  produced by source extraction); whether a given ROI is a real neuron
  is a downstream question that the labeler does not adjudicate.
  - The labeler refuses to load a v1 corpus rather than corrupt
    accumulated label state; pre-publication, just delete or rebuild
    any in-progress label files.
  - `wizards_staff.wizards.cauldron._resolve_label_drops` now reads
    `roi_id` from the corpus and the `Orb.refilter_events` /
    `_apply_event_filters` docstrings reflect the new column name.
  - The labeler's internal event-dict key, the in-memory label key
    tuple, all `_neurons_in_order` / `fwhm_by_neuron` / `current_neuron`
    locals, and the `_get_trace(neuron_id)` parameter were renamed
    `roi_*`. The drop-ledger column `neuron_idx` is unrelated and
    unchanged.
- **`EventLabeler.reject_whole_neuron` → `reject_whole_trace`.** The
  action labels every event on the current ROI's ΔF/F trace as False;
  it does NOT mark the ROI itself bad (whole-ROI rejection lives in
  the outlier-detection layer). **The deprecated alias has been
  removed** rather than carried as a `DeprecationWarning`-emitting
  forwarder — pre-publication, two ways to do the same thing is just
  two surfaces to maintain. Callers must use `reject_whole_trace`.
  Companion changes:
  - Button label "Reject whole neuron (n)" → "Reject whole trace (w)".
  - Keyboard shortcut `n` → `w`.
  - Corpus `notes` value for bulk-reject rows is now
    `whole_trace_reject` (was `whole_neuron_reject`). No code keys on
    this string; the rename is for honesty.
- **EventLabeler `ordering="by_neuron_then_time"` →
  `ordering="by_roi_then_time"`.** The default ordering string was
  renamed to match the column rename. The old string is no longer
  accepted; callers must update.
- **EventLabeler UI labels: `Neuron N of M (id=…)` → `ROI N of M
  (id=…)`** in the progress strip; trace-plot title `Sample S · neuron
  N · event K` → `Sample S · ROI N · event K`; "Reject whole trace"
  tooltip rewritten to spell out that it only affects the *current*
  ROI and does not mark the ROI itself bad.

### Changed (behavior change — results on existing datasets may differ)

- **Outlier detection (neuron level): replaced the amplitude detector
  with a low-PNR detector.** The previous
  `detect_neuron_outliers` flagged neurons via a two-sided modified
  Z-score on max/mean/std of ΔF/F across the population. In datasets
  with many noise-dominated neurons, this pulled the population
  reference distribution toward the noise floor and caused clean,
  high-activity neurons to be incorrectly flagged as high-side
  outliers — the *opposite* of the intended behavior. The detector is
  now a one-sided low-tail test on `log(PNR)`, where PNR is the
  peak-to-noise ratio computed using a robust event-immune noise
  estimator (`sigma_hf = 1.4826 * median(|diff(dff)|) / sqrt(2)`).
  Constant or near-constant traces are reported with
  `reason="flat_trace"`. See `wizards_staff/stats/outliers.py` for
  details.
  - Re-running `Orb.run_all` on a previously analysed dataset can yield
    a different set of flagged neurons than before. In particular,
    populations dominated by noise neurons will now reject the noise
    neurons (correct) instead of the responders (wrong).
  - Default `outlier_threshold` is unchanged (3.5).

### Renamed / deprecated

- `wizards_staff.stats.outliers.detect_neuron_outliers` →
  `detect_low_pnr_neurons`. The old name remains as a deprecated alias
  that emits a `DeprecationWarning` and forwards to the new
  implementation.
- `Orb.run_all(outlier_methods=...)`: detector key `"amplitude"` →
  `"low_pnr"`. The old key is still accepted with a
  `DeprecationWarning`.
- `combine_neuron_qc(amplitude_result=...)` →
  `combine_neuron_qc(low_pnr_result=...)`. The old keyword still works
  with a `DeprecationWarning` for one release cycle.

### Added

- **Per-event drop ledger.**
  `_apply_event_filters` now records, alongside its existing aggregate
  counts, an audit trail of every rejected event:
  `wizards_staff.wizards.shard.Shard._event_drop_log` is a list of
  dicts with `sample_id`, `neuron_idx`, `event_idx`, `peak_amplitude`
  (may be NaN/Inf — those are valid drop reasons), `fwhm_frames` (may
  be NaN/Inf), and `drop_reason` (one of `nan_inf`,
  `amplitude_below_min`, `amplitude_above_max`, `fwhm_below_min`,
  `fwhm_above_max`, `human_label_false`,
  `human_label_disagreement_drop`). When an event would be dropped
  for multiple reasons, the FIRST reason in the order
  `nan_inf > amplitude bounds > fwhm bounds > human label` wins —
  documented in the function docstring. The new
  `Orb.event_drop_log` property concatenates all per-shard ledgers
  into a DataFrame keyed by `sample_id` (returns an empty DataFrame
  with the canonical columns when no events were dropped).
  `Orb.refilter_events` regenerates the ledger from scratch on every
  call so it always reflects the current filter configuration —
  never an appended history. When `save_files=True`,
  `Orb.save_results` writes `event_drop_log.csv` to `output_dir`
  alongside the other CSVs. The `generate_run_report` markdown
  report grew a "Drop reasons" subsection with per-reason counts and
  a pointer to the saved ledger CSV. Tests in
  `tests/wizards/test_event_filters.py` cover every drop-reason
  category, the first-reason-wins ordering, no-double-counting,
  cross-shard concatenation, refilter regeneration, CSV export
  round-trip, and the empty-ledger schema. Existing aggregate logging
  (`amplitude filter dropped X/Y peaks…`) is unchanged — the ledger
  is supplementary.
- New per-neuron QC fields on `detect_low_pnr_neurons`’s
  `neuron_scores` DataFrame: `pnr`, `sigma_hf`,
  `log_pnr_modified_zscore`, `is_low_pnr`, `reason` (one of `"ok"`,
  `"low_pnr"`, `"flat_trace"`).
- `tests/stats/test_outliers.py` covering the new detector, the
  contamination-resistance regression, the flat-trace branch, the
  one-sided semantics, and the legacy aliases.
- **Indicator-aware waveform outlier detector.**
  `detect_waveform_outliers` now accepts an `indicator` parameter
  along with explicit `template_rise_ms`, `template_decay_ms`,
  `template_total_ms`, and `peak_height` overrides. A new
  module-level `INDICATOR_PRESETS` dict ships starting-point
  rise/decay/peak-height defaults for `GCaMP6f` (legacy default),
  `GCaMP6s`, `GCaMP6m`, `GCaMP7f`, `jGCaMP8f`, `jGCaMP8m`,
  `jGCaMP8s`, `jRGECO1a`, `jRCaMP1a`, and `GCaMP3` (citations in
  the source). Explicit kwargs override preset values, so
  `indicator="GCaMP6s", template_decay_ms=2000` keeps the GCaMP6s
  rise/peak-height but uses a 2000 ms decay. Unknown indicator
  names raise `ValueError` listing every available preset.
  Previously the detector hardcoded GCaMP6f-like kinetics
  (`rise_ms=50`, `decay_ms=400`, `peak_height=0.10`) and silently
  miscalibrated on data from other indicators — real events would
  be flagged as shape outliers because they didn't match the
  template, and the absolute peak threshold was inappropriate for
  red indicators with smaller ΔF/F excursions. `Orb.run_all` /
  `wizards_staff.wizards.cauldron.run_all` and the CLI grew
  matching `--indicator` / `--template-rise-ms` /
  `--template-decay-ms` / `--template-total-ms` / `--peak-height`
  flags. Default behavior is unchanged: `indicator=None` keeps the
  legacy GCaMP6f-like defaults and existing analyses are
  bit-identical.
- New tests in `tests/stats/test_outliers.py`:
  `test_waveform_template_parameterized`,
  `test_indicator_preset_lookup`,
  `test_indicator_preset_unknown_raises`,
  `test_explicit_override_wins`,
  `test_default_unchanged`, and
  `test_template_mismatch_predictable_degradation` (regression
  guard that GCaMP6s-shaped events correlate better with the
  GCaMP6s template than the GCaMP6f template, asserting the
  parameter actually flows through to `_make_calcium_template`).
