"""
Notebook-based hand-labeling tool for Wizards-Staff calcium events.

The :class:`EventLabeler` walks a biologist through detected calcium
events on a single :class:`~wizards_staff.wizards.shard.Shard` and
collects True / False / Unsure judgements via a small ipywidgets UI.
Labels are appended to a canonical CSV corpus that is shared across
datasets and labelers.

User-facing workflow (trace-first):
    1. The labeler opens on a per-ROI **overview** showing the full
       ΔF/F trace with every detected event marked by its current
       label state (green=True, red=False, gray=Unsure, orange=
       unlabeled — the unlabeled color must NOT match the trace line
       blue or the markers vanish into the trace).
    2. From the overview the biologist picks one trace-level outcome:

         * **Investigate trace** (``i``; legacy: ``d``) — open the
           per-event view to inspect and label events one at a time.
           Auto-returns to the overview after the last event of the
           ROI is labeled (or via ``b`` / "Back to trace" at any
           time).
         * **Reject whole trace** (``r``) — label every still-unlabeled
           event on this ROI as ``False`` with note
           ``whole_trace_reject``, then jump to the next ROI's overview.
         * **Skip trace** (``s``) — record that the labeler reviewed
           this trace and chose not to label any events individually,
           then advance to the next ROI. A trace-level row
           (``event_idx = TRACE_ACTION_EVENT_IDX``, ``notes =
           "trace_skip"``) is written to the corpus so downstream
           consumers can distinguish "reviewed and clean" from "never
           opened" — important for inter-rater agreement and for
           avoiding sampling bias in the training corpus. Individual
           events on the skipped trace are NOT given event-level labels.

       ``p`` / ``n`` navigate between trace overviews without writing
       anything; this is purely a viewing aid. ``s`` (skip) is the
       review-and-advance equivalent.

       In the per-event view, ``r`` / ``w`` (reject whole trace) is
       intentionally NOT bound: bulk-rejecting the remaining events of
       a trace mid-investigation is destructive and irreversible, so
       the user has to back out to overview to invoke it. This
       prevents a single fat-finger from silently False-labelling 10+
       events.

Design notes:
    * ``ipywidgets`` is imported lazily inside :meth:`EventLabeler.display`
      so the module is safe to import in headless environments (Lizard
      Wizard's CLI invocations of Wizards-Staff, batch jobs, etc.).
    * Plot rendering is done into an ipywidgets ``Output`` widget; no
      ``plt.show()`` is ever called.
    * Persistence is per-action and atomic: every label triggers a full
      atomic re-write of the corpus CSV via ``os.replace`` so a crash
      mid-write cannot corrupt the file or lose previous work.
    * The trace-first UI assumes events for a given ROI are visited
      contiguously. :meth:`EventLabeler.display` therefore reorders
      events to ``by_roi_then_time`` (with a warning) when the labeler
      was constructed with one of the calibration-only orderings such
      as ``stratified`` or ``amplitude_descending``.
    * ``reject_whole_trace`` is a *two-press* action by default: the
      first press arms, the second consecutive press commits. Any
      other action disarms the pending rejection. This prevents a
      stray ``r`` keystroke from silently False-labelling 10+ events
      with no undo path.
    * ``labeler_id`` is canonicalized at construction time (lower-cased,
      whitespace-trimmed) so common typos collapse to the same identity.
      Legacy corpora can be normalized on disk with
      :meth:`EventLabeler.migrate_corpus`.
    * The trace plot draws from ``shard._filtered_idx_cache`` when it
      exists (populated by ``_run_all``) so the displayed ΔF/F line is
      guaranteed to correspond to the events being labeled. When the
      cache is missing the labeler falls back to a default
      ``spatial_filtering`` recompute and logs a loud warning; if the
      trace can't be loaded at all the labeler refuses to enter
      drill view to prevent labeling-without-signal.
"""

from __future__ import annotations

# import
## batteries
import csv
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
## 3rd party
import numpy as np
import pandas as pd

# Public surface
__all__ = ["EventLabeler"]

# Module-level constants
CORPUS_VERSION: int = 2
CORPUS_COLUMNS: Tuple[str, ...] = (
    "corpus_version",
    "sample_id",
    "roi_id",
    "event_idx",
    "label",
    "labeler_id",
    "timestamp",
    "notes",
    "peak_amplitude",
    "fwhm_frames",
    "sampling_rate",
    "indicator",
    "microscope",
    "cell_type",
    "experiment_id",
    "wizards_staff_version",
)
VALID_LABELS: Tuple[str, ...] = ("True", "False", "Unsure")
ORDERINGS: Tuple[str, ...] = (
    "by_roi_then_time",
    "stratified",
    "amplitude_ascending",
    "amplitude_descending",
)

# Sentinel ``event_idx`` value used to mark trace-level review rows in
# the corpus (e.g. "this labeler skipped this trace"). Event indices for
# real, individually-labeled events are always >= 0, so the negative
# sentinel cleanly separates the two row classes without a schema bump.
TRACE_ACTION_EVENT_IDX: int = -1
# Canonical ``notes`` values for trace-level review rows. Skip records
# that the labeler reviewed the trace and chose not to label any events
# individually — distinct from "labeler never opened this trace". Reject
# records that the labeler reviewed the trace and bulk-rejected every
# unlabeled event on it (the per-event False rows still live alongside
# this record with notes='whole_trace_reject'; the trace-level row exists
# so the "was this trace reviewed at the trace level?" predicate is a
# single sentinel-row scan regardless of which trace-level action ran).
NOTE_TRACE_SKIP: str = "trace_skip"
NOTE_TRACE_REJECT: str = "trace_reject"
# Legacy per-event note marker used by reject_whole_trace BEFORE the
# trace-level sentinel row was added. Older corpora won't have the
# event_idx=-1 trace_reject row, so we use this on load to backfill
# the trace-level review record (own labeler) and the
# ``traces_touched_by_others`` set (other labelers).
NOTE_WHOLE_TRACE_REJECT: str = "whole_trace_reject"


def _wizards_staff_version() -> str:
    """Return the installed wizards_staff version string, or 'unknown'."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("wizards_staff")
        except PackageNotFoundError:
            return "unknown"
    except Exception:
        return "unknown"


def _canonicalize_labeler_id(raw: str) -> str:
    """
    Normalize a free-form labeler_id string for stable equality.

    Two labelers should not silently fork inter-rater agreement just
    because one of them typed an extra space or capitalized differently.
    The canonical form lower-cases and strips surrounding whitespace,
    and collapses internal runs of whitespace to a single underscore.

    The original string is *not* stored — the canonical form is what
    persists to the corpus. This means once a labeler_id is committed
    to a corpus row, the same human always resolves to the same id
    regardless of how they typed it on subsequent sessions.
    """
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    parts = s.split()
    return "_".join(parts) if parts else ""


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """
    Atomically write ``df`` to ``path`` as CSV.

    Writes to a uniquely-named temp file in the same directory, then uses
    :func:`os.replace` to swap it into place. ``os.replace`` is atomic on
    POSIX and on modern Windows for files on the same filesystem, so a
    crash either leaves the previous contents intact or installs the
    fully-written new file — never a half-written corpus.

    Args:
        df: DataFrame to serialize.
        path: Destination CSV path.
    """
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".event_labeler_", suffix=".csv.tmp", dir=parent
    )
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            # QUOTE_NONNUMERIC keeps the "True"/"False"/"Unsure" labels
            # round-trippable through a vanilla ``pd.read_csv`` — without
            # quoting, pandas infers the label column as bool whenever
            # only "True"/"False" appear in it.
            df.to_csv(fh, index=False, quoting=csv.QUOTE_NONNUMERIC)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup; never mask the original exception.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


class EventLabeler:
    """
    Hand-label calcium events on a single Wizards-Staff shard.

    The labeler reads detected events from
    ``shard._raw_peak_amplitude_data`` and ``shard._raw_fwhm_data``
    (see :func:`wizards_staff.wizards.cauldron._run_all` for the
    producer) and presents them to a biologist via a trace-first UI in
    the Jupyter notebook. The entry point for every ROI is a full-trace
    **overview** with each detected event marked by its current label
    state; the biologist either *investigates* the trace (per-event
    view) to vet events one at a time, rejects the whole trace, or
    skips to the next trace without writing anything (see the module
    docstring for the full key map).

    Labels are persisted to a canonical CSV at ``corpus_path`` after every
    user action. The corpus is keyed on the
    ``(sample_id, roi_id, event_idx, labeler_id)`` tuple, so re-labeling
    an event by the same labeler updates the existing row in place; labels
    by other labelers are preserved untouched. See the module docstring
    for the corpus schema.

    Args:
        shard: Wizards-Staff :class:`~wizards_staff.wizards.shard.Shard`
            with ``_raw_peak_amplitude_data`` and ``_raw_fwhm_data``
            populated by ``_run_all`` (or an equivalent producer).
        corpus_path: Path to the canonical CSV corpus, typically on shared
            storage. Created if it does not exist; appended to (and rows
            updated in place) on every label action.
        labeler_id: Identifier for the human labeler. Stored on every row
            for later inter-rater analysis.
        context: Per-session metadata stored on every row of the corpus.
            Recognized keys are ``sampling_rate``, ``indicator``,
            ``microscope``, ``cell_type``, ``experiment_id``. Unknown
            keys are accepted but ignored.
        window_scale: Width of the trace window centered on the current
            event, expressed as a multiple of the event's FWHM (in
            frames). Used as a display hint only — the window is
            clipped to the trace bounds. ``None`` (the default) means
            "use the indicator-aware default": if
            ``context["indicator"]`` is one of the named presets
            (GCaMP6f/6m/6s, GCaMP7f/7s, jGCaMP8f/8m/8s, jRGECO1a,
            jRCaMP1a, GCaMP3) the matching multiplier is used,
            otherwise the legacy default of 8.0. Passing an explicit
            float bypasses the presets entirely.
        ordering: One of ``"by_roi_then_time"`` (default; biologist-
            friendly), ``"stratified"`` (one event per amplitude quintile
            within each ROI), ``"amplitude_ascending"``, or
            ``"amplitude_descending"``.
        filtered_idx: Optional sequence mapping local ROI indices
            (the "Neuron" key in ``_raw_peak_amplitude_data`` — the raw
            event lists still use the legacy column name) to absolute
            component indices into ``dff_dat``. If ``None`` (default) and
            ``shard`` has the inputs needed for spatial filtering, the
            labeler will recompute it lazily on first plot. Provide
            explicitly when the shard's spatial filtering would be
            expensive to recompute or when running with mocked data.
    """

    CORPUS_VERSION: int = CORPUS_VERSION
    CORPUS_COLUMNS: Tuple[str, ...] = CORPUS_COLUMNS
    VALID_LABELS: Tuple[str, ...] = VALID_LABELS
    ORDERINGS: Tuple[str, ...] = ORDERINGS
    # Default window_scale used when the constructor argument is not
    # passed. Exposed as a class constant so ``_effective_window_scale``
    # can distinguish "user accepted the default" from "user passed a
    # value that happens to match the default" — only the former opts
    # into the indicator-aware override path.
    _DEFAULT_WINDOW_SCALE: float = 8.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        shard: Any,
        corpus_path: str,
        labeler_id: str,
        context: Optional[Dict[str, Any]] = None,
        window_scale: Optional[float] = None,
        ordering: str = "by_roi_then_time",
        filtered_idx: Optional[Sequence[int]] = None,
        quiet: bool = False,
    ) -> None:
        if not labeler_id or not isinstance(labeler_id, str):
            raise ValueError("labeler_id must be a non-empty string")
        if ordering not in self.ORDERINGS:
            raise ValueError(
                f"ordering must be one of {self.ORDERINGS}; got {ordering!r}"
            )
        # ``window_scale=None`` means "use the indicator-aware default";
        # an explicit numeric value bypasses the indicator presets in
        # ``_effective_window_scale``. Validation still rejects <= 0.
        window_scale_was_explicit = window_scale is not None
        if window_scale is None:
            window_scale = EventLabeler._DEFAULT_WINDOW_SCALE
        if window_scale <= 0:
            raise ValueError("window_scale must be > 0")

        # Canonicalize labeler_id so common typos ("Alice", "alice ",
        # "ALICE") collapse to the same identity. The canonical form is
        # what lands in the corpus.
        canonical = _canonicalize_labeler_id(labeler_id)
        if not canonical:
            raise ValueError("labeler_id must contain at least one non-whitespace character")
        if canonical != labeler_id:
            # Surface the rewrite so a labeler doesn't think their id
            # was silently changed. INFO not WARNING — this is the
            # designed behavior, not a bug.
            (getattr(shard, "_logger", None) or __import__("logging").getLogger(__name__)).info(
                f"EventLabeler: normalized labeler_id {labeler_id!r} -> {canonical!r}"
            )

        self.shard = shard
        self.corpus_path = os.path.abspath(corpus_path)
        self.labeler_id = canonical
        self._labeler_id_raw = labeler_id
        self.context: Dict[str, Any] = dict(context) if context else {}
        self.window_scale = float(window_scale)
        # Sentinel so ``_effective_window_scale`` can tell "user accepted
        # the default" from "user explicitly chose a value". Only the
        # default-acceptance path consults the indicator presets.
        self._window_scale_is_default = not window_scale_was_explicit
        self.ordering = ordering

        self._sample_id: str = str(getattr(shard, "sample_name", "unknown"))
        self._logger = getattr(shard, "_logger", None)
        if self._logger is None:
            import logging

            self._logger = logging.getLogger(__name__)

        # When quiet=True, the construction-time INFO logs (zero-event
        # ROIs, "skipped N events outside active Layer-2 bounds") are
        # downgraded to DEBUG. Used by :class:`MultiShardLabeler` for
        # probe-only constructions across every shard in a dataset
        # so the user doesn't get one INFO line per shard at startup.
        # WARNING-level logs (data-integrity issues like non-numeric
        # amplitudes or legacy labeler_id aliases) are NOT silenced —
        # those indicate genuine problems regardless of caller intent.
        self._quiet: bool = bool(quiet)

        self._wizards_staff_version = _wizards_staff_version()

        # Filtered-index mapping from local "Neuron" key -> absolute dff_dat row.
        self._filtered_idx: Optional[List[int]] = (
            [int(i) for i in filtered_idx] if filtered_idx is not None else None
        )

        # Build chronological per-ROI event list and apply ordering.
        self._events: List[Dict[str, Any]] = []
        self._build_event_list()
        self._rois_in_order: List[int] = self._unique_rois_in_order()
        self._apply_ordering()

        # In-memory label state for THIS labeler, THIS sample. Keyed by
        # (roi_id, event_idx). Loaded from corpus on init; updated on
        # every action; flushed to disk after every action.
        self._labels: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # Set of (roi_id, event_idx) keys touched by other labelers.
        # Used only to render a subtle indicator; the other labelers' label
        # values are intentionally NOT exposed to avoid biasing this user.
        self._touched_by_others: set = set()
        # Trace-level review actions for THIS labeler, keyed by roi_id.
        # A non-empty entry means the labeler explicitly reviewed this
        # trace at the trace level (e.g. ``skip_trace``) and chose not
        # to (or hasn't yet) label its events individually. This is the
        # canonical "was this trace reviewed?" record used downstream
        # for inter-rater agreement and sampling-bias correction.
        self._trace_actions: Dict[int, Dict[str, Any]] = {}
        # Set of roi_ids that other labelers have reviewed at the trace
        # level. As with ``_touched_by_others`` we intentionally do not
        # surface the other labeler's chosen action to avoid biasing.
        self._traces_touched_by_others: set = set()
        self._load_corpus()

        # Cursor into self._events.
        self._cursor: int = 0

        # ROI cursor for the trace-first overview. Indexes into
        # self._rois_in_order. Kept in sync with self._cursor on drill-
        # mode transitions (see _sync_roi_cursor_to_event).
        self._roi_cursor: int = 0

        # Resume-on-open: land on the first ROI with unfinished events
        # rather than always at index 0. Without this, a labeler who
        # has already reviewed (or rejected) ROIs 0..N-1 across earlier
        # sessions opens to a confusing "(previously rejected)"
        # overview at ROI 0 and has to manually press 'n' until they
        # find unreviewed work. The multi-shard wrapper's
        # ``_restore_child_state`` will overwrite this if the labeler
        # has a precise saved cursor for this sample, so explicit
        # navigation always wins over the resume heuristic.
        if self._rois_in_order and self._events:
            resume_idx = self._first_unreviewed_roi_cursor(0)
            if resume_idx is not None:
                self._roi_cursor = resume_idx
                roi_id = self._rois_in_order[self._roi_cursor]
                self._cursor = self._first_event_idx_for_roi(roi_id)

        # UI view state. ``"overview"`` shows the full ΔF/F trace for
        # the current ROI with every detected event marked by its label
        # state; ``"drill"`` shows the windowed per-event labeling UI.
        # The internal sentinel is still ``"drill"`` for backward
        # compatibility with callers that inspect ``self._view``; the
        # user-facing terminology was renamed to "investigate trace"
        # but the sentinel value is API surface that we don't churn.
        # Default to ``"overview"`` so display() lands the biologist
        # on the trace-first entry point described in the user guide.
        self._view: str = "overview"

        # Cached widgets / figure handles, populated on display().
        self._widgets: Dict[str, Any] = {}
        self._fig = None
        self._ax_trace = None
        self._ax_minimap = None
        self._ipy = None  # ipywidgets module handle (lazy)

        # _save reuses the DataFrame of "rows owned by other (sample_id,
        # labeler_id) tuples" between saves, keyed by the corpus file's
        # mtime + size. Invalidated whenever the file changes underneath
        # us (another labeler wrote) or whenever we ourselves rewrite
        # (mtime updates after our own os.replace). Keeps per-label save
        # latency flat as the shared corpus grows past ~10k rows.
        self._preserved_cache: Optional[pd.DataFrame] = None
        self._preserved_cache_key: Optional[Tuple[float, int]] = None

        # Reject-confirmation state. ``reject_whole_trace`` arms on the
        # first call and commits on the second consecutive call for the
        # same ROI; any other action (drill_in, skip_trace, label_current,
        # next_trace, prev_trace, back_to_overview) disarms it. This
        # prevents a stray ``r`` keystroke from silently False-labelling
        # every event on the trace without any feedback loop.
        self._reject_armed_roi: Optional[int] = None

        # Single-step undo snapshot for the most recent
        # ``reject_whole_trace`` *commit*. Holds the per-event label
        # state and trace-action state that existed BEFORE the bulk
        # rejection so :meth:`undo_trace_rejection` can restore them.
        # Cleared on the next state-mutating action (label_current,
        # skip_trace, another reject_whole_trace commit) — pure
        # navigation does NOT clear it, because a biologist who
        # rejects a trace, scrolls forward to the next one, and then
        # realizes they were wrong should still be able to undo.
        # The snapshot is in-memory only: it does NOT survive a
        # kernel restart or a labeler-instance recreation (e.g. when
        # the multi-image wrapper switches images and back), since
        # the corpus only carries the post-rejection state and we
        # would have no way to reconstruct the pre-rejection labels
        # for events that had no prior row. Documented in the
        # method docstring.
        self._last_reject_snapshot: Optional[Dict[str, Any]] = None

        # Optional callback invoked at the end of every successful
        # ``_save()`` (i.e. after any action that mutates label or
        # trace-action state). Used by :class:`MultiShardLabeler` to
        # auto-advance to the next image when the current one is fully
        # reviewed; not exercised in the single-shard UI. Must be
        # cheap and exception-safe — exceptions are logged at WARNING
        # but not propagated, so a buggy subscriber never blocks a
        # save from completing.
        self.on_state_change: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------------
    # Event list construction
    # ------------------------------------------------------------------
    def _build_event_list(self) -> None:
        """
        Walk the shard's raw event lists and build ``self._events``.

        Each event is a dict with keys ``roi_id`` (the local index used
        in the raw lists — stored under the legacy ``"Neuron"`` key in
        the raw rows), ``event_idx`` (chronological index within that
        ROI), ``peak_position`` (column index into the per-ROI trace),
        ``peak_amplitude``, ``fwhm_frames``, ``fwhm_back``,
        ``fwhm_fwd``.

        Skips:
            * ROIs with zero events (logged at INFO),
            * events with non-numeric or NaN/Inf amplitude (logged at
              WARNING),
            * events that fall outside the shard's currently-active
              Layer-2 amplitude / FWHM bounds (when ``_apply_event_filters``
              has stashed an active config). These events would be dropped
              from every per-event metric anyway, and labels can only
              narrow the surviving set further, so reviewing them is pure
              wasted effort. Aggregate skip count is logged at INFO.

        ``event_idx`` always stores the position in the raw amplitude
        list, NOT the position among surfaced events. This is load-
        bearing: the corpus key ``(sample_id, roi_id, event_idx,
        labeler_id)`` and Layer-3 label drops in
        ``_apply_event_filters`` both index positionally into the raw
        list, so renumbering surfaced events would silently misalign
        labels with the events they describe whenever the bounds change
        between labeling and refilter.
        """
        raw_peaks = list(getattr(self.shard, "_raw_peak_amplitude_data", []) or [])
        raw_fwhm = list(getattr(self.shard, "_raw_fwhm_data", []) or [])
        # Raw rows still carry the legacy "Neuron" key — that's the
        # producer's column name in spellbook / cauldron and is not part
        # of the labeler's public schema. We translate to roi_id below.
        fwhm_by_roi: Dict[Any, Dict[str, Any]] = {
            row.get("Neuron"): row for row in raw_fwhm
        }

        # Active Layer-2 bound configuration (stashed by
        # ``_apply_event_filters`` on every run/refilter). ``getattr``
        # with defaults keeps backward compatibility with shard-likes
        # that have not been through ``_apply_event_filters`` yet (e.g.
        # synthetic SimpleNamespace shards in tests, or shards loaded
        # from a pickle predating this field).
        active_filter = bool(getattr(self.shard, "_active_filter_events", False))
        min_amp = getattr(self.shard, "_active_min_event_amplitude", None)
        max_amp = getattr(self.shard, "_active_max_event_amplitude", None)
        min_fwhm = getattr(self.shard, "_active_min_event_fwhm", None)
        max_fwhm = getattr(self.shard, "_active_max_event_fwhm", None)
        apply_amp_filter = active_filter and (min_amp is not None or max_amp is not None)
        apply_fwhm_filter = active_filter and (min_fwhm is not None or max_fwhm is not None)
        n_skipped_bounds = 0

        for row in raw_peaks:
            roi_id_raw = row.get("Neuron")
            try:
                roi_id = int(roi_id_raw)
            except (TypeError, ValueError):
                self._logger.warning(
                    f"EventLabeler: skipping ROI with non-integer id "
                    f"{roi_id_raw!r} in {self._sample_id}"
                )
                continue

            amplitudes = list(row.get("Peak Amplitudes", []) or [])
            positions = list(row.get("Peak Positions", []) or [])
            if not amplitudes:
                # Suppressed under quiet mode (probe-only multi-shard
                # construction) so the wrapper doesn't emit one of these
                # per shard at startup. Kept at INFO for normal
                # single-shard use, where seeing "ROI N had no detected
                # events" once is genuinely useful diagnostic output.
                _zero_log = (
                    self._logger.debug if self._quiet else self._logger.info
                )
                _zero_log(
                    f"EventLabeler: ROI {roi_id} in "
                    f"{self._sample_id} has zero events; skipping."
                )
                continue

            fwhm_row = fwhm_by_roi.get(roi_id_raw, {}) or {}
            fwhm_values = list(fwhm_row.get("FWHM Values", []) or [])
            fwhm_back = list(fwhm_row.get("FWHM Backward Positions", []) or [])
            fwhm_fwd = list(fwhm_row.get("FWHM Forward Positions", []) or [])

            for raw_event_idx, amp in enumerate(amplitudes):
                try:
                    amp_f = float(amp)
                except (TypeError, ValueError):
                    self._logger.warning(
                        f"EventLabeler: skipping non-numeric amplitude "
                        f"sample={self._sample_id} roi={roi_id} "
                        f"event_idx={raw_event_idx}"
                    )
                    continue
                if not np.isfinite(amp_f):
                    self._logger.warning(
                        f"EventLabeler: skipping NaN/Inf amplitude "
                        f"sample={self._sample_id} roi={roi_id} "
                        f"event_idx={raw_event_idx}"
                    )
                    continue

                if raw_event_idx < len(positions):
                    try:
                        pos_i = int(positions[raw_event_idx])
                    except (TypeError, ValueError):
                        pos_i = -1
                else:
                    pos_i = -1

                if raw_event_idx < len(fwhm_values):
                    try:
                        fwhm_v = float(fwhm_values[raw_event_idx])
                    except (TypeError, ValueError):
                        fwhm_v = float("nan")
                else:
                    fwhm_v = float("nan")

                # Layer-2 bounds check — mirrors what
                # ``_apply_event_filters`` does, so the labeler surfaces
                # exactly the set of events that downstream metrics
                # actually keep. Treat non-finite FWHM as failing any
                # active FWHM bound (consistent with the cauldron-side
                # NaN scrub).
                if apply_amp_filter:
                    if min_amp is not None and amp_f < min_amp:
                        n_skipped_bounds += 1
                        continue
                    if max_amp is not None and amp_f > max_amp:
                        n_skipped_bounds += 1
                        continue
                if apply_fwhm_filter:
                    if not np.isfinite(fwhm_v):
                        n_skipped_bounds += 1
                        continue
                    if min_fwhm is not None and fwhm_v < min_fwhm:
                        n_skipped_bounds += 1
                        continue
                    if max_fwhm is not None and fwhm_v > max_fwhm:
                        n_skipped_bounds += 1
                        continue

                fb = (
                    int(fwhm_back[raw_event_idx])
                    if raw_event_idx < len(fwhm_back)
                    and fwhm_back[raw_event_idx] is not None
                    else None
                )
                ff = (
                    int(fwhm_fwd[raw_event_idx])
                    if raw_event_idx < len(fwhm_fwd)
                    and fwhm_fwd[raw_event_idx] is not None
                    else None
                )

                self._events.append(
                    {
                        "roi_id": roi_id,
                        "event_idx": raw_event_idx,
                        "peak_position": pos_i,
                        "peak_amplitude": amp_f,
                        "fwhm_frames": fwhm_v,
                        "fwhm_back": fb,
                        "fwhm_fwd": ff,
                    }
                )

        if n_skipped_bounds > 0:
            # Suppressed under quiet mode (multi-shard probe-only
            # constructions). Kept at INFO for normal single-shard use:
            # "sample X had 69 events outside the active bounds" is
            # information the labeler typically wants to see ONCE per
            # shard they actually open. The multi-image wrapper opens
            # only one shard at a time for review and emits this for
            # that shard non-quietly, so the message still surfaces
            # exactly when it's actionable.
            _bounds_log = (
                self._logger.debug if self._quiet else self._logger.info
            )
            _bounds_log(
                f"EventLabeler: sample={self._sample_id} skipped "
                f"{n_skipped_bounds} event(s) outside the active Layer-2 "
                f"bounds "
                f"(amplitude=[{min_amp}, {max_amp}], "
                f"FWHM=[{min_fwhm}, {max_fwhm}]). These would be dropped "
                f"by every per-event metric anyway; labels can only "
                f"narrow the surviving set further."
            )

    def _unique_rois_in_order(self) -> List[int]:
        """Return the unique ROI ids in first-encounter order."""
        seen: List[int] = []
        seen_set: set = set()
        for ev in self._events:
            n = ev["roi_id"]
            if n not in seen_set:
                seen.append(n)
                seen_set.add(n)
        return seen

    def _apply_ordering(self) -> None:
        """Reorder ``self._events`` according to ``self.ordering``."""
        if self.ordering == "by_roi_then_time":
            # Already in this order from _build_event_list (raw lists are
            # per-ROI in chronological order). Do an explicit sort on
            # (roi_position_in_rois_in_order, event_idx) to be robust
            # against future changes to the raw layout.
            roi_rank = {n: i for i, n in enumerate(self._rois_in_order)}
            self._events.sort(
                key=lambda e: (roi_rank[e["roi_id"]], e["event_idx"])
            )
        elif self.ordering == "amplitude_ascending":
            self._events.sort(key=lambda e: e["peak_amplitude"])
        elif self.ordering == "amplitude_descending":
            self._events.sort(key=lambda e: -e["peak_amplitude"])
        elif self.ordering == "stratified":
            self._events = self._stratified_order(self._events)
        else:  # pragma: no cover  — guarded in __init__
            raise ValueError(self.ordering)

    @staticmethod
    def _stratified_order(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Produce a stratified-by-amplitude ordering, walking ROIs.

        Within each ROI the events are bucketed into amplitude quintiles
        and emitted round-robin (highest quintile first within each round)
        until exhausted. This biases early labels toward sampling the full
        amplitude distribution of every ROI, which is useful for
        threshold-calibration runs.
        """
        by_roi: Dict[int, List[Dict[str, Any]]] = {}
        order: List[int] = []
        for ev in events:
            n = ev["roi_id"]
            if n not in by_roi:
                by_roi[n] = []
                order.append(n)
            by_roi[n].append(ev)

        out: List[Dict[str, Any]] = []
        for n in order:
            group = sorted(by_roi[n], key=lambda e: e["peak_amplitude"])
            if not group:
                continue
            n_quintiles = min(5, len(group))
            buckets: List[List[Dict[str, Any]]] = [[] for _ in range(n_quintiles)]
            for i, ev in enumerate(group):
                # Even-spread bucket assignment: index proportional to rank.
                b = min(int(i * n_quintiles / len(group)), n_quintiles - 1)
                buckets[b].append(ev)
            # Round-robin highest -> lowest quintile so a calibration run
            # that stops early still sees the full amplitude range.
            buckets_rr = list(reversed(buckets))
            while any(buckets_rr):
                for b in buckets_rr:
                    if b:
                        # Pop from the end so the *largest* amplitude within
                        # each bucket is emitted first; combined with the
                        # outer "highest-quintile-first" ordering this gives
                        # a calibration walk that surfaces the strongest
                        # candidates immediately.
                        out.append(b.pop())
        return out

    # ------------------------------------------------------------------
    # Corpus I/O
    # ------------------------------------------------------------------
    def _empty_corpus(self) -> pd.DataFrame:
        """Return an empty corpus DataFrame with the schema columns."""
        return pd.DataFrame({c: pd.Series(dtype="object") for c in self.CORPUS_COLUMNS})

    def _read_corpus_raw(self) -> pd.DataFrame:
        """
        Load the corpus CSV from disk, validating its version.

        Returns an empty (schema-only) DataFrame if the file does not
        exist. Raises ``RuntimeError`` if the file exists but contains
        rows whose ``corpus_version`` does not match :data:`CORPUS_VERSION`
        — silently overwriting a different-version corpus would corrupt
        accumulated labels from other sessions, so the labeler refuses to
        proceed.
        """
        if not os.path.exists(self.corpus_path):
            return self._empty_corpus()

        try:
            df = pd.read_csv(self.corpus_path)
        except pd.errors.EmptyDataError:
            return self._empty_corpus()

        if df.empty:
            return self._empty_corpus()

        if "corpus_version" not in df.columns:
            raise RuntimeError(
                f"Corpus at {self.corpus_path!r} is missing the "
                f"'corpus_version' column. This labeler expects "
                f"corpus_version={self.CORPUS_VERSION}. Refusing to "
                f"overwrite. Move or migrate the existing file before "
                f"continuing."
            )

        try:
            versions = pd.to_numeric(df["corpus_version"], errors="raise")
        except Exception as exc:
            raise RuntimeError(
                f"Corpus at {self.corpus_path!r} has non-numeric values "
                f"in 'corpus_version'. Refusing to proceed."
            ) from exc

        bad = versions[versions != self.CORPUS_VERSION]
        if not bad.empty:
            unique = sorted(set(int(v) for v in bad.unique()))
            raise RuntimeError(
                f"Corpus at {self.corpus_path!r} contains "
                f"corpus_version={unique} but this labeler only "
                f"understands corpus_version={self.CORPUS_VERSION}. "
                f"Migrate the corpus to v{self.CORPUS_VERSION} (or point "
                f"corpus_path at a new file) before re-running."
            )

        # Ensure all schema columns are present (older v1 files written by
        # earlier callers may have a subset; fill the rest with empty).
        for col in self.CORPUS_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[list(self.CORPUS_COLUMNS)].copy()

    def _load_corpus(self) -> None:
        """
        Populate the in-memory state from disk.

        Rows with ``event_idx >= 0`` are event-level labels and feed
        ``self._labels`` / ``self._touched_by_others``. Rows with
        ``event_idx == TRACE_ACTION_EVENT_IDX`` are trace-level review
        records (e.g. skip) and feed ``self._trace_actions`` /
        ``self._traces_touched_by_others``.
        """
        df = self._read_corpus_raw()
        if df.empty:
            return

        sample_mask = df["sample_id"].astype(str) == self._sample_id
        if not sample_mask.any():
            return
        sample_df = df.loc[sample_mask]

        # Warn on near-collisions: any disk labeler_id that canonicalizes
        # to our id but isn't byte-equal is a legacy mixed-case row that
        # *should* belong to us. We still treat it as ours via canonical
        # comparison, but surface the mismatch so an operator can decide
        # whether to migrate (re-canonicalize on disk).
        disk_ids = (
            sample_df["labeler_id"].dropna().astype(str).unique().tolist()
            if "labeler_id" in sample_df.columns
            else []
        )
        legacy_aliases = [
            d for d in disk_ids
            if d != self.labeler_id
            and _canonicalize_labeler_id(d) == self.labeler_id
        ]
        if legacy_aliases:
            self._logger.warning(
                f"EventLabeler: corpus contains labeler_id values "
                f"{legacy_aliases!r} that canonicalize to {self.labeler_id!r}; "
                f"treating them as the same labeler. Run "
                f"EventLabeler.migrate_corpus(...) to normalize on disk."
            )

        for _, row in sample_df.iterrows():
            try:
                roi_id = int(row["roi_id"])
                event_idx = int(row["event_idx"])
            except (TypeError, ValueError):
                continue
            row_labeler_raw = str(row.get("labeler_id", ""))
            row_labeler = _canonicalize_labeler_id(row_labeler_raw)
            row_notes = (
                "" if pd.isna(row.get("notes")) else str(row.get("notes", ""))
            )
            row_timestamp = (
                ""
                if pd.isna(row.get("timestamp"))
                else str(row.get("timestamp", ""))
            )

            if event_idx == TRACE_ACTION_EVENT_IDX:
                # Trace-level review row.
                if row_labeler == self.labeler_id:
                    self._trace_actions[roi_id] = {
                        "notes": row_notes or NOTE_TRACE_SKIP,
                        "timestamp": row_timestamp,
                    }
                else:
                    self._traces_touched_by_others.add(roi_id)
                continue

            key = (roi_id, event_idx)
            if row_labeler == self.labeler_id:
                self._labels[key] = {
                    "label": str(row.get("label", "")),
                    "notes": row_notes,
                    "timestamp": row_timestamp,
                }
            else:
                self._touched_by_others.add(key)

            # Backward-compat: corpora written before the trace-level
            # sentinel row was introduced encode "reject whole trace"
            # only as N event rows with notes='whole_trace_reject'. To
            # keep the trace-level "was reviewed?" predicate symmetric
            # for legacy data, synthesize a trace_action entry from
            # those notes if no explicit sentinel row exists yet.
            if row_notes == NOTE_WHOLE_TRACE_REJECT:
                if row_labeler == self.labeler_id:
                    self._trace_actions.setdefault(
                        roi_id,
                        {
                            "notes": NOTE_TRACE_REJECT,
                            "timestamp": row_timestamp,
                        },
                    )
                else:
                    self._traces_touched_by_others.add(roi_id)

    def _row_for_trace_action(
        self,
        roi_id: int,
        notes: str,
        timestamp: str,
    ) -> Dict[str, Any]:
        """
        Build a corpus row representing a trace-level review action.

        Trace-level rows use ``event_idx = TRACE_ACTION_EVENT_IDX`` (a
        negative sentinel) and an empty ``label``; the action itself
        (e.g. ``trace_skip``) lives in the ``notes`` column. This keeps
        the schema unchanged so older corpus readers continue to work —
        they just see a row with an out-of-range event index that they
        can filter out with ``event_idx >= 0``.
        """
        ctx = self.context
        return {
            "corpus_version": self.CORPUS_VERSION,
            "sample_id": self._sample_id,
            "roi_id": int(roi_id),
            "event_idx": int(TRACE_ACTION_EVENT_IDX),
            "label": "",
            "labeler_id": self.labeler_id,
            "timestamp": timestamp,
            "notes": notes,
            "peak_amplitude": float("nan"),
            "fwhm_frames": float("nan"),
            "sampling_rate": ctx.get("sampling_rate", ""),
            "indicator": ctx.get("indicator", ""),
            "microscope": ctx.get("microscope", ""),
            "cell_type": ctx.get("cell_type", ""),
            "experiment_id": ctx.get("experiment_id", ""),
            "wizards_staff_version": self._wizards_staff_version,
        }

    def _row_for_label(
        self,
        roi_id: int,
        event_idx: int,
        label: str,
        notes: str,
        timestamp: str,
    ) -> Dict[str, Any]:
        """Build a single corpus row dict for the given label state."""
        # Look up the event for amplitude/FWHM context (cheap linear scan
        # — corpora are small enough that the dict overhead isn't worth
        # it). Fall back to NaN if the event isn't found.
        peak_amp: float = float("nan")
        fwhm: float = float("nan")
        for ev in self._events:
            if ev["roi_id"] == roi_id and ev["event_idx"] == event_idx:
                peak_amp = float(ev["peak_amplitude"])
                fwhm = float(ev["fwhm_frames"])
                break

        ctx = self.context
        return {
            "corpus_version": self.CORPUS_VERSION,
            "sample_id": self._sample_id,
            "roi_id": int(roi_id),
            "event_idx": int(event_idx),
            "label": label,
            "labeler_id": self.labeler_id,
            "timestamp": timestamp,
            "notes": notes,
            "peak_amplitude": peak_amp,
            "fwhm_frames": fwhm,
            "sampling_rate": ctx.get("sampling_rate", ""),
            "indicator": ctx.get("indicator", ""),
            "microscope": ctx.get("microscope", ""),
            "cell_type": ctx.get("cell_type", ""),
            "experiment_id": ctx.get("experiment_id", ""),
            "wizards_staff_version": self._wizards_staff_version,
        }

    def _corpus_fs_key(self) -> Optional[Tuple[float, int]]:
        """
        Cache key for the on-disk corpus: (mtime_ns, size_bytes).

        Returns ``None`` if the file does not exist or stat() fails. The
        key is used to short-circuit the full re-read in ``_save`` when
        nothing has changed on disk since our last write.
        """
        try:
            st = os.stat(self.corpus_path)
        except OSError:
            return None
        return (float(st.st_mtime_ns), int(st.st_size))

    def _refresh_other_labeler_signals(self, existing: pd.DataFrame) -> None:
        """
        Recompute ``_touched_by_others`` / ``_traces_touched_by_others``
        from ``existing`` so a long-running session reflects other
        labelers' concurrent writes.

        Called from ``_save`` after the (potentially fresh) corpus read.
        We do NOT touch ``_labels`` or ``_trace_actions`` — those are
        owned by this session and must not be clobbered by stale
        on-disk state during last-write-wins concurrency.
        """
        if existing.empty:
            self._touched_by_others = set()
            self._traces_touched_by_others = set()
            return
        sample_mask = existing["sample_id"].astype(str) == self._sample_id
        if not sample_mask.any():
            self._touched_by_others = set()
            self._traces_touched_by_others = set()
            return
        sample_df = existing.loc[sample_mask]
        disk_canon = (
            sample_df["labeler_id"].astype(str).map(_canonicalize_labeler_id)
        )
        others_df = sample_df.loc[disk_canon != self.labeler_id]

        touched: set = set()
        traces_touched: set = set()
        for _, row in others_df.iterrows():
            try:
                roi_id = int(row["roi_id"])
                event_idx = int(row["event_idx"])
            except (TypeError, ValueError):
                continue
            row_notes = "" if pd.isna(row.get("notes")) else str(row.get("notes", ""))
            if event_idx == TRACE_ACTION_EVENT_IDX:
                traces_touched.add(roi_id)
            else:
                touched.add((roi_id, event_idx))
                if row_notes == NOTE_WHOLE_TRACE_REJECT:
                    traces_touched.add(roi_id)
        self._touched_by_others = touched
        self._traces_touched_by_others = traces_touched

    def _save(self) -> None:
        """
        Atomically rewrite the corpus CSV, preserving rows from other
        labelers and other samples.

        Concurrent labelers on shared storage get last-write-wins
        semantics; this is documented as accepted for v1. We refresh
        ``_touched_by_others`` / ``_traces_touched_by_others`` from the
        same read so the audit hints in the UI don't go stale during
        long sessions.

        Performance: the "rows owned by another (sample_id, labeler_id)"
        slice of the corpus is reused between saves via an mtime+size
        cache, so we don't re-parse a multi-thousand-row CSV on every
        click as the shared corpus grows.
        """
        fs_key = self._corpus_fs_key()
        existing: Optional[pd.DataFrame] = None
        preserved: Optional[pd.DataFrame] = None

        if (
            self._preserved_cache is not None
            and self._preserved_cache_key is not None
            and fs_key is not None
            and fs_key == self._preserved_cache_key
        ):
            # Hot path: file hasn't changed since our last write, so the
            # preserved slice is still valid. We still need to refresh
            # the other-labeler hints from somewhere — but they only
            # change when ``existing`` changes, which hasn't happened.
            preserved = self._preserved_cache.copy()

        if preserved is None:
            existing = self._read_corpus_raw()
            # Drop our (sample_id, labeler_id) rows from the loaded
            # corpus — we'll reinstate them from in-memory state below.
            # This makes the CSV rewrite an idempotent function of
            # (other-labelers' rows, other-samples' rows, our in-memory
            # labels).
            if not existing.empty:
                # Canonicalize labeler_id for the comparison so that
                # legacy mixed-case / whitespace-padded rows owned by
                # this same human are recognized as ours (and therefore
                # re-canonicalized on the rewrite below).
                disk_canon = (
                    existing["labeler_id"].astype(str).map(_canonicalize_labeler_id)
                )
                keep_mask = ~(
                    (existing["sample_id"].astype(str) == self._sample_id)
                    & (disk_canon == self.labeler_id)
                )
                preserved = existing.loc[keep_mask].copy()
            else:
                preserved = self._empty_corpus()
            # Refresh other-labeler hints from the (just-read) existing
            # frame so a colleague's concurrent writes surface in the UI.
            self._refresh_other_labeler_signals(existing)

        new_rows: List[Dict[str, Any]] = []
        for (roi_id, event_idx), entry in self._labels.items():
            new_rows.append(
                self._row_for_label(
                    roi_id=roi_id,
                    event_idx=event_idx,
                    label=entry.get("label", ""),
                    notes=entry.get("notes", "") or "",
                    timestamp=entry.get("timestamp", "") or "",
                )
            )
        for roi_id, entry in self._trace_actions.items():
            new_rows.append(
                self._row_for_trace_action(
                    roi_id=roi_id,
                    notes=entry.get("notes", "") or NOTE_TRACE_SKIP,
                    timestamp=entry.get("timestamp", "") or "",
                )
            )

        if new_rows:
            new_df = pd.DataFrame(new_rows, columns=list(self.CORPUS_COLUMNS))
            out = pd.concat([preserved, new_df], ignore_index=True)
        else:
            out = preserved

        out = out[list(self.CORPUS_COLUMNS)]
        _atomic_write_csv(out, self.corpus_path)

        # Refresh the preserved-rows cache key to the freshly written
        # file's fingerprint so the next save can short-circuit the
        # read. We keep the same ``preserved`` DataFrame since by
        # construction it contains exactly the rows that aren't ours.
        new_key = self._corpus_fs_key()
        if new_key is not None:
            self._preserved_cache = preserved.copy()
            self._preserved_cache_key = new_key

        if self.on_state_change is not None:
            try:
                self.on_state_change()
            except Exception as exc:  # pragma: no cover  — best-effort hook
                self._logger.warning(
                    f"EventLabeler.on_state_change callback raised "
                    f"{type(exc).__name__}: {exc}. Save itself succeeded; "
                    f"ignoring the callback failure."
                )

    # ------------------------------------------------------------------
    # Public data API
    # ------------------------------------------------------------------
    @classmethod
    def migrate_corpus(
        cls,
        in_path: str,
        out_path: str,
        to_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Migrate a labels corpus from an older schema to the current one.

        The migration is a pipeline of version-to-version transforms; this
        scaffold ships with the v1 -> v2 transform (label canonicalization
        and labeler_id normalization). When :data:`CORPUS_VERSION` bumps
        in the future, add a new ``_migrate_v<N>_to_v<N+1>`` staticmethod
        and the chain will pick it up automatically.

        The function is **non-destructive**: ``in_path`` is read but never
        modified, and ``out_path`` is written via the same atomic-rename
        machinery as the live labeler. If ``in_path == out_path`` the
        original is replaced atomically.

        Args:
            in_path: Path to the source corpus CSV.
            out_path: Path to write the migrated CSV. May equal
                ``in_path`` for an in-place migration.
            to_version: Target ``corpus_version`` for the output. Defaults
                to :data:`CORPUS_VERSION` (the version this labeler
                writes). Must be >= the source version.

        Returns:
            Dict with keys:
                ``from_version`` — detected source version (or ``None``
                if the source was empty / missing a version column);
                ``to_version`` — version actually written;
                ``rows_in`` — rows read;
                ``rows_out`` — rows written;
                ``labeler_id_renames`` — count of rows whose labeler_id
                was canonicalized.

        Raises:
            RuntimeError: If the source contains mixed corpus_version
                values or a version greater than the requested
                ``to_version``.
        """
        if to_version is None:
            to_version = cls.CORPUS_VERSION
        if not os.path.exists(in_path):
            return {
                "from_version": None,
                "to_version": to_version,
                "rows_in": 0,
                "rows_out": 0,
                "labeler_id_renames": 0,
            }

        try:
            df = pd.read_csv(in_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()

        rows_in = len(df)

        from_version: Optional[int] = None
        if not df.empty and "corpus_version" in df.columns:
            versions = pd.to_numeric(df["corpus_version"], errors="raise")
            unique_versions = sorted(set(int(v) for v in versions.unique()))
            if len(unique_versions) > 1:
                raise RuntimeError(
                    f"Corpus at {in_path!r} contains mixed corpus_version "
                    f"values {unique_versions}; cannot migrate. Split by "
                    f"version and migrate each subset separately."
                )
            from_version = unique_versions[0]
            if from_version > to_version:
                raise RuntimeError(
                    f"Corpus at {in_path!r} has corpus_version="
                    f"{from_version} which is newer than the requested "
                    f"to_version={to_version}; refusing to downgrade."
                )

        renames = 0
        if not df.empty and "labeler_id" in df.columns:
            original = df["labeler_id"].astype(str)
            canonical = original.map(_canonicalize_labeler_id)
            renames = int((original != canonical).sum())
            df["labeler_id"] = canonical

        # Bring the schema up to the target column set, filling missing
        # cells with empty strings. Drop any columns that aren't part of
        # the canonical schema (preserves forward-compat by ignoring
        # unknown columns rather than failing the migration).
        if not df.empty:
            for col in CORPUS_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            df = df[list(CORPUS_COLUMNS)]
            df["corpus_version"] = int(to_version)
        else:
            df = pd.DataFrame({c: pd.Series(dtype="object") for c in CORPUS_COLUMNS})

        _atomic_write_csv(df, out_path)

        return {
            "from_version": from_version,
            "to_version": int(to_version),
            "rows_in": rows_in,
            "rows_out": len(df),
            "labeler_id_renames": renames,
        }

    @classmethod
    def load_corpus(cls, corpus_path: str) -> pd.DataFrame:
        """
        Load a corpus CSV with the column dtypes the labeler intended.

        Reading the corpus with a vanilla :func:`pandas.read_csv` is mostly
        fine, but the ``label`` column is silently inferred as ``bool``
        whenever the file happens to contain only ``"True"`` / ``"False"``
        values (a common case for small calibration runs). This helper
        reads the file with explicit dtypes so the ``label`` column is
        always returned as ``str``, ``corpus_version`` as ``int``, and the
        identity columns as ``int``.

        Args:
            corpus_path: Path to the canonical corpus CSV.

        Returns:
            DataFrame with the canonical schema, suitable for stratified
            calibration analysis or downstream classifier training.
        """
        if not os.path.exists(corpus_path):
            return pd.DataFrame({c: pd.Series(dtype="object") for c in CORPUS_COLUMNS})
        df = pd.read_csv(
            corpus_path,
            dtype={
                "label": str,
                "labeler_id": str,
                "sample_id": str,
                "notes": str,
                "indicator": str,
                "microscope": str,
                "cell_type": str,
                "experiment_id": str,
                "wizards_staff_version": str,
            },
        )
        return df

    def export_labels(self) -> pd.DataFrame:
        """
        Return a DataFrame of this session's labels for this shard.

        The returned frame uses the same schema as the corpus CSV. It
        contains only rows for ``self._sample_id`` and
        ``self.labeler_id`` — including any trace-level review rows
        (``event_idx == TRACE_ACTION_EVENT_IDX``) written by
        :meth:`skip_trace`. Filter on ``event_idx >= 0`` if you want
        only event-level labels.

        Returns:
            DataFrame with columns matching :data:`CORPUS_COLUMNS`.
        """
        rows: List[Dict[str, Any]] = []
        for (roi_id, event_idx), entry in self._labels.items():
            rows.append(
                self._row_for_label(
                    roi_id=roi_id,
                    event_idx=event_idx,
                    label=entry.get("label", ""),
                    notes=entry.get("notes", "") or "",
                    timestamp=entry.get("timestamp", "") or "",
                )
            )
        for roi_id, entry in self._trace_actions.items():
            rows.append(
                self._row_for_trace_action(
                    roi_id=roi_id,
                    notes=entry.get("notes", "") or NOTE_TRACE_SKIP,
                    timestamp=entry.get("timestamp", "") or "",
                )
            )
        if not rows:
            return self._empty_corpus()
        return pd.DataFrame(rows, columns=list(self.CORPUS_COLUMNS))

    @property
    def events(self) -> List[Dict[str, Any]]:
        """Read-only view of the ordered event list."""
        return list(self._events)

    @property
    def total_events(self) -> int:
        return len(self._events)

    @property
    def labeled_count(self) -> int:
        return sum(1 for v in self._labels.values() if v.get("label"))

    @property
    def sample_id(self) -> str:
        """Read-only access to the sample id this labeler is bound to."""
        return self._sample_id

    @property
    def unfinished_count(self) -> int:
        """
        Count of events on this shard that this labeler has neither
        directly labeled nor implicitly accepted via a trace-level
        review (skip / reject) of their ROI.

        Used by the multi-image wrapper for "Image N of M reviewed"
        progress chips and for auto-advance-on-completion. Computed
        live from in-memory state; cheap (single pass over events).
        """
        if not self._events:
            return 0
        reviewed_rois: set = set(self._trace_actions.keys())
        unfinished = 0
        for ev in self._events:
            key = (ev["roi_id"], ev["event_idx"])
            entry = self._labels.get(key)
            if entry is not None and entry.get("label"):
                continue
            if ev["roi_id"] in reviewed_rois:
                continue
            unfinished += 1
        return unfinished

    @property
    def is_complete(self) -> bool:
        """
        True when every detectable event on this shard has been either
        directly labeled by this labeler or implicitly accepted via a
        trace-level review of its ROI (skip or reject).

        Empty shards (zero labelable events) are trivially complete.
        """
        return self.unfinished_count == 0

    def _roi_unfinished_count(self, roi_id: int) -> int:
        """
        Count events on ``roi_id`` that this labeler has not yet
        reviewed (directly via a label, or implicitly via a
        trace-level skip / reject of the ROI).

        Used to decide whether ``roi_id`` should be skipped during
        post-action auto-advance and resume-on-open. A trace-level
        review (skip or reject) collapses the count to 0 regardless
        of any individual event labels — which is correct: the whole
        point of skip / reject is "I've decided about this whole
        trace; don't make me look at it again."
        """
        if roi_id in self._trace_actions:
            return 0
        count = 0
        for ev in self._events:
            if ev["roi_id"] != roi_id:
                continue
            key = (roi_id, ev["event_idx"])
            entry = self._labels.get(key)
            if entry is not None and entry.get("label"):
                continue
            count += 1
        return count

    def _first_unreviewed_roi_cursor(
        self, start_inclusive: int = 0
    ) -> Optional[int]:
        """
        Walk ``_rois_in_order`` forward looking for the first index
        whose ROI still has unfinished events for this labeler.

        Returns ``None`` if every ROI from ``start_inclusive`` onward
        is fully reviewed. Callers that need a forward landing spot
        in that case should fall back to the existing
        next-in-order behavior; the multi-shard wrapper detects
        ``is_complete`` and auto-advances to the next image, so the
        "all reviewed" case isn't normally hit during productive
        labeling.
        """
        for i in range(start_inclusive, len(self._rois_in_order)):
            if self._roi_unfinished_count(self._rois_in_order[i]) > 0:
                return i
        return None

    # ------------------------------------------------------------------
    # Labeling actions (callable from tests as well as UI)
    # ------------------------------------------------------------------
    def label_current(self, label: str, notes: str = "") -> None:
        """
        Record a label for the event under the cursor and advance.

        Behaviour by view:
            * **Per-event view** (``self._view == "drill"`` — the
              internal sentinel still uses the legacy name): the label
              is recorded for the event under ``self._cursor``, then
              the cursor advances by one. If this label was on the
              last event of the current ROI the labeler returns to
              the trace overview (snapped to the next ROI when one
              exists) rather than silently crossing into a different
              trace.
            * **Overview view**: labeling an individual event from the
              overview is a UX-level "investigate this trace and label
              this event" shortcut. The labeler implicitly switches to
              the per-event view first (anchored to the current ROI's
              first event), so the label always lands on a
              deterministic, user-visible event. Without this, a
              programmatic call would silently advance ``self._cursor``
              across ROI boundaries.

        Args:
            label: One of :data:`VALID_LABELS`.
            notes: Optional free-text annotation stored with the label.
        """
        if label not in self.VALID_LABELS:
            raise ValueError(
                f"label must be one of {self.VALID_LABELS}; got {label!r}"
            )
        if not self._events:
            self._logger.info("EventLabeler: no events to label.")
            return
        # Any label action disarms a pending reject_whole_trace: the
        # biologist's intent has clearly shifted from "this whole
        # trace is junk" to "let me look at individual events".
        self._reject_armed_roi = None
        # Labeling an event also closes the undo window for the
        # most-recent trace rejection — once the biologist starts
        # editing labels by hand, "undo last reject" becomes
        # ambiguous (does it apply on top of the new edits, or
        # blow them away?), so we drop the snapshot.
        self._last_reject_snapshot = None
        if self._view != "drill":
            # Implicit "investigate trace" so the label lands on a
            # deterministic event (the first event of the current ROI)
            # instead of whatever stale _cursor value happened to be
            # lying around.
            self.investigate_trace()
            # ``investigate_trace`` may refuse (e.g. missing trace) and
            # stay in overview view. In that case label_current can't
            # operate safely — bail without a fallback so we don't
            # silently label whatever event ``_cursor`` is pointing at.
            if self._view != "drill":
                self._logger.warning(
                    "EventLabeler: label_current() from overview was "
                    "refused because investigate_trace() could not open "
                    "the per-event view (likely missing ΔF/F trace)."
                )
                return
        ev = self._events[self._cursor]
        key = (ev["roi_id"], ev["event_idx"])
        self._labels[key] = {
            "label": label,
            "notes": notes or "",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._save()
        self._advance_event(+1)

    def reject_whole_trace(self, *, confirm: bool = False) -> None:
        """
        Label every still-unlabeled event on the current ROI's trace
        as False.

        **Two-press confirmation.** Because this action is destructive
        (it can label 10+ events False in a single keystroke), the
        default behavior is two-step:

            * First call: arms the action for the current ROI and
              returns without writing anything. The overview header
              and details panel advertise the armed state.
            * Second call (within the same overview, no intervening
              action): commits the rejection.

        Any other navigation or labeling action (``drill_in``,
        ``skip_trace``, ``label_current``, ``next_trace``,
        ``prev_trace``, ``back_to_overview``) disarms the pending
        rejection — the biologist must explicitly press ``r`` twice
        in a row to commit. ``confirm=True`` bypasses the gate (used
        by tests and by callers that have already prompted the user).

        Named "reject whole trace" because the ROI itself isn't being
        marked bad here — only every detected event on that ROI's
        ΔF/F trace is being labeled False. Whole-ROI rejection
        (marking the component itself bad) belongs in the
        outlier-detection layer, not in event labeling.

        The note ``"whole_trace_reject"`` is recorded so this bulk
        action can be distinguished from individually-rejected events
        later. After the bulk write, the cursor advances to the first
        event on the next ROI (or to the end of the queue if none
        remains).

        **Undo.** The most recent successful commit is reversible via
        :meth:`undo_trace_rejection`, which restores every event's
        prior label state and the prior trace-action sentinel for
        that ROI. The undo window closes as soon as the labeler
        performs another state-mutating action (``label_current``,
        ``skip_trace``, or another ``reject_whole_trace`` commit).
        Pure navigation (``next_trace`` / ``prev_trace`` /
        ``investigate_trace`` / ``back_to_overview``) does not close
        it, so a labeler can scroll forward and still take the
        rejection back. The snapshot is in-memory only and does NOT
        survive a kernel restart or a labeler-instance recreation
        (e.g. switching images in the multi-shard wrapper and back).
        """
        if not self._events:
            return
        current_roi = self._events[self._cursor]["roi_id"]

        # Confirmation gate. Disabled when ``confirm=True``; otherwise
        # the first press arms and the second consecutive press commits.
        if not confirm:
            if self._reject_armed_roi != current_roi:
                self._reject_armed_roi = current_roi
                self._logger.info(
                    f"EventLabeler: reject_whole_trace ARMED for ROI "
                    f"{current_roi} ({self._sample_id}). Press 'r' (or "
                    f"click 'Reject whole trace') again to confirm; "
                    f"any other action cancels."
                )
                self._refresh_ui()
                return
        # Confirmation accepted (or bypassed) — clear the arm and
        # commit.
        self._reject_armed_roi = None
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Capture the pre-mutation state so a subsequent
        # ``undo_trace_rejection`` can fully restore. We snapshot
        # every event on this ROI (not just the ones we're about to
        # mutate), because undo also needs to know which events were
        # explicitly skipped vs. unlabeled in the prior state — the
        # only sound restore is "put each event back to whatever
        # entry it had before this commit, and remove the entry if
        # there wasn't one". The snapshot replaces any earlier one;
        # we only support a single level of undo (the most recent
        # rejection), which is the simplest correct contract.
        events_before: List[Tuple[int, Optional[Dict[str, Any]]]] = []
        for ev in self._events:
            if ev["roi_id"] != current_roi:
                continue
            key = (ev["roi_id"], ev["event_idx"])
            prior = self._labels.get(key)
            events_before.append(
                (
                    ev["event_idx"],
                    dict(prior) if prior is not None else None,
                )
            )
        self._last_reject_snapshot = {
            "roi_id": current_roi,
            "events_before": events_before,
            "trace_action_before": (
                dict(self._trace_actions[current_roi])
                if current_roi in self._trace_actions
                else None
            ),
            "cursor_before": int(self._cursor),
            "roi_cursor_before": int(self._roi_cursor),
            "view_before": str(self._view),
        }

        touched = 0
        for ev in self._events:
            if ev["roi_id"] != current_roi:
                continue
            key = (ev["roi_id"], ev["event_idx"])
            if key in self._labels and self._labels[key].get("label"):
                continue
            self._labels[key] = {
                "label": "False",
                "notes": "whole_trace_reject",
                "timestamp": ts,
            }
            touched += 1
        # Record the trace-level review intent so the "was this trace
        # reviewed at the trace level?" predicate is a single sentinel-
        # row scan whether the action was skip or reject. This overwrites
        # any prior trace_skip on the same ROI (the labeler has changed
        # their mind from "looks fine" to "all bad") — the per-event
        # rows on disk already encode the audit trail.
        self._trace_actions[current_roi] = {
            "notes": NOTE_TRACE_REJECT,
            "timestamp": ts,
        }
        # Always save (we touched at least the trace_action; touched==0
        # would mean every event was already labeled, which is rare but
        # we still want the trace_reject sentinel on disk).
        self._save()
        if touched:
            self._logger.info(
                f"EventLabeler: rejected {touched} unlabeled events on "
                f"ROI {current_roi}'s trace ({self._sample_id})."
            )
        # Advance past the current ROI to the next *unreviewed* one,
        # falling back to the literal next ROI if every later trace
        # is already done (so the cursor visibly moves forward and
        # the multi-shard wrapper can detect ``is_complete`` to
        # auto-advance to the next image). Always return to overview
        # view so the next user action picks a fresh trace-level
        # outcome.
        target = self._first_unreviewed_roi_cursor(self._roi_cursor + 1)
        if target is not None:
            self._roi_cursor = target
            self._cursor = self._first_event_idx_for_roi(
                self._rois_in_order[target]
            )
        else:
            next_idx = self._cursor
            while (
                next_idx < len(self._events)
                and self._events[next_idx]["roi_id"] == current_roi
            ):
                next_idx += 1
            if next_idx < len(self._events):
                self._cursor = next_idx
                next_roi = self._events[next_idx]["roi_id"]
                if next_roi in self._rois_in_order:
                    self._roi_cursor = self._rois_in_order.index(next_roi)
            else:
                self._cursor = max(0, len(self._events) - 1)
        self._view = "overview"
        self._refresh_ui()

    def undo_trace_rejection(self) -> bool:
        """
        Reverse the most recent :meth:`reject_whole_trace` commit.

        Restores every event's label entry on the rejected ROI to
        whatever it was BEFORE the commit (deleting the entry if
        there was none), removes / restores the trace-level review
        sentinel for that ROI, and snaps the cursor + view back to
        where they were when the commit fired.

        After undo, the snapshot is cleared — undo is single-step,
        not a stack. A second call without an intervening rejection
        is a no-op (returns False with an info-level log).

        Returns:
            True if there was an undoable rejection and it was
            reversed; False if there was nothing to undo (e.g.
            the labeler hasn't rejected anything this session,
            or the snapshot was cleared by a subsequent
            ``label_current`` / ``skip_trace`` / second
            ``reject_whole_trace``).

        The undo window opens at every successful
        ``reject_whole_trace`` commit and is closed by:

            * ``label_current`` — biologist is now editing
              individual labels; "undo the last bulk reject"
              becomes ambiguous.
            * ``skip_trace`` — a competing trace-level intent.
            * a second ``reject_whole_trace`` commit — only the
              MOST RECENT rejection is undoable.

        Pure navigation (next/prev trace, investigate, back) does
        NOT close the window, since navigation doesn't change
        which labels exist.
        """
        snap = self._last_reject_snapshot
        if snap is None:
            self._logger.info(
                "EventLabeler: undo_trace_rejection requested but no "
                "rejection is undoable (the most recent action was "
                "not a reject_whole_trace, or the undo window has "
                "since been closed by another labeling action)."
            )
            self._refresh_ui()
            return False

        roi_id = int(snap["roi_id"])

        # Restore per-event label state. Iterating the snapshot
        # rather than self._events keeps undo correct even if the
        # event list has been reordered between commit and undo
        # (it shouldn't, but the snapshot is the canonical record).
        for event_idx, prior_entry in snap["events_before"]:
            key = (roi_id, int(event_idx))
            if prior_entry is None:
                self._labels.pop(key, None)
            else:
                self._labels[key] = dict(prior_entry)

        # Restore the trace-level review sentinel.
        prior_action = snap["trace_action_before"]
        if prior_action is None:
            self._trace_actions.pop(roi_id, None)
        else:
            self._trace_actions[roi_id] = dict(prior_action)

        # Restore cursor / view so the biologist lands back where
        # they were instead of stranded at the next ROI's overview.
        cursor_before = int(snap["cursor_before"])
        if 0 <= cursor_before < len(self._events):
            self._cursor = cursor_before
        roi_cursor_before = int(snap["roi_cursor_before"])
        if 0 <= roi_cursor_before < len(self._rois_in_order):
            self._roi_cursor = roi_cursor_before
        view_before = str(snap["view_before"])
        if view_before in ("overview", "drill"):
            self._view = view_before

        # Drop the snapshot first so a hypothetical re-entrant call
        # via the on_state_change hook (e.g. the multi-shard wrapper
        # invalidating its cache) doesn't see partially-undone state.
        self._last_reject_snapshot = None
        # Persist. _save() rewrites the corpus from in-memory state,
        # so the previously-written False rows for this trace's
        # newly-restored events disappear from disk in the same atomic
        # rewrite. Other labelers' rows are preserved as always.
        self._save()
        self._logger.info(
            f"EventLabeler: undid the most recent reject_whole_trace "
            f"on ROI {roi_id} ({self._sample_id})."
        )
        self._refresh_ui()
        return True

    def skip_trace(self) -> None:
        """
        Record that the labeler reviewed this trace and chose not to
        label any events individually, then advance to the next ROI.

        Persistence:
            Writes a single trace-level review row to the corpus with
            ``event_idx = TRACE_ACTION_EVENT_IDX`` and
            ``notes = "trace_skip"``. **Individual events on the skipped
            trace are not given event-level labels** — skip means "the
            auto-detection on this trace looks reasonable and I don't
            want to confirm each event one by one," not "every event is
            True."

        Why we persist a row rather than just advancing the cursor:
            A no-write skip would make skipped traces indistinguishable
            from never-opened traces, which breaks inter-rater agreement
            analyses, biases sampling toward problem traces, and erases
            the audit trail of what each labeler actually reviewed. The
            sentinel row preserves all three.

        Interaction with prior reject:
            If the labeler previously rejected this trace, skip is a
            no-op (with a warning). Reject is a stronger commitment —
            silently downgrading it to a skip would lie about the
            trace's state on disk (event rows would still be all
            False). To un-reject, investigate the trace and change
            individual labels.

        Behaviour at the last ROI: still records the review (so the
        biologist gets credit for vetting the last trace) and is
        otherwise a no-op on the cursor.
        """
        if not self._rois_in_order:
            return
        # Switching focus away from a pending reject is a disarm.
        self._reject_armed_roi = None
        # Skip is a state-mutating action (writes a trace_skip row),
        # so it closes the undo window on the most-recent rejection
        # for the same reasons label_current does.
        self._last_reject_snapshot = None
        roi_id = self._rois_in_order[self._roi_cursor]
        prior = self._trace_actions.get(roi_id)
        if prior and prior.get("notes") == NOTE_TRACE_REJECT:
            self._logger.warning(
                f"EventLabeler: skip_trace ignored on ROI {roi_id} "
                f"({self._sample_id}) — the trace was previously "
                f"rejected. Investigate the trace to change individual labels if "
                f"you want to take back the rejection."
            )
            # Still advance the cursor so the keyboard shortcut behaves
            # predictably; the refusal is communicated via the logger
            # rather than blocking navigation.
            self._advance_to_next_unreviewed_or_next_in_order()
            return
        self._trace_actions[roi_id] = {
            "notes": NOTE_TRACE_SKIP,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._save()
        self._advance_to_next_unreviewed_or_next_in_order()

    def _advance_to_next_unreviewed_or_next_in_order(self) -> None:
        """
        Helper used by post-action advance paths to land on the next
        ROI with unfinished events, falling back to the literal next
        ROI when there is none.

        Without the skip-reviewed behavior, biologists who labeled
        their way to here would keep landing on traces they already
        finished (a frequent confusion: "why am I being shown this
        rejected trace?"). With it, the labeler walks past
        fully-reviewed traces automatically while still moving the
        cursor visibly forward in the "everything past here is also
        done" case so something perceptible happens after the action.
        """
        target = self._first_unreviewed_roi_cursor(self._roi_cursor + 1)
        if target is not None:
            self._goto_roi(target)
        else:
            self._goto_roi(self._roi_cursor + 1)

    def next_trace(self) -> None:
        """
        Move the overview to the next ROI without writing anything.

        This is pure navigation — use it to peek ahead at upcoming
        traces. If you mean "I've reviewed this trace and the auto-
        detection looks fine, move on," call :meth:`skip_trace` instead
        so the review is recorded in the corpus.

        No-op at the last ROI.
        """
        self._reject_armed_roi = None
        self._goto_roi(self._roi_cursor + 1)

    def prev_trace(self) -> None:
        """
        Move the overview to the previous ROI without writing anything.

        No-op at the first ROI.
        """
        self._reject_armed_roi = None
        self._goto_roi(self._roi_cursor - 1)

    def next_unreviewed_trace(self) -> bool:
        """
        Move the overview to the next ROI with unfinished events,
        skipping past traces this labeler has already labeled,
        skipped, or rejected.

        Returns True if the cursor moved, False if every later trace
        on this shard is already reviewed (in which case the cursor
        stays put). Pure navigation — does not write to the corpus.

        This is the keyboard / button shortcut for biologists who
        want to actively jump past completed traces without going
        through a labeling action first. Companion to
        :meth:`next_trace` (which walks every trace, including
        reviewed ones — kept that way so reviewed traces remain
        reachable for verification).
        """
        self._reject_armed_roi = None
        target = self._first_unreviewed_roi_cursor(self._roi_cursor + 1)
        if target is None:
            self._refresh_ui()
            return False
        self._goto_roi(target)
        return True

    def investigate_trace(self) -> None:
        """
        Open the per-event view for the current ROI's trace at its
        first event. No-op if the labeler has no events.

        This is the action the biologist takes to "look at this trace
        closely and label its events one at a time" — the per-event
        view replaces the overview's bird's-eye plot with a window
        centered on each event in turn. The view stays scoped to the
        current ROI; stepping past the last event of the ROI returns
        to the overview at the next ROI.

        Refuses to open the per-event view if the ΔF/F trace for the
        current ROI cannot be loaded (missing ``dff_dat`` or no
        resolvable ``filtered_idx``). Labeling without the visual
        context is a bug magnet — the per-event keys would still
        operate on the correct event indices, but the biologist would
        be deciding True / False without the underlying signal to
        compare against. ``reject_whole_trace`` and ``skip_trace``
        remain available from the overview because rejecting a trace
        you can't visualize ("no signal at all") is still a
        legitimate decision.
        """
        if not self._rois_in_order or not self._events:
            return
        roi_id = self._rois_in_order[self._roi_cursor]

        trace = self._get_trace(roi_id)
        if trace is None:
            self._logger.warning(
                f"EventLabeler: refusing to investigate ROI {roi_id} of "
                f"{self._sample_id} — no ΔF/F trace available. Reject "
                f"or skip the trace from the overview, or pass "
                f"filtered_idx explicitly to EventLabeler."
            )
            return

        self._reject_armed_roi = None
        self._cursor = self._first_event_idx_for_roi(roi_id)
        # The internal sentinel is still ``"drill"`` for backward
        # compatibility with callers that inspect ``_view``. The
        # user-facing terminology — buttons, banner, progress widget
        # — uses "investigate" / "per-event view" everywhere.
        self._view = "drill"
        self._refresh_ui()

    def drill_in(self) -> None:
        """
        Deprecated alias for :meth:`investigate_trace`.

        The original UI used "drill in" terminology; biologists found
        it opaque, so the action is now called "investigate trace".
        This alias is preserved so scripts written against the older
        public API continue to work. New code should call
        :meth:`investigate_trace` directly.
        """
        self.investigate_trace()

    def back_to_overview(self) -> None:
        """
        Return to the trace overview, snapped to the ROI of the current
        event. Safe to call from any view.
        """
        self._reject_armed_roi = None
        if self._events:
            roi = self._events[self._cursor]["roi_id"]
            if roi in self._rois_in_order:
                self._roi_cursor = self._rois_in_order.index(roi)
        self._view = "overview"
        self._refresh_ui()

    # ------------------------------------------------------------------
    # Internal navigation helpers
    # ------------------------------------------------------------------
    def _sampling_rate(self) -> Optional[float]:
        """
        Best-effort lookup of the recording's sampling rate (Hz).

        Returns ``None`` if neither ``context["sampling_rate"]`` is
        usable nor the shard exposes a non-zero ``_recording_frate``.
        Used by the seconds-formatting helpers; never raises.
        """
        ctx_rate = self.context.get("sampling_rate")
        if ctx_rate is not None:
            try:
                rate = float(ctx_rate)
                if np.isfinite(rate) and rate > 0:
                    return rate
            except (TypeError, ValueError):
                pass
        shard_rate = getattr(self.shard, "_recording_frate", 0) or 0
        try:
            rate = float(shard_rate)
            if np.isfinite(rate) and rate > 0:
                return rate
        except (TypeError, ValueError):
            pass
        return None

    def _frame_to_seconds_str(self, frame: int) -> str:
        """
        Format ``frame`` as ``"<frame> (X.XXs)"`` when a sampling rate
        is available; otherwise just ``"<frame>"``.

        Biologists think in seconds — especially when comparing against
        published indicator kinetics quoted in ms. We never drop the
        raw frame index because that's still the source-of-truth for
        the corpus and for follow-up debugging.
        """
        rate = self._sampling_rate()
        if rate is None:
            return f"{frame}"
        return f"{frame} ({frame / rate:.2f}s)"

    def _trace_review_state(self, roi_id: int) -> Optional[str]:
        """
        Classify the labeler's most-recent trace-level intent for ``roi_id``.

        Returns one of:
            * ``"skipped"``   — labeler skipped this trace and has not
              labeled any individual events on it since (the skip is the
              most recent action).
            * ``"rejected"``  — labeler rejected this trace at the trace
              level.
            * ``"revisited"`` — labeler previously skipped this trace
              but has since labeled at least one event individually
              (the skip is stale).
            * ``None``        — no trace-level review action recorded.

        Why this lives here: both the overview title and the details
        panel need the same predicate, and the "skip then drill" stale-
        annotation bug surfaces if either site computes it ad-hoc.
        """
        entry = self._trace_actions.get(roi_id)
        if entry is None:
            return None
        action_notes = entry.get("notes", "")
        if action_notes == NOTE_TRACE_REJECT:
            return "rejected"
        any_event_labeled = any(
            v.get("label")
            for (rid, _eidx), v in self._labels.items()
            if rid == roi_id
        )
        if any_event_labeled:
            return "revisited"
        return "skipped"

    def _first_event_idx_for_roi(self, roi_id: int) -> int:
        """Return the index into ``self._events`` of the first event on ``roi_id``."""
        for i, ev in enumerate(self._events):
            if ev["roi_id"] == roi_id:
                return i
        return 0

    def _goto_roi(self, new_roi_cursor: int) -> None:
        """
        Move ``self._roi_cursor`` to ``new_roi_cursor`` (clamped) and snap
        ``self._cursor`` to the first event on the resulting ROI. Always
        sets the view back to ``"overview"`` so trace-level navigation
        cannot strand the user in a drill view on a different ROI.
        """
        if not self._rois_in_order:
            return
        clamped = max(0, min(len(self._rois_in_order) - 1, new_roi_cursor))
        self._roi_cursor = clamped
        if self._events:
            roi_id = self._rois_in_order[self._roi_cursor]
            self._cursor = self._first_event_idx_for_roi(roi_id)
        self._view = "overview"
        self._refresh_ui()

    def _sync_roi_cursor_to_event(self) -> None:
        """Snap ``self._roi_cursor`` to the ROI of the current event."""
        if not self._events:
            return
        roi = self._events[self._cursor]["roi_id"]
        if roi in self._rois_in_order:
            self._roi_cursor = self._rois_in_order.index(roi)

    def _advance_event(self, step: int) -> None:
        """
        Step the event cursor in drill view.

        Forward steps that cross out of the current ROI return to the
        trace overview at the next ROI rather than silently switching
        drilled traces. Backward steps clamp to the first event of the
        current ROI so ``k`` cannot rewind into a previously-labeled
        trace from drill mode. In overview view a forward step still
        advances ``self._cursor`` (used by programmatic callers) but does
        not change the view.
        """
        if not self._events:
            return

        if self._view != "drill":
            new_cursor = max(
                0, min(len(self._events) - 1, self._cursor + step)
            )
            self._cursor = new_cursor
            self._sync_roi_cursor_to_event()
            self._refresh_ui()
            return

        current_roi = self._events[self._cursor]["roi_id"]
        if step > 0:
            new_cursor = self._cursor + step
            if (
                new_cursor >= len(self._events)
                or self._events[new_cursor]["roi_id"] != current_roi
            ):
                # Crossed past the last event of this ROI. Land on
                # the next ROI with unfinished work rather than the
                # literal next ROI in order — the biologist just
                # finished labeling this trace and shouldn't be
                # forced through traces they already vetted.
                target = self._first_unreviewed_roi_cursor(
                    self._roi_cursor + 1
                )
                if target is not None:
                    self._roi_cursor = target
                    self._cursor = self._first_event_idx_for_roi(
                        self._rois_in_order[target]
                    )
                elif new_cursor < len(self._events):
                    self._cursor = new_cursor
                    self._sync_roi_cursor_to_event()
                else:
                    self._cursor = len(self._events) - 1
                self._view = "overview"
                self._refresh_ui()
                return
            self._cursor = new_cursor
        else:
            first = self._first_event_idx_for_roi(current_roi)
            self._cursor = max(first, self._cursor + step)
        self._refresh_ui()

    # ------------------------------------------------------------------
    # Trace / display helpers
    # ------------------------------------------------------------------
    def _resolve_filtered_idx(self) -> Optional[List[int]]:
        """
        Best-effort recovery of the filtered_idx mapping.

        Resolution order:
            1. Explicit ``filtered_idx`` passed to the constructor.
            2. ``shard._filtered_idx_cache`` populated by ``_run_all``
               (authoritative — guaranteed to match the spatial filter
               that produced the raw event lists this labeler is
               showing).
            3. Recompute via ``shard.spatial_filtering`` with the legacy
               defaults (``p_th=75``, ``size_threshold=20000``). Logs a
               loud warning because the trace plot may visually
               disagree with the event positions if the original run
               used different thresholds.

        Returns ``None`` if every fallback fails; callers must handle
        the ``None`` case gracefully (e.g. by skipping the trace plot).
        """
        if self._filtered_idx is not None:
            return self._filtered_idx

        cached = getattr(self.shard, "_filtered_idx_cache", None) or []
        cache_params = getattr(self.shard, "_filtered_idx_params", None)
        if cached and cache_params is not None:
            self._filtered_idx = [int(i) for i in cached]
            self._logger.debug(
                f"EventLabeler: using shard-cached filtered_idx "
                f"(params={cache_params})."
            )
            return self._filtered_idx

        # Recompute path: this is the last resort. The biologist almost
        # certainly ran ``_run_all`` already (the raw event lists this
        # labeler reads from come from there), so reaching this branch
        # means the cache wasn't populated (older shard, custom flow,
        # or a test fixture). Log loudly so a visual misalignment
        # between displayed trace and event markers can be diagnosed.
        try:
            idx = self.shard.spatial_filtering(
                p_th=75, size_threshold=20000, plot=False, silence=True
            )
            self._filtered_idx = [int(i) for i in idx]
            self._logger.warning(
                f"EventLabeler: shard {self._sample_id} has no cached "
                f"filtered_idx; recomputed with default p_th=75, "
                f"size_threshold=20000. If _run_all was invoked with "
                f"different thresholds the displayed ΔF/F trace may "
                f"NOT correspond to the events shown. Pass filtered_idx "
                f"explicitly to EventLabeler to suppress this warning."
            )
            return self._filtered_idx
        except Exception as exc:
            self._logger.warning(
                f"EventLabeler: could not derive filtered_idx for "
                f"{self._sample_id}: {exc}. Trace plot will be skipped."
            )
            return None

    def _get_trace(self, roi_id: int) -> Optional[np.ndarray]:
        """Return the ΔF/F trace for the given local ROI id, or None."""
        try:
            dff = self.shard.get_input("dff_dat", req=True)
        except Exception as exc:
            self._logger.warning(
                f"EventLabeler: failed to load dff_dat for "
                f"{self._sample_id}: {exc}"
            )
            return None
        if dff is None:
            return None
        filtered_idx = self._resolve_filtered_idx()
        if filtered_idx is None:
            return None
        if roi_id < 0 or roi_id >= len(filtered_idx):
            self._logger.warning(
                f"EventLabeler: roi_id={roi_id} out of bounds for "
                f"filtered_idx (len={len(filtered_idx)})"
            )
            return None
        absolute = filtered_idx[roi_id]
        if absolute < 0 or absolute >= dff.shape[0]:
            self._logger.warning(
                f"EventLabeler: absolute component index {absolute} out "
                f"of bounds for dff_dat (rows={dff.shape[0]})"
            )
            return None
        return np.asarray(dff[absolute, :], dtype=float)

    # Minimum effective fwhm (in frames) used when scaling the trace
    # window so that very narrow events still get a few frames of
    # context on either side. Named so the `5` isn't a magic number.
    _MIN_EFFECTIVE_FWHM_FRAMES: int = 5
    # Minimum window size in frames. Below this we collapse to the
    # full trace rather than show a useless ~3-frame slice.
    _MIN_WINDOW_FRAMES: int = 5

    # Indicator-specific window_scale presets. Slow indicators
    # (GCaMP6s, jGCaMP7s, jRCaMP1a) have longer FWHM in frames, so
    # the same window_scale=8.0 that works for GCaMP6f produces a
    # window that's too wide to be useful. These values are tuned so
    # the displayed window is ~3-5x the indicator's typical decay
    # constant — enough context to see the rise and fall, narrow
    # enough that the event is recognizably central in the plot.
    # Indicators not in this map fall through to the constructor's
    # ``window_scale`` argument unchanged.
    _INDICATOR_WINDOW_SCALE: Dict[str, float] = {
        "GCaMP6f":  8.0,
        "GCaMP6m":  5.0,
        "GCaMP6s":  3.0,
        "GCaMP7f":  10.0,
        "GCaMP7s":  2.5,
        "jGCaMP8m": 12.0,
        "jGCaMP8s": 10.0,
        "jGCaMP8f": 15.0,
        "jRGECO1a": 5.0,
        "jRCaMP1a": 3.0,
        "GCaMP3":   4.0,
    }

    def _effective_window_scale(self) -> float:
        """
        Resolve the per-event window half-width multiplier to use.

        Priority:
            1. If the labeler was constructed with a non-default
               ``window_scale`` we honor it (explicit user choice wins).
            2. Otherwise, if ``context["indicator"]`` matches one of the
               indicator presets, use that.
            3. Otherwise fall back to the constructor default (8.0).

        The "explicit user choice" detection is sentinel-based:
        :attr:`_window_scale_is_default` is set in ``__init__`` when
        the constructor argument equals the class default.
        """
        if not getattr(self, "_window_scale_is_default", False):
            return float(self.window_scale)
        indicator = self.context.get("indicator")
        if indicator and indicator in self._INDICATOR_WINDOW_SCALE:
            return float(self._INDICATOR_WINDOW_SCALE[indicator])
        return float(self.window_scale)

    def _window_for_event(
        self, ev: Dict[str, Any], n_frames: int
    ) -> Tuple[int, int]:
        """
        Compute (start, end) frame indices of the trace window around ``ev``.

        Window half-width is ``window_scale * max(fwhm_frames,
        _MIN_EFFECTIVE_FWHM_FRAMES)``. The effective ``window_scale`` is
        resolved via :meth:`_effective_window_scale`, which lets slow
        indicators (GCaMP6s, jGCaMP7s, ...) use a smaller multiplier so
        the displayed window stays close to a few decay constants
        instead of zooming out to a needle-in-haystack view.

        Clipped to ``[0, n_frames]``; for very short traces the window
        collapses to the whole trace.
        """
        scale = self._effective_window_scale()
        fwhm = ev.get("fwhm_frames")
        if fwhm is None or not np.isfinite(fwhm) or fwhm <= 0:
            half = int(round(scale * 10))
        else:
            half = max(
                1,
                int(round(scale * max(float(fwhm), self._MIN_EFFECTIVE_FWHM_FRAMES))),
            )
        peak = ev.get("peak_position", 0)
        if not isinstance(peak, (int, np.integer)) or peak < 0:
            peak = 0
        start = max(0, int(peak) - half)
        end = min(n_frames, int(peak) + half + 1)
        if end - start < self._MIN_WINDOW_FRAMES and n_frames > 0:
            return 0, n_frames
        return start, end

    # ------------------------------------------------------------------
    # ipywidgets UI (lazy import boundary)
    # ------------------------------------------------------------------
    def display(self) -> None:
        """
        Render the labeling UI in the current Jupyter notebook.

        Imports ``ipywidgets`` lazily so the surrounding module remains
        importable in headless environments. Raises ``ImportError`` with a
        clear install hint if the optional dependency is missing.

        Returns ``None`` (rather than the root widget) so that calling
        ``labeler.display()`` as the last expression in a notebook cell
        produces a single rendering of the UI. Returning the root widget
        used to cause a second auto-render of the cell's return value;
        the explicit ``IPython.display.display(root)`` below is the only
        path that puts the UI on screen. The constructed root widget is
        also retained on ``self._widgets["root"]`` for callers that want
        to inspect it programmatically.
        """
        self._build_root_widget()
        try:
            from IPython.display import display as _ipy_display
        except ImportError as exc:  # pragma: no cover  — IPython ships with notebooks
            raise ImportError(
                "EventLabeler.display() requires IPython."
            ) from exc
        _ipy_display(self._widgets["root"])

    def _build_root_widget(self) -> Any:
        """
        Build the labeler's ipywidgets tree without rendering it.

        Used by both :meth:`display` (which then explicitly displays
        the returned root) and :class:`MultiShardLabeler` (which embeds
        the root inside its own wrapper VBox via ``body.children = ...``
        and is therefore responsible for the single render itself).
        Splitting "build" from "display" prevents the multi-image
        wrapper from rendering the child labeler twice — once via the
        child's own ``IPython.display.display`` call and once via the
        wrapper's body container — which was visible as duplicated UI
        in the notebook output.

        Idempotent in practice: subsequent calls would rebuild a fresh
        widget tree, so callers should treat this as a one-shot helper
        per labeler instance.
        """
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise ImportError(
                "EventLabeler.display() requires ipywidgets. Install with: "
                "pip install 'wizards_staff[labeling]'"
            ) from exc

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover  — matplotlib is core dep
            raise ImportError(
                "EventLabeler.display() requires matplotlib."
            ) from exc

        self._ipy = widgets

        # The trace-first UI assumes events for a given ROI are visited
        # contiguously. Reorder for this session if the constructor was
        # called with a different ordering; the ORDERINGS attribute stays
        # available for programmatic / calibration use of the class.
        if self.ordering != "by_roi_then_time":
            self._logger.warning(
                f"EventLabeler.display() requires by_roi_then_time "
                f"ordering for the trace-first UI; reordering events for "
                f"this session (constructor value was {self.ordering!r})."
            )
            self.ordering = "by_roi_then_time"
            self._apply_ordering()
            self._cursor = 0
            self._roi_cursor = 0
            self._view = "overview"

        # Output widget that hosts the matplotlib figure. We never call
        # plt.show(); rendering happens inside the Output capture context.
        plot_out = widgets.Output(
            layout=widgets.Layout(height="540px", border="1px solid #ddd")
        )
        with plot_out:
            self._fig, (self._ax_trace, self._ax_minimap) = plt.subplots(
                2,
                1,
                figsize=(10, 5.4),
                gridspec_kw={"height_ratios": [4, 1]},
            )
            self._fig.tight_layout()
        # Detach the figure from pyplot's global tracking so the inline
        # backend's end-of-cell ``flush_figures`` hook doesn't render
        # this figure a second time as a standalone cell output. The
        # figure object itself stays alive (we hold a reference via
        # ``self._fig``) and is re-rendered on demand inside ``plot_out``
        # by ``_draw_figure``.
        plt.close(self._fig)

        # Drill-view buttons (per-event labeling).
        btn_true = widgets.Button(
            description="True (t)",
            button_style="success",
            tooltip="Confirm this is a real event",
        )
        btn_false = widgets.Button(
            description="False (f)",
            button_style="danger",
            tooltip="Reject this event",
        )
        btn_unsure = widgets.Button(
            description="Unsure (u)",
            button_style="warning",
            tooltip="Mark as ambiguous",
        )
        btn_prev = widgets.Button(
            description="Prev event (k)",
            tooltip="Step back one event within the current ROI.",
        )
        btn_next = widgets.Button(
            description="Next event (j)",
            tooltip=(
                "Step forward one event. Past the last event of this ROI, "
                "returns to the trace overview at the next ROI."
            ),
        )
        btn_back = widgets.Button(
            description="Back to trace (b)",
            tooltip="Return to the trace overview for the current ROI.",
        )

        # Overview-view buttons (trace-level outcomes).
        btn_drill_in = widgets.Button(
            description="Investigate trace (i)",
            button_style="primary",
            tooltip=(
                "Look at this trace's events one at a time so you can "
                "remove false positives. Press 'i' (legacy: 'd')."
            ),
        )
        btn_reject_trace = widgets.Button(
            description="Reject whole trace (r)",
            button_style="danger",
            tooltip=(
                "Label every unlabeled event on the CURRENT ROI's trace "
                "as False (only this ROI — not other ROIs, not the "
                "whole sample). Does not mark the ROI itself bad — "
                "that's outlier-detection territory."
            ),
        )
        btn_skip_trace = widgets.Button(
            description="Skip trace (s)",
            tooltip=(
                "Record that you reviewed this trace and the auto-"
                "detection looks fine, then advance to the next ROI. "
                "Writes a single trace-level review row to the corpus "
                "(event_idx=-1, notes='trace_skip') so downstream code "
                "can tell 'reviewed and clean' from 'never opened' — "
                "but does NOT label any individual events True."
            ),
        )
        btn_undo_reject = widgets.Button(
            description="Undo last reject (z)",
            tooltip=(
                "Reverse the most recent 'Reject whole trace' commit, "
                "restoring every event's prior label and the prior "
                "trace-level sentinel. Available only until you label "
                "an event, skip a trace, or reject another trace — "
                "single-step undo, not a stack."
            ),
            disabled=True,
        )
        btn_prev_trace = widgets.Button(
            description="Prev trace (p)",
            tooltip="Move the overview to the previous ROI.",
        )
        btn_next_trace = widgets.Button(
            description="Next trace (n)",
            tooltip=(
                "Move the overview to the next ROI in order, "
                "INCLUDING reviewed ones. Use this to flip back "
                "through your work to verify."
            ),
        )
        btn_next_unreviewed_trace = widgets.Button(
            description="Next unreviewed (m)",
            tooltip=(
                "Jump to the next ROI with unfinished events, "
                "skipping past traces you have already labeled, "
                "skipped, or rejected. Companion to 'Next trace' — "
                "use this when you want to keep moving forward "
                "through new work."
            ),
        )

        notes = widgets.Text(
            value="",
            placeholder="Optional notes for the next label",
            description="Notes:",
            layout=widgets.Layout(width="60%"),
        )

        # "Command box" — the cheapest way to get keyboard shortcuts in a
        # standard ipywidgets stack without custom JS. The biologist clicks
        # into this single-character text field and types one of the keys
        # listed in the placeholder. Dispatch is view-aware (see
        # _on_cmd_change below). After every action we attempt to
        # re-focus this box via ``cmd.focus()`` so the biologist doesn't
        # have to keep clicking back into it — see ``_refocus_cmd``.
        cmd = widgets.Text(
            value="",
            placeholder=(
                "Press a key — overview: i/r/z/s/p/n/m; per-event: t/f/u/j/k/b/z"
            ),
            description="Keys:",
            layout=widgets.Layout(width="40%"),
        )

        progress = widgets.HTML()
        details = widgets.HTML()

        # Permanent help banner. The placeholder on ``cmd`` is too narrow
        # in standard notebooks to carry both the keymap and the
        # semantic caveats biologists trip over (Unsure is non-binding;
        # True cannot recover an event already dropped by the amplitude
        # / FWHM bounds layer). Surface those facts up front rather than
        # waiting for someone to file a bug.
        help_banner = widgets.HTML(
            value=(
                "<div style='font-size:0.85em;color:#333;background:#fffae6;"
                "border:1px solid #e6cf66;padding:6px 8px;border-radius:4px'>"
                "<b>Key map.</b> "
                "Overview: <code>i</code> investigate trace, "
                "<code>r</code> reject whole trace (press twice to confirm), "
                "<code>z</code> undo last reject, "
                "<code>s</code> skip, <code>p</code>/<code>n</code> prev/next trace "
                "(walks every trace, including reviewed ones), "
                "<code>m</code> next unreviewed trace (skips ones you've labeled / "
                "skipped / rejected). "
                "Per-event view: <code>t</code> True, <code>f</code> False, "
                "<code>u</code> Unsure, <code>j</code>/<code>k</code> next/prev event, "
                "<code>b</code> back to overview, <code>z</code> undo last reject."
                "<br/>"
                "<b>Plot legend.</b> Each detected event is drawn as a "
                "<em>dot at its peak</em> (color = label state: orange = unlabeled, "
                "green = True, red = False, gray = Unsure). The "
                "<em>shaded orange band</em> is the analysis-computed FWHM "
                "(full width at half max) extent of that event. "
                "Only <em>unlabeled</em> events get the FWHM shading — once "
                "you label an event the band disappears so the plot doesn't "
                "get crowded; the shading also disappears when the analysis "
                "couldn't compute valid FWHM bounds for that event. "
                "Long sustained transients can produce FWHM bands that span "
                "most of the trace; this is correct behavior, not a bug."
                "<br/>"
                "<b>Semantics.</b> Only <code>False</code> labels remove events from the "
                "analysis. <code>Unsure</code> is stored for calibration but does "
                "<em>not</em> filter. <code>True</code> records your agreement but "
                "<em>cannot</em> recover an event already dropped by the amplitude/FWHM "
                "bounds (labels can only narrow the surviving set, never widen it)."
                "</div>"
            )
        )

        def _on_label(label: str) -> None:
            self.label_current(label, notes=notes.value)
            notes.value = ""

        btn_true.on_click(lambda _b: _on_label("True"))
        btn_false.on_click(lambda _b: _on_label("False"))
        btn_unsure.on_click(lambda _b: _on_label("Unsure"))
        btn_prev.on_click(lambda _b: self._advance_event(-1))
        btn_next.on_click(lambda _b: self._advance_event(+1))
        btn_back.on_click(lambda _b: self.back_to_overview())

        btn_drill_in.on_click(lambda _b: self.investigate_trace())
        btn_reject_trace.on_click(lambda _b: self.reject_whole_trace())
        btn_skip_trace.on_click(lambda _b: self.skip_trace())
        btn_undo_reject.on_click(lambda _b: self.undo_trace_rejection())
        btn_prev_trace.on_click(lambda _b: self.prev_trace())
        btn_next_trace.on_click(lambda _b: self.next_trace())
        btn_next_unreviewed_trace.on_click(
            lambda _b: self.next_unreviewed_trace()
        )

        def _keymap_for_view() -> Dict[str, Any]:
            # Legacy aliases kept for muscle memory / scripts:
            #   - ``w`` was the original key for "reject whole trace"
            #     (pre-trace-first UX). Kept in overview only —
            #     intentionally NOT mapped in per-event view because
            #     bulk-rejecting mid-investigation is destructive and
            #     irreversible, so the user has to back out to overview
            #     to invoke it.
            #   - ``d`` was the original key for "drill in" before the
            #     action was renamed to "investigate trace" (``i``).
            #     Kept for biologists with existing muscle memory.
            if self._view == "overview":
                return {
                    "i": self.investigate_trace,
                    "d": self.investigate_trace,
                    "r": self.reject_whole_trace,
                    "w": self.reject_whole_trace,
                    "s": self.skip_trace,
                    "z": self.undo_trace_rejection,
                    "p": self.prev_trace,
                    "n": self.next_trace,
                    "m": self.next_unreviewed_trace,
                }
            return {
                "t": lambda: _on_label("True"),
                "f": lambda: _on_label("False"),
                "u": lambda: _on_label("Unsure"),
                "j": lambda: self._advance_event(+1),
                "k": lambda: self._advance_event(-1),
                "b": self.back_to_overview,
                # ``z`` works in per-event view too, on the off
                # chance the labeler invokes undo while still
                # investigating: it doesn't move the cursor of
                # its own accord, just restores labels.
                "z": self.undo_trace_rejection,
            }

        def _on_cmd_change(change: Dict[str, Any]) -> None:
            value = change.get("new", "") or ""
            if not value:
                return
            # Always reset the box, regardless of dispatch outcome.
            cmd.value = ""
            # Dispatch EVERY character, not just the first. ipywidgets
            # coalesces keystrokes typed in quick succession into a
            # single ``value`` change — so the two-press reject confirm
            # ("press 'r' again to confirm") frequently arrives as a
            # single ``"rr"`` change. Taking only ``value[:1]`` silently
            # dropped the confirming press, leaving the trace armed and
            # the "REJECT ARMED" badge stuck on screen. Walking each
            # character commits the second ``r`` (and is generally
            # correct: no buffered keystroke is ever dropped). The
            # keymap is recomputed per character because a key can flip
            # the view mid-sequence (e.g. ``i`` opens the per-event
            # view, after which ``t``/``f``/``u`` should apply).
            for ch in value.strip().lower():
                action = _keymap_for_view().get(ch)
                if action is not None:
                    action()

        cmd.observe(_on_cmd_change, names="value")

        overview_controls = widgets.HBox(
            [
                btn_drill_in,
                btn_reject_trace,
                btn_undo_reject,
                btn_skip_trace,
                btn_prev_trace,
                btn_next_trace,
                btn_next_unreviewed_trace,
            ]
        )
        drill_controls = widgets.HBox(
            [btn_true, btn_false, btn_unsure, btn_prev, btn_next, btn_back]
        )
        meta = widgets.HBox([notes, cmd])

        root = widgets.VBox(
            [help_banner, progress, plot_out, overview_controls, drill_controls, meta, details]
        )

        self._widgets = {
            "root": root,
            "plot_out": plot_out,
            "progress": progress,
            "details": details,
            "notes": notes,
            "cmd": cmd,
            "btn_true": btn_true,
            "btn_false": btn_false,
            "btn_unsure": btn_unsure,
            "btn_prev": btn_prev,
            "btn_next": btn_next,
            "btn_back": btn_back,
            "btn_drill_in": btn_drill_in,
            "btn_reject_trace": btn_reject_trace,
            "btn_undo_reject": btn_undo_reject,
            "btn_skip_trace": btn_skip_trace,
            "btn_prev_trace": btn_prev_trace,
            "btn_next_trace": btn_next_trace,
            "btn_next_unreviewed_trace": btn_next_unreviewed_trace,
            "overview_controls": overview_controls,
            "drill_controls": drill_controls,
        }

        self._refresh_ui()
        return root

    def _refresh_ui(self) -> None:
        """Repaint the figure and progress strings, if the UI is built."""
        if not self._widgets:
            return
        self._apply_view_visibility()
        self._refresh_undo_button()
        self._refresh_progress()
        self._refresh_figure()
        self._refresh_details()
        self._refocus_cmd()

    def _refresh_undo_button(self) -> None:
        """Enable / disable 'Undo last reject' based on snapshot state."""
        btn = self._widgets.get("btn_undo_reject") if self._widgets else None
        if btn is None:
            return
        btn.disabled = self._last_reject_snapshot is None

    def _refocus_cmd(self) -> None:
        """
        Return keyboard focus to the command box after every action.

        ipywidgets >= 8 exposes ``Widget.focus()``; older versions
        don't, in which case the user has to click back into the box
        manually. We swallow any failure silently — auto-refocus is a
        comfort feature, not a correctness one — but log at DEBUG so a
        developer who notices the missing focus can find the reason.

        Without this, the single biggest UX wart of the labeler is
        that the user must click into the "Keys:" field before EVERY
        keystroke. With it, an entire labeling session can be done
        without ever leaving the keyboard.
        """
        cmd = self._widgets.get("cmd") if self._widgets else None
        if cmd is None:
            return
        focus = getattr(cmd, "focus", None)
        if focus is None:
            return
        try:
            focus()
        except Exception as exc:  # pragma: no cover  — focus() is best-effort
            self._logger.debug(
                f"EventLabeler: cmd.focus() failed ({exc!r}); user will "
                f"need to click back into the command box manually."
            )

    def _apply_view_visibility(self) -> None:
        """Show/hide the overview vs drill control rows based on ``self._view``."""
        ov = self._widgets.get("overview_controls")
        dc = self._widgets.get("drill_controls")
        if ov is None or dc is None:
            return
        if self._view == "overview":
            ov.layout.display = "flex"
            dc.layout.display = "none"
        else:
            ov.layout.display = "none"
            dc.layout.display = "flex"

    def _refresh_progress(self) -> None:
        prog = self._widgets.get("progress")
        if prog is None:
            return
        if not self._events:
            prog.value = (
                f"<b>{self._sample_id}</b>: no labelable events on this shard."
            )
            return
        total = len(self._events)
        labeled = self.labeled_count
        total_rois = len(self._rois_in_order) or 1

        if self._view == "overview":
            roi_id = self._rois_in_order[self._roi_cursor]
            same_roi = [e for e in self._events if e["roi_id"] == roi_id]
            on_trace_labeled = sum(
                1
                for e in same_roi
                if self._labels.get((e["roi_id"], e["event_idx"]), {}).get("label")
            )
            armed_hint = (
                " &middot; <span style='color:#fff;background:#d62728;"
                "padding:1px 6px;border-radius:3px'>"
                "REJECT ARMED &mdash; press 'r' again to confirm</span>"
                if self._reject_armed_roi == roi_id
                else ""
            )
            prog.value = (
                f"<b>{self._sample_id}</b> &middot; "
                f"ROI {self._roi_cursor + 1} of {total_rois} "
                f"(id={roi_id}) &middot; "
                f"view=<b>trace overview</b> &middot; "
                f"{on_trace_labeled}/{len(same_roi)} labeled on this trace "
                f"&middot; <b>{labeled}/{total}</b> total{armed_hint}"
            )
            return

        ev = self._events[self._cursor]
        roi_rank = self._rois_in_order.index(ev["roi_id"]) + 1
        same_roi = [e for e in self._events if e["roi_id"] == ev["roi_id"]]
        try:
            within = same_roi.index(ev) + 1
        except ValueError:
            within = 1
        prog.value = (
            f"<b>{self._sample_id}</b> &middot; "
            f"ROI {roi_rank} of {total_rois} "
            f"(id={ev['roi_id']}) &middot; "
            f"view=<b>investigating trace</b> &middot; "
            f"Event {within} of {len(same_roi)} &middot; "
            f"<b>{labeled}/{total}</b> total labeled"
        )

    def _refresh_details(self) -> None:
        details = self._widgets.get("details")
        if details is None:
            return
        if not self._events:
            details.value = ""
            return

        if self._view == "overview":
            roi_id = (
                self._rois_in_order[self._roi_cursor]
                if self._rois_in_order
                else None
            )
            if roi_id is None:
                details.value = ""
                return
            same_roi = [e for e in self._events if e["roi_id"] == roi_id]
            counts = {"True": 0, "False": 0, "Unsure": 0, "unreviewed": 0}
            other_touch = 0
            for e in same_roi:
                key = (e["roi_id"], e["event_idx"])
                entry = self._labels.get(key)
                lbl = entry.get("label") if entry else ""
                if lbl in counts:
                    counts[lbl] += 1
                else:
                    counts["unreviewed"] += 1
                if key in self._touched_by_others:
                    other_touch += 1
            other_hint = (
                f" &middot; <span style='color:#888'>"
                f"{other_touch} touched by other labelers</span>"
                if other_touch
                else ""
            )
            review_state = self._trace_review_state(roi_id)
            other_reviewed = roi_id in self._traces_touched_by_others
            self_hint_text = {
                "skipped": "you previously skipped this trace",
                "rejected": "you previously rejected this trace",
                "revisited": "you skipped this trace, then revisited it",
                None: "",
            }[review_state]
            skip_hints: List[str] = []
            if self_hint_text:
                skip_hints.append(
                    f"<span style='color:#0a7'>{self_hint_text}</span>"
                )
            if other_reviewed:
                skip_hints.append(
                    "<span style='color:#888'>another labeler reviewed this trace at the trace level</span>"
                )
            skip_hint_html = (
                " &middot; " + " &middot; ".join(skip_hints) if skip_hints else ""
            )
            details.value = (
                f"<span style='color:#2ca02c'>True: {counts['True']}</span> &middot; "
                f"<span style='color:#d62728'>False: {counts['False']}</span> &middot; "
                f"<span style='color:#7f7f7f'>Unsure: {counts['Unsure']}</span> &middot; "
                f"<span style='color:#ff7f0e'>unreviewed: {counts['unreviewed']}</span>"
                f"{other_hint}{skip_hint_html}"
            )
            return

        ev = self._events[self._cursor]
        key = (ev["roi_id"], ev["event_idx"])
        existing = self._labels.get(key)
        prior_self = (
            f"<span style='color:#0a7'>self prior label: "
            f"<b>{existing['label']}</b></span>"
            if existing and existing.get("label")
            else "<span style='color:#999'>not yet labeled by you</span>"
        )
        other_hint = (
            "<span style='color:#888'> &middot; another labeler has touched "
            "this event</span>"
            if key in self._touched_by_others
            else ""
        )
        rate = self._sampling_rate()
        fwhm_str = f"{ev['fwhm_frames']:.2f} frames"
        if rate is not None and np.isfinite(ev["fwhm_frames"]):
            fwhm_str += f" ({(ev['fwhm_frames'] / rate) * 1000:.0f}ms)"
        details.value = (
            f"peak_amp={ev['peak_amplitude']:.4f} &middot; "
            f"fwhm={fwhm_str} &middot; "
            f"peak_frame={self._frame_to_seconds_str(ev['peak_position'])} "
            f"&middot; {prior_self}{other_hint}"
        )

    def _refresh_figure(self) -> None:
        """Repaint the matplotlib figure for the current view."""
        if self._view == "overview":
            self._refresh_overview_figure()
        else:
            self._refresh_event_figure()

    def _refresh_event_figure(self) -> None:
        plot_out = self._widgets.get("plot_out")
        if plot_out is None or self._fig is None:
            return
        if not self._events:
            return

        ev = self._events[self._cursor]
        trace = self._get_trace(ev["roi_id"])

        ax_t = self._ax_trace
        ax_m = self._ax_minimap
        ax_t.clear()
        ax_m.clear()
        # The minimap may have been hidden by the overview view; ensure
        # it is visible again before drawing the windowed view.
        ax_m.set_visible(True)
        ax_t.set_visible(True)

        if trace is None or trace.size == 0:
            ax_t.text(
                0.5,
                0.5,
                "ΔF/F trace unavailable\n"
                "(pass filtered_idx to EventLabeler if you ran with\n"
                "non-default p_th/size_threshold).",
                ha="center",
                va="center",
                transform=ax_t.transAxes,
            )
            ax_t.set_xticks([])
            ax_t.set_yticks([])
            ax_m.set_xticks([])
            ax_m.set_yticks([])
            self._draw_figure(plot_out)
            return

        n_frames = trace.size
        start, end = self._window_for_event(ev, n_frames)

        # Pad with NaN if the requested window extends past the trace.
        # (window_for_event already clips, but this branch covers the
        # documented "event window clipped at trace start/end" edge case
        # by ensuring the displayed window never visually misaligns the
        # peak marker.)
        x = np.arange(start, end)
        y = trace[start:end]

        ax_t.plot(x, y, color="#1f77b4", linewidth=1.0)
        # Shade FWHM window if available.
        fb = ev.get("fwhm_back")
        ff = ev.get("fwhm_fwd")
        if (
            fb is not None
            and ff is not None
            and fb >= 0
            and ff >= fb
            and ff < n_frames
        ):
            ax_t.axvspan(fb, ff, alpha=0.15, color="#ff7f0e")
        # Peak marker.
        peak = ev.get("peak_position", 0)
        if 0 <= peak < n_frames:
            ax_t.axvline(peak, color="#d62728", linewidth=0.8, linestyle="--")
            ax_t.plot([peak], [trace[peak]], marker="o", color="#d62728")

        # Previously-labeled events on this ROI as colored ticks along
        # the bottom of the top axis.
        ymin, ymax = ax_t.get_ylim()
        tick_y = ymin + 0.02 * (ymax - ymin)
        for other_ev in self._events:
            if other_ev["roi_id"] != ev["roi_id"]:
                continue
            okey = (other_ev["roi_id"], other_ev["event_idx"])
            entry = self._labels.get(okey)
            if not entry or not entry.get("label"):
                continue
            color = {
                "True": "#2ca02c",
                "False": "#d62728",
                "Unsure": "#7f7f7f",
            }.get(entry["label"], "#000000")
            ox = other_ev.get("peak_position", -1)
            if start <= ox < end:
                ax_t.plot([ox], [tick_y], marker="|", color=color, markersize=14)

        rate = self._sampling_rate()
        peak_time_str = (
            f" @ {ev['peak_position'] / rate:.2f}s"
            if rate is not None and 0 <= ev["peak_position"]
            else ""
        )
        ax_t.set_title(
            f"Sample {self._sample_id} · ROI {ev['roi_id']} "
            f"· event {ev['event_idx']}{peak_time_str}"
        )
        ax_t.set_xlabel("frame" if rate is None else f"frame  (rate={rate:g} Hz)")
        ax_t.set_ylabel("ΔF/F")
        ax_t.set_xlim(start, max(end - 1, start + 1))
        self._add_drill_legend(ax_t)

        # Minimap.
        ax_m.plot(np.arange(n_frames), trace, color="#444", linewidth=0.6)
        ax_m.set_xlim(0, n_frames)
        ymin_m, ymax_m = ax_m.get_ylim()
        ax_m.add_patch(
            __import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (start, ymin_m),
                max(end - start, 1),
                ymax_m - ymin_m,
                facecolor="#1f77b4",
                alpha=0.2,
                edgecolor="#1f77b4",
            )
        )
        ax_m.set_yticks([])
        ax_m.set_xlabel("full trace")

        self._draw_figure(plot_out)

    def _refresh_overview_figure(self) -> None:
        """
        Plot the full ΔF/F trace for the current ROI with every detected
        event marked by its current label state.

        Color encoding (chosen so unlabeled markers contrast sharply
        with the blue trace line — biologists should be able to spot
        "what still needs my attention" at a glance):

            * ``True``     -> green   (``#2ca02c``)
            * ``False``    -> red     (``#d62728``)
            * ``Unsure``   -> gray    (``#7f7f7f``)
            * unlabeled    -> orange  (``#ff7f0e``), with a faint
              same-color FWHM shade so the eye is drawn to outstanding
              work even when the peak is small.

        The previously-skipped state (a labeler hit ``s`` on this trace)
        is rendered as a subtle title annotation rather than per-event
        markers — skip is a trace-level review action, not an event-
        level label, so it shouldn't overload the per-event colors.
        """
        plot_out = self._widgets.get("plot_out")
        if plot_out is None or self._fig is None:
            return
        if not self._rois_in_order or not self._events:
            return

        roi_id = self._rois_in_order[self._roi_cursor]
        trace = self._get_trace(roi_id)

        ax_t = self._ax_trace
        ax_m = self._ax_minimap
        ax_t.clear()
        ax_m.clear()
        # The full trace IS the overview, so the minimap subplot is
        # redundant — hide it so the trace gets the whole figure area.
        ax_m.set_visible(False)
        ax_t.set_visible(True)

        if trace is None or trace.size == 0:
            ax_t.text(
                0.5,
                0.5,
                "ΔF/F trace unavailable\n"
                "(pass filtered_idx to EventLabeler if you ran with\n"
                "non-default p_th/size_threshold).",
                ha="center",
                va="center",
                transform=ax_t.transAxes,
            )
            ax_t.set_xticks([])
            ax_t.set_yticks([])
            self._draw_figure(plot_out)
            return

        n_frames = trace.size
        ax_t.plot(np.arange(n_frames), trace, color="#1f77b4", linewidth=0.8)

        # Distinct hues for each state. The "unlabeled" color must NOT
        # match the trace line color (#1f77b4) or the markers vanish
        # into the trace — that was the v1 bug. Orange picks the
        # opposite end of the matplotlib default cycle.
        label_colors = {
            "True": "#2ca02c",
            "False": "#d62728",
            "Unsure": "#7f7f7f",
        }
        unlabeled_color = "#ff7f0e"

        same_roi = [e for e in self._events if e["roi_id"] == roi_id]
        for e in same_roi:
            pos = e.get("peak_position", -1)
            if not (0 <= pos < n_frames):
                continue
            key = (e["roi_id"], e["event_idx"])
            entry = self._labels.get(key)
            label = entry.get("label") if entry else ""
            color = label_colors.get(label, unlabeled_color)
            is_unlabeled = not label

            # Shade FWHM for unlabeled events so they stand out as work
            # remaining. Labeled events get a thin axvline only to keep
            # the overview readable when ROIs are dense.
            fb = e.get("fwhm_back")
            ff = e.get("fwhm_fwd")
            if (
                is_unlabeled
                and fb is not None
                and ff is not None
                and 0 <= fb <= ff < n_frames
            ):
                ax_t.axvspan(fb, ff, alpha=0.15, color=unlabeled_color)
            ax_t.axvline(pos, color=color, linewidth=0.6, alpha=0.5)
            # Unlabeled events use a slightly larger marker with a thin
            # white edge so they stay legible even when they sit on top
            # of the orange axvline at low ΔF/F.
            if is_unlabeled:
                ax_t.plot(
                    [pos],
                    [trace[pos]],
                    marker="o",
                    color=color,
                    markersize=8,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                )
            else:
                ax_t.plot(
                    [pos],
                    [trace[pos]],
                    marker="o",
                    color=color,
                    markersize=6,
                )

        review_state = self._trace_review_state(roi_id)
        review_marker = {
            "skipped": " · (previously skipped)",
            "rejected": " · (previously rejected)",
            "revisited": " · (skipped, then revisited)",
            None: "",
        }[review_state]
        title = (
            f"Sample {self._sample_id} · ROI {roi_id} · "
            f"trace overview ({len(same_roi)} detected event"
            f"{'s' if len(same_roi) != 1 else ''}){review_marker}"
        )
        rate = self._sampling_rate()
        ax_t.set_title(title)
        ax_t.set_xlabel("frame" if rate is None else f"frame  (rate={rate:g} Hz)")
        ax_t.set_ylabel("ΔF/F")
        ax_t.set_xlim(0, n_frames)
        self._add_overview_legend(ax_t, label_colors, unlabeled_color)

        self._draw_figure(plot_out)

    @staticmethod
    def _add_drill_legend(ax_t: Any) -> None:
        """
        Add a compact in-plot legend to the per-event (drill) axes.

        Drill view uses red for the *current* event's peak (dashed
        vertical line + filled circle), an orange axvspan for the
        FWHM extent, and colored "|" tick marks at the bottom of the
        axes for already-labeled peer events on the same trace.
        Without this legend, biologists routinely conflated the red
        dot with the orange shading or assumed the bottom ticks were
        x-axis decorations.
        """
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        handles = [
            mlines.Line2D(
                [], [],
                color="#d62728", marker="o", linestyle="--",
                markersize=5, label="current event peak",
            ),
            mpatches.Patch(
                facecolor="#ff7f0e", alpha=0.15,
                label="FWHM extent",
            ),
            mlines.Line2D(
                [], [],
                color="#2ca02c", marker="|", linestyle="None",
                markersize=10, label="peer label: True",
            ),
            mlines.Line2D(
                [], [],
                color="#d62728", marker="|", linestyle="None",
                markersize=10, label="peer label: False",
            ),
            mlines.Line2D(
                [], [],
                color="#7f7f7f", marker="|", linestyle="None",
                markersize=10, label="peer label: Unsure",
            ),
        ]
        ax_t.legend(
            handles=handles,
            loc="best",
            fontsize=7,
            framealpha=0.85,
            ncol=2,
            handlelength=1.2,
            borderpad=0.3,
            handletextpad=0.4,
            columnspacing=0.8,
        )

    @staticmethod
    def _add_overview_legend(
        ax_t: Any,
        label_colors: Dict[str, str],
        unlabeled_color: str,
    ) -> None:
        """
        Add a compact in-plot legend to the trace-overview axes.

        The previous version of the overview rendered orange dots,
        a thin vertical orange line, AND a wide orange axvspan for
        unlabeled events with no on-plot key. Biologists could not
        infer that "dot = peak position" and "shaded region = FWHM
        extent" are different annotations of the same event — and
        because the FWHM shading is omitted for already-labeled
        events (and for events whose FWHM bounds the analysis
        couldn't compute), the shading appearing on some traces
        and not others looked like a bug.

        The legend is intentionally compact (small font, two columns,
        ``loc='best'`` so matplotlib picks the least obstructive
        corner). It documents every glyph the overview emits so the
        encoding is fully self-describing.
        """
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        handles = [
            mlines.Line2D(
                [], [],
                color=unlabeled_color, marker="o", linestyle="None",
                markersize=7, markeredgecolor="white", markeredgewidth=0.6,
                label="peak (unlabeled)",
            ),
            mpatches.Patch(
                facecolor=unlabeled_color, alpha=0.15,
                label="FWHM extent (unlabeled only)",
            ),
            mlines.Line2D(
                [], [],
                color=label_colors["True"], marker="o", linestyle="None",
                markersize=5, label="True",
            ),
            mlines.Line2D(
                [], [],
                color=label_colors["False"], marker="o", linestyle="None",
                markersize=5, label="False",
            ),
            mlines.Line2D(
                [], [],
                color=label_colors["Unsure"], marker="o", linestyle="None",
                markersize=5, label="Unsure",
            ),
        ]
        ax_t.legend(
            handles=handles,
            loc="best",
            fontsize=7,
            framealpha=0.85,
            ncol=2,
            handlelength=1.2,
            borderpad=0.3,
            handletextpad=0.4,
            columnspacing=0.8,
        )

    def _draw_figure(self, plot_out: Any) -> None:
        """Re-draw the cached figure inside ``plot_out``, no plt.show()."""
        if self._fig is None:
            return
        # Re-render by clearing the Output and re-displaying the figure.
        # We use IPython.display.display(figure) inside the Output
        # capture, which renders inline without ever touching plt.show().
        from IPython.display import display as _ipy_display

        plot_out.clear_output(wait=True)
        with plot_out:
            self._fig.tight_layout()
            _ipy_display(self._fig)
