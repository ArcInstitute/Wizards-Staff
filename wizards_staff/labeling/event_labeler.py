"""
Notebook-based hand-labeling tool for Wizards-Staff calcium events.

The :class:`EventLabeler` walks a biologist through individually-detected
calcium events on a single :class:`~wizards_staff.wizards.shard.Shard`,
collecting True / False / Unsure judgements via a small ipywidgets UI.
Labels are appended to a canonical CSV corpus that is shared across
datasets and labelers.

Design notes:
    * ``ipywidgets`` is imported lazily inside :meth:`EventLabeler.display`
      so the module is safe to import in headless environments (Lizard
      Wizard's CLI invocations of Wizards-Staff, batch jobs, etc.).
    * Plot rendering is done into an ipywidgets ``Output`` widget; no
      ``plt.show()`` is ever called.
    * Persistence is per-action and atomic: every label triggers a full
      atomic re-write of the corpus CSV via ``os.replace`` so a crash
      mid-write cannot corrupt the file or lose previous work.
"""

from __future__ import annotations

# import
## batteries
import csv
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
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
    (see :func:`wizards_staff.wizards.cauldron._run_all` for the producer)
    and presents them one at a time in a Jupyter notebook for the
    biologist to confirm or reject.

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
            frames). Defaults to 8.0. Used as a display hint only — the
            window is clipped to the trace bounds.
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

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        shard: Any,
        corpus_path: str,
        labeler_id: str,
        context: Optional[Dict[str, Any]] = None,
        window_scale: float = 8.0,
        ordering: str = "by_roi_then_time",
        filtered_idx: Optional[Sequence[int]] = None,
    ) -> None:
        if not labeler_id or not isinstance(labeler_id, str):
            raise ValueError("labeler_id must be a non-empty string")
        if ordering not in self.ORDERINGS:
            raise ValueError(
                f"ordering must be one of {self.ORDERINGS}; got {ordering!r}"
            )
        if window_scale <= 0:
            raise ValueError("window_scale must be > 0")

        self.shard = shard
        self.corpus_path = os.path.abspath(corpus_path)
        self.labeler_id = labeler_id
        self.context: Dict[str, Any] = dict(context) if context else {}
        self.window_scale = float(window_scale)
        self.ordering = ordering

        self._sample_id: str = str(getattr(shard, "sample_name", "unknown"))
        self._logger = getattr(shard, "_logger", None)
        if self._logger is None:
            import logging

            self._logger = logging.getLogger(__name__)

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
        self._load_corpus()

        # Cursor into self._events.
        self._cursor: int = 0

        # Cached widgets / figure handles, populated on display().
        self._widgets: Dict[str, Any] = {}
        self._fig = None
        self._ax_trace = None
        self._ax_minimap = None
        self._ipy = None  # ipywidgets module handle (lazy)

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
              WARNING).
        """
        raw_peaks = list(getattr(self.shard, "_raw_peak_amplitude_data", []) or [])
        raw_fwhm = list(getattr(self.shard, "_raw_fwhm_data", []) or [])
        # Raw rows still carry the legacy "Neuron" key — that's the
        # producer's column name in spellbook / cauldron and is not part
        # of the labeler's public schema. We translate to roi_id below.
        fwhm_by_roi: Dict[Any, Dict[str, Any]] = {
            row.get("Neuron"): row for row in raw_fwhm
        }

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
                self._logger.info(
                    f"EventLabeler: ROI {roi_id} in "
                    f"{self._sample_id} has zero events; skipping."
                )
                continue

            fwhm_row = fwhm_by_roi.get(roi_id_raw, {}) or {}
            fwhm_values = list(fwhm_row.get("FWHM Values", []) or [])
            fwhm_back = list(fwhm_row.get("FWHM Backward Positions", []) or [])
            fwhm_fwd = list(fwhm_row.get("FWHM Forward Positions", []) or [])

            for event_idx, amp in enumerate(amplitudes):
                try:
                    amp_f = float(amp)
                except (TypeError, ValueError):
                    self._logger.warning(
                        f"EventLabeler: skipping non-numeric amplitude "
                        f"sample={self._sample_id} roi={roi_id} "
                        f"event_idx={event_idx}"
                    )
                    continue
                if not np.isfinite(amp_f):
                    self._logger.warning(
                        f"EventLabeler: skipping NaN/Inf amplitude "
                        f"sample={self._sample_id} roi={roi_id} "
                        f"event_idx={event_idx}"
                    )
                    continue

                if event_idx < len(positions):
                    try:
                        pos_i = int(positions[event_idx])
                    except (TypeError, ValueError):
                        pos_i = -1
                else:
                    pos_i = -1

                if event_idx < len(fwhm_values):
                    try:
                        fwhm_v = float(fwhm_values[event_idx])
                    except (TypeError, ValueError):
                        fwhm_v = float("nan")
                else:
                    fwhm_v = float("nan")

                fb = (
                    int(fwhm_back[event_idx])
                    if event_idx < len(fwhm_back)
                    and fwhm_back[event_idx] is not None
                    else None
                )
                ff = (
                    int(fwhm_fwd[event_idx])
                    if event_idx < len(fwhm_fwd)
                    and fwhm_fwd[event_idx] is not None
                    else None
                )

                self._events.append(
                    {
                        "roi_id": roi_id,
                        "event_idx": event_idx,
                        "peak_position": pos_i,
                        "peak_amplitude": amp_f,
                        "fwhm_frames": fwhm_v,
                        "fwhm_back": fb,
                        "fwhm_fwd": ff,
                    }
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
        """Populate ``self._labels`` and ``self._touched_by_others`` from disk."""
        df = self._read_corpus_raw()
        if df.empty:
            return

        sample_mask = df["sample_id"].astype(str) == self._sample_id
        if not sample_mask.any():
            return
        sample_df = df.loc[sample_mask]

        for _, row in sample_df.iterrows():
            try:
                key = (int(row["roi_id"]), int(row["event_idx"]))
            except (TypeError, ValueError):
                continue
            row_labeler = str(row.get("labeler_id", ""))
            if row_labeler == self.labeler_id:
                self._labels[key] = {
                    "label": str(row.get("label", "")),
                    "notes": "" if pd.isna(row.get("notes")) else str(row.get("notes", "")),
                    "timestamp": "" if pd.isna(row.get("timestamp")) else str(row.get("timestamp", "")),
                }
            else:
                self._touched_by_others.add(key)

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

    def _save(self) -> None:
        """
        Atomically rewrite the corpus CSV, preserving rows from other
        labelers and other samples.

        Concurrent labelers on shared storage get last-write-wins
        semantics; this is documented as accepted for v1.
        """
        existing = self._read_corpus_raw()
        # Drop our (sample_id, labeler_id) rows from the loaded corpus —
        # we'll reinstate them from in-memory state below. This makes the
        # CSV rewrite an idempotent function of (other-labelers' rows,
        # other-samples' rows, our in-memory labels).
        if not existing.empty:
            keep_mask = ~(
                (existing["sample_id"].astype(str) == self._sample_id)
                & (existing["labeler_id"].astype(str) == self.labeler_id)
            )
            preserved = existing.loc[keep_mask].copy()
        else:
            preserved = self._empty_corpus()

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

        if new_rows:
            new_df = pd.DataFrame(new_rows, columns=list(self.CORPUS_COLUMNS))
            out = pd.concat([preserved, new_df], ignore_index=True)
        else:
            out = preserved

        out = out[list(self.CORPUS_COLUMNS)]
        _atomic_write_csv(out, self.corpus_path)

    # ------------------------------------------------------------------
    # Public data API
    # ------------------------------------------------------------------
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
        ``self.labeler_id``; rows from other labelers or other samples
        on disk are not included.

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

    # ------------------------------------------------------------------
    # Labeling actions (callable from tests as well as UI)
    # ------------------------------------------------------------------
    def label_current(self, label: str, notes: str = "") -> None:
        """
        Record a label for the event under the cursor and advance.

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
        ev = self._events[self._cursor]
        key = (ev["roi_id"], ev["event_idx"])
        self._labels[key] = {
            "label": label,
            "notes": notes or "",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._save()
        self._advance(+1)

    def reject_whole_trace(self) -> None:
        """
        Label every still-unlabeled event on the current ROI's trace
        as False.

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
        """
        if not self._events:
            return
        current_roi = self._events[self._cursor]["roi_id"]
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
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
        if touched:
            self._save()
            self._logger.info(
                f"EventLabeler: rejected {touched} unlabeled events on "
                f"ROI {current_roi}'s trace ({self._sample_id})."
            )
        # Advance past the current ROI.
        next_idx = self._cursor
        while (
            next_idx < len(self._events)
            and self._events[next_idx]["roi_id"] == current_roi
        ):
            next_idx += 1
        self._cursor = min(next_idx, max(0, len(self._events) - 1))
        self._refresh_ui()

    def _advance(self, step: int) -> None:
        """Move the cursor by ``step`` and refresh the UI if displayed."""
        if not self._events:
            return
        self._cursor = max(0, min(len(self._events) - 1, self._cursor + step))
        self._refresh_ui()

    # ------------------------------------------------------------------
    # Trace / display helpers
    # ------------------------------------------------------------------
    def _resolve_filtered_idx(self) -> Optional[List[int]]:
        """
        Best-effort recovery of the filtered_idx mapping.

        Returns ``None`` if neither an explicit mapping nor the inputs
        needed to recompute it are available; callers must handle the
        ``None`` case gracefully (e.g. by skipping the trace plot).
        """
        if self._filtered_idx is not None:
            return self._filtered_idx
        try:
            # Defaults match the legacy run_all defaults; if the user ran
            # with non-default p_th / size_threshold they should pass
            # filtered_idx explicitly to EventLabeler.
            idx = self.shard.spatial_filtering(
                p_th=75, size_threshold=20000, plot=False, silence=True
            )
            self._filtered_idx = [int(i) for i in idx]
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

    def _window_for_event(
        self, ev: Dict[str, Any], n_frames: int
    ) -> Tuple[int, int]:
        """
        Compute (start, end) frame indices of the trace window around ``ev``.

        Window half-width is ``window_scale * max(fwhm_frames, 5)``.
        Clipped to ``[0, n_frames]``; for very short traces the window
        collapses to the whole trace.
        """
        fwhm = ev.get("fwhm_frames")
        if fwhm is None or not np.isfinite(fwhm) or fwhm <= 0:
            half = int(round(self.window_scale * 10))
        else:
            half = max(1, int(round(self.window_scale * max(float(fwhm), 5))))
        peak = ev.get("peak_position", 0)
        if not isinstance(peak, (int, np.integer)) or peak < 0:
            peak = 0
        start = max(0, int(peak) - half)
        end = min(n_frames, int(peak) + half + 1)
        if end - start < 5 and n_frames > 0:
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

        try:
            from IPython.display import display as _ipy_display
        except ImportError as exc:  # pragma: no cover  — IPython ships with notebooks
            raise ImportError(
                "EventLabeler.display() requires IPython."
            ) from exc

        self._ipy = widgets

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

        # Buttons.
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
        btn_reject_trace = widgets.Button(
            description="Reject whole trace (w)",
            tooltip=(
                "Label every unlabeled event on the CURRENT ROI's trace "
                "as False (only this ROI — not other ROIs, not the "
                "whole sample). Does not mark the ROI itself bad — "
                "that's outlier-detection territory."
            ),
        )
        btn_prev = widgets.Button(description="Prev (k)")
        btn_next = widgets.Button(description="Next (j)")

        notes = widgets.Text(
            value="",
            placeholder="Optional notes for the next label",
            description="Notes:",
            layout=widgets.Layout(width="60%"),
        )

        # "Command box" — the cheapest way to get keyboard shortcuts in a
        # standard ipywidgets stack without custom JS. The biologist clicks
        # into this single-character text field and types t/f/u/j/k/w.
        cmd = widgets.Text(
            value="",
            placeholder="Click here, then press t/f/u/j/k/w",
            description="Keys:",
            layout=widgets.Layout(width="40%"),
        )

        progress = widgets.HTML()
        details = widgets.HTML()

        # Wire up callbacks.
        def _on_label(label: str) -> None:
            self.label_current(label, notes=notes.value)
            notes.value = ""

        btn_true.on_click(lambda _b: _on_label("True"))
        btn_false.on_click(lambda _b: _on_label("False"))
        btn_unsure.on_click(lambda _b: _on_label("Unsure"))
        btn_reject_trace.on_click(lambda _b: self.reject_whole_trace())
        btn_prev.on_click(lambda _b: self._advance(-1))
        btn_next.on_click(lambda _b: self._advance(+1))

        def _on_cmd_change(change: Dict[str, Any]) -> None:
            value = change.get("new", "") or ""
            if not value:
                return
            ch = value.strip().lower()[:1]
            # Always reset the box, regardless of dispatch outcome.
            cmd.value = ""
            mapping = {
                "t": lambda: _on_label("True"),
                "f": lambda: _on_label("False"),
                "u": lambda: _on_label("Unsure"),
                "j": lambda: self._advance(+1),
                "k": lambda: self._advance(-1),
                "w": lambda: self.reject_whole_trace(),
            }
            action = mapping.get(ch)
            if action is not None:
                action()

        cmd.observe(_on_cmd_change, names="value")

        controls = widgets.HBox([btn_true, btn_false, btn_unsure, btn_reject_trace])
        nav = widgets.HBox([btn_prev, btn_next, progress])
        meta = widgets.HBox([notes, cmd])

        root = widgets.VBox([progress, plot_out, controls, nav, meta, details])

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
            "btn_reject_trace": btn_reject_trace,
            "btn_prev": btn_prev,
            "btn_next": btn_next,
        }

        self._refresh_ui()
        _ipy_display(root)

    def _refresh_ui(self) -> None:
        """Repaint the figure and progress strings, if the UI is built."""
        if not self._widgets:
            return
        self._refresh_progress()
        self._refresh_figure()
        self._refresh_details()

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
        ev = self._events[self._cursor]
        roi_rank = self._rois_in_order.index(ev["roi_id"]) + 1
        total_rois = len(self._rois_in_order) or 1
        # Per-ROI event progress (chronological within ROI).
        same_roi = [e for e in self._events if e["roi_id"] == ev["roi_id"]]
        try:
            within = same_roi.index(ev) + 1
        except ValueError:
            within = 1
        prog.value = (
            f"<b>{self._sample_id}</b> &middot; "
            f"ROI {roi_rank} of {total_rois} "
            f"(id={ev['roi_id']}) &middot; "
            f"Event {within} of {len(same_roi)} &middot; "
            f"<b>{labeled}/{total}</b> total labeled "
            f"&middot; ordering={self.ordering}"
        )

    def _refresh_details(self) -> None:
        details = self._widgets.get("details")
        if details is None:
            return
        if not self._events:
            details.value = ""
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
        details.value = (
            f"peak_amp={ev['peak_amplitude']:.4f} &middot; "
            f"fwhm={ev['fwhm_frames']:.2f} frames &middot; "
            f"peak_frame={ev['peak_position']} &middot; {prior_self}{other_hint}"
        )

    def _refresh_figure(self) -> None:
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

        ax_t.set_title(
            f"Sample {self._sample_id} &middot; ROI {ev['roi_id']} "
            f"&middot; event {ev['event_idx']}".replace("&middot;", "·")
        )
        ax_t.set_xlabel("frame")
        ax_t.set_ylabel("ΔF/F")
        ax_t.set_xlim(start, max(end - 1, start + 1))

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
