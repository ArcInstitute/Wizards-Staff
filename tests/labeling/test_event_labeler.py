"""
Tests for :mod:`wizards_staff.labeling.event_labeler`.

These tests deliberately avoid importing the top-level ``wizards_staff``
package so they remain runnable in environments where the heavy-weight
core dependencies (``caiman``, ``tensorflow``, etc.) are not installed.
The tests load ``event_labeler.py`` as a stand-alone module via
:mod:`importlib.util`, mirroring the way the labeling subpackage is
intended to be importable in headless contexts.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest import mock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Module loader — keeps the test suite independent of caiman / tensorflow.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
EVENT_LABELER_PATH = REPO_ROOT / "wizards_staff" / "labeling" / "event_labeler.py"


def _load_event_labeler_module(extra_sys_modules: Optional[Dict[str, Any]] = None):
    """
    Import ``event_labeler.py`` as a standalone module.

    Optionally inject ``extra_sys_modules`` into :data:`sys.modules` for the
    duration of the import (used to simulate ipywidgets being absent or
    present without disturbing the global Python environment).
    """
    spec = importlib.util.spec_from_file_location(
        "wizards_staff_labeling_event_labeler_under_test",
        str(EVENT_LABELER_PATH),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    saved: Dict[str, Any] = {}
    if extra_sys_modules:
        for k, v in extra_sys_modules.items():
            saved[k] = sys.modules.get(k)
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    try:
        spec.loader.exec_module(module)
    finally:
        if extra_sys_modules:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
    return module


@pytest.fixture(scope="module")
def event_labeler_module():
    return _load_event_labeler_module()


@pytest.fixture
def EventLabeler(event_labeler_module):
    return event_labeler_module.EventLabeler


# ---------------------------------------------------------------------------
# Fake shard helpers.
# ---------------------------------------------------------------------------
def make_fake_shard(
    sample_name: str = "sampleA",
    roi_specs: Optional[List[Dict[str, Any]]] = None,
    n_frames: int = 200,
    inject_bad_event: bool = False,
):
    """
    Build a minimal Shard-like object that the EventLabeler can consume.

    Args:
        sample_name: Stored as ``shard.sample_name``.
        roi_specs: Sequence of dicts each describing one ROI with
            keys ``roi_id`` (local index), ``positions``, ``amplitudes``,
            ``fwhm``. Lists must be the same length within an ROI.
        n_frames: Width of the synthetic ``dff_dat`` matrix.
        inject_bad_event: If True, append a NaN amplitude to the first
            ROI to exercise the scrubbing path.
    """
    if roi_specs is None:
        roi_specs = [
            {
                "roi_id": 0,
                "positions": [10, 50, 120],
                "amplitudes": [0.5, 1.2, 0.3],
                "fwhm": [4.0, 6.0, 3.0],
            },
            {
                "roi_id": 1,
                "positions": [25, 80],
                "amplitudes": [0.9, 0.4],
                "fwhm": [5.0, 4.5],
            },
        ]

    raw_peaks: List[Dict[str, Any]] = []
    raw_fwhm: List[Dict[str, Any]] = []
    for spec in roi_specs:
        amps = list(spec["amplitudes"])
        positions = list(spec["positions"])
        fwhms = list(spec["fwhm"])
        if inject_bad_event and spec is roi_specs[0]:
            amps = amps + [float("nan")]
            positions = positions + [n_frames - 1]
            fwhms = fwhms + [3.0]
        # Raw rows still use the legacy "Neuron" key (the spellbook /
        # cauldron producer hasn't been renamed); the labeler translates
        # this to the public ``roi_id`` corpus column.
        raw_peaks.append(
            {
                "Sample": sample_name,
                "Neuron": spec["roi_id"],
                "Peak Amplitudes": amps,
                "Peak Positions": positions,
                "is_outlier": False,
            }
        )
        raw_fwhm.append(
            {
                "Sample": sample_name,
                "Neuron": spec["roi_id"],
                "FWHM Backward Positions": [max(0, p - int(f)) for p, f in zip(positions, fwhms)],
                "FWHM Forward Positions": [min(n_frames - 1, p + int(f)) for p, f in zip(positions, fwhms)],
                "FWHM Values": fwhms,
                "Spike Counts": [1] * len(amps),
                "is_outlier": False,
            }
        )

    n_rois = len(roi_specs)
    rng = np.random.default_rng(0)
    dff = rng.normal(0.0, 0.05, size=(n_rois, n_frames))
    for i, spec in enumerate(roi_specs):
        for p, a in zip(spec["positions"], spec["amplitudes"]):
            if 0 <= p < n_frames:
                dff[i, p] += a

    inputs = {"dff_dat": dff}

    shard = SimpleNamespace(
        sample_name=sample_name,
        _raw_peak_amplitude_data=raw_peaks,
        _raw_fwhm_data=raw_fwhm,
        _logger=logging.getLogger(f"test.shard.{sample_name}"),
        get_input=lambda name, req=False: inputs.get(name),
        spatial_filtering=lambda **_kw: list(range(n_rois)),
    )
    return shard


# ---------------------------------------------------------------------------
# 1. Module imports without ipywidgets installed.
# ---------------------------------------------------------------------------
def test_module_imports_without_ipywidgets():
    """
    The labeling module must be importable in headless environments where
    ``ipywidgets`` is not installed. We simulate that by hiding any
    cached ``ipywidgets`` module and then loading event_labeler.py with a
    custom import hook that raises ImportError on ``import ipywidgets``.
    """
    blocker = mock.MagicMock()
    blocker.find_spec = lambda name, *a, **kw: None  # noop

    saved_ipy = sys.modules.pop("ipywidgets", None)
    try:
        # Re-execute the module load with ipywidgets absent.
        module = _load_event_labeler_module(extra_sys_modules={"ipywidgets": None})
        # Importing must succeed even though ipywidgets is missing.
        assert hasattr(module, "EventLabeler")

        # Calling display() without ipywidgets must raise a clear error.
        shard = make_fake_shard()
        labeler = module.EventLabeler(
            shard=shard,
            corpus_path=str(Path(os.devnull)),  # never written by display()
            labeler_id="alice",
        )
        # Patch the import inside display() to raise as if ipywidgets is
        # missing: stash a sentinel that triggers ImportError on attr.
        with mock.patch.dict(sys.modules, {"ipywidgets": None}):
            with pytest.raises(ImportError) as excinfo:
                labeler.display()
            assert "wizards_staff[labeling]" in str(excinfo.value)
    finally:
        if saved_ipy is not None:
            sys.modules["ipywidgets"] = saved_ipy


# ---------------------------------------------------------------------------
# 2. Corpus round-trip: label, save, reload, verify.
# ---------------------------------------------------------------------------
def test_corpus_roundtrip(EventLabeler, tmp_path):
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard()
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(corpus),
        labeler_id="alice",
        context={"sampling_rate": 30, "indicator": "GCaMP6f"},
    )

    labeler.label_current("True", notes="clean peak")
    labeler.label_current("False")
    labeler.label_current("Unsure", notes="ambiguous shape")

    assert corpus.exists()
    df = pd.read_csv(corpus)
    assert len(df) == 3
    assert set(df["label"]) == {"True", "False", "Unsure"}
    # Schema has all expected columns in the documented order.
    assert list(df.columns) == list(EventLabeler.CORPUS_COLUMNS)
    # Context values made it onto every row.
    assert (df["sampling_rate"].astype(str) == "30").all()
    assert (df["indicator"] == "GCaMP6f").all()

    # Reload the labeler and confirm prior labels are restored.
    shard2 = make_fake_shard()
    labeler2 = EventLabeler(
        shard=shard2,
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    assert labeler2.labeled_count == 3
    exported = labeler2.export_labels()
    assert len(exported) == 3
    assert set(exported["label"]) == {"True", "False", "Unsure"}


# ---------------------------------------------------------------------------
# 3. Atomic write: simulate a crash mid-write.
# ---------------------------------------------------------------------------
def test_atomic_write_preserves_corpus_on_crash(EventLabeler, tmp_path):
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard()
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(corpus),
        labeler_id="alice",
    )

    # First successful write establishes a known-good baseline.
    labeler.label_current("True")
    baseline_bytes = corpus.read_bytes()
    baseline_df = pd.read_csv(corpus)
    assert len(baseline_df) == 1

    # Simulate a crash in os.replace: the temp file is written but the
    # rename never lands. Verify that (a) the call propagates the error,
    # and (b) the original corpus is left intact (no partial overwrite,
    # no empty file).
    real_replace = os.replace

    def boom(_src, _dst):
        raise RuntimeError("simulated power loss")

    with mock.patch("os.replace", side_effect=boom):
        with pytest.raises(RuntimeError, match="simulated power loss"):
            labeler.label_current("False")

    after_bytes = corpus.read_bytes()
    assert after_bytes == baseline_bytes, (
        "Atomic write contract violated: corpus changed despite os.replace failure"
    )

    # No leftover .tmp files in the corpus directory.
    leftover = list(tmp_path.glob(".event_labeler_*.csv.tmp"))
    assert leftover == [], f"Temp files leaked on crash: {leftover}"

    # Labeler can still recover. The in-memory label for the "crashed"
    # event is retained, so the next successful save flushes both the
    # original True (still in-memory) and the post-crash relabel.
    labeler.label_current("Unsure")
    df_after = pd.read_csv(corpus)
    # Two events have been touched (0,0) → True and (0,1) → Unsure
    # (overwriting the in-memory False that never made it to disk).
    assert len(df_after) == 2
    by_event = {(int(r["roi_id"]), int(r["event_idx"])): r["label"] for _, r in df_after.iterrows()}
    assert by_event[(0, 0)] == "True"
    assert by_event[(0, 1)] == "Unsure"


# ---------------------------------------------------------------------------
# 4. Resume: prior labels restored.
# ---------------------------------------------------------------------------
def test_resume_restores_prior_labels(EventLabeler, tmp_path):
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard()
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    labeler.label_current("True")
    labeler.label_current("False")

    # Reopen and confirm cursor starts at 0 but prior events are remembered
    # and visible via the in-memory map / export_labels.
    labeler2 = EventLabeler(
        shard=make_fake_shard(), corpus_path=str(corpus), labeler_id="alice"
    )
    assert labeler2.labeled_count == 2
    keys = {(r["roi_id"], r["event_idx"]) for _, r in labeler2.export_labels().iterrows()}
    assert (0, 0) in keys and (0, 1) in keys


def test_other_labelers_are_not_pre_populated(EventLabeler, tmp_path):
    """
    Labels written by another labeler must not pre-populate the current
    labeler's session, but the labeler should still note that the events
    have been touched (without revealing the other label).
    """
    corpus = tmp_path / "labels.csv"
    # First labeler writes a few labels.
    bob = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(corpus),
        labeler_id="bob",
    )
    bob.label_current("True")
    bob.label_current("False")

    # Second labeler opens the same corpus.
    alice = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    assert alice.labeled_count == 0
    # Touched-by-others set should contain bob's events.
    assert (0, 0) in alice._touched_by_others
    assert (0, 1) in alice._touched_by_others


# ---------------------------------------------------------------------------
# 5. Re-labeling updates in place, does not append.
# ---------------------------------------------------------------------------
def test_relabel_updates_in_place(EventLabeler, tmp_path):
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard()
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    # Label first event.
    labeler.label_current("True", notes="initial")
    assert pd.read_csv(corpus).shape[0] == 1

    # Re-label the same event by rewinding the cursor.
    labeler._cursor = 0
    labeler.label_current("False", notes="changed my mind")

    # Use the canonical loader so the label column is read as str even
    # when pandas would otherwise infer bool from a single-row file.
    df = EventLabeler.load_corpus(str(corpus))
    # Still exactly one row for (sampleA, ROI 0, event 0, alice).
    matching = df[
        (df["sample_id"] == "sampleA")
        & (df["roi_id"] == 0)
        & (df["event_idx"] == 0)
        & (df["labeler_id"] == "alice")
    ]
    assert len(matching) == 1
    assert matching.iloc[0]["label"] == "False"
    assert matching.iloc[0]["notes"] == "changed my mind"


# ---------------------------------------------------------------------------
# 6. Schema version mismatch fails loudly.
# ---------------------------------------------------------------------------
def test_schema_version_mismatch_raises(EventLabeler, tmp_path):
    corpus = tmp_path / "labels.csv"
    # Write a corpus with a different version. We use the canonical
    # column set so the only mismatch is the version itself.
    bogus = pd.DataFrame(
        [
            {col: "" for col in EventLabeler.CORPUS_COLUMNS},
        ]
    )
    bogus["corpus_version"] = 99
    bogus["sample_id"] = "old_sample"
    bogus["roi_id"] = 0
    bogus["event_idx"] = 0
    bogus["label"] = "True"
    bogus["labeler_id"] = "old_user"
    bogus.to_csv(corpus, index=False)

    with pytest.raises(RuntimeError, match="corpus_version"):
        EventLabeler(
            shard=make_fake_shard(),
            corpus_path=str(corpus),
            labeler_id="alice",
        )


def test_corpus_missing_version_column_raises(EventLabeler, tmp_path):
    corpus = tmp_path / "labels.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(corpus, index=False)
    with pytest.raises(RuntimeError, match="corpus_version"):
        EventLabeler(
            shard=make_fake_shard(),
            corpus_path=str(corpus),
            labeler_id="alice",
        )


# ---------------------------------------------------------------------------
# 7. Ordering modes produce expected orderings.
# ---------------------------------------------------------------------------
def _ordering_specs():
    return [
        {
            "roi_id": 0,
            "positions": [10, 50, 120],
            "amplitudes": [0.1, 0.5, 0.9],
            "fwhm": [3.0, 4.0, 5.0],
        },
        {
            "roi_id": 1,
            "positions": [25, 80],
            "amplitudes": [0.3, 0.7],
            "fwhm": [3.0, 4.0],
        },
    ]


def test_ordering_by_roi_then_time(EventLabeler, tmp_path):
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
        ordering="by_roi_then_time",
    )
    keys = [(e["roi_id"], e["event_idx"]) for e in labeler.events]
    assert keys == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]


def test_ordering_amplitude_ascending_descending(EventLabeler, tmp_path):
    shard = make_fake_shard(roi_specs=_ordering_specs())
    asc = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "asc.csv"),
        labeler_id="alice",
        ordering="amplitude_ascending",
    )
    amps_asc = [e["peak_amplitude"] for e in asc.events]
    assert amps_asc == sorted(amps_asc)

    desc = EventLabeler(
        shard=make_fake_shard(roi_specs=_ordering_specs()),
        corpus_path=str(tmp_path / "desc.csv"),
        labeler_id="alice",
        ordering="amplitude_descending",
    )
    amps_desc = [e["peak_amplitude"] for e in desc.events]
    assert amps_desc == sorted(amps_desc, reverse=True)


def test_ordering_stratified_walks_rois(EventLabeler, tmp_path):
    # Many events per ROI so the quintile bucketing has work to do.
    specs = [
        {
            "roi_id": 0,
            "positions": list(range(10, 110, 10)),
            "amplitudes": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "fwhm": [3.0] * 10,
        },
        {
            "roi_id": 1,
            "positions": [15, 35, 55, 75, 95],
            "amplitudes": [0.2, 0.4, 0.6, 0.8, 1.0],
            "fwhm": [3.0] * 5,
        },
    ]
    shard = make_fake_shard(roi_specs=specs, n_frames=200)
    lab = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "strat.csv"),
        labeler_id="alice",
        ordering="stratified",
    )
    keys = [(e["roi_id"], e["event_idx"]) for e in lab.events]
    # Same total event count as inputs.
    assert len(keys) == 15
    # All ROI-0 events come before any ROI-1 event (stratified walks
    # ROIs in encounter order).
    n0 = [k for k in keys if k[0] == 0]
    n1 = [k for k in keys if k[0] == 1]
    assert keys == n0 + n1
    # First emitted ROI-0 event should be from the highest quintile
    # (round-robin starts from the top to maximise calibration coverage).
    first_n0_event_idx = n0[0][1]
    assert lab.events[0]["peak_amplitude"] == pytest.approx(1.0)
    assert first_n0_event_idx == 9


# ---------------------------------------------------------------------------
# 8. "Reject whole trace" labels all remaining events on that ROI's trace.
# ---------------------------------------------------------------------------
def test_reject_whole_trace(EventLabeler, tmp_path):
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    # Manually label the first event True, then reject the whole trace.
    labeler.label_current("True")
    # Cursor now at ROI 0, event 1.
    # ``confirm=True`` bypasses the two-press confirmation gate — that
    # behavior is exercised separately in ``test_reject_whole_trace_*
    # _gate``.
    labeler.reject_whole_trace(confirm=True)

    df = EventLabeler.load_corpus(str(corpus))
    # Event-level rows only — the trace-level review row (event_idx=-1)
    # is checked separately below.
    n0_events = df[(df["roi_id"] == 0) & (df["event_idx"] >= 0)].sort_values(
        "event_idx"
    )
    assert list(n0_events["label"]) == ["True", "False", "False"]
    rejects = n0_events[n0_events["label"] == "False"]
    assert (rejects["notes"] == "whole_trace_reject").all()
    # The first event was already labeled True and must be untouched.
    assert n0_events.iloc[0]["label"] == "True"
    assert n0_events.iloc[0]["notes"] != "whole_trace_reject"

    # reject_whole_trace also writes a trace-level review row so the
    # "was this trace reviewed at the trace level?" predicate is a
    # single sentinel-row scan regardless of skip vs reject.
    n0_trace = df[(df["roi_id"] == 0) & (df["event_idx"] == -1)]
    assert len(n0_trace) == 1
    assert n0_trace.iloc[0]["notes"] == "trace_reject"

    # Cursor advanced past ROI 0.
    assert labeler._events[labeler._cursor]["roi_id"] == 1
    # ROI cursor tracked the advance; view returned to overview.
    assert labeler._rois_in_order[labeler._roi_cursor] == 1
    assert labeler._view == "overview"


# ---------------------------------------------------------------------------
# 8a. Undo for the most recent reject_whole_trace commit.
# ---------------------------------------------------------------------------
def test_undo_trace_rejection_restores_prior_state(EventLabeler, tmp_path):
    """
    After reject_whole_trace, undo_trace_rejection must restore every
    event's prior label entry, the prior trace-action sentinel, and
    the cursor / view that were active at commit time.
    """
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    # Pre-existing state on ROI 0: one event labeled True, one Unsure.
    labeler.label_current("True")
    labeler.label_current("Unsure")

    cursor_before = labeler._cursor
    roi_cursor_before = labeler._roi_cursor
    view_before = labeler._view

    labeler.reject_whole_trace(confirm=True)

    # Sanity: the rejection actually fired.
    assert labeler._trace_actions.get(0, {}).get("notes") == "trace_reject"
    rejected_keys = [
        (e["roi_id"], e["event_idx"]) for e in labeler._events
        if e["roi_id"] == 0
    ]
    n_false = sum(
        1 for k in rejected_keys
        if labeler._labels.get(k, {}).get("label") == "False"
    )
    assert n_false == 1, "exactly one previously-unlabeled event was rejected"

    # Undo.
    assert labeler.undo_trace_rejection() is True

    # Trace sentinel removed (there was no prior trace_action).
    assert 0 not in labeler._trace_actions
    # Pre-existing labels restored intact.
    assert labeler._labels[(0, 0)]["label"] == "True"
    assert labeler._labels[(0, 1)]["label"] == "Unsure"
    # The previously-unlabeled event is back to having no entry.
    assert (0, 2) not in labeler._labels
    # Cursor / view restored.
    assert labeler._cursor == cursor_before
    assert labeler._roi_cursor == roi_cursor_before
    assert labeler._view == view_before

    # Snapshot is consumed — second undo is a no-op.
    assert labeler.undo_trace_rejection() is False


def test_undo_persists_to_corpus(EventLabeler, tmp_path):
    """The undo's _save() rewrites the corpus to the restored state."""
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    labeler.reject_whole_trace(confirm=True)

    df_after_reject = EventLabeler.load_corpus(str(corpus))
    assert (
        df_after_reject[
            (df_after_reject["roi_id"] == 0)
            & (df_after_reject["event_idx"] >= 0)
            & (df_after_reject["label"] == "False")
        ].shape[0]
        > 0
    )
    assert (
        df_after_reject[
            (df_after_reject["roi_id"] == 0)
            & (df_after_reject["event_idx"] == -1)
        ].shape[0]
        == 1
    )

    assert labeler.undo_trace_rejection() is True

    df_after_undo = EventLabeler.load_corpus(str(corpus))
    # Every False-labeled row from the rejection is gone from disk.
    n0_events = df_after_undo[
        (df_after_undo["roi_id"] == 0)
        & (df_after_undo["event_idx"] >= 0)
    ]
    assert n0_events.empty
    # The trace-level reject sentinel is gone too.
    n0_sentinels = df_after_undo[
        (df_after_undo["roi_id"] == 0)
        & (df_after_undo["event_idx"] == -1)
    ]
    assert n0_sentinels.empty


def test_undo_preserves_other_labelers_rows(EventLabeler, tmp_path):
    """Undo must not erase rows owned by other labelers."""
    corpus = tmp_path / "labels.csv"
    bob = EventLabeler(
        shard=make_fake_shard(roi_specs=_ordering_specs()),
        corpus_path=str(corpus),
        labeler_id="bob",
    )
    bob.label_current("True")
    bob.label_current("False")

    alice = EventLabeler(
        shard=make_fake_shard(roi_specs=_ordering_specs()),
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    alice.reject_whole_trace(confirm=True)
    assert alice.undo_trace_rejection() is True

    df = EventLabeler.load_corpus(str(corpus))
    bob_rows = df[df["labeler_id"] == "bob"]
    assert len(bob_rows) == 2, (
        "undo must rewrite the corpus from alice's in-memory state "
        "while preserving bob's rows untouched"
    )


def test_undo_preserves_prior_skip_trace_action(EventLabeler, tmp_path):
    """If the trace had a prior trace_skip, undo restores it (not removes it)."""
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    # First skip the trace, then navigate back and reject it. The
    # reject overwrites the prior skip; undo must put the skip back.
    labeler.skip_trace()
    assert labeler._trace_actions[0]["notes"] == "trace_skip"
    labeler.prev_trace()
    labeler.reject_whole_trace(confirm=True)
    assert labeler._trace_actions[0]["notes"] == "trace_reject"

    assert labeler.undo_trace_rejection() is True
    assert labeler._trace_actions[0]["notes"] == "trace_skip"


def test_undo_window_closed_by_label_current(EventLabeler, tmp_path):
    """Labeling an event closes the undo window for the prior reject."""
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    labeler.reject_whole_trace(confirm=True)
    assert labeler._last_reject_snapshot is not None
    # Move into per-event view on the next ROI and label something.
    labeler.investigate_trace()
    labeler.label_current("True")
    assert labeler._last_reject_snapshot is None
    # Undo is now a no-op.
    assert labeler.undo_trace_rejection() is False


def test_undo_window_closed_by_skip_trace(EventLabeler, tmp_path):
    """Skipping a trace also closes the undo window for the prior reject."""
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    labeler.reject_whole_trace(confirm=True)
    assert labeler._last_reject_snapshot is not None
    # Cursor is now on ROI 1's first event in overview view; skip it.
    labeler.skip_trace()
    assert labeler._last_reject_snapshot is None
    assert labeler.undo_trace_rejection() is False


def test_undo_window_replaced_by_second_reject(EventLabeler, tmp_path):
    """Only the most recent reject is undoable — earlier ones aren't."""
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    labeler.reject_whole_trace(confirm=True)
    snap1 = labeler._last_reject_snapshot
    assert snap1 is not None and snap1["roi_id"] == 0

    # Cursor is now on ROI 1; reject again.
    labeler.reject_whole_trace(confirm=True)
    snap2 = labeler._last_reject_snapshot
    assert snap2 is not None and snap2["roi_id"] == 1
    assert snap2 is not snap1, "second reject must replace, not stack"

    # Undo should reverse only the most recent (ROI 1).
    assert labeler.undo_trace_rejection() is True
    assert 1 not in labeler._trace_actions
    # ROI 0's rejection from earlier remains in place.
    assert labeler._trace_actions[0]["notes"] == "trace_reject"


def test_undo_after_navigation_still_works(EventLabeler, tmp_path):
    """Pure navigation (next_trace) must NOT close the undo window."""
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    labeler.reject_whole_trace(confirm=True)
    # reject_whole_trace's own commit advances the cursor onto ROI 1's
    # overview. From there a manual next/prev shouldn't close undo.
    labeler.next_trace()
    labeler.prev_trace()
    labeler.investigate_trace()
    labeler.back_to_overview()
    assert labeler._last_reject_snapshot is not None
    assert labeler.undo_trace_rejection() is True
    assert 0 not in labeler._trace_actions


def test_undo_with_no_snapshot_is_noop(EventLabeler, tmp_path):
    corpus = tmp_path / "labels.csv"
    labeler = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    assert labeler.undo_trace_rejection() is False
    # The corpus may legitimately have been touched by _save during a
    # spurious in-memory state, but the labeled count must remain 0.
    assert labeler.labeled_count == 0


# ---------------------------------------------------------------------------
# 8c. In-plot legends explain the dot vs. shaded-FWHM encoding.
# ---------------------------------------------------------------------------
def test_overview_figure_has_legend(EventLabeler, tmp_path):
    """
    The trace overview must paint an in-plot legend so biologists
    don't have to guess at the dot vs. axvspan encoding. Pinned so a
    refactor that removes ``_add_overview_legend`` doesn't silently
    revert the UX fix that made the encoding self-describing.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    shard = make_fake_shard()
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    fig, (ax_t, ax_m) = plt.subplots(2, 1)
    try:
        labeler._fig = fig
        labeler._ax_trace = ax_t
        labeler._ax_minimap = ax_m
        labeler._widgets = {"plot_out": object()}
        # ``_refresh_*_figure`` ends with a ``_draw_figure`` call that
        # lazy-imports IPython.display; the test env may not have
        # IPython installed. Stubbing keeps the test focused on the
        # legend (the artist mutations on ax_t happen before
        # _draw_figure runs).
        with mock.patch.object(labeler, "_draw_figure"):
            labeler._refresh_overview_figure()
        legend = ax_t.get_legend()
        assert legend is not None, "overview must paint an in-plot legend"
        labels = [t.get_text() for t in legend.get_texts()]
        # The two visual elements that confused biologists most —
        # the orange dot and the orange axvspan — must be explicitly
        # named in the legend.
        assert any("peak" in lbl for lbl in labels)
        assert any("FWHM" in lbl for lbl in labels)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 8d. Auto-skip past already-reviewed traces
#     Resume-on-open and post-action advance must skip over ROIs that
#     this labeler has already labeled / skipped / rejected. Manual
#     next_trace / prev_trace still walk every trace (so reviewed work
#     remains reachable for verification).
# ---------------------------------------------------------------------------
def _three_roi_specs():
    """Three ROIs of three events each — handy for skip/reject tests."""
    return [
        {
            "roi_id": 0,
            "positions": [10, 50, 90],
            "amplitudes": [0.5, 0.7, 0.9],
            "fwhm": [3.0, 4.0, 5.0],
        },
        {
            "roi_id": 1,
            "positions": [25, 65, 105],
            "amplitudes": [0.3, 0.5, 0.7],
            "fwhm": [3.0, 4.0, 5.0],
        },
        {
            "roi_id": 2,
            "positions": [35, 75, 115],
            "amplitudes": [0.4, 0.6, 0.8],
            "fwhm": [3.0, 4.0, 5.0],
        },
    ]


def test_resume_lands_on_first_unreviewed_roi(EventLabeler, tmp_path):
    """Re-opening an EventLabeler resumes on the first ROI with unfinished work."""
    corpus = tmp_path / "labels.csv"
    # Pre-reject ROI 0 entirely so it has no unfinished events.
    pre = EventLabeler(
        shard=make_fake_shard(roi_specs=_three_roi_specs()),
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    pre.reject_whole_trace(confirm=True)
    assert 0 in pre._trace_actions

    # Reopen — without resume-on-open the cursor would land on ROI 0
    # (now in "(previously rejected)" state). With it, the cursor
    # lands on the first ROI with unfinished work, i.e. ROI 1.
    fresh = EventLabeler(
        shard=make_fake_shard(roi_specs=_three_roi_specs()),
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    assert fresh._rois_in_order[fresh._roi_cursor] == 1
    assert fresh._events[fresh._cursor]["roi_id"] == 1


def test_resume_falls_back_when_all_reviewed(EventLabeler, tmp_path):
    """When every ROI is reviewed, resume falls back to ROI 0 cleanly."""
    corpus = tmp_path / "labels.csv"
    pre = EventLabeler(
        shard=make_fake_shard(roi_specs=_three_roi_specs()),
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    pre.reject_whole_trace(confirm=True)  # ROI 0
    pre.reject_whole_trace(confirm=True)  # ROI 1
    pre.reject_whole_trace(confirm=True)  # ROI 2

    fresh = EventLabeler(
        shard=make_fake_shard(roi_specs=_three_roi_specs()),
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    # No unreviewed ROI exists; cursor stays at the default 0 so the
    # labeler still has a valid render target. The wrapper would
    # detect ``is_complete`` here and auto-advance to the next image.
    assert fresh._roi_cursor == 0
    assert fresh.is_complete is True


def test_skip_trace_skips_past_already_reviewed_traces(EventLabeler, tmp_path):
    """skip_trace's post-action advance must hop over reviewed ROIs."""
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_three_roi_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    # Pre-reject ROI 1 so the auto-advance from ROI 0's skip should
    # land on ROI 2, not ROI 1.
    labeler._goto_roi(1)
    labeler.reject_whole_trace(confirm=True)
    assert 1 in labeler._trace_actions

    # Back to ROI 0, skip it — should skip past ROI 1 and land on ROI 2.
    labeler._goto_roi(0)
    labeler.skip_trace()
    assert labeler._rois_in_order[labeler._roi_cursor] == 2


def test_reject_whole_trace_skips_past_already_reviewed_traces(
    EventLabeler, tmp_path
):
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_three_roi_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    labeler._goto_roi(1)
    labeler.skip_trace()  # ROI 1 marked reviewed
    assert 1 in labeler._trace_actions

    labeler._goto_roi(0)
    labeler.reject_whole_trace(confirm=True)
    # Should land on ROI 2, not ROI 1.
    assert labeler._rois_in_order[labeler._roi_cursor] == 2


def test_advance_event_post_last_event_skips_reviewed(EventLabeler, tmp_path):
    """
    Drill-mode auto-return after labeling the last event of a trace
    must also hop over already-reviewed ROIs. Without this fix, a
    biologist who finishes labeling ROI 0's events lands on ROI 1's
    overview even when ROI 1 was already rejected in a prior session.
    """
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_three_roi_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    # Pre-reject ROI 1.
    labeler._goto_roi(1)
    labeler.reject_whole_trace(confirm=True)
    assert 1 in labeler._trace_actions

    # Back to ROI 0, label every event one by one. The label after
    # the LAST event triggers _advance_event's auto-return, which
    # must skip past the reviewed ROI 1 onto ROI 2.
    labeler._goto_roi(0)
    labeler.investigate_trace()
    labeler.label_current("True")
    labeler.label_current("True")
    labeler.label_current("True")  # last event of ROI 0
    assert labeler._rois_in_order[labeler._roi_cursor] == 2


def test_next_trace_walks_every_trace_even_reviewed(EventLabeler, tmp_path):
    """
    Manual next_trace / prev_trace must NOT auto-skip reviewed traces.

    Biologists need to be able to flip back through their work to
    verify; the skip-reviewed behavior is reserved for productive
    labeling auto-advance, not pure navigation.
    """
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_three_roi_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    # Pre-reject ROI 1.
    labeler._goto_roi(1)
    labeler.reject_whole_trace(confirm=True)
    labeler._goto_roi(0)

    labeler.next_trace()
    assert labeler._rois_in_order[labeler._roi_cursor] == 1, (
        "next_trace must walk every ROI in order — landing on the "
        "rejected ROI 1 is the desired behavior so the labeler can "
        "verify their prior decision."
    )
    labeler.prev_trace()
    assert labeler._rois_in_order[labeler._roi_cursor] == 0


def test_next_unreviewed_trace_jumps_past_reviewed(EventLabeler, tmp_path):
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_three_roi_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    # Skip ROI 1; from ROI 0, "Next unreviewed trace" should land on ROI 2.
    labeler._goto_roi(1)
    labeler.skip_trace()
    labeler._goto_roi(0)

    moved = labeler.next_unreviewed_trace()
    assert moved is True
    assert labeler._rois_in_order[labeler._roi_cursor] == 2


def test_next_unreviewed_trace_returns_false_when_none_remain(
    EventLabeler, tmp_path
):
    """Returns False (no movement) when every later ROI is already done."""
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_three_roi_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    labeler._goto_roi(1)
    labeler.skip_trace()
    labeler._goto_roi(2)
    labeler.skip_trace()
    # Nothing unreviewed past ROI 0 now.
    labeler._goto_roi(0)
    moved = labeler.next_unreviewed_trace()
    assert moved is False
    assert labeler._roi_cursor == 0


def test_drill_figure_has_legend(EventLabeler, tmp_path):
    """The per-event view must also include a self-describing legend."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    shard = make_fake_shard()
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    labeler._view = "drill"
    fig, (ax_t, ax_m) = plt.subplots(2, 1)
    try:
        labeler._fig = fig
        labeler._ax_trace = ax_t
        labeler._ax_minimap = ax_m
        labeler._widgets = {"plot_out": object()}
        with mock.patch.object(labeler, "_draw_figure"):
            labeler._refresh_event_figure()
        legend = ax_t.get_legend()
        assert legend is not None, "drill view must paint an in-plot legend"
        labels = [t.get_text() for t in legend.get_texts()]
        assert any("peak" in lbl for lbl in labels)
        assert any("FWHM" in lbl for lbl in labels)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 8b. Trace-first navigation: skip_trace, next/prev_trace, drill_in,
#     back_to_overview, and the label-past-last-event auto-return.
# ---------------------------------------------------------------------------
def test_skip_trace_writes_trace_action_row_and_advances(EventLabeler, tmp_path):
    """
    Skip must leave a footprint in the corpus so downstream consumers
    can distinguish "labeler reviewed this trace and was happy" from
    "labeler never opened this trace." The row uses event_idx=-1 and
    notes='trace_skip'; no event-level labels are written.
    """
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    assert labeler._view == "overview"
    assert labeler._rois_in_order[labeler._roi_cursor] == 0

    labeler.skip_trace()

    assert labeler._events[labeler._cursor]["roi_id"] == 1
    assert labeler._events[labeler._cursor]["event_idx"] == 0
    assert labeler._rois_in_order[labeler._roi_cursor] == 1

    assert labeler.labeled_count == 0
    assert 0 in labeler._trace_actions
    assert labeler._trace_actions[0]["notes"] == "trace_skip"

    df = EventLabeler.load_corpus(str(corpus))
    skip_rows = df[
        (df["sample_id"] == "sampleA")
        & (df["labeler_id"] == "alice")
        & (df["roi_id"] == 0)
        & (df["event_idx"] == -1)
    ]
    assert len(skip_rows) == 1
    assert skip_rows.iloc[0]["notes"] == "trace_skip"
    assert skip_rows.iloc[0]["label"] in ("", "nan") or pd.isna(
        skip_rows.iloc[0]["label"]
    )
    event_rows = df[(df["roi_id"] == 0) & (df["event_idx"] >= 0)]
    assert event_rows.empty


def test_skip_trace_record_survives_reload(EventLabeler, tmp_path):
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    alice = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    alice.skip_trace()
    assert 0 in alice._trace_actions

    alice2 = EventLabeler(
        shard=make_fake_shard(roi_specs=_ordering_specs()),
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    assert 0 in alice2._trace_actions
    assert alice2._trace_actions[0]["notes"] == "trace_skip"
    assert alice2.labeled_count == 0


def test_skip_trace_record_does_not_leak_to_other_labelers(
    EventLabeler, tmp_path
):
    corpus = tmp_path / "labels.csv"
    alice = EventLabeler(
        shard=make_fake_shard(roi_specs=_ordering_specs()),
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    alice.skip_trace()

    bob = EventLabeler(
        shard=make_fake_shard(roi_specs=_ordering_specs()),
        corpus_path=str(corpus),
        labeler_id="bob",
    )
    # Bob does not see alice's skip in his own action map.
    assert 0 not in bob._trace_actions
    # But Bob can see that *some* other labeler has reviewed this trace
    # at the trace level, mirroring the existing _touched_by_others
    # signal for event-level labels.
    assert 0 in bob._traces_touched_by_others


def test_skip_trace_at_last_roi_still_records_review(EventLabeler, tmp_path):
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    labeler.next_trace()
    assert labeler._rois_in_order[labeler._roi_cursor] == 1
    last_cursor = labeler._cursor

    # Skip at the last ROI: cursor doesn't move (no next trace) but the
    # review IS still recorded — the biologist gets credit for vetting
    # the last trace just like any other.
    labeler.skip_trace()
    assert labeler._rois_in_order[labeler._roi_cursor] == 1
    assert labeler._cursor == last_cursor
    assert labeler._view == "overview"

    assert 1 in labeler._trace_actions
    df = EventLabeler.load_corpus(str(corpus))
    assert (
        (df["roi_id"] == 1) & (df["event_idx"] == -1) & (df["labeler_id"] == "alice")
    ).sum() == 1


def test_next_trace_does_not_write_anything(EventLabeler, tmp_path):
    """
    next_trace is pure navigation; it must NOT record a trace review.
    This is the semantic distinction from skip_trace.
    """
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    labeler.next_trace()
    labeler.prev_trace()
    labeler.next_trace()

    assert labeler._trace_actions == {}
    assert labeler.labeled_count == 0
    # No writes happened, so the CSV may not exist yet — but if it does
    # it should be empty.
    if corpus.exists():
        df = EventLabeler.load_corpus(str(corpus))
        assert df.empty


def test_next_and_prev_trace_navigation(EventLabeler, tmp_path):
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    # Start: ROI 0.
    assert labeler._rois_in_order[labeler._roi_cursor] == 0

    # Forward.
    labeler.next_trace()
    assert labeler._rois_in_order[labeler._roi_cursor] == 1
    assert labeler._events[labeler._cursor]["roi_id"] == 1
    assert labeler._events[labeler._cursor]["event_idx"] == 0

    # Forward at the last ROI: clamp, no IndexError.
    labeler.next_trace()
    assert labeler._rois_in_order[labeler._roi_cursor] == 1

    # Backward.
    labeler.prev_trace()
    assert labeler._rois_in_order[labeler._roi_cursor] == 0
    assert labeler._events[labeler._cursor]["roi_id"] == 0
    assert labeler._events[labeler._cursor]["event_idx"] == 0

    # Backward at the first ROI: clamp, no IndexError.
    labeler.prev_trace()
    assert labeler._rois_in_order[labeler._roi_cursor] == 0


def test_drill_in_then_back_to_overview_preserves_roi(EventLabeler, tmp_path):
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    # Move to ROI 1 in overview.
    labeler.next_trace()
    assert labeler._rois_in_order[labeler._roi_cursor] == 1
    assert labeler._view == "overview"

    # Drill in -> event view at first event of ROI 1.
    labeler.drill_in()
    assert labeler._view == "drill"
    assert labeler._events[labeler._cursor]["roi_id"] == 1
    assert labeler._events[labeler._cursor]["event_idx"] == 0

    # Step forward within ROI 1; should not leave ROI 1 yet.
    labeler._advance_event(+1)
    assert labeler._view == "drill"
    assert labeler._events[labeler._cursor]["roi_id"] == 1

    # Back to overview returns to ROI 1.
    labeler.back_to_overview()
    assert labeler._view == "overview"
    assert labeler._rois_in_order[labeler._roi_cursor] == 1


def test_label_past_last_event_returns_to_overview_at_next_roi(
    EventLabeler, tmp_path
):
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    # Drill into ROI 0.
    labeler.drill_in()
    assert labeler._view == "drill"

    # Label all three events on ROI 0.
    labeler.label_current("True")
    labeler.label_current("False")
    # The third label is the last event on ROI 0 — labelling it should
    # cross out of ROI 0 into the next ROI's overview.
    assert labeler._view == "drill"
    labeler.label_current("True")

    assert labeler._view == "overview"
    assert labeler._rois_in_order[labeler._roi_cursor] == 1
    assert labeler._events[labeler._cursor]["roi_id"] == 1
    assert labeler.labeled_count == 3


def test_drill_advance_event_backward_clamps_to_first_event_of_roi(
    EventLabeler, tmp_path
):
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    # Move to ROI 1 and drill in.
    labeler.next_trace()
    labeler.drill_in()
    assert labeler._events[labeler._cursor]["roi_id"] == 1
    assert labeler._events[labeler._cursor]["event_idx"] == 0

    # Stepping backward from the first event of ROI 1 must NOT cross
    # back into ROI 0 — the drill view is scoped to a single trace.
    labeler._advance_event(-1)
    assert labeler._events[labeler._cursor]["roi_id"] == 1
    assert labeler._events[labeler._cursor]["event_idx"] == 0
    assert labeler._view == "drill"


def test_initial_state_is_overview_at_first_roi(EventLabeler, tmp_path):
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    assert labeler._view == "overview"
    assert labeler._roi_cursor == 0
    assert labeler._rois_in_order[labeler._roi_cursor] == 0


# ---------------------------------------------------------------------------
# 8c. Trace-action audit interactions: reject↔skip overwrites, stale-skip
#     suppression, label_current from overview, and legacy backfill.
# ---------------------------------------------------------------------------
def test_reject_whole_trace_writes_trace_action_row(EventLabeler, tmp_path):
    """
    Reject must leave the same trace-level sentinel row that skip does,
    so the 'was this trace reviewed?' predicate is one scan regardless
    of which trace-level action ran.
    """
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    labeler.reject_whole_trace(confirm=True)

    df = EventLabeler.load_corpus(str(corpus))
    trace_rows = df[(df["event_idx"] == -1) & (df["labeler_id"] == "alice")]
    assert len(trace_rows) == 1
    assert trace_rows.iloc[0]["roi_id"] == 0
    assert trace_rows.iloc[0]["notes"] == "trace_reject"

    # In-memory mirror.
    assert labeler._trace_actions[0]["notes"] == "trace_reject"


def test_reject_overwrites_prior_skip_trace_action(EventLabeler, tmp_path):
    """
    Reject is a stronger commitment than skip, so a subsequent reject
    must overwrite the trace_action record (otherwise the corpus would
    report 'skipped' while every event row is False).
    """
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    labeler.skip_trace()
    assert labeler._trace_actions[0]["notes"] == "trace_skip"

    # Move back to ROI 0 and reject.
    labeler.prev_trace()
    assert labeler._rois_in_order[labeler._roi_cursor] == 0
    labeler.reject_whole_trace(confirm=True)

    # In-memory: reject won.
    assert labeler._trace_actions[0]["notes"] == "trace_reject"
    # On disk: exactly one trace-level row for ROI 0, and it's reject.
    df = EventLabeler.load_corpus(str(corpus))
    n0_trace = df[(df["roi_id"] == 0) & (df["event_idx"] == -1)]
    assert len(n0_trace) == 1
    assert n0_trace.iloc[0]["notes"] == "trace_reject"


def test_skip_after_reject_is_refused(EventLabeler, tmp_path, caplog):
    """
    Skip after reject would lie about state on disk (event rows would
    still be all False but the trace_action would say 'skipped'). The
    labeler logs a warning and leaves the trace_action as 'trace_reject'
    rather than silently downgrading.
    """
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    labeler.reject_whole_trace(confirm=True)
    assert labeler._trace_actions[0]["notes"] == "trace_reject"

    labeler.prev_trace()
    assert labeler._rois_in_order[labeler._roi_cursor] == 0

    with caplog.at_level(logging.WARNING):
        labeler.skip_trace()

    # trace_action unchanged.
    assert labeler._trace_actions[0]["notes"] == "trace_reject"
    # Cursor still advanced (navigation is predictable even on refusal).
    assert labeler._rois_in_order[labeler._roi_cursor] == 1
    # Warning explains the refusal.
    assert any(
        "previously rejected" in r.message for r in caplog.records
    )


def test_label_current_from_overview_drills_in_and_labels_first_event(
    EventLabeler, tmp_path
):
    """
    Calling label_current from overview view must NOT silently advance
    the event cursor across ROI boundaries (the previous behavior).
    Instead it implicitly drills in and labels the first event of the
    current ROI so the result is deterministic.
    """
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    # Move overview to ROI 1, then label from overview.
    labeler.next_trace()
    assert labeler._view == "overview"
    assert labeler._rois_in_order[labeler._roi_cursor] == 1

    labeler.label_current("True")

    df = EventLabeler.load_corpus(str(corpus))
    matching = df[
        (df["roi_id"] == 1)
        & (df["event_idx"] == 0)
        & (df["labeler_id"] == "alice")
    ]
    assert len(matching) == 1
    assert matching.iloc[0]["label"] == "True"


def test_skip_annotation_clears_when_event_subsequently_labeled(
    EventLabeler, tmp_path
):
    """
    The labeler's trace-review-state predicate is the source of truth
    for the overview '(previously skipped)' annotation. After a skip
    followed by an event label on the same trace, the state must flip
    from 'skipped' to 'revisited' (the skip is no longer the most recent
    intent).
    """
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    labeler.skip_trace()
    # State for ROI 0: just skipped.
    assert labeler._trace_review_state(0) == "skipped"

    # Go back, drill in, label one event.
    labeler.prev_trace()
    labeler.drill_in()
    labeler.label_current("True")

    # Skip record still exists on disk (audit trail preserved) but the
    # predicate now reports the state as "revisited" so the UI does NOT
    # show the stale '(previously skipped)' annotation.
    assert 0 in labeler._trace_actions
    assert labeler._trace_review_state(0) == "revisited"


def test_legacy_whole_trace_reject_backfills_trace_action_on_load(
    EventLabeler, tmp_path
):
    """
    Corpora written before the trace-level sentinel row was introduced
    encode 'reject whole trace' only as N event rows with
    notes='whole_trace_reject'. On load, the labeler must backfill a
    trace_action record so the post-#3 'was reviewed?' predicate
    returns true for legacy data too.
    """
    corpus = tmp_path / "labels.csv"
    # Hand-craft a legacy-shaped corpus: alice rejected ROI 0 (event
    # rows only, no sentinel), bob rejected ROI 1 the same way.
    legacy = pd.DataFrame(
        [
            {
                "corpus_version": 2,
                "sample_id": "sampleA",
                "roi_id": 0,
                "event_idx": 0,
                "label": "False",
                "labeler_id": "alice",
                "timestamp": "2025-01-01T00:00:00+00:00",
                "notes": "whole_trace_reject",
                "peak_amplitude": 0.1,
                "fwhm_frames": 3.0,
                "sampling_rate": "",
                "indicator": "",
                "microscope": "",
                "cell_type": "",
                "experiment_id": "",
                "wizards_staff_version": "legacy",
            },
            {
                "corpus_version": 2,
                "sample_id": "sampleA",
                "roi_id": 1,
                "event_idx": 0,
                "label": "False",
                "labeler_id": "bob",
                "timestamp": "2025-01-01T00:00:00+00:00",
                "notes": "whole_trace_reject",
                "peak_amplitude": 0.1,
                "fwhm_frames": 3.0,
                "sampling_rate": "",
                "indicator": "",
                "microscope": "",
                "cell_type": "",
                "experiment_id": "",
                "wizards_staff_version": "legacy",
            },
        ]
    )
    legacy.to_csv(corpus, index=False)

    shard = make_fake_shard(roi_specs=_ordering_specs())
    alice = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )
    # alice's own reject is backfilled into trace_actions.
    assert 0 in alice._trace_actions
    assert alice._trace_actions[0]["notes"] == "trace_reject"
    # bob's reject shows up as a trace-level touch (without leaking the action).
    assert 1 in alice._traces_touched_by_others


def test_overview_review_state_predicate(EventLabeler, tmp_path):
    """Comprehensive smoke for _trace_review_state across all branches."""
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    # No trace_action recorded yet.
    assert labeler._trace_review_state(0) is None

    # Skip → "skipped".
    labeler.skip_trace()
    assert labeler._trace_review_state(0) == "skipped"

    # Drill back in and label an event → "revisited".
    labeler.prev_trace()
    labeler.drill_in()
    labeler.label_current("True")
    assert labeler._trace_review_state(0) == "revisited"

    # Move to ROI 1 and reject → "rejected" (independent of label state).
    labeler.next_trace()
    labeler.reject_whole_trace(confirm=True)
    assert labeler._trace_review_state(1) == "rejected"


# ---------------------------------------------------------------------------
# 9. Edge cases.
# ---------------------------------------------------------------------------
def test_roi_with_zero_events_is_skipped(EventLabeler, tmp_path, caplog):
    specs = [
        {"roi_id": 0, "positions": [], "amplitudes": [], "fwhm": []},
        {
            "roi_id": 1,
            "positions": [10, 50],
            "amplitudes": [0.5, 0.7],
            "fwhm": [3.0, 4.0],
        },
    ]
    shard = make_fake_shard(roi_specs=specs)
    with caplog.at_level(logging.INFO):
        lab = EventLabeler(
            shard=shard,
            corpus_path=str(tmp_path / "labels.csv"),
            labeler_id="alice",
        )
    # Only ROI 1's events show up.
    assert all(e["roi_id"] == 1 for e in lab.events)
    assert lab.total_events == 2


def test_nan_amplitudes_are_skipped_with_warning(EventLabeler, tmp_path, caplog):
    shard = make_fake_shard(inject_bad_event=True)
    with caplog.at_level(logging.WARNING):
        lab = EventLabeler(
            shard=shard,
            corpus_path=str(tmp_path / "labels.csv"),
            labeler_id="alice",
        )
    # 5 valid events from default fixture; the NaN appended to ROI 0
    # is dropped.
    assert lab.total_events == 5


# ---------------------------------------------------------------------------
# Layer-2 bound skipping.
# ---------------------------------------------------------------------------
def test_active_amplitude_bounds_skip_subthreshold_events(
    EventLabeler, tmp_path, caplog
):
    """
    When ``_apply_event_filters`` has stashed an active amplitude
    bound on the shard, the labeler must not surface events that
    fall outside it: those events are dropped by every per-event
    metric anyway, so labeling them is wasted effort.
    """
    specs = [
        {
            "roi_id": 0,
            "positions": [10, 50, 120],
            # 0.05 is below the active floor of 0.1; the other two
            # are above it and must be surfaced.
            "amplitudes": [0.05, 1.2, 0.3],
            "fwhm": [4.0, 6.0, 3.0],
        },
        {
            "roi_id": 1,
            "positions": [25, 80],
            "amplitudes": [0.9, 0.04],  # second event below the floor
            "fwhm": [5.0, 4.5],
        },
    ]
    shard = make_fake_shard(roi_specs=specs)
    shard._active_filter_events = True
    shard._active_min_event_amplitude = 0.1
    shard._active_max_event_amplitude = 10.0
    shard._active_min_event_fwhm = None
    shard._active_max_event_fwhm = None

    with caplog.at_level(logging.INFO):
        lab = EventLabeler(
            shard=shard,
            corpus_path=str(tmp_path / "labels.csv"),
            labeler_id="alice",
        )

    # Two sub-threshold events dropped; three surface.
    assert lab.total_events == 3
    # The surviving events must preserve their raw event_idx — this
    # is the load-bearing invariant that keeps corpus keys and the
    # _apply_event_filters Layer-3 positional lookup in sync. ROI 0's
    # surviving events are raw indices 1 and 2 (index 0 was filtered);
    # ROI 1's surviving event is raw index 0 (index 1 was filtered).
    by_roi = {}
    for ev in lab.events:
        by_roi.setdefault(ev["roi_id"], []).append(ev["event_idx"])
    assert sorted(by_roi[0]) == [1, 2]
    assert sorted(by_roi[1]) == [0]

    # The aggregate skip count is surfaced at INFO so labelers can
    # tell why their queue is shorter than the raw event count.
    skip_records = [
        r for r in caplog.records
        if "skipped" in r.getMessage() and "active Layer-2 bounds" in r.getMessage()
    ]
    assert len(skip_records) == 1
    assert "2" in skip_records[0].getMessage()


def test_active_bounds_inactive_when_filter_events_false(
    EventLabeler, tmp_path, caplog
):
    """
    ``filter_events=False`` means the bounds are recorded for
    diagnostics but not enforced — the labeler must surface every
    raw event regardless of the stashed bound values. Otherwise a
    user running with bounds-off would silently lose events the
    metrics still keep.
    """
    specs = [
        {
            "roi_id": 0,
            "positions": [10, 50, 120],
            "amplitudes": [0.05, 1.2, 0.3],
            "fwhm": [4.0, 6.0, 3.0],
        },
    ]
    shard = make_fake_shard(roi_specs=specs)
    # Bounds stashed (refilter_events records them for diagnostics)
    # but the master switch is OFF.
    shard._active_filter_events = False
    shard._active_min_event_amplitude = 0.1
    shard._active_max_event_amplitude = 10.0

    with caplog.at_level(logging.INFO):
        lab = EventLabeler(
            shard=shard,
            corpus_path=str(tmp_path / "labels.csv"),
            labeler_id="alice",
        )

    assert lab.total_events == 3
    # No skip message when the filter is inactive.
    skips = [
        r for r in caplog.records
        if "active Layer-2 bounds" in r.getMessage()
    ]
    assert skips == []


def test_active_fwhm_bound_drops_short_events(EventLabeler, tmp_path):
    """An active FWHM floor drops events with FWHM below the floor."""
    specs = [
        {
            "roi_id": 0,
            "positions": [10, 50, 120],
            "amplitudes": [0.5, 1.2, 0.3],
            # event 1 has fwhm=1 (sub-threshold), event 2 has fwhm=0
            # (also sub-threshold); only event 0 survives.
            "fwhm": [4.0, 1.0, 0.0],
        },
    ]
    shard = make_fake_shard(roi_specs=specs)
    shard._active_filter_events = True
    shard._active_min_event_amplitude = None
    shard._active_max_event_amplitude = None
    shard._active_min_event_fwhm = 2.0
    shard._active_max_event_fwhm = None

    lab = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    assert lab.total_events == 1
    surviving = lab.events[0]
    assert surviving["roi_id"] == 0
    assert surviving["event_idx"] == 0


def test_missing_active_bounds_attributes_preserve_legacy_behavior(
    EventLabeler, tmp_path
):
    """
    A shard that has never been through ``_apply_event_filters``
    (or one loaded from a pickle predating the active-bounds fields)
    must still work — the labeler should default to surfacing every
    raw event. Verified by deleting the attributes from the fake
    shard entirely.
    """
    shard = make_fake_shard()
    for attr in (
        "_active_filter_events",
        "_active_min_event_amplitude",
        "_active_max_event_amplitude",
        "_active_min_event_fwhm",
        "_active_max_event_fwhm",
    ):
        if hasattr(shard, attr):
            delattr(shard, attr)

    lab = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    # Default fixture has 5 events across two ROIs.
    assert lab.total_events == 5


def test_short_trace_window_collapses_to_full_trace(EventLabeler, tmp_path):
    shard = make_fake_shard(n_frames=20)
    lab = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
        window_scale=8.0,
    )
    ev = lab.events[0]
    start, end = lab._window_for_event(ev, n_frames=20)
    assert start == 0 and end == 20


def test_export_labels_returns_only_this_session(EventLabeler, tmp_path):
    corpus = tmp_path / "labels.csv"
    bob = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(corpus),
        labeler_id="bob",
    )
    bob.label_current("True")
    bob.label_current("False")

    alice = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    alice.label_current("Unsure")

    exported = alice.export_labels()
    assert len(exported) == 1
    assert exported.iloc[0]["labeler_id"] == "alice"
    assert exported.iloc[0]["label"] == "Unsure"
    # Bob's rows are still on disk untouched.
    on_disk = pd.read_csv(corpus)
    assert (on_disk["labeler_id"] == "bob").sum() == 2
    assert (on_disk["labeler_id"] == "alice").sum() == 1


def test_invalid_label_raises(EventLabeler, tmp_path):
    lab = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    with pytest.raises(ValueError, match="label must be one of"):
        lab.label_current("Maybe")


def test_invalid_ordering_raises(EventLabeler, tmp_path):
    with pytest.raises(ValueError, match="ordering must be one of"):
        EventLabeler(
            shard=make_fake_shard(),
            corpus_path=str(tmp_path / "labels.csv"),
            labeler_id="alice",
            ordering="random",
        )


def test_empty_labeler_id_raises(EventLabeler, tmp_path):
    with pytest.raises(ValueError, match="labeler_id"):
        EventLabeler(
            shard=make_fake_shard(),
            corpus_path=str(tmp_path / "labels.csv"),
            labeler_id="",
        )


# ---------------------------------------------------------------------------
# 10. New behaviors introduced by the audit-driven cleanup
# ---------------------------------------------------------------------------


# --- #10 labeler_id canonicalization ----------------------------------------
def test_labeler_id_is_canonicalized(EventLabeler, tmp_path):
    """
    Case / whitespace variants of the same name must collapse to a single
    canonical identity so inter-rater agreement isn't silently fractured
    across "Alice" / "alice" / "  ALICE  ".
    """
    corpus = tmp_path / "labels.csv"
    a = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(corpus),
        labeler_id="  Alice  ",
    )
    assert a.labeler_id == "alice"
    a.label_current("True")

    # Open with a different casing — must recognize prior rows as ours.
    b = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(corpus),
        labeler_id="ALICE",
    )
    assert b.labeler_id == "alice"
    assert b.labeled_count == 1


def test_labeler_id_whitespace_only_raises(EventLabeler, tmp_path):
    with pytest.raises(ValueError, match="labeler_id"):
        EventLabeler(
            shard=make_fake_shard(),
            corpus_path=str(tmp_path / "labels.csv"),
            labeler_id="   ",
        )


def test_legacy_mixedcase_labeler_id_emits_collision_warning(
    EventLabeler, tmp_path, caplog
):
    """
    A corpus written by an older labeler that didn't canonicalize will
    contain mixed-case labeler_id values. New sessions must recognize
    them as the same labeler AND surface a one-time warning so an
    operator can decide whether to run ``migrate_corpus``.
    """
    corpus = tmp_path / "labels.csv"
    legacy = pd.DataFrame(
        [
            {
                "corpus_version": 2,
                "sample_id": "sampleA",
                "roi_id": 0,
                "event_idx": 0,
                "label": "True",
                "labeler_id": "Alice",
                "timestamp": "2025-01-01T00:00:00+00:00",
                "notes": "",
                "peak_amplitude": 0.5,
                "fwhm_frames": 4.0,
                "sampling_rate": "",
                "indicator": "",
                "microscope": "",
                "cell_type": "",
                "experiment_id": "",
                "wizards_staff_version": "legacy",
            }
        ]
    )
    legacy.to_csv(corpus, index=False)

    with caplog.at_level(logging.WARNING):
        lab = EventLabeler(
            shard=make_fake_shard(),
            corpus_path=str(corpus),
            labeler_id="alice",
        )
    assert lab.labeled_count == 1
    assert any("canonicalize" in r.message for r in caplog.records)


# --- #7 migrate_corpus ------------------------------------------------------
def test_migrate_corpus_canonicalizes_labeler_ids(EventLabeler, tmp_path):
    """
    ``migrate_corpus`` rewrites all labeler_id values to their canonical
    form so legacy "Alice" rows end up byte-equal to fresh "alice" rows.
    Migration is non-destructive: source file unchanged when out_path
    differs.
    """
    src = tmp_path / "legacy.csv"
    dst = tmp_path / "migrated.csv"
    rows = []
    for i, who in enumerate(["Alice", "ALICE", "alice ", "Bob"]):
        rows.append(
            {
                "corpus_version": 2,
                "sample_id": "sampleA",
                "roi_id": 0,
                "event_idx": i,
                "label": "True",
                "labeler_id": who,
                "timestamp": "2025-01-01T00:00:00+00:00",
                "notes": "",
                "peak_amplitude": 0.1 * (i + 1),
                "fwhm_frames": 3.0,
                "sampling_rate": "",
                "indicator": "",
                "microscope": "",
                "cell_type": "",
                "experiment_id": "",
                "wizards_staff_version": "legacy",
            }
        )
    pd.DataFrame(rows).to_csv(src, index=False)
    src_bytes = src.read_bytes()

    report = EventLabeler.migrate_corpus(str(src), str(dst))
    assert report["from_version"] == 2
    assert report["to_version"] == EventLabeler.CORPUS_VERSION
    assert report["rows_in"] == 4
    assert report["rows_out"] == 4
    # All four rows are non-canonical ("Alice"/"ALICE"/"alice " all
    # canonicalize to "alice", and "Bob" canonicalizes to "bob").
    assert report["labeler_id_renames"] == 4

    out = pd.read_csv(dst)
    assert sorted(out["labeler_id"].unique()) == ["alice", "bob"]
    # Non-destructive: source file untouched.
    assert src.read_bytes() == src_bytes


def test_migrate_corpus_missing_file_returns_zero_report(EventLabeler, tmp_path):
    report = EventLabeler.migrate_corpus(
        str(tmp_path / "nope.csv"), str(tmp_path / "out.csv")
    )
    assert report["rows_in"] == 0
    assert report["rows_out"] == 0
    assert report["from_version"] is None


def test_migrate_corpus_refuses_downgrade(EventLabeler, tmp_path):
    src = tmp_path / "future.csv"
    pd.DataFrame(
        [
            {
                **{c: "" for c in EventLabeler.CORPUS_COLUMNS},
                "corpus_version": 999,
                "sample_id": "sampleA",
                "roi_id": 0,
                "event_idx": 0,
                "label": "True",
                "labeler_id": "alice",
            }
        ]
    ).to_csv(src, index=False)
    with pytest.raises(RuntimeError, match="newer than"):
        EventLabeler.migrate_corpus(str(src), str(tmp_path / "out.csv"), to_version=2)


# --- #3 reject confirmation gate --------------------------------------------
def test_reject_whole_trace_arms_on_first_press_and_commits_on_second(
    EventLabeler, tmp_path
):
    """
    A single ``r`` arms the action and writes nothing; a second
    consecutive ``r`` commits. Bypass via ``confirm=True`` for callers
    that already prompted the user.
    """
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )

    # First press: arms.
    labeler.reject_whole_trace()
    assert labeler._reject_armed_roi == 0
    assert labeler.labeled_count == 0
    assert 0 not in labeler._trace_actions

    # Second consecutive press: commits.
    labeler.reject_whole_trace()
    assert labeler._reject_armed_roi is None
    assert labeler._trace_actions[0]["notes"] == "trace_reject"
    df = EventLabeler.load_corpus(str(corpus))
    rejected = df[(df["roi_id"] == 0) & (df["event_idx"] >= 0)]
    assert (rejected["label"] == "False").all()


def test_reject_whole_trace_arm_is_cleared_by_navigation(
    EventLabeler, tmp_path
):
    """
    Navigation or labeling between two ``r`` presses must disarm the
    pending reject — otherwise a stray keystroke a minute later could
    silently commit the action.
    """
    corpus = tmp_path / "labels.csv"
    shard = make_fake_shard(roi_specs=_ordering_specs())
    labeler = EventLabeler(
        shard=shard, corpus_path=str(corpus), labeler_id="alice"
    )

    labeler.reject_whole_trace()
    assert labeler._reject_armed_roi == 0

    labeler.next_trace()
    assert labeler._reject_armed_roi is None

    # Going back to ROI 0 and pressing once must re-arm, not commit.
    labeler.prev_trace()
    labeler.reject_whole_trace()
    assert labeler._reject_armed_roi == 0
    assert 0 not in labeler._trace_actions
    df_path_exists = corpus.exists()
    if df_path_exists:
        df = EventLabeler.load_corpus(str(corpus))
        assert df.empty or (df["event_idx"] == -1).any() is False


# --- #5 indicator-aware window_scale ----------------------------------------
def test_window_scale_defaults_to_indicator_preset(EventLabeler, tmp_path):
    """
    When window_scale is not passed explicitly, the indicator name in
    ``context`` selects an appropriate multiplier. Passing an explicit
    value bypasses the preset.
    """
    shard = make_fake_shard()
    fast = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "fast.csv"),
        labeler_id="alice",
        context={"indicator": "GCaMP6f"},
    )
    slow = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(tmp_path / "slow.csv"),
        labeler_id="alice",
        context={"indicator": "GCaMP6s"},
    )
    # Same indicator family; slower variant should produce a narrower
    # multiplier so the window doesn't blow up.
    assert fast._effective_window_scale() > slow._effective_window_scale()

    explicit = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(tmp_path / "explicit.csv"),
        labeler_id="alice",
        context={"indicator": "GCaMP6s"},
        window_scale=99.0,
    )
    # Explicit value wins over indicator preset.
    assert explicit._effective_window_scale() == 99.0


def test_window_scale_unknown_indicator_uses_default(EventLabeler, tmp_path):
    lab = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
        context={"indicator": "Mystery1z"},
    )
    assert lab._effective_window_scale() == EventLabeler._DEFAULT_WINDOW_SCALE


# --- #4 cached filtered_idx on shard ----------------------------------------
def test_resolve_filtered_idx_prefers_shard_cache(EventLabeler, tmp_path):
    """
    When the shard exposes ``_filtered_idx_cache`` (populated by
    ``_run_all``), the labeler must use it verbatim instead of calling
    ``spatial_filtering`` with default thresholds.
    """
    shard = make_fake_shard()
    # Simulate _run_all's cache stash. The cache value is intentionally
    # different from what shard.spatial_filtering would return so we can
    # detect which path the labeler took.
    shard._filtered_idx_cache = [42, 43]
    shard._filtered_idx_params = {"p_th": 50.0, "size_threshold": 12345}
    lab = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    assert lab._resolve_filtered_idx() == [42, 43]


def test_drill_in_refuses_when_trace_unavailable(EventLabeler, tmp_path, caplog):
    """
    Labeling without a visible trace is a bug magnet. drill_in must
    refuse if ``_get_trace`` returns None and stay in overview view.
    """
    shard = make_fake_shard()
    # Force _get_trace failure by removing dff_dat.
    shard.get_input = lambda name, req=False: None
    lab = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    assert lab._view == "overview"
    with caplog.at_level(logging.WARNING):
        lab.drill_in()
    assert lab._view == "overview"
    assert any("refusing to investigate" in r.message for r in caplog.records)


# --- #2 seconds display -----------------------------------------------------
def test_frame_to_seconds_str_uses_context_sampling_rate(EventLabeler, tmp_path):
    lab = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
        context={"sampling_rate": 30},
    )
    # 60 frames @ 30 Hz = 2.00s
    assert "(2.00s)" in lab._frame_to_seconds_str(60)
    assert "60" in lab._frame_to_seconds_str(60)


def test_frame_to_seconds_str_falls_back_to_frame_only(EventLabeler, tmp_path):
    lab = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    # No sampling_rate -> no parenthetical.
    assert lab._frame_to_seconds_str(60) == "60"


# --- #8 / #9 save-time refresh and preserved cache --------------------------
def test_save_refreshes_touched_by_others_after_concurrent_write(
    EventLabeler, tmp_path
):
    """
    Another labeler writing to the same corpus during this labeler's
    session must be reflected in ``_touched_by_others`` after the
    next save — without restarting the session.
    """
    corpus = tmp_path / "labels.csv"
    alice = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    alice.label_current("True")  # alice owns (0, 0)
    assert (0, 1) not in alice._touched_by_others

    # Simulate a concurrent labeler writing a row for (0, 1).
    import time
    time.sleep(0.01)  # ensure mtime granularity differs
    bob = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(corpus),
        labeler_id="bob",
    )
    # label_current from overview implicitly drills in (resetting the
    # cursor to event 0 of the current ROI). Drill in explicitly first
    # so the cursor we set actually targets the event we want.
    bob.drill_in()
    bob._cursor = 1  # event_idx 1 within ROI 0
    bob.label_current("False")

    # Alice does another label — _save reads the updated corpus, refreshes
    # her view of which events others have touched.
    alice.drill_in()
    alice._cursor = 2
    alice.label_current("Unsure")
    assert (0, 1) in alice._touched_by_others


def test_investigate_trace_is_the_canonical_name_for_drill_in(
    EventLabeler, tmp_path
):
    """
    ``investigate_trace`` is the new public name; ``drill_in`` is kept
    as a thin alias so older scripts continue to work. Both must put
    the labeler into the per-event view at event 0 of the current
    ROI.
    """
    shard = make_fake_shard(roi_specs=_ordering_specs())
    a = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "a.csv"),
        labeler_id="alice",
    )
    a.next_trace()
    a.investigate_trace()
    assert a._view == "drill"
    assert a._events[a._cursor]["roi_id"] == 1
    assert a._events[a._cursor]["event_idx"] == 0

    # Legacy alias.
    b = EventLabeler(
        shard=make_fake_shard(roi_specs=_ordering_specs()),
        corpus_path=str(tmp_path / "b.csv"),
        labeler_id="alice",
    )
    b.next_trace()
    b.drill_in()
    assert b._view == "drill"
    assert b._events[b._cursor]["roi_id"] == 1
    assert b._events[b._cursor]["event_idx"] == 0


def test_overview_key_i_and_legacy_d_both_investigate_trace(
    EventLabeler, tmp_path
):
    """
    In overview view, pressing ``i`` opens the per-event view; the
    legacy ``d`` key still works so existing biologists' muscle memory
    survives the rename.
    """
    shard = make_fake_shard(roi_specs=_ordering_specs())
    lab = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    # Simulate the keyboard dispatch directly. ``display()`` is the
    # only place that builds the keymap, but the underlying methods
    # are wired the same way the cmd-box dispatcher calls them.
    assert lab._view == "overview"
    lab.investigate_trace()
    assert lab._view == "drill"

    lab.back_to_overview()
    lab.drill_in()
    assert lab._view == "drill"


def test_save_uses_preserved_cache_on_repeated_writes(EventLabeler, tmp_path):
    """
    When the corpus file hasn't changed under us, _save should reuse
    its cached ``preserved`` slice instead of re-reading the file.
    """
    corpus = tmp_path / "labels.csv"
    lab = EventLabeler(
        shard=make_fake_shard(),
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    lab.label_current("True")
    first_key = lab._preserved_cache_key
    assert first_key is not None

    # Patch _read_corpus_raw to fail loudly — proves the next save
    # didn't touch disk for the read.
    real_read = lab._read_corpus_raw

    def boom():
        raise AssertionError("preserved cache should have short-circuited the read")

    lab._read_corpus_raw = boom  # type: ignore[assignment]
    try:
        lab.label_current("False")
    finally:
        lab._read_corpus_raw = real_read  # type: ignore[assignment]

    # Cache key advances because we just wrote (mtime changed).
    assert lab._preserved_cache_key != first_key
