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
    labeler.reject_whole_trace()

    df = EventLabeler.load_corpus(str(corpus))
    n0 = df[df["roi_id"] == 0].sort_values("event_idx")
    assert list(n0["label"]) == ["True", "False", "False"]
    rejects = n0[n0["label"] == "False"]
    assert (rejects["notes"] == "whole_trace_reject").all()
    # The first event was already labeled True and must be untouched.
    assert n0.iloc[0]["label"] == "True"
    assert n0.iloc[0]["notes"] != "whole_trace_reject"

    # Cursor advanced past ROI 0.
    assert labeler._events[labeler._cursor]["roi_id"] == 1


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
