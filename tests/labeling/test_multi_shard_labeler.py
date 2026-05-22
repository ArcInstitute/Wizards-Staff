"""
Tests for :mod:`wizards_staff.labeling.multi_shard_labeler`.

These tests load the labeling submodules as standalone modules via
``importlib.util`` so the test suite stays runnable in environments
without the heavy-weight Wizards-Staff core dependencies (``caiman``,
``tensorflow``). The pattern mirrors ``tests/labeling/test_event_labeler.py``.

The wrapper itself is exercised here in headless mode — no ``display()``
calls — because the auto-advance, resume-on-reopen, snapshot/restore,
and progress-summary logic are all driven through the public API
without needing ipywidgets. A separate small block confirms
``display()`` raises a clean ImportError when ipywidgets is absent.
"""

from __future__ import annotations

import importlib
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
# Module loader
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
EVENT_LABELER_PATH = REPO_ROOT / "wizards_staff" / "labeling" / "event_labeler.py"
MULTI_SHARD_PATH = REPO_ROOT / "wizards_staff" / "labeling" / "multi_shard_labeler.py"


def _load_modules(extra_sys_modules: Optional[Dict[str, Any]] = None):
    """
    Import event_labeler.py and multi_shard_labeler.py as a coherent pair.

    multi_shard_labeler imports from ``wizards_staff.labeling.event_labeler``
    so we install the loaded event_labeler module under that exact name
    before exec'ing the wrapper.

    All sys.modules mutations are reverted in the ``finally`` block —
    including the parent-package stubs we install — so other test
    modules that perform genuine ``import wizards_staff.wizards`` calls
    later in the same pytest session see the real package, not our
    SimpleNamespace stub. Without this strict restoration, a test like
    ``tests/wizards/test_event_filters.py`` that runs after this one
    fails with ``ModuleNotFoundError: ... 'wizards_staff' is not a
    package``.
    """
    keys_to_save = list((extra_sys_modules or {}).keys()) + [
        "wizards_staff",
        "wizards_staff.labeling",
        "wizards_staff.labeling.event_labeler",
        "wizards_staff.labeling.multi_shard_labeler",
    ]
    saved_sys: Dict[str, Any] = {k: sys.modules.get(k) for k in keys_to_save}

    try:
        if extra_sys_modules:
            for k, v in extra_sys_modules.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        # Stub out wizards_staff parents so absolute imports resolve.
        ws_pkg = sys.modules.get("wizards_staff") or SimpleNamespace()
        ws_label_pkg = sys.modules.get("wizards_staff.labeling") or SimpleNamespace()
        sys.modules["wizards_staff"] = ws_pkg
        sys.modules["wizards_staff.labeling"] = ws_label_pkg

        spec_evt = importlib.util.spec_from_file_location(
            "wizards_staff.labeling.event_labeler",
            str(EVENT_LABELER_PATH),
        )
        assert spec_evt is not None and spec_evt.loader is not None
        mod_evt = importlib.util.module_from_spec(spec_evt)
        sys.modules["wizards_staff.labeling.event_labeler"] = mod_evt
        spec_evt.loader.exec_module(mod_evt)
        ws_label_pkg.event_labeler = mod_evt
        ws_label_pkg.EventLabeler = mod_evt.EventLabeler

        spec_msl = importlib.util.spec_from_file_location(
            "wizards_staff.labeling.multi_shard_labeler",
            str(MULTI_SHARD_PATH),
        )
        assert spec_msl is not None and spec_msl.loader is not None
        mod_msl = importlib.util.module_from_spec(spec_msl)
        sys.modules["wizards_staff.labeling.multi_shard_labeler"] = mod_msl
        spec_msl.loader.exec_module(mod_msl)
        ws_label_pkg.multi_shard_labeler = mod_msl
        ws_label_pkg.MultiShardLabeler = mod_msl.MultiShardLabeler

        return mod_evt, mod_msl
    finally:
        # Strict cleanup: revert every key we touched, in either
        # direction (module → module, module → None, None → module).
        for k, prev in saved_sys.items():
            if prev is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = prev


@pytest.fixture(scope="module")
def labeler_modules():
    return _load_modules()


@pytest.fixture
def EventLabeler(labeler_modules):
    return labeler_modules[0].EventLabeler


@pytest.fixture
def MultiShardLabeler(labeler_modules):
    return labeler_modules[1].MultiShardLabeler


# ---------------------------------------------------------------------------
# Fake shard helpers (mirror the single-shard test fixture)
# ---------------------------------------------------------------------------
def make_fake_shard(
    sample_name: str,
    roi_specs: Optional[List[Dict[str, Any]]] = None,
    n_frames: int = 200,
):
    """Build a minimal shard-like object the labeler can consume."""
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
                "FWHM Backward Positions": [
                    max(0, p - int(f)) for p, f in zip(positions, fwhms)
                ],
                "FWHM Forward Positions": [
                    min(n_frames - 1, p + int(f)) for p, f in zip(positions, fwhms)
                ],
                "FWHM Values": fwhms,
                "Spike Counts": [1] * len(amps),
                "is_outlier": False,
            }
        )

    n_rois = max(1, len(roi_specs))
    rng = np.random.default_rng(hash(sample_name) % (2**32))
    dff = rng.normal(0.0, 0.05, size=(n_rois, n_frames))
    for i, spec in enumerate(roi_specs):
        for p, a in zip(spec["positions"], spec["amplitudes"]):
            if 0 <= p < n_frames:
                dff[i, p] += a

    inputs = {"dff_dat": dff}

    return SimpleNamespace(
        sample_name=sample_name,
        _raw_peak_amplitude_data=raw_peaks,
        _raw_fwhm_data=raw_fwhm,
        _logger=logging.getLogger(f"test.shard.{sample_name}"),
        get_input=lambda name, req=False: inputs.get(name),
        spatial_filtering=lambda **_kw: list(range(n_rois)),
    )


def make_empty_shard(sample_name: str):
    """Build a shard with zero events (Layer-2 dropped everything)."""
    return SimpleNamespace(
        sample_name=sample_name,
        _raw_peak_amplitude_data=[],
        _raw_fwhm_data=[],
        _logger=logging.getLogger(f"test.shard.{sample_name}"),
        get_input=lambda name, req=False: None,
        spatial_filtering=lambda **_kw: [],
    )


def label_every_event(child) -> None:
    """Label every event on ``child`` as True (drives is_complete -> True)."""
    while not child.is_complete:
        before = child.unfinished_count
        child.label_current("True")
        # Defensive guard: if label_current doesn't reduce the
        # unfinished count we'd hang. The labeler advances the cursor
        # automatically; this assertion catches any regression where
        # labeling silently fails to record.
        assert child.unfinished_count < before, (
            "label_current did not reduce unfinished_count; possible "
            "regression in EventLabeler.label_current"
        )


# ---------------------------------------------------------------------------
# 1. Construction & basic API
# ---------------------------------------------------------------------------
def test_requires_at_least_one_shard(MultiShardLabeler, tmp_path):
    with pytest.raises(ValueError, match="at least one shard"):
        MultiShardLabeler(
            shards=[],
            corpus_path=str(tmp_path / "labels.csv"),
            labeler_id="alice",
        )


def test_basic_construction_and_properties(MultiShardLabeler, tmp_path):
    shards = [make_fake_shard("a"), make_fake_shard("b"), make_fake_shard("c")]
    wrapper = MultiShardLabeler(
        shards=shards,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    assert wrapper.n_shards == 3
    assert wrapper.current_index == 0
    assert wrapper.current_shard.sample_name == "a"


def test_corpus_path_is_canonicalized(MultiShardLabeler, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wrapper = MultiShardLabeler(
        shards=[make_fake_shard("a")],
        corpus_path="labels.csv",
        labeler_id="alice",
    )
    # Wrapper stores absolute path so a chdir between cells doesn't
    # move the labeler's corpus underneath it.
    assert os.path.isabs(wrapper.corpus_path)


# ---------------------------------------------------------------------------
# 2. EventLabeler.is_complete / unfinished_count semantics
# ---------------------------------------------------------------------------
def test_is_complete_predicate(EventLabeler, tmp_path):
    """is_complete tracks event-level labels and trace-level reviews."""
    shard = make_fake_shard("a")
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    assert not labeler.is_complete
    assert labeler.unfinished_count == labeler.total_events

    # Label every event on every ROI -> complete.
    while not labeler.is_complete:
        labeler.label_current("True")
    assert labeler.is_complete
    assert labeler.unfinished_count == 0


def test_skip_trace_counts_as_review(EventLabeler, tmp_path):
    """skip_trace makes that ROI's events count as 'done' for is_complete."""
    shard = make_fake_shard(
        "a",
        roi_specs=[
            {
                "roi_id": 0,
                "positions": [10, 50],
                "amplitudes": [0.5, 0.7],
                "fwhm": [4.0, 4.0],
            },
            {
                "roi_id": 1,
                "positions": [20],
                "amplitudes": [0.4],
                "fwhm": [3.0],
            },
        ],
    )
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    # Skip both traces (the labeler skips the *current* trace, then
    # advances to the next overview ROI).
    labeler.skip_trace()
    labeler.skip_trace()
    assert labeler.is_complete
    assert labeler.unfinished_count == 0


def test_empty_shard_is_trivially_complete(EventLabeler, tmp_path):
    labeler = EventLabeler(
        shard=make_empty_shard("a"),
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    assert labeler.total_events == 0
    assert labeler.is_complete


# ---------------------------------------------------------------------------
# 3. on_state_change callback fires on save
# ---------------------------------------------------------------------------
def test_on_state_change_fires_after_label(EventLabeler, tmp_path):
    shard = make_fake_shard("a")
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    calls: List[bool] = []
    labeler.on_state_change = lambda: calls.append(labeler.is_complete)
    labeler.label_current("True")
    assert len(calls) == 1
    # Label every remaining event; last call should see is_complete=True.
    while not labeler.is_complete:
        labeler.label_current("True")
    assert calls[-1] is True


def test_on_state_change_exception_does_not_break_save(EventLabeler, tmp_path, caplog):
    shard = make_fake_shard("a")
    labeler = EventLabeler(
        shard=shard,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    labeler.on_state_change = lambda: 1 / 0  # noqa: B023 — intentional
    with caplog.at_level(logging.WARNING):
        labeler.label_current("True")
    # Save still landed despite the callback raising.
    df = pd.read_csv(tmp_path / "labels.csv")
    assert len(df) == 1


# ---------------------------------------------------------------------------
# 4. Resume-on-reopen
# ---------------------------------------------------------------------------
def test_start_at_default_lands_on_first_unfinished(MultiShardLabeler, EventLabeler, tmp_path):
    """Reopening lands on the first image with unfinished work."""
    corpus = tmp_path / "labels.csv"
    shards = [
        make_fake_shard("a"),
        make_fake_shard("b"),
        make_fake_shard("c"),
    ]
    # Pre-label every event on shard 'a' so it's complete.
    pre = EventLabeler(
        shard=shards[0],
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    label_every_event(pre)
    assert pre.is_complete

    wrapper = MultiShardLabeler(
        shards=[make_fake_shard("a"), make_fake_shard("b"), make_fake_shard("c")],
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    # 'a' is done, so we resume on 'b'.
    assert wrapper.current_shard.sample_name == "b"


def test_start_at_explicit_int_and_name(MultiShardLabeler, tmp_path):
    shards = [make_fake_shard("a"), make_fake_shard("b"), make_fake_shard("c")]
    wrapper_idx = MultiShardLabeler(
        shards=shards,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
        start_at=2,
    )
    assert wrapper_idx.current_shard.sample_name == "c"

    wrapper_name = MultiShardLabeler(
        shards=[make_fake_shard("a"), make_fake_shard("b"), make_fake_shard("c")],
        corpus_path=str(tmp_path / "labels2.csv"),
        labeler_id="alice",
        start_at="b",
    )
    assert wrapper_name.current_index == 1


def test_start_at_invalid(MultiShardLabeler, tmp_path):
    shards = [make_fake_shard("a")]
    with pytest.raises(IndexError):
        MultiShardLabeler(
            shards=shards,
            corpus_path=str(tmp_path / "labels.csv"),
            labeler_id="alice",
            start_at=5,
        )
    with pytest.raises(KeyError):
        MultiShardLabeler(
            shards=[make_fake_shard("a")],
            corpus_path=str(tmp_path / "labels2.csv"),
            labeler_id="alice",
            start_at="nope",
        )


# ---------------------------------------------------------------------------
# 5. Navigation: next / prev / goto / next_unfinished
# ---------------------------------------------------------------------------
def test_next_prev_basic(MultiShardLabeler, tmp_path):
    shards = [make_fake_shard("a"), make_fake_shard("b"), make_fake_shard("c")]
    wrapper = MultiShardLabeler(
        shards=shards,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    assert wrapper.next_image() is True
    assert wrapper.current_shard.sample_name == "b"
    assert wrapper.next_image() is True
    assert wrapper.current_shard.sample_name == "c"
    assert wrapper.next_image() is False  # no further
    assert wrapper.prev_image() is True
    assert wrapper.current_shard.sample_name == "b"


def test_next_skips_empty_shards(MultiShardLabeler, tmp_path):
    """Empty shards (no labelable events) are skipped on next/prev."""
    shards = [
        make_fake_shard("a"),
        make_empty_shard("b_empty"),
        make_fake_shard("c"),
    ]
    wrapper = MultiShardLabeler(
        shards=shards,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    assert wrapper.current_shard.sample_name == "a"
    moved = wrapper.next_image()
    assert moved is True
    assert wrapper.current_shard.sample_name == "c"


def test_next_unfinished_image(MultiShardLabeler, EventLabeler, tmp_path):
    corpus = tmp_path / "labels.csv"
    # Pre-complete shard 'b' so next_unfinished skips over it.
    pre = EventLabeler(
        shard=make_fake_shard("b"),
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    label_every_event(pre)

    wrapper = MultiShardLabeler(
        shards=[make_fake_shard("a"), make_fake_shard("b"), make_fake_shard("c")],
        corpus_path=str(corpus),
        labeler_id="alice",
        start_at=0,
    )
    assert wrapper.next_unfinished_image() is True
    assert wrapper.current_shard.sample_name == "c"


def test_goto_image_by_int_and_name(MultiShardLabeler, tmp_path):
    wrapper = MultiShardLabeler(
        shards=[make_fake_shard("a"), make_fake_shard("b"), make_fake_shard("c")],
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    wrapper.goto_image(2)
    assert wrapper.current_shard.sample_name == "c"
    wrapper.goto_image("a")
    assert wrapper.current_shard.sample_name == "a"


def test_goto_image_invalid(MultiShardLabeler, tmp_path):
    wrapper = MultiShardLabeler(
        shards=[make_fake_shard("a")],
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    with pytest.raises(IndexError):
        wrapper.goto_image(99)
    with pytest.raises(KeyError):
        wrapper.goto_image("missing")


# ---------------------------------------------------------------------------
# 6. Snapshot / restore: cursor preserved across image switches
# ---------------------------------------------------------------------------
def test_cursor_preserved_across_switches(MultiShardLabeler, tmp_path):
    wrapper = MultiShardLabeler(
        shards=[make_fake_shard("a"), make_fake_shard("b")],
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    # Force-construct the active child by going through goto_image
    # (display() isn't safe to call without ipywidgets).
    wrapper._child = wrapper._make_child(wrapper.current_shard)
    wrapper._child._cursor = 2
    # Switch out; on return we expect the cursor restored.
    wrapper.next_image()
    wrapper.prev_image()
    assert wrapper._child is not None
    assert wrapper._child._cursor == 2


# ---------------------------------------------------------------------------
# 7. Auto-advance: completing a shard advances to the next unfinished
# ---------------------------------------------------------------------------
def test_auto_advance_on_completion(MultiShardLabeler, tmp_path):
    wrapper = MultiShardLabeler(
        shards=[make_fake_shard("a"), make_fake_shard("b"), make_fake_shard("c")],
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
        auto_advance=True,
    )
    # Construct the active child and wire auto-advance manually
    # (display() would do this but pulls in ipywidgets).
    wrapper._child = wrapper._make_child(wrapper.current_shard)
    wrapper._wire_auto_advance(wrapper._child)
    # Pretend the wrapper has been display()'d: stub a body container
    # so _switch_to's _render_active_view path doesn't crash. The
    # body just needs a 'children' attribute the wrapper can write
    # to; we use a SimpleNamespace and accept that _render_active_view
    # will short-circuit because the child already has its widgets
    # built lazily.
    wrapper._widgets = {}

    # Label every event on shard 'a'. The on_state_change hook should
    # fire after every save, and once the shard is complete the
    # wrapper auto-advances to 'b'.
    label_every_event(wrapper._child)
    assert wrapper.current_shard.sample_name == "b"


def test_auto_advance_disabled_keeps_current_shard(MultiShardLabeler, tmp_path):
    wrapper = MultiShardLabeler(
        shards=[make_fake_shard("a"), make_fake_shard("b")],
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
        auto_advance=False,
    )
    wrapper._child = wrapper._make_child(wrapper.current_shard)
    wrapper._wire_auto_advance(wrapper._child)
    label_every_event(wrapper._child)
    # auto_advance disabled => no movement.
    assert wrapper.current_shard.sample_name == "a"


def test_auto_advance_on_last_shard_does_not_crash(
    MultiShardLabeler, tmp_path
):
    """Completing the final shard with auto-advance on must not crash."""
    wrapper = MultiShardLabeler(
        shards=[make_fake_shard("a")],
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
        auto_advance=True,
    )
    wrapper._child = wrapper._make_child(wrapper.current_shard)
    wrapper._wire_auto_advance(wrapper._child)
    wrapper._widgets = {}  # display() not called
    label_every_event(wrapper._child)
    # The wrapper should have computed "everything is done" without
    # an exception. The completion banner is only rendered when
    # widgets are live; here we simply assert the index didn't move.
    assert wrapper.current_index == 0


# ---------------------------------------------------------------------------
# 8. Progress summary
# ---------------------------------------------------------------------------
def test_progress_summary_aggregates_across_shards(
    MultiShardLabeler, EventLabeler, tmp_path
):
    corpus = tmp_path / "labels.csv"
    # Pre-complete shard 'a'.
    pre = EventLabeler(
        shard=make_fake_shard("a"),
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    total_a = pre.total_events
    label_every_event(pre)

    wrapper = MultiShardLabeler(
        shards=[make_fake_shard("a"), make_fake_shard("b")],
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    summary = wrapper.progress_summary()
    assert summary["n_images"] == 2
    assert summary["n_reviewed_images"] == 1
    assert summary["n_total_events"] >= total_a
    assert summary["n_labeled_events"] == total_a
    assert summary["all_complete"] is False


def test_progress_summary_all_complete(MultiShardLabeler, EventLabeler, tmp_path):
    corpus = tmp_path / "labels.csv"
    for name in ("a", "b"):
        pre = EventLabeler(
            shard=make_fake_shard(name),
            corpus_path=str(corpus),
            labeler_id="alice",
        )
        label_every_event(pre)

    wrapper = MultiShardLabeler(
        shards=[make_fake_shard("a"), make_fake_shard("b")],
        corpus_path=str(corpus),
        labeler_id="alice",
    )
    summary = wrapper.progress_summary()
    assert summary["all_complete"] is True
    assert summary["n_reviewed_images"] == 2


# ---------------------------------------------------------------------------
# 9. Quiet probes: multi-shard construction must not log INFO per shard
# ---------------------------------------------------------------------------
def test_quiet_construction_silences_per_shard_info_logs(
    EventLabeler, tmp_path, caplog
):
    """``EventLabeler(quiet=True)`` downgrades the noisy INFO logs."""
    # Build a shard with one zero-event ROI (triggers the "ROI N has
    # zero events" INFO log) plus a Layer-2 bound that drops one
    # event (triggers the "skipped 1 event(s) outside the active
    # Layer-2 bounds" INFO log).
    shard = make_fake_shard(
        sample_name="loud",
        roi_specs=[
            {
                "roi_id": 0,
                "positions": [10],
                "amplitudes": [0.5],
                "fwhm": [4.0],
            },
            {
                "roi_id": 1,
                "positions": [],
                "amplitudes": [],
                "fwhm": [],
            },
        ],
    )
    shard._active_filter_events = True
    shard._active_min_event_amplitude = 0.4
    shard._active_max_event_amplitude = None
    shard._active_min_event_fwhm = None
    shard._active_max_event_fwhm = None

    # Loud baseline: with quiet=False both INFO messages fire.
    with caplog.at_level(logging.INFO, logger="test.shard.loud"):
        EventLabeler(
            shard=shard,
            corpus_path=str(tmp_path / "labels.csv"),
            labeler_id="alice",
            quiet=False,
        )
    loud_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("zero events" in m for m in loud_msgs), (
        "quiet=False should still emit the zero-event ROI INFO message"
    )

    caplog.clear()
    # Quiet baseline: same shard, same logger, quiet=True. The
    # construction-time INFOs are suppressed.
    with caplog.at_level(logging.INFO, logger="test.shard.loud"):
        EventLabeler(
            shard=shard,
            corpus_path=str(tmp_path / "labels2.csv"),
            labeler_id="alice",
            quiet=True,
        )
    quiet_infos = [
        r.message for r in caplog.records if r.levelno == logging.INFO
    ]
    assert all("zero events" not in m for m in quiet_infos), (
        "quiet=True must downgrade the zero-event INFO log to DEBUG"
    )


def test_multi_shard_probe_uses_quiet_construction(
    MultiShardLabeler, tmp_path, caplog
):
    """The wrapper's startup probes must not log INFO per shard."""
    # Three shards, each with one zero-event ROI to make construction
    # loud at INFO. If the wrapper probes with quiet=True (as it
    # should) we see no INFOs from these constructions.
    shards = [
        make_fake_shard(
            sample_name=name,
            roi_specs=[
                {
                    "roi_id": 0,
                    "positions": [10],
                    "amplitudes": [0.5],
                    "fwhm": [4.0],
                },
                {
                    "roi_id": 1,
                    "positions": [],
                    "amplitudes": [],
                    "fwhm": [],
                },
            ],
        )
        for name in ("a", "b", "c")
    ]
    with caplog.at_level(logging.INFO):
        MultiShardLabeler(
            shards=shards,
            corpus_path=str(tmp_path / "labels.csv"),
            labeler_id="alice",
        )
    infos = [r.message for r in caplog.records if r.levelno == logging.INFO]
    # The wrapper itself does not emit any INFO logs during
    # construction; any "zero events" message would have to come
    # from a probe-mode EventLabeler. There must be zero of them.
    assert all("zero events" not in m for m in infos), (
        f"MultiShardLabeler probes leaked per-shard INFO logs: "
        f"{[m for m in infos if 'zero events' in m]}"
    )


# ---------------------------------------------------------------------------
# 10. Probe cache: navigation does not re-walk every shard each click
# ---------------------------------------------------------------------------
def test_probe_cache_avoids_repeat_constructions(
    MultiShardLabeler, tmp_path, monkeypatch
):
    shards = [make_fake_shard(name) for name in ("a", "b", "c", "d")]
    wrapper = MultiShardLabeler(
        shards=shards,
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    # ``_resolve_start_index`` short-circuits at the first unfinished
    # shard, so after construction the cache holds 'a' only — we
    # never paid for probes on shards 'b', 'c', 'd' yet. This is the
    # design: we don't waste work walking shards we haven't been
    # asked about.
    assert set(wrapper._probe_cache.keys()) == {"a"}

    # Force a full progress probe — this is what the header refresh
    # does — and confirm the cache fills in for all shards. After
    # this point any further progress / navigation reads should be
    # cache hits, not fresh constructions.
    wrapper.progress_summary()
    assert set(wrapper._probe_cache.keys()) == {"a", "b", "c", "d"}

    # Spy on _make_child to count fresh constructions during nav.
    orig_make_child = wrapper._make_child
    n_constructions = {"count": 0}

    def _spy_make_child(shard, *, quiet=False):
        n_constructions["count"] += 1
        return orig_make_child(shard, quiet=quiet)

    monkeypatch.setattr(wrapper, "_make_child", _spy_make_child)

    # Now navigate. _switch_to constructs the active child (one per
    # move), but progress refresh and skip-empty checks should hit
    # the cache for every other shard.
    wrapper.next_image()           # _switch_to(1)         => +1 active-child
    wrapper.next_image()           # _switch_to(2)         => +1 active-child
    wrapper.prev_image()           # _switch_to(1)         => +1 active-child
    wrapper.progress_summary()     # all-cached, no probes
    wrapper.next_unfinished_image()# may _switch_to(...)   => up to +1 active-child

    # Up to four _switch_to calls => four active-child constructions.
    # Without the cache, each navigation step would *also* fan out
    # one probe construction per other shard (3 per call across 5
    # nav-style calls => 15 extra constructions on top). With the
    # cache: only the active-child ones.
    assert n_constructions["count"] <= 4, (
        f"Expected at most 4 active-child constructions during nav, "
        f"got {n_constructions['count']} — probe cache is not being "
        f"consulted (probes leaked through to _make_child)"
    )


def test_probe_cache_invalidated_on_save(MultiShardLabeler, tmp_path):
    """A save on the active child must invalidate that shard's cache."""
    wrapper = MultiShardLabeler(
        shards=[make_fake_shard("a"), make_fake_shard("b")],
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    # Build the active child and wire the on_state_change hook the
    # way display() would. Stub the wrapper's widgets dict so
    # _refresh_header bails cleanly.
    wrapper._child = wrapper._make_child(wrapper.current_shard)
    wrapper._widgets = {}
    wrapper._wire_auto_advance(wrapper._child)

    # Force-cache the active shard's probe.
    summary_before = wrapper.progress_summary()
    assert "a" in wrapper._probe_cache

    # Label one event — the on_state_change hook should drop the
    # cache entry for "a".
    wrapper._child.label_current("True")
    assert "a" not in wrapper._probe_cache, (
        "save on active child must invalidate its probe cache entry"
    )

    # The next progress_summary refresh should reflect the new label.
    summary_after = wrapper.progress_summary()
    assert summary_after["n_labeled_events"] == summary_before["n_labeled_events"] + 1


# ---------------------------------------------------------------------------
# 11. Display() raises a clear ImportError without ipywidgets
# ---------------------------------------------------------------------------
def test_display_raises_clean_error_without_ipywidgets(MultiShardLabeler, tmp_path):
    wrapper = MultiShardLabeler(
        shards=[make_fake_shard("a")],
        corpus_path=str(tmp_path / "labels.csv"),
        labeler_id="alice",
    )
    with mock.patch.dict(sys.modules, {"ipywidgets": None}):
        with pytest.raises(ImportError) as excinfo:
            wrapper.display()
        assert "wizards_staff[labeling]" in str(excinfo.value)
