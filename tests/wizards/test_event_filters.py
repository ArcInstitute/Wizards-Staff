"""Tests for ``wizards_staff.wizards.cauldron._apply_event_filters``.

Covers the regression where per-event filtering only propagated to peak /
FWHM data and left rise / fall / peak-to-peak / FRPM reflecting the raw
z-score crossings (so two metrics on the same shard described different
event sets).

The module under test imports caiman / tensorflow transitively, so we
load just ``cauldron`` and its dependencies via ``importlib`` from the
file system, monkey-patching the heavyweight optional imports. This
mirrors the pattern used by ``tests/stats/test_outliers.py``.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Module loader — keeps the test suite independent of caiman / tensorflow
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_stub(name: str, attrs: dict | None = None) -> types.ModuleType:
    """Install (or refresh) a stub module under ``name`` in ``sys.modules``."""
    mod = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _load_cauldron():
    """Load ``wizards_staff.wizards.cauldron`` without dragging caiman in."""
    if "cauldron_under_test" in sys.modules:
        return sys.modules["cauldron_under_test"]

    # Stub out heavy optional dependencies that cauldron transitively
    # imports through spellbook / shard / stats. We never call into them
    # in these tests, so empty stubs are fine.
    caiman_root = _install_stub("caiman")
    caiman_se = _install_stub("caiman.source_extraction")
    caiman_cnmf = _install_stub("caiman.source_extraction.cnmf")
    deconv = _install_stub(
        "caiman.source_extraction.cnmf.deconvolution",
        attrs={
            "estimate_time_constant": lambda *a, **kw: np.zeros(2),
            "GetSn": lambda *a, **kw: 1.0,
            "axcov": lambda *a, **kw: np.zeros((10, 1)),
            "constrained_foopsi": lambda *a, **kw: (None,) * 7,
        },
    )
    caiman_cnmf.deconvolution = deconv
    caiman_se.cnmf = caiman_cnmf
    caiman_root.source_extraction = caiman_se

    # Tensorflow / matplotlib backends pulled in by plotting.* — stub
    # only the bits cauldron's import touches.
    _install_stub("ipywidgets")
    # caiman pulls in skimage, but skimage is real; let it import.

    pkg_root = REPO_ROOT / "wizards_staff"

    def _load(mod_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    # Load cauldron via the real package machinery so its relative
    # imports resolve — but stub the heavy submodules first.
    sys.path.insert(0, str(REPO_ROOT))
    try:
        # Import the cauldron module via standard import.
        from wizards_staff.wizards import cauldron  # type: ignore
    finally:
        try:
            sys.path.remove(str(REPO_ROOT))
        except ValueError:
            pass

    sys.modules["cauldron_under_test"] = cauldron
    return cauldron


@pytest.fixture(scope="module")
def cauldron():
    return _load_cauldron()


# ---------------------------------------------------------------------------
# Synthetic shard helpers
# ---------------------------------------------------------------------------
def _make_shard(
    sample_name: str = "S1",
    *,
    per_neuron_events: list[list[dict]] | None = None,
    n_frames: int = 600,
    frate: int = 10,
):
    """Build a SimpleNamespace shard pre-populated with raw event lists.

    Each entry in ``per_neuron_events`` is a list of event dicts with
    keys: ``amp``, ``peak``, ``rise_time``, ``rise_pos``, ``fall_time``,
    ``fall_pos``, ``fwhm``, ``fwhm_back``, ``fwhm_fwd``, ``spike_count``.
    Entries are listed in walk order — i.e. the i-th entry is the i-th
    event for that neuron and shares its index across every metric.
    """
    if per_neuron_events is None:
        per_neuron_events = []

    raw_peak = []
    raw_fwhm = []
    raw_rise = []
    raw_fall = []
    raw_p2p = []
    raw_frpm = []

    for neuron_idx, events in enumerate(per_neuron_events):
        amps = [e["amp"] for e in events]
        peak_pos = [e["peak"] for e in events]
        rise_times = [e["rise_time"] for e in events]
        rise_pos = [e["rise_pos"] for e in events]
        fall_times = [e["fall_time"] for e in events]
        fall_pos = [e["fall_pos"] for e in events]
        fwhm_vals = [e["fwhm"] for e in events]
        fwhm_back = [e["fwhm_back"] for e in events]
        fwhm_fwd = [e["fwhm_fwd"] for e in events]
        spike_counts = [e.get("spike_count", 1) for e in events]

        raw_peak.append({
            "Sample": sample_name,
            "Neuron": neuron_idx,
            "Peak Amplitudes": list(amps),
            "Peak Positions": list(peak_pos),
            "is_outlier": False,
        })
        raw_fwhm.append({
            "Sample": sample_name,
            "Neuron": neuron_idx,
            "FWHM Backward Positions": list(fwhm_back),
            "FWHM Forward Positions": list(fwhm_fwd),
            "FWHM Values": list(fwhm_vals),
            "Spike Counts": list(spike_counts),
            "is_outlier": False,
        })
        raw_rise.append({
            "Sample": sample_name,
            "Neuron": neuron_idx,
            "Rise Times": list(rise_times),
            "Rise Positions": list(rise_pos),
            "is_outlier": False,
        })
        raw_fall.append({
            "Sample": sample_name,
            "Neuron": neuron_idx,
            "Fall Times": list(fall_times),
            "Fall Positions": list(fall_pos),
            "is_outlier": False,
        })
        # Raw intervals are simple diffs of peak positions.
        intervals = (
            list(np.diff(np.asarray(peak_pos)).astype(float))
            if len(peak_pos) >= 2
            else []
        )
        raw_p2p.append({
            "Sample": sample_name,
            "Neuron": neuron_idx,
            "Peak Positions": list(peak_pos),
            "Inter-Spike Intervals": intervals,
            "is_outlier": False,
        })
        n_events = len(events)
        raw_frpm.append({
            "Sample": sample_name,
            "Neuron": neuron_idx,
            "Neuron Index": neuron_idx,
            "Firing Rate Per Min": (
                n_events * 60.0 * frate / n_frames if n_frames else float("nan")
            ),
            "N Events": n_events,
            "N Frames": n_frames,
            "Frate": frate,
            "is_outlier": False,
        })

    return SimpleNamespace(
        sample_name=sample_name,
        _raw_peak_amplitude_data=raw_peak,
        _raw_fwhm_data=raw_fwhm,
        _raw_rise_time_data=raw_rise,
        _raw_fall_time_data=raw_fall,
        _raw_peak_to_peak_data=raw_p2p,
        _raw_frpm_data=raw_frpm,
        _peak_amplitude_data=[],
        _max_peak_amplitude_data=[],
        _fwhm_data=[],
        _rise_time_data=[],
        _fall_time_data=[],
        _peak_to_peak_data=[],
        _frpm_data=[],
        _recording_n_frames=n_frames,
        _recording_frate=frate,
        _logger=logging.getLogger(f"test.shard.{sample_name}"),
    )


def _ev(amp, peak, rise=4, fwhm=5):
    """Build one canonical event dict with sensible defaults."""
    return {
        "amp": float(amp),
        "peak": int(peak),
        "rise_time": int(rise),
        "rise_pos": int(peak) + 1,
        "fall_time": int(rise) * 2,
        "fall_pos": int(peak),
        "fwhm": float(fwhm),
        "fwhm_back": int(peak) - int(fwhm) // 2,
        "fwhm_fwd": int(peak) + int(fwhm) // 2,
        "spike_count": 1,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_keep_mask_consistency(cauldron):
    """After filtering, every per-event metric has the same length per neuron."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.02, 100), _ev(1.0, 200), _ev(15.0, 300)],
        [_ev(0.3, 80), _ev(0.4, 160), _ev(0.5, 240)],
    ])
    summary = cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=2,
        max_event_fwhm=None,
    )

    # Neuron 0: drop event 1 (amp=0.02 < 0.05) and event 3 (amp=15 > 10).
    # Neuron 1: keep all three.
    assert [len(r["Peak Amplitudes"]) for r in shard._peak_amplitude_data] == [2, 3]
    assert [len(r["FWHM Values"]) for r in shard._fwhm_data] == [2, 3]
    assert [len(r["Rise Times"]) for r in shard._rise_time_data] == [2, 3]
    assert [len(r["Fall Times"]) for r in shard._fall_time_data] == [2, 3]
    # peak-to-peak has n-1 intervals after dropping.
    assert [len(r["Inter-Spike Intervals"]) for r in shard._peak_to_peak_data] == [1, 2]
    # frpm has one row per neuron.
    assert len(shard._frpm_data) == 2
    assert summary["total_events_after"] == 5
    assert summary["total_events_before"] == 7
    # Surviving events match by peak position.
    assert shard._peak_amplitude_data[0]["Peak Positions"] == [50, 200]
    assert shard._rise_time_data[0]["Rise Positions"] == [51, 201]


def test_nan_inf_scrub_propagates(cauldron):
    """The unconditional NaN/Inf scrub must touch every metric, not just peak/FWHM."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(float("nan"), 100), _ev(1.0, 200)],
    ])
    summary = cauldron._apply_event_filters(
        shard,
        filter_events=False,  # bounds disabled — only NaN scrub fires
        min_event_amplitude=None,
        max_event_amplitude=None,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )

    assert summary["dropped_by_amplitude_nan"] == 1
    assert len(shard._peak_amplitude_data[0]["Peak Amplitudes"]) == 2
    assert len(shard._fwhm_data[0]["FWHM Values"]) == 2
    # Regression: rise/fall/peak-to-peak/frpm must also drop the bad event.
    assert len(shard._rise_time_data[0]["Rise Times"]) == 2
    assert len(shard._fall_time_data[0]["Fall Times"]) == 2
    assert len(shard._peak_to_peak_data[0]["Inter-Spike Intervals"]) == 1
    assert shard._frpm_data[0]["N Events"] == 2


def test_amplitude_bounds_propagate(cauldron):
    """Real but out-of-bounds amplitudes drop from every per-event metric."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(20.0, 100), _ev(1.0, 200)],  # event 1 above max
    ])
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    for attr, col in (
        ("_peak_amplitude_data", "Peak Amplitudes"),
        ("_fwhm_data", "FWHM Values"),
        ("_rise_time_data", "Rise Times"),
        ("_fall_time_data", "Fall Times"),
    ):
        assert len(getattr(shard, attr)[0][col]) == 2, attr
    assert shard._peak_amplitude_data[0]["Peak Positions"] == [50, 200]


def test_fwhm_bounds_propagate(cauldron):
    """FWHM bound rejections cascade to every per-event metric."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50, fwhm=10), _ev(0.5, 100, fwhm=1), _ev(0.5, 200, fwhm=8)],
    ])
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=None,
        max_event_amplitude=None,
        min_event_fwhm=2,
        max_event_fwhm=None,
    )
    # Event with FWHM=1 is dropped.
    for attr, col in (
        ("_peak_amplitude_data", "Peak Amplitudes"),
        ("_fwhm_data", "FWHM Values"),
        ("_rise_time_data", "Rise Times"),
        ("_fall_time_data", "Fall Times"),
    ):
        assert len(getattr(shard, attr)[0][col]) == 2, attr


def test_peak_to_peak_recomputed(cauldron):
    """Inter-event intervals are gaps between *surviving* peaks."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 10), _ev(0.5, 20), _ev(0.02, 30), _ev(0.5, 40), _ev(0.5, 50)],
    ])
    # Drop event at peak=30 (amp=0.02 < min).
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    intervals = shard._peak_to_peak_data[0]["Inter-Spike Intervals"]
    assert intervals == [10.0, 20.0, 10.0]


def test_frpm_uses_filtered_events(cauldron):
    """Filtered FRPM = surviving event count per minute, not raw."""
    n_frames = 600  # 60 s at 10 fps
    frate = 10
    # 100 raw events with 30 below min amplitude → 70 should survive.
    events = []
    rng = np.random.default_rng(0)
    for i in range(70):
        events.append(_ev(0.5, 5 + i * 5))
    for i in range(30):
        events.append(_ev(0.001, 5 + (70 + i) * 5))
    shard = _make_shard(
        per_neuron_events=[events],
        n_frames=n_frames,
        frate=frate,
    )
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    expected = 70 * 60.0 * frate / n_frames  # = 70.0 events / minute
    row = shard._frpm_data[0]
    assert row["N Events"] == 70
    assert row["Firing Rate Per Min"] == pytest.approx(expected)


def test_filter_events_false_still_scrubs_nan(cauldron):
    """filter_events=False still applies the unconditional NaN/Inf scrub."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(float("inf"), 100), _ev(1.0, 200)],
    ])
    summary = cauldron._apply_event_filters(
        shard,
        filter_events=False,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=2,
        max_event_fwhm=None,
    )
    # Bounds are off; only the NaN/Inf scrub fired.
    assert summary["apply_amp_filter"] is False
    assert summary["dropped_by_amplitude_nan"] == 1
    assert summary["dropped_by_amplitude_bounds"] == 0
    for attr, col in (
        ("_peak_amplitude_data", "Peak Amplitudes"),
        ("_fwhm_data", "FWHM Values"),
        ("_rise_time_data", "Rise Times"),
        ("_fall_time_data", "Fall Times"),
    ):
        assert len(getattr(shard, attr)[0][col]) == 2, attr


def test_raw_lists_unchanged_by_filter(cauldron):
    """``_apply_event_filters`` is a pure function of raw lists — never mutates them."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.02, 100), _ev(1.0, 200), _ev(15.0, 300)],
    ])
    snapshot = {
        attr: [dict(row) for row in getattr(shard, attr)]
        for attr in (
            "_raw_peak_amplitude_data",
            "_raw_fwhm_data",
            "_raw_rise_time_data",
            "_raw_fall_time_data",
            "_raw_peak_to_peak_data",
            "_raw_frpm_data",
        )
    }
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=2,
        max_event_fwhm=None,
    )
    for attr, expected in snapshot.items():
        actual = getattr(shard, attr)
        assert len(actual) == len(expected), attr
        for got_row, want_row in zip(actual, expected):
            assert dict(got_row) == dict(want_row), attr


def test_idempotent_filter(cauldron):
    """Applying the filter twice with the same bounds is a no-op on the second pass."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.02, 100), _ev(1.0, 200)],
    ])
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    first = [dict(row) for row in shard._peak_amplitude_data]
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    second = [dict(row) for row in shard._peak_amplitude_data]
    assert first == second


def test_loosening_bounds_recovers_events(cauldron):
    """Re-applying with looser bounds recovers events that the tighter pass dropped.

    Validates the ``Orb.refilter_events`` contract: filter is a pure
    function of raw lists, so changing bounds re-derives a fresh view.
    """
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.02, 100), _ev(1.0, 200)],
    ])
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    assert len(shard._peak_amplitude_data[0]["Peak Amplitudes"]) == 2

    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.001,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    assert len(shard._peak_amplitude_data[0]["Peak Amplitudes"]) == 3
    assert len(shard._rise_time_data[0]["Rise Times"]) == 3
    assert len(shard._fall_time_data[0]["Fall Times"]) == 3
    assert shard._peak_to_peak_data[0]["Inter-Spike Intervals"] == [50.0, 100.0]


def test_propagation_count_matches_intersection(cauldron):
    """The combined keep-mask cardinality equals the (amplitude AND FWHM) intersection."""
    shard = _make_shard(per_neuron_events=[
        # event 0: amp ok,    fwhm ok      -> keep
        # event 1: amp small, fwhm ok      -> drop (amp)
        # event 2: amp ok,    fwhm tiny    -> drop (fwhm)
        # event 3: amp small, fwhm tiny    -> drop (both)
        # event 4: amp ok,    fwhm ok      -> keep
        [
            _ev(0.5, 10),
            _ev(0.001, 20),
            _ev(0.5, 30, fwhm=1),
            _ev(0.001, 40, fwhm=1),
            _ev(0.5, 50),
        ],
    ])
    summary = cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=2,
        max_event_fwhm=None,
    )
    assert summary["total_events_before"] == 5
    assert summary["total_events_after"] == 2
    assert summary["dropped_by_amplitude_bounds"] == 2  # events 1 and 3
    assert summary["dropped_by_fwhm_bounds"] == 2       # events 2 and 3
    # No NaN values in this fixture.
    assert summary["dropped_by_amplitude_nan"] == 0
    assert summary["dropped_by_fwhm_nan"] == 0


def test_refilter_events_refreshes_all_metrics(cauldron):
    """``Orb.refilter_events`` updates rise/fall/peak-to-peak/FRPM, not just peak/FWHM."""
    from wizards_staff.wizards.orb import Orb

    # Build a minimal Orb-like object with one shard whose raw lists are
    # pre-populated. We bypass __init__ via __new__ to avoid the metadata
    # / file-categorisation dance that's irrelevant to refilter_events.
    orb = Orb.__new__(Orb)
    orb._logger = logging.getLogger("test.orb.refilter")
    orb._remove_outlier = False
    orb._show_outlier = False
    orb._shards = {}
    # Initialise every cached DataFrame slot to None so the lazy
    # _get_shard_data path rebuilds from the shard lists each time.
    for cache in (
        "_rise_time_data",
        "_fall_time_data",
        "_fwhm_data",
        "_frpm_data",
        "_peak_amplitude_data",
        "_max_peak_amplitude_data",
        "_peak_to_peak_data",
    ):
        setattr(orb, cache, None)
    orb.metadata = pd.DataFrame({
        "Sample": ["S1"],
        "Well": ["A1"],
        "Frate": [10],
    })

    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.02, 100), _ev(1.0, 200), _ev(15.0, 300)],
    ])
    orb._shards["S1"] = shard

    # First refilter pass with tight bounds: drops the 0.02 and 15.0 events.
    orb.refilter_events(
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=2,
        max_event_fwhm=None,
    )
    assert len(shard._peak_amplitude_data[0]["Peak Amplitudes"]) == 2
    assert len(shard._rise_time_data[0]["Rise Times"]) == 2
    assert len(shard._fall_time_data[0]["Fall Times"]) == 2
    assert len(shard._peak_to_peak_data[0]["Inter-Spike Intervals"]) == 1
    assert shard._frpm_data[0]["N Events"] == 2

    # Caches must have been invalidated so subsequent reads see the
    # new view. Manually peek at the protected slots.
    assert orb._rise_time_data is None
    assert orb._fall_time_data is None
    assert orb._peak_to_peak_data is None
    assert orb._frpm_data is None

    # Looser pass: recover the 0.02 amplitude event.
    orb.refilter_events(
        filter_events=True,
        min_event_amplitude=0.001,
        max_event_amplitude=20.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    assert len(shard._peak_amplitude_data[0]["Peak Amplitudes"]) == 4
    assert len(shard._rise_time_data[0]["Rise Times"]) == 4
    assert len(shard._fall_time_data[0]["Fall Times"]) == 4
    assert shard._frpm_data[0]["N Events"] == 4


def test_walk_order_invariant_on_synthetic_trace(cauldron):
    """The shared canonical walk produces aligned event counts across spellbook functions.

    Regression for the bug where ``calc_fwhm_spikes`` and ``calc_fall_tm``
    used wider index advances and dropped events that ``calc_rise_tm`` /
    ``calc_peak_amplitude`` / ``calc_peak_to_peak`` caught — making a
    positional keep-mask unsound.
    """
    from wizards_staff.wizards import spellbook

    T = 200
    calcium = np.zeros(T)
    # Three closely-spaced events in the same neuron.
    calcium[20:25] = np.linspace(0, 2.0, 5)
    for i in range(25, 60):
        calcium[i] = max(0, calcium[i - 1] * 0.97)
    calcium[60:65] = np.linspace(calcium[59], calcium[59] + 2.0, 5)
    for i in range(65, 100):
        calcium[i] = max(0, calcium[i - 1] * 0.95)
    calcium[100:105] = np.linspace(calcium[99], calcium[99] + 2.0, 5)
    for i in range(105, T):
        calcium[i] = max(0, calcium[i - 1] * 0.92)
    spikes = np.zeros(T)
    spikes[20] = 5.0
    spikes[60] = 5.0
    spikes[100] = 5.0

    rise_t, rise_p = spellbook.calc_rise_tm(
        calcium[None, :], spikes[None, :], zscore_threshold=3
    )
    fall_t, fall_p = spellbook.calc_fall_tm(
        calcium[None, :], spikes[None, :], zscore_threshold=3
    )
    fwhm_b, fwhm_f, fwhm_v, fwhm_c = spellbook.calc_fwhm_spikes(
        calcium[None, :], spikes[None, :], zscore_threshold=3
    )
    pk_a, pk_p = spellbook.calc_peak_amplitude(
        calcium[None, :], spikes[None, :], zscore_threshold=3
    )
    p2p = spellbook.calc_peak_to_peak(
        calcium[None, :], spikes[None, :], zscore_threshold=3
    )

    n_rise = len(rise_t[0])
    assert n_rise == 3, n_rise
    assert len(fall_t[0]) == n_rise
    assert len(fwhm_v[0]) == n_rise
    assert len(pk_a[0]) == n_rise
    assert len(p2p[0]) == n_rise - 1


# ---------------------------------------------------------------------------
# Tests: labels-corpus integration (third filter layer)
# ---------------------------------------------------------------------------
def _write_corpus(
    path: Path,
    rows: list[dict],
) -> Path:
    """Write a minimal labels corpus CSV that ``_apply_event_filters`` can consume."""
    columns = (
        "corpus_version",
        "sample_id",
        "roi_id",
        "event_idx",
        "label",
        "labeler_id",
        "timestamp",
        "notes",
    )
    if not rows:
        df = pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    else:
        df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)
    return path


def test_labels_corpus_drops_events(cauldron, tmp_path):
    """A single label=False row drops that event from every per-event metric."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.5, 100), _ev(0.5, 200)],
    ])
    corpus = _write_corpus(
        tmp_path / "corpus.csv",
        [
            {
                "corpus_version": 2,
                "sample_id": "S1",
                "roi_id": 0,
                "event_idx": 1,  # the middle event
                "label": "False",
                "labeler_id": "alice",
                "timestamp": "",
                "notes": "",
            },
        ],
    )
    summary = cauldron._apply_event_filters(
        shard,
        filter_events=False,
        min_event_amplitude=None,
        max_event_amplitude=None,
        min_event_fwhm=None,
        max_event_fwhm=None,
        labels_corpus=corpus,
    )
    assert summary["apply_label_filter"] is True
    assert summary["dropped_by_labels"] == 1
    assert summary["total_events_after"] == 2
    # Surviving events match by peak position (event_idx=1 had peak=100).
    assert shard._peak_amplitude_data[0]["Peak Positions"] == [50, 200]
    for attr, col in (
        ("_peak_amplitude_data", "Peak Amplitudes"),
        ("_fwhm_data", "FWHM Values"),
        ("_rise_time_data", "Rise Times"),
        ("_fall_time_data", "Fall Times"),
    ):
        assert len(getattr(shard, attr)[0][col]) == 2, attr
    assert shard._peak_to_peak_data[0]["Inter-Spike Intervals"] == [150.0]
    assert shard._frpm_data[0]["N Events"] == 2


def test_labels_corpus_unsure_kept(cauldron, tmp_path):
    """label=Unsure is treated as not-labeled and never causes a drop."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.5, 100), _ev(0.5, 200)],
    ])
    corpus = _write_corpus(
        tmp_path / "corpus.csv",
        [
            {
                "corpus_version": 2,
                "sample_id": "S1",
                "roi_id": 0,
                "event_idx": 1,
                "label": "Unsure",
                "labeler_id": "alice",
                "timestamp": "",
                "notes": "",
            },
        ],
    )
    summary = cauldron._apply_event_filters(
        shard,
        filter_events=False,
        min_event_amplitude=None,
        max_event_amplitude=None,
        min_event_fwhm=None,
        max_event_fwhm=None,
        labels_corpus=corpus,
    )
    assert summary["dropped_by_labels"] == 0
    assert summary["label_unsure_only"] == 1
    assert summary["total_events_after"] == 3
    assert len(shard._peak_amplitude_data[0]["Peak Amplitudes"]) == 3


def test_labels_corpus_disagreement_drop(cauldron, tmp_path):
    """on_disagreement='drop' resolves a True/False conflict by dropping."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.5, 100), _ev(0.5, 200)],
    ])
    corpus = _write_corpus(
        tmp_path / "corpus.csv",
        [
            {
                "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
                "event_idx": 1, "label": "True", "labeler_id": "alice",
                "timestamp": "", "notes": "",
            },
            {
                "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
                "event_idx": 1, "label": "False", "labeler_id": "bob",
                "timestamp": "", "notes": "",
            },
        ],
    )
    summary = cauldron._apply_event_filters(
        shard,
        filter_events=False,
        min_event_amplitude=None,
        max_event_amplitude=None,
        min_event_fwhm=None,
        max_event_fwhm=None,
        labels_corpus=corpus,
        on_disagreement="drop",
    )
    assert summary["dropped_by_labels"] == 1
    assert summary["label_disagreements"] == 1
    assert shard._peak_amplitude_data[0]["Peak Positions"] == [50, 200]


def test_labels_corpus_disagreement_keep(cauldron, tmp_path):
    """on_disagreement='keep' resolves a True/False conflict by keeping."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.5, 100), _ev(0.5, 200)],
    ])
    corpus = _write_corpus(
        tmp_path / "corpus.csv",
        [
            {
                "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
                "event_idx": 1, "label": "True", "labeler_id": "alice",
                "timestamp": "", "notes": "",
            },
            {
                "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
                "event_idx": 1, "label": "False", "labeler_id": "bob",
                "timestamp": "", "notes": "",
            },
        ],
    )
    summary = cauldron._apply_event_filters(
        shard,
        filter_events=False,
        min_event_amplitude=None,
        max_event_amplitude=None,
        min_event_fwhm=None,
        max_event_fwhm=None,
        labels_corpus=corpus,
        on_disagreement="keep",
    )
    assert summary["dropped_by_labels"] == 0
    assert summary["label_disagreements"] == 1
    assert summary["total_events_after"] == 3


def test_labels_corpus_disagreement_majority(cauldron, tmp_path):
    """on_disagreement='majority' picks the side with more votes; ties drop."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.5, 100), _ev(0.5, 200)],
    ])
    # Event 0: 2 False vs 1 True  -> drop (majority False).
    # Event 2: 1 False vs 2 True  -> keep (majority True).
    rows = []
    for labeler in ("alice", "bob"):
        rows.append({
            "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
            "event_idx": 0, "label": "False", "labeler_id": labeler,
            "timestamp": "", "notes": "",
        })
    rows.append({
        "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
        "event_idx": 0, "label": "True", "labeler_id": "carol",
        "timestamp": "", "notes": "",
    })
    rows.append({
        "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
        "event_idx": 2, "label": "False", "labeler_id": "alice",
        "timestamp": "", "notes": "",
    })
    for labeler in ("bob", "carol"):
        rows.append({
            "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
            "event_idx": 2, "label": "True", "labeler_id": labeler,
            "timestamp": "", "notes": "",
        })
    corpus = _write_corpus(tmp_path / "corpus.csv", rows)
    cauldron._apply_event_filters(
        shard,
        filter_events=False,
        min_event_amplitude=None,
        max_event_amplitude=None,
        min_event_fwhm=None,
        max_event_fwhm=None,
        labels_corpus=corpus,
        on_disagreement="majority",
    )
    # Event at peak=50 dropped; events at peak=100 and peak=200 survive.
    assert shard._peak_amplitude_data[0]["Peak Positions"] == [100, 200]


def test_labels_corpus_missing_file(cauldron, caplog, tmp_path):
    """A missing corpus path warns but does not crash; all events pass through."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.5, 100), _ev(0.5, 200)],
    ])
    missing = tmp_path / "definitely-not-here.csv"
    with caplog.at_level(logging.WARNING):
        summary = cauldron._apply_event_filters(
            shard,
            filter_events=False,
            min_event_amplitude=None,
            max_event_amplitude=None,
            min_event_fwhm=None,
            max_event_fwhm=None,
            labels_corpus=missing,
        )
    assert summary["dropped_by_labels"] == 0
    assert summary["total_events_after"] == 3
    assert any(
        "does not exist" in rec.message and "label-based filtering" in rec.message
        for rec in caplog.records
    )


def test_labels_corpus_with_other_filters(cauldron, tmp_path):
    """Labels compose with NaN/Inf scrub and amplitude bounds into a single keep-mask."""
    shard = _make_shard(per_neuron_events=[
        # event 0: amp ok                       -> keep (no label)
        # event 1: amp NaN                      -> drop (NaN scrub)
        # event 2: amp out-of-bounds (too big)  -> drop (amplitude bounds)
        # event 3: amp ok, labeled False        -> drop (label)
        # event 4: amp ok                       -> keep (no label)
        [
            _ev(0.5, 10),
            _ev(float("nan"), 20),
            _ev(15.0, 30),
            _ev(0.5, 40),
            _ev(0.5, 50),
        ],
    ])
    corpus = _write_corpus(
        tmp_path / "corpus.csv",
        [
            {
                "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
                "event_idx": 3, "label": "False", "labeler_id": "alice",
                "timestamp": "", "notes": "",
            },
            # Also label event 2 False — it would already be dropped by
            # the amplitude bounds, so dropped_by_labels must NOT
            # double-count it.
            {
                "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
                "event_idx": 2, "label": "False", "labeler_id": "alice",
                "timestamp": "", "notes": "",
            },
        ],
    )
    summary = cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
        labels_corpus=corpus,
    )
    assert summary["dropped_by_amplitude_nan"] == 1
    assert summary["dropped_by_amplitude_bounds"] == 1
    # Only event 3 was *additionally* dropped by labels; event 2 was
    # already going to be dropped by the amplitude bounds.
    assert summary["dropped_by_labels"] == 1
    assert summary["total_events_before"] == 5
    assert summary["total_events_after"] == 2
    assert shard._peak_amplitude_data[0]["Peak Positions"] == [10, 50]


def test_refilter_events_with_labels(cauldron, tmp_path):
    """Orb.refilter_events(labels_corpus=...) refreshes every per-event metric."""
    from wizards_staff.wizards.orb import Orb

    orb = Orb.__new__(Orb)
    orb._logger = logging.getLogger("test.orb.refilter.labels")
    orb._remove_outlier = False
    orb._show_outlier = False
    orb._shards = {}
    for cache in (
        "_rise_time_data",
        "_fall_time_data",
        "_fwhm_data",
        "_frpm_data",
        "_peak_amplitude_data",
        "_max_peak_amplitude_data",
        "_peak_to_peak_data",
    ):
        setattr(orb, cache, None)
    orb.metadata = pd.DataFrame({
        "Sample": ["S1"],
        "Well": ["A1"],
        "Frate": [10],
    })

    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.5, 100), _ev(0.5, 200), _ev(0.5, 300)],
    ])
    orb._shards["S1"] = shard

    corpus = _write_corpus(
        tmp_path / "corpus.csv",
        [
            {
                "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
                "event_idx": 1, "label": "False", "labeler_id": "alice",
                "timestamp": "", "notes": "",
            },
            {
                "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
                "event_idx": 2, "label": "False", "labeler_id": "alice",
                "timestamp": "", "notes": "",
            },
        ],
    )

    orb.refilter_events(
        filter_events=False,
        min_event_amplitude=None,
        max_event_amplitude=None,
        min_event_fwhm=None,
        max_event_fwhm=None,
        labels_corpus=corpus,
    )
    # Two of four events labeled False -> two survive across every
    # per-event metric.
    assert len(shard._peak_amplitude_data[0]["Peak Amplitudes"]) == 2
    assert len(shard._rise_time_data[0]["Rise Times"]) == 2
    assert len(shard._fall_time_data[0]["Fall Times"]) == 2
    assert len(shard._fwhm_data[0]["FWHM Values"]) == 2
    # Peak-to-peak intervals are recomputed on surviving peaks
    # (peak=50 -> peak=300 -> 250 frames).
    assert shard._peak_to_peak_data[0]["Inter-Spike Intervals"] == [250.0]
    assert shard._frpm_data[0]["N Events"] == 2

    # Caches must be invalidated so subsequent property reads see the
    # fresh views.
    assert orb._rise_time_data is None
    assert orb._fall_time_data is None
    assert orb._fwhm_data is None
    assert orb._frpm_data is None
    assert orb._peak_amplitude_data is None
    assert orb._peak_to_peak_data is None

    # Re-running refilter_events without labels_corpus recovers all events.
    orb.refilter_events(
        filter_events=False,
        min_event_amplitude=None,
        max_event_amplitude=None,
        min_event_fwhm=None,
        max_event_fwhm=None,
        labels_corpus=None,
    )
    assert len(shard._peak_amplitude_data[0]["Peak Amplitudes"]) == 4


# ---------------------------------------------------------------------------
# Tests: per-event drop ledger (audit trail)
# ---------------------------------------------------------------------------
_LEDGER_COLS = [
    "sample_id", "neuron_idx", "event_idx",
    "peak_amplitude", "fwhm_frames", "drop_reason",
]


def _ledger_row(shard, *, neuron_idx, event_idx):
    """Return the (single) ledger row matching ``(neuron_idx, event_idx)``."""
    matches = [
        r for r in shard._event_drop_log
        if r["neuron_idx"] == neuron_idx and r["event_idx"] == event_idx
    ]
    assert len(matches) == 1, (
        f"expected exactly one ledger row for (neuron={neuron_idx}, "
        f"event={event_idx}); got {matches}"
    )
    return matches[0]


def test_drop_ledger_nan_inf(cauldron):
    """A NaN amplitude lands in the ledger with drop_reason='nan_inf'."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(float("nan"), 100), _ev(0.5, 200)],
    ])
    cauldron._apply_event_filters(
        shard,
        filter_events=False,
        min_event_amplitude=None,
        max_event_amplitude=None,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    assert len(shard._event_drop_log) == 1
    row = _ledger_row(shard, neuron_idx=0, event_idx=1)
    assert row["sample_id"] == "S1"
    assert row["drop_reason"] == "nan_inf"
    assert np.isnan(row["peak_amplitude"])


def test_drop_ledger_amplitude_below(cauldron):
    """An amplitude below the lower bound is recorded as 'amplitude_below_min'."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.001, 100), _ev(0.5, 200)],
    ])
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    assert len(shard._event_drop_log) == 1
    row = _ledger_row(shard, neuron_idx=0, event_idx=1)
    assert row["drop_reason"] == "amplitude_below_min"
    assert row["peak_amplitude"] == pytest.approx(0.001)


def test_drop_ledger_amplitude_above(cauldron):
    """An amplitude above the upper bound is recorded as 'amplitude_above_max'."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(20.0, 100), _ev(0.5, 200)],
    ])
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    assert len(shard._event_drop_log) == 1
    row = _ledger_row(shard, neuron_idx=0, event_idx=1)
    assert row["drop_reason"] == "amplitude_above_max"
    assert row["peak_amplitude"] == pytest.approx(20.0)


def test_drop_ledger_fwhm_below(cauldron):
    """A FWHM below the lower bound is recorded as 'fwhm_below_min'."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50, fwhm=5), _ev(0.5, 100, fwhm=1), _ev(0.5, 200, fwhm=5)],
    ])
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=None,
        max_event_amplitude=None,
        min_event_fwhm=2,
        max_event_fwhm=None,
    )
    assert len(shard._event_drop_log) == 1
    row = _ledger_row(shard, neuron_idx=0, event_idx=1)
    assert row["drop_reason"] == "fwhm_below_min"
    assert row["fwhm_frames"] == pytest.approx(1.0)


def test_drop_ledger_fwhm_above(cauldron):
    """A FWHM above the upper bound is recorded as 'fwhm_above_max'."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50, fwhm=5), _ev(0.5, 100, fwhm=300), _ev(0.5, 200, fwhm=5)],
    ])
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=None,
        max_event_amplitude=None,
        min_event_fwhm=2,
        max_event_fwhm=200,
    )
    assert len(shard._event_drop_log) == 1
    row = _ledger_row(shard, neuron_idx=0, event_idx=1)
    assert row["drop_reason"] == "fwhm_above_max"
    assert row["fwhm_frames"] == pytest.approx(300.0)


def test_drop_ledger_human_label(cauldron, tmp_path):
    """An event labeled False in the corpus is recorded as 'human_label_false'."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.5, 100), _ev(0.5, 200)],
    ])
    corpus = _write_corpus(
        tmp_path / "corpus.csv",
        [
            {
                "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
                "event_idx": 1, "label": "False", "labeler_id": "alice",
                "timestamp": "", "notes": "",
            },
        ],
    )
    cauldron._apply_event_filters(
        shard,
        filter_events=False,
        min_event_amplitude=None,
        max_event_amplitude=None,
        min_event_fwhm=None,
        max_event_fwhm=None,
        labels_corpus=corpus,
    )
    assert len(shard._event_drop_log) == 1
    row = _ledger_row(shard, neuron_idx=0, event_idx=1)
    assert row["drop_reason"] == "human_label_false"


def test_drop_ledger_human_label_disagreement(cauldron, tmp_path):
    """A True/False conflict resolved by on_disagreement='drop' is flagged
    distinctly so users can audit how many rejections came from inter-rater
    disagreement vs. consensus."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.5, 100), _ev(0.5, 200)],
    ])
    corpus = _write_corpus(
        tmp_path / "corpus.csv",
        [
            {
                "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
                "event_idx": 1, "label": "True", "labeler_id": "alice",
                "timestamp": "", "notes": "",
            },
            {
                "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
                "event_idx": 1, "label": "False", "labeler_id": "bob",
                "timestamp": "", "notes": "",
            },
        ],
    )
    cauldron._apply_event_filters(
        shard,
        filter_events=False,
        min_event_amplitude=None,
        max_event_amplitude=None,
        min_event_fwhm=None,
        max_event_fwhm=None,
        labels_corpus=corpus,
        on_disagreement="drop",
    )
    assert len(shard._event_drop_log) == 1
    row = _ledger_row(shard, neuron_idx=0, event_idx=1)
    assert row["drop_reason"] == "human_label_disagreement_drop"


def test_drop_ledger_first_reason_wins(cauldron):
    """An event with both NaN amplitude AND below-min amplitude wins 'nan_inf'.

    Impossible in practice (a NaN can't be 'below' anything numerically),
    but tests the ordering: ``nan_inf`` is checked before the bounds
    layer, so a NaN amplitude is attributed to ``nan_inf`` regardless of
    what the bound check would have said.
    """
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(float("nan"), 100), _ev(0.5, 200)],
    ])
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    row = _ledger_row(shard, neuron_idx=0, event_idx=1)
    assert row["drop_reason"] == "nan_inf"


def test_drop_ledger_no_double_count(cauldron, tmp_path):
    """An event dropped by the amplitude bound AND labeled False has ONE
    row attributed to 'amplitude_below_min' (the earlier-checked layer)."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.001, 100), _ev(0.5, 200)],
    ])
    corpus = _write_corpus(
        tmp_path / "corpus.csv",
        [
            {
                "corpus_version": 2, "sample_id": "S1", "roi_id": 0,
                "event_idx": 1, "label": "False", "labeler_id": "alice",
                "timestamp": "", "notes": "",
            },
        ],
    )
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
        labels_corpus=corpus,
    )
    assert len(shard._event_drop_log) == 1
    row = _ledger_row(shard, neuron_idx=0, event_idx=1)
    assert row["drop_reason"] == "amplitude_below_min"


def _make_orb_with_shards(shards):
    """Build a barebones Orb with the given shards installed for testing."""
    from wizards_staff.wizards.orb import Orb

    orb = Orb.__new__(Orb)
    orb._logger = logging.getLogger("test.orb.event_drop_log")
    orb._remove_outlier = False
    orb._show_outlier = False
    orb._shards = {sh.sample_name: sh for sh in shards}
    for cache in (
        "_rise_time_data",
        "_fall_time_data",
        "_fwhm_data",
        "_frpm_data",
        "_peak_amplitude_data",
        "_max_peak_amplitude_data",
        "_peak_to_peak_data",
        "_event_drop_log",
        "_outlier_data",
        "_pwc_plots",
        "_df_mn_pwc",
        "_df_mn_pwc_intra",
        "_df_mn_pwc_inter",
    ):
        setattr(orb, cache, None)
    orb._pwc_plots = {}
    orb.metadata = pd.DataFrame({
        "Sample": [sh.sample_name for sh in shards],
        "Well": ["A1"] * len(shards),
        "Frate": [10] * len(shards),
    })
    return orb


def test_drop_ledger_orb_concat(cauldron):
    """``orb.event_drop_log`` concatenates per-shard ledgers, keyed by sample_id."""
    shard_a = _make_shard(
        sample_name="S1",
        per_neuron_events=[[_ev(0.5, 50), _ev(0.001, 100)]],
    )
    shard_b = _make_shard(
        sample_name="S2",
        per_neuron_events=[[_ev(0.5, 75), _ev(20.0, 150)]],
    )
    for sh in (shard_a, shard_b):
        cauldron._apply_event_filters(
            sh,
            filter_events=True,
            min_event_amplitude=0.05,
            max_event_amplitude=10.0,
            min_event_fwhm=None,
            max_event_fwhm=None,
        )
    orb = _make_orb_with_shards([shard_a, shard_b])

    log = orb.event_drop_log
    assert list(log.columns) == _LEDGER_COLS
    assert len(log) == 2
    assert set(log["sample_id"]) == {"S1", "S2"}
    by_sample = dict(zip(log["sample_id"], log["drop_reason"]))
    assert by_sample["S1"] == "amplitude_below_min"
    assert by_sample["S2"] == "amplitude_above_max"


def test_drop_ledger_refilter_regenerates(cauldron):
    """``refilter_events`` regenerates the ledger from scratch — never appends."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.001, 100), _ev(20.0, 200), _ev(0.5, 300)],
    ])
    orb = _make_orb_with_shards([shard])

    orb.refilter_events(
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    log_first = orb.event_drop_log
    assert len(log_first) == 2
    reasons_first = sorted(log_first["drop_reason"].tolist())
    assert reasons_first == ["amplitude_above_max", "amplitude_below_min"]

    # Loosen bounds: the previously-dropped 0.001 amplitude is now in
    # range, so the ledger must SHRINK, not grow.
    orb.refilter_events(
        filter_events=True,
        min_event_amplitude=0.0001,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    log_second = orb.event_drop_log
    assert len(log_second) == 1, (
        "refilter_events must regenerate the ledger, not append to it"
    )
    assert log_second.iloc[0]["drop_reason"] == "amplitude_above_max"


def test_drop_ledger_csv_export(cauldron, tmp_path):
    """``save_results`` writes ``event_drop_log.csv`` matching the in-memory ledger."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.001, 100), _ev(float("nan"), 200)],
    ])
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=None,
        max_event_fwhm=None,
    )
    orb = _make_orb_with_shards([shard])

    out_dir = tmp_path / "out"
    # Save just the ledger to avoid pulling in PWC / mask CSV machinery.
    orb.save_results(str(out_dir), result_names=["event_drop_log"])

    csv_path = out_dir / "event_drop_log.csv"
    assert csv_path.exists(), "event_drop_log.csv must be written"
    on_disk = pd.read_csv(csv_path)
    assert list(on_disk.columns) == _LEDGER_COLS
    in_memory = orb.event_drop_log.reset_index(drop=True)
    assert len(on_disk) == len(in_memory)
    # Compare ledger contents (sort to ignore row ordering, normalise NaN
    # representation across the round-trip).
    on_disk_sorted = on_disk.sort_values(
        ["sample_id", "neuron_idx", "event_idx"]
    ).reset_index(drop=True)
    in_memory_sorted = in_memory.sort_values(
        ["sample_id", "neuron_idx", "event_idx"]
    ).reset_index(drop=True)
    for col in ("sample_id", "neuron_idx", "event_idx", "drop_reason"):
        assert (
            on_disk_sorted[col].astype(str).tolist()
            == in_memory_sorted[col].astype(str).tolist()
        ), col


def test_drop_ledger_empty_when_nothing_dropped(cauldron):
    """Clean events: ledger is an empty DataFrame with the canonical columns."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.5, 100), _ev(0.5, 200)],
    ])
    cauldron._apply_event_filters(
        shard,
        filter_events=True,
        min_event_amplitude=0.05,
        max_event_amplitude=10.0,
        min_event_fwhm=2,
        max_event_fwhm=None,
    )
    assert shard._event_drop_log == []
    orb = _make_orb_with_shards([shard])
    log = orb.event_drop_log
    assert isinstance(log, pd.DataFrame)
    assert log.empty
    assert list(log.columns) == _LEDGER_COLS


# ---------------------------------------------------------------------------
# Tests: FWHM data invariant guard
# ---------------------------------------------------------------------------
def test_fwhm_data_invariant_missing_row_raises(cauldron):
    """Missing _raw_fwhm_data row raises RuntimeError with a useful message.

    Pipeline invariant: ``_run_all`` always populates ``_raw_fwhm_data``
    alongside ``_raw_peak_amplitude_data`` (one row per neuron, equal
    lengths). Violating this is a bug, not a recoverable degraded state
    — silently keeping all events would let NaN/Inf FWHM events leak
    into downstream metrics, so the function fails loudly instead.
    """
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.5, 100), _ev(0.5, 200)],
    ])
    # Simulate the invariant violation: drop the FWHM row entirely.
    shard._raw_fwhm_data = []
    with pytest.raises(RuntimeError) as excinfo:
        cauldron._apply_event_filters(
            shard,
            filter_events=True,
            min_event_amplitude=0.05,
            max_event_amplitude=10.0,
            min_event_fwhm=2,
            max_event_fwhm=None,
        )
    msg = str(excinfo.value)
    assert "FWHM data inconsistent" in msg
    assert "row missing" in msg
    assert "S1" in msg
    assert "report this as a bug" in msg


def test_fwhm_data_invariant_length_mismatch_raises(cauldron):
    """Length mismatch between amplitude and FWHM rows raises RuntimeError."""
    shard = _make_shard(per_neuron_events=[
        [_ev(0.5, 50), _ev(0.5, 100), _ev(0.5, 200)],
    ])
    # Simulate the invariant violation: truncate the FWHM row so it's
    # shorter than the amplitude row.
    shard._raw_fwhm_data[0]["FWHM Values"] = [5.0, 5.0]
    with pytest.raises(RuntimeError) as excinfo:
        cauldron._apply_event_filters(
            shard,
            filter_events=True,
            min_event_amplitude=0.05,
            max_event_amplitude=10.0,
            min_event_fwhm=2,
            max_event_fwhm=None,
        )
    msg = str(excinfo.value)
    assert "FWHM data inconsistent" in msg
    assert "length mismatch" in msg
    assert "3 amplitude events vs 2 FWHM values" in msg
    assert "report this as a bug" in msg
