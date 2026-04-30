"""
Run-summary reporting for ``run_all``.

Generates a human-readable markdown report summarizing a pipeline run:
parameters, per-sample neuron / event counts, outlier and event-filter
impact, and distribution summaries (describe() tables + optional
histograms) for each per-event metric.

Designed to be invoked once at the end of ``run_all`` as a
read-only consumer of an already-populated ``Orb`` (it never mutates
orb state). Prints the report to stdout and, when ``save_files=True``,
also writes ``run_report.md`` and per-metric histograms under
``<output_dir>/run_report/``.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


# Metrics tracked by the report, in the order they should appear.
# Each entry: (section title, orb attribute, column name in the DataFrame, units label).
_METRIC_SPEC: Sequence[tuple] = (
    ("Peak Amplitudes", "peak_amplitude_data", "Peak Amplitudes", "ΔF/F"),
    ("FWHM", "fwhm_data", "FWHM Values", "frames"),
    ("Max Peak df/f (per-neuron)", "max_peak_amplitude_data", "Max Peak df/f", "ΔF/F"),
    ("Peak Amplitude Std (per-neuron)", "max_peak_amplitude_data", "Peak Amplitude Std", "ΔF/F"),
    ("Rise Times", "rise_time_data", "Rise Times", "frames"),
    ("Fall Times", "fall_time_data", "Fall Times", "frames"),
    ("Inter-Spike Intervals", "peak_to_peak_data", "Inter-Spike Intervals", "frames"),
    ("Firing Rate Per Min", "frpm_data", "Firing Rate Per Min", "spikes/min"),
)

# Percentiles used for describe() tables.
_PERCENTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Coerce a Series (possibly object dtype from explode()) to numeric, drop NaNs."""
    return pd.to_numeric(series, errors="coerce").dropna()


def _describe_block(values: pd.Series, units: str) -> str:
    """Return a markdown code block with a describe() summary for ``values``."""
    if values.empty:
        return "  _(no data)_"
    desc = values.describe(percentiles=_PERCENTILES)
    # Format as aligned text.
    lines = [f"  count : {int(desc['count']):,}"]
    for label, key in (
        ("mean  ", "mean"),
        ("std   ", "std"),
        ("min   ", "min"),
        ("5%    ", "5%"),
        ("25%   ", "25%"),
        ("50%   ", "50%"),
        ("75%   ", "75%"),
        ("95%   ", "95%"),
        ("max   ", "max"),
    ):
        v = desc.get(key)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            lines.append(f"  {label}: n/a")
        else:
            lines.append(f"  {label}: {v:.4f} {units}")
    return "```\n" + "\n".join(lines) + "\n```"


def _sample_summary_table(orb: Any) -> str:
    """Build a per-sample markdown table: neurons filtered/outliers/kept, events.

    Every per-event metric in the orb describes the same surviving event
    set after ``_apply_event_filters`` (peak amp / FWHM / rise / fall /
    peak-to-peak / FRPM share a single keep-mask), so the per-sample
    counts pulled here are equal across metrics by construction. We
    cross-check that invariant defensively and surface a footnote when
    it fails — that would indicate a regression in the keep-mask
    propagation.
    """
    outlier_full = getattr(orb, "outlier_data", None)

    def _full(attr_with_outliers: str, attr_fallback: str):
        df = getattr(orb, attr_with_outliers, None)
        if df is None:
            df = getattr(orb, attr_fallback, None)
        return df

    peak_full = _full("peak_amplitude_data_with_outliers", "peak_amplitude_data")
    fwhm_full = _full("fwhm_data_with_outliers", "fwhm_data")
    rise_full = _full("rise_time_data_with_outliers", "rise_time_data")
    fall_full = _full("fall_time_data_with_outliers", "fall_time_data")

    samples = [sh.sample_name for sh in orb.shards]
    if not samples:
        return "  _(no samples processed)_"

    def _count(df, name, value_col):
        if df is None or "Sample" not in df.columns:
            return 0
        sl = df[df["Sample"] == name]
        if "is_outlier" in sl.columns:
            sl = sl[~sl["is_outlier"].fillna(False).astype(bool)]
        return _safe_numeric(sl.get(value_col, pd.Series(dtype=float))).size

    rows = []
    inconsistencies = []
    for name in samples:
        n_spatial = 0
        n_outlier = 0
        n_kept = 0
        if outlier_full is not None and "Sample" in outlier_full.columns:
            oo = outlier_full[outlier_full["Sample"] == name]
            n_spatial = len(oo)
            if "any_outlier" in oo.columns:
                n_outlier = int(oo["any_outlier"].fillna(False).astype(bool).sum())
            n_kept = n_spatial - n_outlier

        counts = {
            "peak":  _count(peak_full, name, "Peak Amplitudes"),
            "fwhm":  _count(fwhm_full, name, "FWHM Values"),
            "rise":  _count(rise_full, name, "Rise Times"),
            "fall":  _count(fall_full, name, "Fall Times"),
        }
        # The keep-mask invariant: all four counts should agree.
        unique = {v for v in counts.values() if v > 0}
        if len(unique) > 1:
            inconsistencies.append((name, counts))

        rows.append((name, n_spatial, n_outlier, n_kept, counts))

    header = (
        "| Sample | Spatially filtered | Outliers | Kept neurons | "
        "Peak events | FWHM events | Rise events | Fall events |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    body_lines = [
        f"| {name} | {n_spatial} | {n_outlier} | {n_kept} | "
        f"{c['peak']} | {c['fwhm']} | {c['rise']} | {c['fall']} |"
        for name, n_spatial, n_outlier, n_kept, c in rows
    ]
    table = header + "\n" + "\n".join(body_lines)
    if inconsistencies:
        details = "; ".join(
            f"{name}: peak={c['peak']} fwhm={c['fwhm']} rise={c['rise']} fall={c['fall']}"
            for name, c in inconsistencies
        )
        table += (
            "\n\n> WARNING: per-event metric counts disagree for some "
            f"samples — the keep-mask should propagate uniformly. "
            f"Details: {details}."
        )
    return table


def _params_block(params: Dict[str, Any]) -> str:
    """Format the key ``run_all`` parameters as a markdown bullet list."""
    tracked_keys = [
        "frate", "zscore_threshold", "percentage_threshold",
        "p_th", "size_threshold",
        "min_clusters", "max_clusters",
        "group_name", "threads",
        "outlier_threshold", "outlier_methods",
        "remove_outlier", "show_outlier",
        "filter_events",
        "min_event_amplitude", "max_event_amplitude",
        "min_event_fwhm", "max_event_fwhm",
    ]
    lines = [
        f"- `{k}` = `{params[k]!r}`"
        for k in tracked_keys
        if k in params
    ]
    return "\n".join(lines) if lines else "  _(no parameters recorded)_"


def _outlier_impact_block(orb: Any) -> str:
    od = getattr(orb, "outlier_data", None)
    if od is None or od.empty or "any_outlier" not in od.columns:
        return "  _(outlier detection produced no results)_"
    flagged = int(od["any_outlier"].fillna(False).astype(bool).sum())
    total = len(od)
    pct = (flagged / total * 100.0) if total else 0.0
    return (
        f"- Neurons flagged as outliers: **{flagged} / {total}** "
        f"({pct:.1f}% of spatially-filtered neurons)"
    )


# Drop-reason categories the ledger uses, in the same first-reason-wins
# order applied by ``_apply_event_filters``. Pinning the row order in
# the report makes runs comparable across datasets and surfaces missing
# categories as zero rows rather than absent rows.
_DROP_REASON_ORDER: Sequence[str] = (
    "nan_inf",
    "amplitude_below_min",
    "amplitude_above_max",
    "fwhm_below_min",
    "fwhm_above_max",
    "human_label_false",
    "human_label_disagreement_drop",
)


def _drop_reasons_block(orb: Any, output_dir: str, save_files: bool) -> str:
    """Build the 'Drop reasons' subsection: per-reason counts + ledger pointer."""
    log = getattr(orb, "event_drop_log", None)
    if log is None or len(log) == 0 or "drop_reason" not in getattr(log, "columns", []):
        body = "  _(no events were dropped — the ledger is empty)_"
    else:
        counts = log["drop_reason"].astype(str).value_counts()
        total = int(counts.sum())
        header = (
            "| Drop reason                    | Count | % of dropped |\n"
            "|--------------------------------|------:|-------------:|"
        )
        seen = set()
        body_lines = []
        # Pinned reasons first so the table layout is stable.
        for reason in _DROP_REASON_ORDER:
            n = int(counts.get(reason, 0))
            if n == 0:
                continue
            pct = (n / total * 100.0) if total else 0.0
            body_lines.append(
                f"| {reason:<30} | {n:>5} | {pct:>11.1f}% |"
            )
            seen.add(reason)
        # Surface any unexpected reasons that aren't in the canonical
        # ordering — defensive against future reason additions / typos.
        for reason in counts.index:
            if reason in seen:
                continue
            n = int(counts[reason])
            pct = (n / total * 100.0) if total else 0.0
            body_lines.append(
                f"| {str(reason):<30} | {n:>5} | {pct:>11.1f}% |"
            )
        body = header + "\n" + "\n".join(body_lines)
    if save_files:
        ledger_note = (
            f"\n\n- Per-event ledger CSV: "
            f"`{os.path.join(output_dir, 'event_drop_log.csv')}` "
            f"(one row per dropped event with `sample_id`, `neuron_idx`, "
            f"`event_idx`, `peak_amplitude`, `fwhm_frames`, `drop_reason`)."
        )
    else:
        ledger_note = (
            "\n\n- Per-event ledger available in-memory via "
            "`orb.event_drop_log` (run with `save_files=True` to also "
            "write `event_drop_log.csv`)."
        )
    return body + ledger_note


def _distribution_section(orb: Any) -> str:
    blocks = []
    for title, attr, col, units in _METRIC_SPEC:
        df = getattr(orb, attr, None)
        if df is None:
            blocks.append(f"### {title}\n  _(no data — `orb.{attr}` is None)_")
            continue
        if col not in df.columns:
            blocks.append(
                f"### {title}\n  _(no data — column `{col}` not present in `orb.{attr}`)_"
            )
            continue
        values = _safe_numeric(df[col])
        blocks.append(f"### {title}\n{_describe_block(values, units)}")
    return "\n\n".join(blocks)


def _save_histograms(orb: Any, out_dir: str) -> Sequence[str]:
    """Save one histogram PNG per available metric. Returns list of written paths."""
    import matplotlib
    # Use a non-interactive backend here to avoid accidentally popping windows
    # during save_files=True batch runs.
    current_backend = matplotlib.get_backend()
    matplotlib.use("Agg", force=True)
    try:
        import matplotlib.pyplot as plt
    finally:
        # Don't permanently swap the user's backend — restore after this block.
        # (We still use plt locally; the restoration happens after savefig.)
        pass

    os.makedirs(out_dir, exist_ok=True)
    written = []
    try:
        for title, attr, col, units in _METRIC_SPEC:
            df = getattr(orb, attr, None)
            if df is None or col not in df.columns:
                continue
            values = _safe_numeric(df[col])
            if values.empty:
                continue

            fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
            ax.hist(values, bins=60)
            ax.set_title(f"{title} (n={len(values):,})")
            ax.set_xlabel(units)
            ax.set_ylabel("Count")
            safe_name = title.lower().replace(" ", "_").replace("/", "_")
            path = os.path.join(out_dir, f"{safe_name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
    finally:
        try:
            matplotlib.use(current_backend, force=True)
        except Exception:
            pass
    return written


def generate_run_report(
    orb: Any,
    params: Dict[str, Any],
    output_dir: str,
    save_files: bool = False,
    report_subdir: str = "run_report",
) -> Optional[str]:
    """
    Build a markdown run-summary report, print it to stdout, and optionally
    save it plus per-metric histograms to disk.

    Parameters
    ----------
    orb
        Orb whose ``run_all`` has completed. Must expose the standard metric
        properties (``peak_amplitude_data``, ``fwhm_data``, ``outlier_data``,
        ``rise_time_data``, ``fall_time_data``, ``peak_to_peak_data``,
        ``frpm_data``, ``max_peak_amplitude_data``).
    params
        Dict of the parameters used for the run. Printed verbatim.
    output_dir
        Base directory to write the report into when ``save_files=True``.
    save_files
        When True, also writes ``<output_dir>/run_report.md`` and histograms
        under ``<output_dir>/<report_subdir>/``.
    report_subdir
        Subdirectory (under ``output_dir``) for saved histograms.

    Returns
    -------
    str or None
        The markdown report string (also printed to stdout). Returns ``None``
        if the orb appears to have no processed shards.
    """
    samples = [sh.sample_name for sh in orb.shards]
    if not samples:
        print("Run report skipped: no shards processed.", flush=True)
        return None

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"# Wizards-Staff run report")
    lines.append("")
    lines.append(f"_Generated: {now}_  ")
    lines.append(f"_Output dir: `{output_dir}`_  ")
    lines.append(f"_Samples processed: **{len(samples)}**_")
    lines.append("")
    lines.append("## Parameters")
    lines.append(_params_block(params))
    lines.append("")
    lines.append("## Per-sample summary")
    lines.append("")
    lines.append(_sample_summary_table(orb))
    lines.append("")
    lines.append("## Outlier impact")
    lines.append(_outlier_impact_block(orb))
    lines.append("")
    if params.get("filter_events"):
        lines.append("## Per-event filter impact")
        lines.append(
            f"- `filter_events=True`; bounds: "
            f"amplitude in "
            f"[{params.get('min_event_amplitude')}, "
            f"{params.get('max_event_amplitude')}] ΔF/F, "
            f"FWHM in "
            f"[{params.get('min_event_fwhm')}, "
            f"{params.get('max_event_fwhm')}] frames."
        )
        lines.append(
            "- A single keep-mask is built per (sample, neuron) from the "
            "amplitude and FWHM bounds (plus the unconditional NaN/Inf "
            "scrub) and applied uniformly to every per-event metric: "
            "`peak_amplitude_data`, `fwhm_data`, `max_peak_amplitude_data`, "
            "`well_peak_amplitude_data`, `rise_time_data`, "
            "`fall_time_data`, `peak_to_peak_data`, `frpm_data`. The event "
            "counts reported above are equal across metrics by construction."
        )
    else:
        lines.append("## Per-event filter impact")
        lines.append(
            "- `filter_events=False` — amplitude/FWHM bounds disabled; "
            "the unconditional NaN/Inf scrub still fires and propagates "
            "to every per-event metric."
        )
    lines.append("")
    lines.append("### Drop reasons")
    lines.append("")
    lines.append(_drop_reasons_block(orb, output_dir, save_files))
    lines.append("")
    lines.append("## Distribution summaries")
    lines.append("")
    lines.append(
        "Distributions below reflect whatever is currently exposed by each "
        "`orb.<metric>_data` accessor, so they already account for "
        "`remove_outlier` / `filter_events` modes where applicable."
    )
    lines.append("")
    lines.append(_distribution_section(orb))
    lines.append("")

    report = "\n".join(lines)

    # Always print to stdout so interactive users see it immediately.
    print("\n" + ("═" * 78), flush=True)
    print("RUN REPORT", flush=True)
    print(("═" * 78) + "\n", flush=True)
    print(report, flush=True)

    if save_files:
        try:
            os.makedirs(output_dir, exist_ok=True)
            md_path = os.path.join(output_dir, "run_report.md")
            with open(md_path, "w", encoding="utf-8") as fh:
                fh.write(report)
                fh.write("\n")
            print(f"\n[report] markdown saved to: {md_path}", flush=True)

            hist_dir = os.path.join(output_dir, report_subdir)
            written = _save_histograms(orb, hist_dir)
            if written:
                print(
                    f"[report] {len(written)} histogram(s) saved to: {hist_dir}",
                    flush=True,
                )
        except Exception as e:
            print(
                f"[report] WARNING: failed to save report artifacts: {e}",
                file=sys.stderr, flush=True,
            )

    return report
