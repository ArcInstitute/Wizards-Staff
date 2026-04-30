Wizards Staff
=============

<img src="./img/wizards_staff.png" alt="drawing" width="400"/>


Calcium imaging analysis toolkit for processing outputs from calcium imaging pipelines (like [Lizard-Wizard](https://github.com/ArcInstitute/Lizard-Wizard)) and extracting advanced metrics, correlations, and visualizations to characterize neural activity.

## Features

- **Comprehensive Metrics Analysis**: Extract rise time, FWHM (Full Width at Half Maximum), and Firing Rate Per Minute (FRPM) metrics from calcium imaging data
- **Advanced Correlation Analysis**: Perform pairwise correlation (PWC) analysis within and between neuron populations
- **Spatial Activity Mapping**: Generate spatial activity maps to visualize active neurons and their clustering
- **K-means Clustering**: Apply clustering algorithms to identify synchronously active neurons
- **Versatile Visualization Tools**: Create publication-quality visualizations for activity traces, spatial components, and clustering results
- **Modular Architecture**: Utilize the `Orb` and `Shard` classes for organized, scalable data processing

## Table of Contents

- [Installation](#installation)
- [Brief Functions Overview](#wizards-staff-python-package---brief-function-overview)

## Installation

### Clone the Repo

To download the latest version of this package, clone it from the repository:

```bash
git clone git@github.com:ArcInstitute/Wizards-Staff.git
cd Wizards-Staff
```

### Create a Virtual Environment (Optional but Recommended)

Its recommended to create a virtual environment for Wizards-Staff as this ensures that your project dependencies are isolated. You can use mamba to create one:

```bash
mamba create -n wizards_staff python=3.11 -y
mamba activate wizards_staff
```

### Install the Package

To install the wizards_staff package within your virtual environment, from the command line run:

```bash
pip install .
```

---

## Quick Start

```python
from wizards_staff import Orb

# Initialize an Orb with your results folder and metadata file
orb = Orb(
    results_folder="/path/to/calcium_imaging_results", 
    metadata_file_path="/path/to/metadata.csv"
)

# Run comprehensive analysis (all metrics)
orb.run_all(
    group_name=None,  # Group samples by this metadata column
    frate=30,           # Frame rate of recording
    show_plots=True,    # Display plots during analysis
    save_files=True     # Save results to disk
)

# Access results as pandas DataFrames
rise_time_df = orb.rise_time_data
fwhm_df = orb.fwhm_data
frpm_df = orb.frpm_data
```

## Data Requirements

### Input Data

Wizards Staff is designed to process outputs from calcium imaging pipelines such as [Lizard-Wizard](https://github.com/ArcInstitute/Lizard-Wizard). The main input data includes:

- Delta F/F0 (dF/F0) matrices
- Spatial footprints of neurons (cnm_A)
- Indices of accepted components (cnm_idx)
- Minimum projection images
- Masks (optional, for shape metrics)

### Metadata Format

A metadata CSV file with the following required columns:
- `Sample`: Unique identifier for each sample, matching filenames
- `Well`: Well identifier (or other grouping variable)
- `Frate`: Frame rate of the recording in frames per second

## Examples

### Calcium Indicator (GCaMP6f, GCaMP6s, jGCaMP8m, jRGECO1a, …)

The waveform outlier detector (`detect_waveform_outliers`) correlates
each transient against a synthetic template whose kinetics depend on
the calcium indicator used in the experiment. The legacy default
matches GCaMP6f (50 ms rise, 400 ms decay, 0.10 ΔF/F peak threshold).
**If you used a different indicator, set the `indicator` parameter** —
otherwise real events get silently flagged as shape outliers because
they don't match the template, and the absolute peak threshold may be
inappropriate for indicators with smaller ΔF/F excursions (e.g. the
red indicators).

```python
orb.run_all(
    group_name="Well",
    indicator="GCaMP6s",   # also: GCaMP6m, GCaMP7f, jGCaMP8f / 8m / 8s,
                           # jRGECO1a, jRCaMP1a, GCaMP3
)
```

Or override individual kinetics on top of a preset:

```python
orb.run_all(
    group_name="Well",
    indicator="GCaMP6s",       # preset rise / peak height
    template_decay_ms=2000.0,  # but with a longer decay than the preset
)
```

The presets in `wizards_staff.stats.outliers.INDICATOR_PRESETS` are
*starting points* drawn from published kinetics under typical
acquisition conditions. Verify against your own measurements when
accuracy matters; if your data argues for a different rise/decay,
override `template_rise_ms` / `template_decay_ms` / `peak_height`
directly. From the CLI use `--indicator GCaMP6s` (and optionally
`--template-rise-ms`, `--template-decay-ms`, `--template-total-ms`,
`--peak-height`).

### Recommended Workflow: Run, Label, Refilter

The standard analysis cycle is three steps: run the automatic pipeline,
hand-review the detected events, then refilter so the labels feed back
into every per-event metric.

```python
from pathlib import Path
from wizards_staff import Orb
from wizards_staff.labeling import EventLabeler

orb = Orb(results_folder="...", metadata_file_path="...")

# 1. Initial run with automatic QC.
orb.run_all(group_name="Well", indicator="GCaMP6s", filter_events=True)

# 2. Open the labeling widget on a shard. The corpus saves automatically.
shard = next(orb.shatter())
corpus = Path("event_labels_corpus.csv")
labeler = EventLabeler(
    shard,
    corpus_path=str(corpus),
    labeler_id="your_initials",
    context={"indicator": "GCaMP6s", "experiment_id": "expt-001"},
)
labeler.display()    # review events: t / f / u keys, or click buttons

# 3. Fold the labels into the analysis (cheap — no re-running of run_all).
orb.refilter_events(
    labels_corpus=str(corpus),
    on_disagreement="drop",   # also: "keep", "majority"
    filter_events=True,       # keep amplitude/FWHM bounds active too
)
```

#### Three-layer event filter

Every per-event metric in Wizards-Staff (`peak_amplitude_data`,
`fwhm_data`, `frpm_data`, `rise_time_data`, `fall_time_data`,
`peak_to_peak_data`) describes the same surviving event set. That set
is the intersection of three filter layers, applied in order:

| Layer | Source | Always on? |
|-------|--------|------------|
| 1. NaN/Inf scrub           | deconvolution artefacts                         | yes |
| 2. Amplitude / FWHM bounds | `min_event_*` / `max_event_*` parameters        | when `filter_events=True` |
| 3. Human labels            | `labels_corpus` CSV from `EventLabeler`         | when `labels_corpus=...` is passed |

```
raw events → NaN/Inf scrub → amplitude/FWHM bounds → human labels
                                                     ↓
                                            surviving events used
                                            in every per-event metric
```

Labels can ONLY drop events. A label of `"True"` cannot recover an
event that layers 1 or 2 already rejected — labels are a strictly
additional rejection layer, not an automatic-rejection override. This
makes the labeling step monotonically conservative: it can only narrow
the surviving set, never widen it.

`"Unsure"` labels are stored in the corpus (useful for downstream
calibration) but are treated as not-labeled and never cause a drop.
When multiple labelers disagree on the same event, `on_disagreement`
chooses the resolution policy:

- `"drop"` (default) — precautionary, drop on any conflict.
- `"keep"` — retain when at least one labeler said True.
- `"majority"` — majority of {True, False} votes; ties drop.

The corpus CSV accumulates across sessions and labelers, so a missing
file path is logged as a warning and ignored rather than crashing.

### Pairwise Correlation Analysis

```python
# Run pairwise correlation analysis
orb.run_pwc(
    group_name="Well",  # Group by this metadata column
    poly=True,          # Apply polynomial fitting
    show_plots=True     # Display correlation plots
)

# Access pairwise correlation results
pwc_df = orb.df_mn_pwc  # Overall pairwise correlations
intra_df = orb.df_mn_pwc_intra  # Intra-group correlations
inter_df = orb.df_mn_pwc_inter  # Inter-group correlations
```

## Documentation

For detailed usage instructions and examples, please refer to:

- [Jupyter Notebook Tutorial](notebooks/tutorial-notebook.ipynb)
- [API Reference](docs/api.md)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This software was developed at the [Arc Institute](https://arcinstitute.org/)