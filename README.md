# Wizards-Staff

Calcium imaging analysis pipeline for processing Lizard-Wizard outputs and performing metrics analysis, clustering, and more. This package was developed for the Arc Institute.

## Features

- Categorize result files by prefixes and extensions.
- Spatially filter and process Calcium Imaging datasets.
- Compute rise time, FWHM, Firing Rate, and Mask Shape metrics.
- Perform pairwise correlation analysis.
- Generate K-means clustering plots and silhouette scores for synchronicity calculations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Functions Overview](#functions-overview)

# Installation

### Clone the Repo

To download the latest version of this package, clone it from the repository:

```bash
git clone git@github.com:ArcInstitute/Wizards-Staff.git
cd Wizards-Staff
```

### Create a Virtual Environmetn (Optional but Recommended)

Its recommended to create a virtual environment for Wizards-Staff as this ensures that your project dependencies are isolated. You can use conda to create one:

```console
conda create -n wizards_staff python=3.11 -y
conda activate wizards_staff
```

### Install the Package

To install the wizards_staff package within your virtual environment, from the command line run:

```console
pip install .
```

## Usage

### Example 1: Running the Full Analysis Pipeline
You can run the entire analysis pipeline on a results folder by calling the `run_all` function:


```python
import wizards_staff

results_folder = './path/to/results'
metadata_path = './path/to/metadata.csv'
frate = 30  # Written in Frames Per Second

# Run the full analysis pipeline:
rise_time_df, fwhm_df, frpm_df, mask_metrics_df, silhouette_scores_df = wizards_staff.run_all(results_folder, metadata_path, frate=75, size_threshold=20000, show_plots = True, save_files = False)
```

## Functions Overview

A more detailed breakdown of the individual functions can be found on [this page](functions_overview.md)

`categorize_files(results_folder)`
Categorizes result files based on their prefixes and extensions.

`load_and_filter_files(categorized_files, p_th=75, size_threshold=20000)`
Loads ΔF/F₀ data and applies spatial filtering based on user-defined thresholds.

`run_all(results_folder, metadata_path, ...)`
Runs the entire analysis pipeline and computes metrics such as rise time, FWHM, FRPM, and mask metrics.

`run_pwc(group_name, metadata_path, results_folder, ...)`
Performs pairwise correlation calculations for neuron groups and generates corresponding plots.

`spatial_filter_and_plot(cn_filter, p_th, size_threshold, ...)`
Applies spatial filtering to components and generates montages of the results.