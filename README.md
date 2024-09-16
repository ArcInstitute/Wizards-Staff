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
- [Brief Functions Overview](#wizards-staff-python-package---brief-function-overview)

## Installation

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

---

## Wizards Staff Python Package - Brief Function Overview

A more detailed breakdown of the individual functions can be found on [this page](functions_overview.md).

### Wizards Cauldron Functions

`run_all(results_folder, metadata_path, ...)`  
Processes the results folder, computes metrics such as rise time, FWHM, FRPM, and mask metrics, and optionally performs pairwise correlation analysis.

---

### Wizards Spellbook Functions

`convert_f_to_cs(fluorescence_data, p=2, noise_range=[0.25, 0.5])`  
Converts fluorescence data to calcium and spike signals using deconvolution.

`calc_rise_tm(calcium_signals, spike_zscores, zscore_threshold=3)`  
Calculates the rise time of calcium signals based on detected spikes.

`calc_fwhm_spikes(calcium_signals, spike_zscores, zscore_threshold=3, percentage_threshold=0.2)`  
Calculates the full width at half maximum (FWHM) of spikes in calcium signals.

`calc_frpm(zscored_spike_events, neuron_ids, fps, zscore_threshold=5)`  
Calculates the firing rate per minute (FRPM) for z-scored spike events.

`calc_mask_shape_metrics(mask_path)`  
Loads a binary mask image and calculates roundness, diameter, and area of the masked object.

---

### Plotting Functions

`plot_spatial_activity_map(im_min, cnm_A, cnm_idx, raw_filename, ...)`  
Plots the activity of neurons by overlaying their spatial footprints on a single image.

`plot_kmeans_heatmap(dff_data, filtered_idx, raw_filename, ...)`  
Generates a K-means clustering heatmap and outputs clustering metrics.

`plot_cluster_activity(dff_data, filtered_idx, raw_filename, ...)`  
Plots the average and detailed activity of clusters in ΔF/F₀ data.

`overlay_images(im_avg, binary_overlay, overlay_color=[255, 255, 0])`  
Creates an overlay image by combining a grayscale background image with a binary overlay.

`plot_montage(images, im_avg, grid_shape, ...)`  
Creates a montage from a list of binary images overlaying them on a grayscale background.

`plot_dff_activity(dff_dat, cnm_idx, frate, raw_filename, ...)`  
Plots ΔF/F₀ activity data for neurons within a specified time range.

---

### Pairwise Correlation (PWC) Functions

`run_pwc(group_name, metadata_path, results_folder, ...)`  
Performs pairwise correlation calculations for neuron groups and generates plots.

`calc_pwc_mn(d_k_in_groups, d_dff, d_nspIDs, dff_cut=0.1, norm_corr=False)`  
Calculates pairwise correlation means for groups based on dF/F matrices and neuron IDs.

`extract_intra_inter_nsp_neurons(conn_matrix, nsp_ids)`  
Extracts intra- and inter-subpopulation connections from a neuron connectivity matrix.

`gen_mn_std_means(mean_pwc_dict)`  
Calculates the mean and standard deviation of the means for pairwise correlations.

`gen_polynomial_fit(data_dict, degree=4)`  
Generates a polynomial fit for input data representing pairwise correlations.

`filter_group_keys(d_k_in_groups, d_dff, d_nspIDs)`  
Filters group keys to ensure valid dF/F data and neuron IDs are retained for analysis.

`plot_pwc_means(d_mn_pwc, title, fname, output_dir, ...)`  
Generates plots of mean pairwise correlations with error bars and optionally saves the plots.

---

### Metadata Functions

`load_and_process_metadata(metadata_path)`  
Loads and preprocesses the metadata CSV file by cleaning the filenames.

`append_metadata_to_dfs(metadata_path, **dataframes)`  
Appends metadata from a CSV file to multiple DataFrames based on filename matches.

---

### Wizards Familiars Functions

`categorize_files(results_folder)`  
Categorizes result files based on their prefixes and extensions.

`load_and_filter_files(categorized_files, p_th=75, size_threshold=20000)`  
Loads ΔF/F₀ data and applies spatial filtering based on user-defined thresholds.

`load_required_files(categorized_files, raw_filename)`  
Loads the necessary files for a given raw filename from the categorized files.

`spatial_filtering(cn_filter, p_th, size_threshold, ...)`  
Applies spatial filtering to components and generates montages of the results.

---
