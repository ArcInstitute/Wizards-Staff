# Wizards Staff Python Package - Function Overview

This document provides an overview of the functions in the `wizards_staff` Python package, including function signatures, arguments, and descriptions of their usage.

---

## Table of Contents

- [Wizards Cauldron Functions](#wizards_cauldronpy---functions-overview)
- [Wizards Spellbook Functions](#wizards_spellbookpy---functions-overview)
- [Plotting Functions](#plottingpy---functions-overview)
- [Pairwise Correlation (PWC) Functions](#pwcpy---functions-overview)
- [Metadata Functions](#metadatapy---function-overview)
- [Wizards Familiars Functions](#wizards-familiarspy---functions-overview)

---

## wizards_cauldron.py - Functions Overview

### `run_all`

```python
run_all(results_folder, metadata_path, frate, zscore_threshold = 3, percentage_threshold = 0.2, p_th = 75, min_clusters=2, 
    max_clusters=10, random_seed = 1111111, group_name = None, poly = False, size_threshold = 20000, show_plots=True, 
    save_files=True, output_dir='./wizard_staff_outputs')
```

This is the main wrapper function that processes an entire results folder. The `run_all` function in this Python file processes imaging results by extracting and calculating metrics related to neuron activity, including rise time, full width at half maximum (FWHM), firing rates, and silhouette scores for K-means clustering. The function organizes the results into DataFrames, plots relevant data visualizations, and saves both the metrics and visualizations. Additionally, it performs pairwise correlation (PWC) analysis if a group name is provided, and optionally saves all outputs to specified directories.

**Arguments:**

- `results_folder (str)`: Path to the results folder.
- `metadata_path (str)`: Path to the metadata CSV file.
- `frate (int)`: Frames per second of the imaging session.
- `zscore_threshold (int)`: Z-score threshold for spike detection (default: 3).
- `percentage_threshold (float)`: Percentage threshold for FWHM calculation (default: 0.2).
- `p_th (float)`: Percentile threshold for image processing (default: 75).
- `min_clusters (int)`: Minimum number of clusters for K-means (default: 2).
- `max_clusters (int)`: Maximum number of clusters for K-means (default: 10).
- `random_seed (int)`: Seed for random number generation (default: 1111111).
- `group_name (str)`: Name of the group to which the data belongs. Required for PWC analysis.
- `poly (bool)`: Flag to control polynomial fitting during PWC analysis (default: `False`).
- `size_threshold (int)`: Size threshold for filtering noise events (default: 20000).
- `show_plots (bool)`: If `True`, displays plots (default: `True`).
- `save_files (bool)`: If `True`, saves files to disk (default: `True`).
- `output_dir (str)`: Directory where output files will be saved (default: `'./wizard_staff_outputs'`).

**Returns:**

- `rise_time_df (pd.DataFrame)`: DataFrame containing rise time metrics.
- `fwhm_df (pd.DataFrame)`: DataFrame containing FWHM metrics.
- `frpm_df (pd.DataFrame)`: DataFrame containing FRPM metrics.
- `mask_metrics_df (pd.DataFrame)`: DataFrame containing mask metrics.
- `silhouette_scores_df (pd.DataFrame)`: DataFrame containing silhouette scores for K-means clustering.

If `group_name` is provided, the following are also returned:

- `df_mn_pwc (pd.DataFrame)`: DataFrame containing overall pairwise correlation metrics.
- `df_mn_pwc_inter (pd.DataFrame)`: DataFrame containing inter-group pairwise correlation metrics.
- `df_mn_pwc_intra (pd.DataFrame)`: DataFrame containing intra-group pairwise correlation metrics.

## wizards_spellbook.py - Functions Overview

The `wizards_spellbook.py` file contains various metric calculation functions for analyzing neuron activity data. The key functions include converting fluorescence signals to calcium and spike signals, calculating the rise time and full width at half maximum (FWHM) of spikes in calcium signals, determining the firing rate per minute (FRPM), and calculating shape metrics (such as roundness, diameter, and area) for spheroid/organoid masks.

### `convert_f_to_cs`

Converts fluorescence data to calcium and spike signals using deconvolution.

**Arguments:**

- `fluorescence_data (ndarray)`: Fluorescence data matrix with neurons as rows and time points as columns.
- `p (int)`: Order of the autoregressive process (default: 2).
- `noise_range (list)`: Range for estimating noise (default: `[0.25, 0.5]`).

**Returns:**

- `calcium_signal (ndarray)`: Calcium signal matrix.
- `spike_signal (ndarray)`: Spike signal matrix.

---

### `calc_rise_tm`

Calculates the rise time of calcium signals based on spike detection.

**Arguments:**

- `calcium_signals (ndarray)`: Calcium signal matrix with neurons as rows and time points as columns.
- `spike_zscores (ndarray)`: Z-scored spike signal matrix.
- `zscore_threshold (float)`: Z-score threshold for spike detection (default: 3).

**Returns:**

- `rise_times (dict)`: Dictionary of rise times for each neuron.
- `rise_positions (dict)`: Dictionary of positions corresponding to the rise times for each neuron.

---

### `calc_fwhm_spikes`

Calculates the full width at half maximum (FWHM) of spikes in calcium signals.

**Arguments:**

- `calcium_signals (ndarray)`: Calcium signal matrix with neurons as rows and time points as columns.
- `spike_zscores (ndarray)`: Z-scored spike signal matrix.
- `zscore_threshold (float)`: Z-score threshold for spike detection (default: 3).
- `percentage_threshold (float)`: Percentage threshold for determining half maximum (default: 0.2).

**Returns:**

- `fwhm_backward_positions (dict)`: Dictionary of backward positions of FWHM for each neuron.
- `fwhm_forward_positions (dict)`: Dictionary of forward positions of FWHM for each neuron.
- `fwhm_values (dict)`: Dictionary of FWHM values for each neuron.
- `spike_counts (dict)`: Dictionary of the number of spikes within the FWHM for each neuron.

---

### `calc_frpm`

Calculates the firing rate per minute (FRPM) for z-scored spike event data.

**Arguments:**

- `zscored_spike_events (ndarray)`: Z-scored spike events with neurons as rows and time points as columns.
- `neuron_ids (ndarray)`: Array containing neuron IDs.
- `fps (int)`: Frames per second of the recording.
- `zscore_threshold (int)`: Z-score threshold for detecting spikes (default: 5).

**Returns:**

- `frpm (float)`: Average firing rate per minute for the dataset.
- `spike_dict (dict)`: Dictionary containing firing rates for each neuron.

---

### `calc_mask_shape_metrics`

Loads a binary mask image and calculates roundness, diameter, and area of the masked object.

**Arguments:**

- `mask_path (str)`: Path to the mask image.

**Returns:**

- `roundness (float)`: Roundness of the masked object.
- `diameter (float)`: Diameter of the masked object.
- `area (float)`: Area of the masked object.

---

## plotting.py - Functions Overview

.py file contains main plotting functions used for creating Wizard Staff plots

### `plot_spatial_activity_map`

**Function**: Plots the activity of neurons by overlaying the spatial footprints on a single image.

```python
plot_spatial_activity_map(im_min, cnm_A, cnm_idx, raw_filename, p_th=75,
 min_clusters=2, max_clusters=10, random_seed=1111111, show_plots=True, 
 save_files=False,clustering=False, dff_data=None, output_dir='./wizard_staff_outputs')
```

**Arguments:**

- `im_min (ndarray)`: Minimum intensity image for overlay.
- `cnm_A (ndarray)`: Spatial footprint matrix of neurons.
- `cnm_idx (ndarray)`: Indices of accepted components.
- `raw_filename (str)`: The raw filename of the image.
- `p_th (float)`: Percentile threshold for image processing (default: 75).
- `min_clusters (int)`: Minimum number of clusters for K-means (default: 2).
- `max_clusters (int)`: Maximum number of clusters for K-means (default: 10).
- `random_seed (int)`: Seed for random number generation (default: 1111111).
- `show_plots (bool)`: If `True`, displays plots (default: `True`).
- `save_files (bool)`: If `True`, saves the overlay image (default: `False`).
- `clustering (bool)`: If `True`, performs K-means clustering.
- `dff_data (ndarray)`: The ΔF/F₀ data array (required if `clustering=True`).
- `output_dir (str)`: Directory for saving output files (default: `'./wizard_staff_outputs'`).

**Returns:**

- `overlay_image (ndarray)`: The combined overlay image.

---

### `plot_kmeans_heatmap`

Generates a K-means clustering heatmap and outputs clustering metrics to a file.

```python
plot_kmeans_heatmap(dff_data, filtered_idx, raw_filename,  output_dir='./wizard_staff_outputs', min_clusters=2,
    max_clusters=10, random_seed=1111111, show_plots=True, save_files = True)
```

**Arguments:**

- `dff_data (ndarray)`: The ΔF/F₀ data array.
- `filtered_idx (ndarray)`: Indices of the filtered data.
- `raw_filename (str)`: The raw filename of the data.
- `min_clusters (int)`: Minimum number of clusters for K-means (default: 2).
- `max_clusters (int)`: Maximum number of clusters for K-means (default: 10).
- `random_seed (int)`: Seed for random number generation (default: 1111111).
- `show_plots (bool)`: If `True`, displays plots (default: `True`).
- `save_files (bool)`: If `True`, saves plots and clustering info (default: `True`).
- `output_dir (str)`: Directory for saving output files (default: `'./wizard_staff_outputs'`).

**Returns:**

- `best_silhouette_score (float)`: The best silhouette score.
- `best_num_clusters (int)`: The optimal number of clusters.

---

### `plot_cluster_activity`

```python
plot_cluster_activity(dff_data, filtered_idx, raw_filename, min_clusters=2, max_clusters=10, random_seed=1111111, 
    norm=False, show_plots=True, save_files = True, output_dir='./wizard_staff_outputs')
```

Plots the average activity of each cluster and detailed activity of a specified cluster.

**Arguments:**

- `dff_data (ndarray)`: The ΔF/F₀ data array.
- `filtered_idx (ndarray)`: Indices of the filtered data.
- `raw_filename (str)`: The raw filename of the data.
- `min_clusters (int)`: Minimum number of clusters for K-means (default: 2).
- `max_clusters (int)`: Maximum number of clusters for K-means (default: 10).
- `random_seed (int)`: Seed for random number generation (default: 1111111).
- `norm (bool)`: If `True`, normalizes data.
- `show_plots (bool)`: If `True`, displays plots (default: `True`).
- `save_files (bool)`: If `True`, saves plots (default: `True`).
- `output_dir (str)`: Directory for saving output files (default: `'./wizard_staff_outputs'`).

---

### `overlay_images`

```python
 overlay_images(im_avg, binary_overlay, overlay_color=[255, 255, 0])
```

Creates an overlay image by combining a grayscale background image with a binary overlay.

**Arguments:**

- `im_avg (ndarray)`: Grayscale average image.
- `binary_overlay (ndarray)`: Binary image for overlay.
- `overlay_color (list)`: RGB color for the overlay (default: `[255, 255, 0]`).

**Returns:**

- `overlay_image (ndarray)`: The combined overlay image.

---

### `plot_montage`

```python
plot_montage(images, im_avg, grid_shape, overlay_color=[255, 255, 0], rescale_intensity=False)
```

Creates a montage from a list of images, overlaying a binary image on a grayscale background.

**Arguments:**

- `images (list)`: List of binary images to be arranged in a montage.
- `im_avg (ndarray)`: Grayscale background image.
- `grid_shape (tuple)`: Shape of the montage grid (rows, columns).
- `overlay_color (list)`: RGB color for the binary overlay (default: `[255, 255, 0]`).
- `rescale_intensity (bool)`: If `True`, rescales intensity values (default: `False`).

**Returns:**

- `montage (ndarray)`: The montage image.

---

### plot_dff_activity

```python
plot_dff_activity(dff_dat, cnm_idx, frate, raw_filename, sz_per_neuron = 0.5, max_z=0.45, 
    begin_tp = 0, end_tp = -1, n_start = 0, n_stop = -1, dff_bar = 1, lw=.5, show_plots=True, 
    save_files = True, output_dir='./wizard_staff_outputs'):
```

Plots the activity data of neurons within a specified time range.

**Arguments***

- `dff_dat (ndarray)`: Activity data matrix with neurons as rows and time points as columns.
- `cnm_idx (array)`: Array of neuron IDs corresponding to the rows of dff_dat.
- `frate (int)`: Frames per second of the data.
- `raw_filename (str)`: The filename of the data. Needed for saving the plot.
- `sz_per_neuron (float)`: Size of each neuron in the plot.
- `max_z (float)`: Maximum ΔF/F₀ intensity for scaling the plot.
- `begin_tp (int)`: Starting time point for the plot.
- `end_tp (int)`: Ending time point for the plot.
- `n_start (int)`: Index of the first cell to plot.
- `n_stop (int)`: Index of the last cell to plot.
- `dff_bar (float)`: Height of the ΔF/F₀ scale bar.
- `lw (float)`: Line width of the lines drawn in the plot.
- `show_plots (bool)`: If True, shows the plots. Default is True.
- `save_files (bool)`: If True, saves the plot to the output directory. Default is True.
- `output_dir (str)`: Directory where the plot will be saved.

**Returns:**

- `None`

---

## pwc.py - Functions Overview

The `pwc.py` file contains functions which calculate pairwise correlations between neuronal signals within user specified groups, analyze intra- and inter-group connectivity, and generate visual plots for these correlations. It also supports polynomial fitting and allows the user to save the computed data and generated plots as files. This functionality is used to understand neuronal group interactions and visualize their connectivity patterns.

---

### `run_pwc`

```python
run_pwc(group_name, metadata_path, results_folder, poly = False, pdeg = 4, lw = 1, lwp = 0.5, psz = 2,
    show_plots=False, save_files = False, output_dir = './wizard_staff_outputs')
```

Main wrapper function for performing the PairWise Correlation calculations. This function processes data, computes metrics, generates plots, and stores them in DataFrames.

**Arguments:**

- `group_name (str)`: Column name to group metadata by.
- `metadata_path (str)`: Path to the metadata CSV file.
- `results_folder (str)`: Path to the results folder.
- `poly (bool)`: Whether to apply polynomial fitting (default: `False`).
- `pdeg (int)`: Degree for polynomial fitting (default: 4).
- `lw (float)`: Line width for plots (default: 1).
- `lwp (float)`: Line width for points (default: 0.5).
- `psz (float)`: Point size for plots (default: 2).
- `show_plots (bool)`: If `True`, displays plots (default: `False`).
- `save_files (bool)`: If `True`, saves plots and dataframes (default: `False`).
- `output_dir (str)`: Directory where output files will be saved (default: `'./wizard_staff_outputs'`).

**Returns:**

- `df_mn_pwc (pd.DataFrame)`: DataFrame containing overall pairwise correlation metrics.
- `df_mn_pwc_inter (pd.DataFrame)`: DataFrame containing inter-group pairwise correlation metrics.
- `df_mn_pwc_intra (pd.DataFrame)`: DataFrame containing intra-group pairwise correlation metrics.

---

### `calc_pwc_mn`

```python
calc_pwc_mn(d_k_in_groups, d_dff, d_nspIDs, dff_cut=0.1, norm_corr=False)
```

Calculates pairwise correlation means for groups.

**Arguments:**

- `d_k_in_groups (dict)`: Dictionary where each key is a group identifier and the value is a list of keys.
- `d_dff (dict)`: Dictionary where each key corresponds to a key in `d_k_in_groups` and the value is a dF/F matrix.
- `d_nspIDs (dict)`: Dictionary where each key corresponds to a key in `d_k_in_groups` and the value is a neuron ID array.
- `dff_cut (float)`: Threshold for filtering dF/F values (default: 0.1).
- `norm_corr (bool)`: Whether to normalize the correlation using Fisher's z-transformation (default: `False`).

**Returns:**

- `d_mn_pwc (dict)`: Dictionary of mean pairwise correlations for each group.
- `d_mn_pwc_inter (dict)`: Dictionary of mean inter-group correlations for each group.
- `d_mn_pwc_intra (dict)`: Dictionary of mean intra-group correlations for each group.

---

### `extract_intra_inter_nsp_neurons`

```python
extract_intra_inter_nsp_neurons(conn_matrix, nsp_ids)
```

Extracts intra-subpopulation and inter-subpopulation connections from a connectivity matrix.

**Arguments:**

- `conn_matrix (np.ndarray)`: A square matrix representing the connectivity between neurons.
- `nsp_ids (np.ndarray)`: An array containing the subpopulation IDs for each neuron.

**Returns:**

- `intra_conn (np.ndarray)`: The upper triangular values of the connectivity matrix for intra-subpopulation connections.
- `inter_conn (np.ndarray)`: The upper triangular values of the connectivity matrix for inter-subpopulation connections.

---

### `gen_mn_std_means`

```python
gen_mn_std_means(mean_pwc_dict)
```

Calculates the mean and standard deviation of the means for each key in the input dictionary.

**Arguments:**

- `mean_pwc_dict (dict)`: Dictionary where each key is associated with an array of mean pairwise correlations.

**Returns:**

- `mean_of_means_dict (dict)`: Dictionary containing the mean of means for each key.
- `std_of_means_dict (dict)`: Dictionary containing the standard deviation of means for each key.

---

### `gen_polynomial_fit`

```python
gen_polynomial_fit(data_dict, degree=4)
```

Generates a polynomial fit for the given data.

**Arguments:**

- `data_dict (dict)`: Dictionary where keys are the independent variable (x) and values are the dependent variable (y).
- `degree (int)`: The degree of the polynomial fit (default: 4).

**Returns:**

- `x_values (np.ndarray)`: Array of x values.
- `y_predicted (np.ndarray)`: Array of predicted y values from the polynomial fit.

---

### `filter_group_keys`

```python
filter_group_keys(d_k_in_groups, d_dff, d_nspIDs)
```

Filters group keys to ensure that only those with valid dF/F data and neuron IDs are retained.

**Arguments:**

- `d_k_in_groups (dict)`: Dictionary mapping group IDs to lists of filenames.
- `d_dff (dict)`: Dictionary containing dF/F data matrices for each filename.
- `d_nspIDs (dict)`: Dictionary containing lists of filtered neuron IDs for each filename.

**Returns:**

- `filtered_d_k_in_groups (dict)`: Filtered dictionary where each group ID maps to a list of valid filenames.

---

### `plot_pwc_means`

```python
plot_pwc_means(d_mn_pwc, title, fname, output_dir, xlabel='Groups', ylabel='Mean Pairwise Correlation', 
    poly = False, lwp = 1, psz = 5, pdeg = 4, show_plots = True, save_files = False)
```

Generates plots of mean pairwise correlations with error bars and optionally saves the plots.

**Arguments:**

- `d_mn_pwc (dict)`: Dictionary containing mean pairwise correlation data.
- `title (str)`: Title of the plot.
- `fname (str)`: Filename for saving the results (without extension).
- `output_dir (str)`: Directory where output files will be saved.
- `xlabel (str)`: Label for the x-axis (default: 'Groups').
- `ylabel (str)`: Label for the y-axis (default: 'Mean Pairwise Correlation').
- `lwp (float)`: Line width for the plot (default: 1).
- `psz (float)`: Point size for the plot (default: 5).
- `pdeg (int)`: Degree of the polynomial fit, if applied (default: 4).
- `show_plots (bool)`: If `True`, displays plots (default: `True`).
- `save_files (bool)`: If `True`, saves plots to files (default: `False`).

**Returns:**

`None`

---

## metadata.py - Function Overview

The `metadata.py` file contains utility functions for loading and processing metadata from a CSV file. It includes functions to load metadata, preprocess it by cleaning file names, and append this metadata to various DataFrames that contain imaging metrics such as rise times, FWHM, FRPM, mask metrics, and silhouette scores.

### `load_and_process_metadata`

```python
load_and_process_metadata(metadata_path)
```

Loads and preprocesses the metadata from a CSV file, cleaning file names by removing certain extensions.

**Arguments:**

- `metadata_path` (str): Path to the metadata CSV file.

**Returns:**

- `pd.DataFrame`: Preprocessed metadata DataFrame with cleaned filenames.

---

### `append_metadata_to_dfs`

```python
append_metadata_to_dfs(metadata_path, **dataframes)
```

Appends metadata to the given dataframes based on the filename match.

**Arguments:**

- `metadata_path` (str): Path to the metadata CSV file.
- `**dataframes`: Dictionary of DataFrames to append metadata to. Each key should be a string describing the metric
                      (e.g., 'frpm', 'fwhm'), and each value should be the corresponding DataFrame.

**Returns:**

- `dict`: A dictionary of DataFrames with appended metadata.

---

## wizards-familiars.py - Functions Overview

The `wizards-familiars.py` file provides common utility functions for categorizing, loading, and spatially filtering imaging data files. It includes methods for organizing results based on file types, loading necessary imaging data, and applying spatial filtering to remove noise from neuron activity data. The file also includes functions to visualize the processed data through montage plots.

---

## categorize_files

```python
categorize_files(results_folder)
```

Categorizes files in the results folder based on their prefixes and extensions.

**Arguments:**

- `results_folder` (str): Path to the folder containing result files.

**Returns:**

- `dict`: Dictionary where keys are raw filenames and values are lists of categorized file paths.

---

### load_and_filter_files

```python
load_and_filter_files(categorized_files, p_th=75, size_threshold=20000)
```

Loads ΔF/F₀ data and applies spatial filtering to filter out noise events based on the size of neuron footprints.

**Arguments:**

- `categorized_files` (dict): Dictionary mapping filenames to their corresponding file paths.
- `p_th` (float): Percentile threshold for image processing (default: 75).
- `size_threshold` (int): Size threshold for filtering out noise events (default: 20000).

**Returns:**

- `d_dff` (dict): Dictionary where each key is a filename, and the value is the loaded ΔF/F₀ data matrix.
- `d_nspIDs` (dict): Dictionary where each key is a filename, and the value is the list of filtered neuron IDs.

---

### load_required_files

```python
load_required_files(categorized_files, raw_filename)
```

Loads necessary files for a given raw filename from the categorized files.

**Arguments:**

- `categorized_files` (dict): Dictionary mapping filenames to their corresponding file paths.
- `raw_filename` (str): The filename to load the files for.

**Returns:**

- `dict`: Dictionary containing loaded files with keys corresponding to the file type.

---

### spatial_filtering

```python
spatial_filtering(cn_filter, p_th, size_threshold, cnm_A, cnm_idx, im_min, plot=True, silence = False)
```

Applies spatial filtering to components, generates montages, and optionally plots the results.

**Arguments:**

- `cn_filter` (str): Path to the masked cn_filter image.
- `p_th` (float): Percentile threshold for image processing.
- `size_threshold` (int): Size threshold for filtering out noise events.
- `cnm_A` (ndarray): Spatial footprint matrix of neurons.
- `im_min` (ndarray): Minimum intensity image for overlay.
- `cnm_idx` (ndarray): Indices of accepted components.
- `plot` (bool): If True, displays the montage plots (default: True).
- `silence` (bool): If False, prints the number of components before and after filtering.

**Returns:**

- `filtered_idx` (list): List of indices of the filtered components.
