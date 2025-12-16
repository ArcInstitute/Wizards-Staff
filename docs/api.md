# Wizards Staff Function Reference

This document provides a comprehensive overview of all functions available in the Wizards Staff package, including function signatures, arguments, descriptions, and usage examples.

## Table of Contents

- [Core Classes and Methods](#core-classes-and-methods)
  - [Orb Class](#orb-class)
  - [Shard Class](#shard-class)
- [Wizards Spellbook Functions](#wizards-spellbook-functions)
- [Plotting Functions](#plotting-functions)
- [Pairwise Correlation (PWC) Functions](#pairwise-correlation-pwc-functions)
- [Metadata Functions](#metadata-functions)
- [Wizards Familiars Functions](#wizards-familiars-functions)

---

## Core Classes and Methods

### Orb Class

The main container class that organizes all samples and provides methods for batch processing.

**Constructor**

```python
Orb(results_folder, metadata_file_path, quiet=False)
```

**Arguments:**

- `results_folder (str)`: Path to the folder containing analysis results
- `metadata_file_path (str)`: Path to the metadata CSV file
- `quiet (bool)`: If True, suppresses informational messages (default: False)

#### Key Methods

##### `run_all`

```python
run_all(group_name=None, frate=None, zscore_threshold=3, percentage_threshold=0.2, 
        p_th=75, min_clusters=2, max_clusters=10, random_seed=1111111, 
        poly=False, size_threshold=20000, show_plots=False, save_files=False, 
        output_dir='wizards_staff_outputs', threads=2, debug=False, **kwargs)
```

Runs a comprehensive analysis on all samples, calculating metrics like rise time, FWHM, firing rates, and clustering.

**Arguments:**

- `group_name (str)`: Column in metadata to group samples by (required for PWC analysis)
- `frate (int)`: Frame rate in frames per second. If None (default), automatically reads from the 'Frate' column in metadata for each sample
- `zscore_threshold (int)`: Z-score threshold for spike detection
- `percentage_threshold (float)`: Threshold for FWHM calculation
- `p_th (float)`: Percentile threshold for spatial filtering
- `min_clusters (int)`: Minimum number of clusters for K-means
- `max_clusters (int)`: Maximum number of clusters for K-means
- `random_seed (int)`: Seed for random number generation
- `poly (bool)`: Whether to use polynomial fitting for PWC
- `size_threshold (int)`: Size threshold for filtering out noise events
- `show_plots (bool)`: Whether to display plots
- `save_files (bool)`: Whether to save output files
- `output_dir (str)`: Directory for output files
- `threads (int)`: Number of parallel processing threads
- `debug (bool)`: Run in debug mode (single-threaded)
- `**kwargs`: Additional keyword arguments for PWC analysis

##### `run_pwc`

```python
run_pwc(group_name, poly=False, p_th=75, size_threshold=20000, pdeg=4, lw=1, 
        lwp=0.5, psz=2, show_plots=False, save_files=False, 
        output_dir='wizard_staff_outputs')
```

Runs pairwise correlation analysis on samples grouped by a metadata column.

**Arguments:**

- `group_name (str)`: Column in metadata to group samples by
- `poly (bool)`: Whether to use polynomial fitting
- `p_th (float)`: Percentile threshold for spatial filtering
- `size_threshold (int)`: Size threshold for filtering
- `pdeg (int)`: Polynomial degree for fitting
- `lw (float)`: Line width for plots
- `lwp (float)`: Line width for points
- `psz (float)`: Point size for plots
- `show_plots (bool)`: Whether to display plots
- `save_files (bool)`: Whether to save output files
- `output_dir (str)`: Directory for output files

##### `shatter`

```python
shatter()
```

Yields each Shard (sample) in the Orb for individual processing.

##### `save_results`

```python
save_results(outdir, result_names=["rise_time_data", "fwhm_data", "frpm_data", 
                                  "mask_metrics_data", "silhouette_scores_data",
                                  "df_mn_pwc", "df_mn_pwc_intra", "df_mn_pwc_inter"])
```

Saves analysis results to disk.

**Arguments:**

- `outdir (str)`: Output directory
- `result_names (list)`: List of result datasets to save

#### Key Properties

- `samples`: Set of sample names
- `input_files`: DataFrame of input file paths
- `input`: DataFrame of input data with metadata
- `rise_time_data`: DataFrame of rise time metrics
- `fwhm_data`: DataFrame of FWHM metrics
- `frpm_data`: DataFrame of firing rate metrics
- `mask_metrics_data`: DataFrame of mask shape metrics
- `silhouette_scores_data`: DataFrame of clustering metrics
- `df_mn_pwc`: DataFrame of pairwise correlations
- `df_mn_pwc_intra`: DataFrame of intra-group correlations
- `df_mn_pwc_inter`: DataFrame of inter-group correlations

### Shard Class

Represents a single sample and provides methods for processing that specific sample.

**Constructor**

```python
Shard(sample_name, metadata, files, quiet=False)
```

**Arguments:**

- `sample_name (str)`: Name of the sample
- `metadata (pd.DataFrame)`: Metadata for this sample
- `files (dict)`: Dictionary mapping data items to file paths and loaders
- `quiet (bool)`: If True, suppresses informational messages

#### Key Methods

##### `get_input`

```python
get_input(item_name, req=False)
```

Retrieves the input item for the given name, loading it if not already loaded.

**Arguments:**
- `item_name (str)`: Name of the data item to retrieve
- `req (bool)`: If True, raises an error if the item is not found

##### `has_file`

```python
has_file(item_name)
```

Checks if a data item is available for this sample.

**Arguments:**

- `item_name (str)`: Name of the data item to check

##### `spatial_filtering`

```python
spatial_filtering(p_th=75, size_threshold=20000, plot=True, silence=False)
```

Applies spatial filtering to components based on size threshold.

**Arguments:**

- `p_th (float)`: Percentile threshold for image processing
- `size_threshold (int)`: Size threshold for filtering out noise events
- `plot (bool)`: Whether to show plots
- `silence (bool)`: Whether to suppress print statements

##### `convert_f_to_cs`

```python
convert_f_to_cs(p=2, noise_range=[0.25, 0.5])
```

Converts fluorescence data to calcium and spike signals.

**Arguments:**

- `p (int)`: Order of the autoregressive process
- `noise_range (list)`: Range for estimating noise

##### `calc_rise_tm`

```python
calc_rise_tm(calcium_signals, spike_events, zscore_threshold=3)
```

Calculates rise time of calcium signals.

**Arguments:**

- `calcium_signals (ndarray)`: Calcium signal matrix
- `spike_events (ndarray)`: Z-scored spike event matrix
- `zscore_threshold (float)`: Z-score threshold for spike detection

##### `calc_fwhm_spikes`

```python
calc_fwhm_spikes(calcium_signals, zscored_spike_events, zscore_threshold=3, percentage_threshold=0.2)
```

Calculates FWHM of spikes in calcium signals.

**Arguments:**

- `calcium_signals (ndarray)`: Calcium signal matrix
- `zscored_spike_events (ndarray)`: Z-scored spike event matrix
- `zscore_threshold (float)`: Z-score threshold for spike detection
- `percentage_threshold (float)`: Threshold for FWHM calculation

##### `calc_frpm`

```python
calc_frpm(zscored_spike_events, filtered_idx, frate, zscore_threshold=5)
```

Calculates firing rate per minute.

**Arguments:**
- `zscored_spike_events (ndarray)`: Z-scored spike events
- `filtered_idx (ndarray)`: Indices of filtered components
- `frate (int)`: Frame rate in frames per second
- `zscore_threshold (int)`: Z-score threshold for spike detection

##### `calc_mask_shape_metrics`

```python
calc_mask_shape_metrics()
```

Calculates shape metrics for the mask.

#### Key Properties

- `input_files`: DataFrame of input file paths
- `rise_time_data`: List of rise time data
- `fwhm_data`: List of FWHM data
- `frpm_data`: List of firing rate data
- `mask_metrics_data`: List of mask shape metrics
- `silhouette_scores_data`: List of silhouette scores

---

## Wizards Spellbook Functions

Functions for signal processing and metric calculations.

### `convert_f_to_cs`

```python
convert_f_to_cs(fluorescence_data, p=2, noise_range=[0.25, 0.5])
```

Converts fluorescence data to calcium and spike signals using deconvolution.

**Arguments:**

- `fluorescence_data (ndarray)`: Fluorescence data matrix with neurons as rows and time points as columns
- `p (int)`: Order of the autoregressive process
- `noise_range (list)`: Range for estimating noise

**Returns:**

- `calcium_signal (ndarray)`: Calcium signal matrix
- `spike_signal (ndarray)`: Spike signal matrix

### `calc_rise_tm`

```python
calc_rise_tm(calcium_signals, spike_zscores, zscore_threshold=3)
```

Calculates the rise time of calcium signals based on spike detection.

**Arguments:**

- `calcium_signals (ndarray)`: Calcium signal matrix with neurons as rows and time points as columns
- `spike_zscores (ndarray)`: Z-scored spike signal matrix
- `zscore_threshold (float)`: Z-score threshold for spike detection

**Returns:**

- `rise_times (dict)`: Dictionary of rise times for each neuron
- `rise_positions (dict)`: Dictionary of positions corresponding to the rise times

### `calc_fwhm_spikes`

```python
calc_fwhm_spikes(calcium_signals, spike_zscores, zscore_threshold=3, percentage_threshold=0.2)
```

Calculates the full width at half maximum (FWHM) of spikes in calcium signals.

**Arguments:**

- `calcium_signals (ndarray)`: Calcium signal matrix with neurons as rows and time points as columns
- `spike_zscores (ndarray)`: Z-scored spike signal matrix
- `zscore_threshold (float)`: Z-score threshold for spike detection
- `percentage_threshold (float)`: Percentage threshold for determining half maximum

**Returns:**
- `fwhm_backward_positions (dict)`: Dictionary of backward positions of FWHM for each neuron
- `fwhm_forward_positions (dict)`: Dictionary of forward positions of FWHM for each neuron
- `fwhm_values (dict)`: Dictionary of FWHM values for each neuron
- `spike_counts (dict)`: Dictionary of the number of spikes within the FWHM for each neuron

### `calc_frpm`

```python
calc_frpm(zscored_spike_events, neuron_ids, fps, zscore_threshold=5)
```

Calculates the firing rate per minute (FRPM) for z-scored spike event data.

**Arguments:**

- `zscored_spike_events (ndarray)`: Z-scored spike events with neurons as rows and time points as columns
- `neuron_ids (ndarray)`: Array containing neuron IDs
- `fps (int)`: Frames per second of the recording
- `zscore_threshold (int)`: Z-score threshold for detecting spikes

**Returns:**

- `frpm (float)`: Average firing rate per minute for the dataset
- `spike_dict (dict)`: Dictionary containing firing rates for each neuron

### `calc_mask_shape_metrics`

```python
calc_mask_shape_metrics(mask_image)
```

Calculates shape metrics (roundness, diameter, area) for a mask image.

**Arguments:**
- `mask_image (ndarray)`: Binary mask image of the spheroid/organoid

**Returns:**
- `dict`: Dictionary containing roundness, diameter, and area of the masked object

---

## Plotting Functions

Functions for visualizing calcium imaging data.

### `plot_spatial_activity_map`

```python
plot_spatial_activity_map(im_min, cnm_A, cnm_idx, sample_name, p_th=75, 
                          min_clusters=2, max_clusters=10, random_seed=1111111, 
                          show_plots=True, save_files=False, clustering=False, 
                          dff_dat=None, output_dir='wizard_staff_outputs')
```

Plots the activity of neurons by overlaying their spatial footprints on a single image.

**Arguments:**

- `im_min (ndarray)`: Minimum intensity image for overlay
- `cnm_A (ndarray)`: Spatial footprint matrix of neurons
- `cnm_idx (ndarray)`: Indices of accepted components
- `sample_name (str)`: The sample name
- `p_th (float)`: Percentile threshold for image processing
- `min_clusters (int)`: Minimum number of clusters for K-means
- `max_clusters (int)`: Maximum number of clusters for K-means
- `random_seed (int)`: Seed for random number generation
- `show_plots (bool)`: Whether to show plots
- `save_files (bool)`: Whether to save files
- `clustering (bool)`: Whether to perform clustering for coloring
- `dff_dat (ndarray)`: ΔF/F₀ data array (required if clustering=True)
- `output_dir (str)`: Output directory

**Returns:**

- `overlay_image (ndarray)`: The combined overlay image

### `plot_kmeans_heatmap`

```python
plot_kmeans_heatmap(dff_dat, filtered_idx, sample_name, 
                   output_dir='wizard_staff_outputs', min_clusters=2,
                   max_clusters=10, random_seed=1111111, 
                   show_plots=True, save_files=True)
```

Generates a K-means clustering heatmap and outputs clustering metrics.

**Arguments:**

- `dff_dat (ndarray)`: The ΔF/F₀ data array
- `filtered_idx (ndarray)`: Indices of filtered components
- `sample_name (str)`: Sample name
- `output_dir (str)`: Output directory
- `min_clusters (int)`: Minimum number of clusters
- `max_clusters (int)`: Maximum number of clusters
- `random_seed (int)`: Random seed for K-means
- `show_plots (bool)`: Whether to show plots
- `save_files (bool)`: Whether to save files

**Returns:**

- `best_silhouette_score (float)`: Best silhouette score
- `best_num_clusters (int)`: Optimal number of clusters

### `plot_cluster_activity`

```python
plot_cluster_activity(dff_dat, filtered_idx, sample_name, 
                     min_clusters=2, max_clusters=10, random_seed=1111111, 
                     norm=False, show_plots=True, save_files=True, 
                     output_dir='wizard_staff_outputs')
```

Plots the average activity of each cluster and detailed activity of a specific cluster.

**Arguments:**

- `dff_dat (ndarray)`: The ΔF/F₀ data array
- `filtered_idx (ndarray)`: Indices of filtered components
- `sample_name (str)`: Sample name
- `min_clusters (int)`: Minimum number of clusters
- `max_clusters (int)`: Maximum number of clusters
- `random_seed (int)`: Random seed for K-means
- `norm (bool)`: Whether to normalize the data
- `show_plots (bool)`: Whether to show plots
- `save_files (bool)`: Whether to save files
- `output_dir (str)`: Output directory

### `plot_dff_activity`

```python
plot_dff_activity(dff_dat, cnm_idx, frate, sample_name, sz_per_neuron=0.5, 
                 max_z=0.45, begin_tp=0, end_tp=-1, n_start=0, n_stop=-1, 
                 dff_bar=1, lw=0.5, show_plots=True, save_files=True, 
                 output_dir='wizard_staff_outputs')
```

Plots ΔF/F₀ activity data for neurons within a specified time range.

**Arguments:**

- `dff_dat (ndarray)`: Activity data matrix
- `cnm_idx (ndarray)`: Neuron indices
- `frate (int)`: Frame rate
- `sample_name (str)`: Sample name
- `sz_per_neuron (float)`: Size per neuron in the plot
- `max_z (float)`: Maximum ΔF/F₀ intensity
- `begin_tp (int)`: Starting time point
- `end_tp (int)`: Ending time point (-1 for all)
- `n_start (int)`: First neuron to plot
- `n_stop (int)`: Last neuron to plot (-1 for all)
- `dff_bar (float)`: Height of the ΔF/F₀ scale bar
- `lw (float)`: Line width
- `show_plots (bool)`: Whether to show plots
- `save_files (bool)`: Whether to save files
- `output_dir (str)`: Output directory

### `overlay_images`

```python
overlay_images(im_avg, binary_overlay, overlay_color=[255, 255, 0])
```

Creates an overlay image with the specified color for the binary overlay.

**Arguments:**

- `im_avg (ndarray)`: Average grayscale image
- `binary_overlay (ndarray)`: Binary overlay image
- `overlay_color (list)`: RGB color for the overlay

**Returns:**

- `overlay_image (ndarray)`: Combined overlay image

### `plot_montage`

```python
plot_montage(images, im_avg, grid_shape, overlay_color=[255, 255, 0],
             rescale_intensity=False)
```

Creates a montage from a list of binary images with overlay on grayscale background.

**Arguments:**

- `images (list)`: List of binary images
- `im_avg (ndarray)`: Average grayscale image
- `grid_shape (tuple)`: Shape of the montage grid
- `overlay_color (list)`: RGB color for the overlay
- `rescale_intensity (bool)`: Whether to rescale intensity

**Returns:**

- `montage (ndarray)`: Montage image

---

## Pairwise Correlation (PWC) Functions

Functions for analyzing correlations between neurons.

### `run_pwc`

```python
run_pwc(orb, group_name, poly=False, p_th=75, size_threshold=20000, pdeg=4, 
        lw=1, lwp=0.5, psz=2, show_plots=False, save_files=False, 
        output_dir='wizard_staff_outputs')
```

Processes data, computes pairwise correlation metrics, and generates plots.

**Arguments:**

- `orb (Orb)`: Orb object containing the data
- `group_name (str)`: Column name to group by
- `poly (bool)`: Whether to use polynomial fitting
- `p_th (float)`: Percentile threshold
- `size_threshold (int)`: Size threshold
- `pdeg (int)`: Polynomial degree
- `lw (float)`: Line width
- `lwp (float)`: Line width for points
- `psz (float)`: Point size
- `show_plots (bool)`: Whether to show plots
- `save_files (bool)`: Whether to save files
- `output_dir (str)`: Output directory

### `calc_pwc_mn`

```python
calc_pwc_mn(d_k_in_groups, d_dff, d_nspIDs, dff_cut=0.1, norm_corr=False)
```

Calculates pairwise correlation means for groups.

**Arguments:**

- `d_k_in_groups (dict)`: Dictionary mapping group IDs to lists of keys
- `d_dff (dict)`: Dictionary mapping keys to dF/F matrices
- `d_nspIDs (dict)`: Dictionary mapping keys to neuron ID arrays
- `dff_cut (float)`: Threshold for filtering dF/F values
- `norm_corr (bool)`: Whether to normalize correlations

**Returns:**

- `d_mn_pwc (dict)`: Dictionary of mean pairwise correlations
- `d_mn_pwc_inter (dict)`: Dictionary of mean inter-group correlations
- `d_mn_pwc_intra (dict)`: Dictionary of mean intra-group correlations

### `extract_intra_inter_nsp_neurons`

```python
extract_intra_inter_nsp_neurons(conn_matrix, nsp_ids)
```

Extracts intra-subpopulation and inter-subpopulation connections.

**Arguments:**

- `conn_matrix (ndarray)`: Connectivity matrix
- `nsp_ids (ndarray)`: Subpopulation IDs for each neuron

**Returns:**

- `intra_conn (ndarray)`: Intra-subpopulation connections
- `inter_conn (ndarray)`: Inter-subpopulation connections

### `gen_mn_std_means`

```python
gen_mn_std_means(mean_pwc_dict)
```

Calculates the mean and standard deviation of the means.

**Arguments:**

- `mean_pwc_dict (dict)`: Dictionary of mean pairwise correlations

**Returns:**

- `mean_of_means_dict (dict)`: Mean of means for each key
- `std_of_means_dict (dict)`: Standard deviation of means for each key

### `gen_polynomial_fit`

```python
gen_polynomial_fit(data_dict, degree=4)
```

Generates a polynomial fit for the given data.

**Arguments:**

- `data_dict (dict)`: Dictionary of x-y data
- `degree (int)`: Polynomial degree

**Returns:**

- `x_values (ndarray)`: X values
- `y_predicted (ndarray)`: Predicted Y values

### `filter_group_keys`

```python
filter_group_keys(d_k_in_groups, d_dff, d_nspIDs)
```

Filters group keys to ensure valid dF/F data and neuron IDs.

**Arguments:**

- `d_k_in_groups (dict)`: Dictionary mapping group IDs to lists of keys
- `d_dff (dict)`: Dictionary mapping keys to dF/F matrices
- `d_nspIDs (dict)`: Dictionary mapping keys to neuron ID arrays

**Returns:**

- `filtered_d_k_in_groups (dict)`: Filtered dictionary

### `plot_pwc_means`

```python
plot_pwc_means(d_mn_pwc, title, fname, output_dir=None, xlabel='Groups', 
               ylabel='Mean Pairwise Correlation', poly=False, lwp=1, 
               psz=5, pdeg=4, show_plots=True, save_files=True)
```

Generates plots of mean pairwise correlations with error bars.

**Arguments:**

- `d_mn_pwc (dict)`: Dictionary of mean pairwise correlations
- `title (str)`: Plot title
- `fname (str)`: Filename for saving
- `output_dir (str)`: Output directory
- `xlabel (str)`: X-axis label
- `ylabel (str)`: Y-axis label
- `poly (bool)`: Whether to use polynomial fitting
- `lwp (float)`: Line width for points
- `psz (float)`: Point size
- `pdeg (int)`: Polynomial degree
- `show_plots (bool)`: Whether to show plots
- `save_files (bool)`: Whether to save files

**Returns:**

- `fig (Figure)`: Matplotlib figure object

---

## Metadata Functions

Functions for handling metadata.

### `load_and_process_metadata`

```python
load_and_process_metadata(metadata_path)
```

Loads and preprocesses the metadata CSV file.

**Arguments:**

- `metadata_path (str)`: Path to the metadata CSV file

**Returns:**

- `metadata_df (DataFrame)`: Preprocessed metadata DataFrame

### `append_metadata_to_dfs`

```python
append_metadata_to_dfs(metadata_path, **dataframes)
```

Appends metadata to multiple DataFrames based on filename matches.

**Arguments:**

- `metadata_path (str)`: Path to the metadata CSV file
- `**dataframes`: Dictionary of DataFrames to append metadata to

**Returns:**

- `updated_dfs (dict)`: Dictionary of DataFrames with appended metadata

---

## Wizards Familiars Functions

Functions for file handling and processing.

### `spatial_filtering`

```python
spatial_filtering(p_th, size_threshold, cnm_A, cnm_idx, im_min, 
                  plot=True, silence=False)
```

Applies spatial filtering to components based on size threshold.

**Arguments:**

- `p_th (float)`: Percentile threshold for image processing
- `size_threshold (int)`: Size threshold for filtering out noise events
- `cnm_A (ndarray)`: Spatial footprint matrix of neurons
- `cnm_idx (ndarray)`: Indices of accepted components
- `im_min (ndarray)`: Minimum intensity image
- `plot (bool)`: Whether to show plots
- `silence (bool)`: Whether to suppress print statements

**Returns:**

- `filtered_idx (list)`: List of indices of filtered components

### `load_and_filter_files`

```python
load_and_filter_files(categorized_files, p_th=75, size_threshold=20000)
```

Loads ΔF/F₀ data and applies spatial filtering.

**Arguments:**

- `categorized_files (dict)`: Dictionary mapping filenames to file paths
- `p_th (float)`: Percentile threshold for image processing
- `size_threshold (int)`: Size threshold for filtering out noise events

**Returns:**

- `d_dff (dict)`: Dictionary mapping filenames to ΔF/F₀ data matrices
- `d_nspIDs (dict)`: Dictionary mapping filenames to filtered neuron IDs

### `categorize_files`

```python
categorize_files(results_folder)
```

Categorizes result files based on their prefixes and extensions.

**Arguments:**

- `results_folder (str)`: Path to the results folder

**Returns:**

- `categorized_files (dict)`: Dictionary mapping filenames to categorized file paths

### `load_required_files`

```python
load_required_files(categorized_files, raw_filename)
```

Loads necessary files for a given raw filename from the categorized files.

**Arguments:**

- `categorized_files (dict)`: Dictionary mapping filenames to file paths
- `raw_filename (str)`: Raw filename to load files for

**Returns:**

- `file_data (dict)`: Dictionary containing loaded files