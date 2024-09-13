# Functions Overview

This document provides an overview of the functions in the `wizards_staff` Python package, including function signatures, arguments, and descriptions of their usage.

---

## Table of Contents

- [Plotting Functions](#plotting-functions)
- [Usage](#usage)
- [Functions Overview](#functions-overview)

---

### Plotting Functions

### `plot_spatial_activity_map`

**Function**: Plots the activity of neurons by overlaying the spatial footprints on a single image.

```python
plot_spatial_activity_map(im_min, cnm_A, cnm_idx, raw_filename, p_th=75, min_clusters=2, max_clusters=10, random_seed=1111111, show_plots=True, save_files=False,clustering=False, dff_data=None, output_dir='./wizard_staff_outputs')
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

Creates an overlay image by combining a grayscale background image with a binary overlay.

**Arguments:**

- `im_avg (ndarray)`: Grayscale average image.
- `binary_overlay (ndarray)`: Binary image for overlay.
- `overlay_color (list)`: RGB color for the overlay (default: `[255, 255, 0]`).

**Returns:**

- `overlay_image (ndarray)`: The combined overlay image.

---

### `plot_montage`

Creates a montage from a list of images, overlaying a binary image on a grayscale background.

**Arguments:**

- `images (list)`: List of binary images to be arranged in a montage.
- `im_avg (ndarray)`: Grayscale background image.
- `grid_shape (tuple)`: Shape of the montage grid (rows, columns).
- `overlay_color (list)`: RGB color for the binary overlay (default: `[255, 255, 0]`).
- `rescale_intensity (bool)`: If `True`, rescales intensity values (default: `False`).

**Returns:**

- `montage (ndarray)`: The montage image.