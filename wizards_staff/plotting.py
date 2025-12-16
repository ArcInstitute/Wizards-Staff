# import
## batteries
import os
import sys
import random
import warnings
from typing import List, Tuple, Optional
## 3rd party
import numpy as np
import pandas as pd
from skimage.io import imread
from sklearn.metrics import silhouette_score
from scipy.cluster.vq import kmeans2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 
from matplotlib import colors

# Functions
def plot_spatial_activity_map(im_min: np.ndarray, cnm_A: np.ndarray, cnm_idx: np.ndarray, 
                              sample_name: str, p_th: float=75, min_clusters: int=2, 
                              max_clusters: int=10, random_seed: int=1111111,
                              show_plots: bool=True, save_files: bool=False,
                              clustering: bool=False, dff_dat: np.ndarray=None,
                              output_dir: str='wizard_staff_outputs') -> Optional[np.ndarray]:
    """
    Plot the activity of neurons by overlaying the spatial footprints on a single image.
    
    Args:
        im_min: Minimum intensity image for overlay.
        cnm_A: Spatial footprint matrix of neurons.
        cnm_idx: Indices of accepted components.
        sample_name: The sample name.
        p_th: Percentile threshold for image processing.
        min_clusters: The minimum number of clusters to try. Default is 2.
        max_clusters: The maximum number of clusters to try. Default is 10.
        random_seed: The seed for random number generation in K-means. Default is 1111111.
        show_plots: If True, shows the plots. Default is True.
        save_files: If True, saves the overlay image to the output directory. Default is True.
        clustering: If True, perform K-means clustering and color ROIs by cluster.
        dff_dat: The dF/F data array, required if clustering=True.
        output_dir: Directory where output files will be saved.
    
    Returns:
        overlay_image: The combined overlay image.
    """
    # Load the minimum intensity image
    im_shape = im_min.shape
    im_sz = [im_shape[0], im_shape[1]]
    
    # Initialize an image to store the overlay of all neuron activities
    overlay_image = np.zeros((im_sz[0], im_sz[1], 3), dtype=np.uint8)
    
    # Generate color map using nipyspectral
    cmap = plt.get_cmap('nipy_spectral')
    norm = colors.Normalize(vmin=0, vmax=len(cnm_idx))

    if clustering and dff_dat is not None:
        # Perform clustering on the filtered data
        data_t = dff_dat[cnm_idx]

        # If empty, return None  # TODO: check with Jesse about this
        if data_t.shape[0] == 0:
            return None

        best_silhouette_score = -1
        best_num_clusters = 2
        best_labels = None

        # Try K-means with different numbers of clusters
        for num_clusters in range(min_clusters, max_clusters + 1):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                # Perform K-means clustering
                centroids, labels = kmeans2(data_t, num_clusters, seed = random_seed)
            
            # Calculate silhouette score
            if len(np.unique(labels)) > 1:
                silhouette_avg = silhouette_score(data_t, labels)
            else:
                silhouette_avg = -1

            # Update if this is the best silhouette score
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_num_clusters = num_clusters
                best_labels = labels
        
        # Return None if no clusters are found
        if best_labels is None:
            print('No clusters found. No overlay image will be generated.', file=sys.stderr)
            return None

        # Assign colors to clusters
        unique_clusters = np.unique(best_labels)
        cluster_colors = {}
        for cluster in unique_clusters:
            color_idx = random.uniform(0.0, 1.0)  # Random float between 0 and 1
            color = np.array(cmap(color_idx)[:3]) * 255  # Select a random color from the full colormap
            cluster_colors[cluster] = color

    # Apply spatial filtering and generate the overlay image
    for i, idx in enumerate(cnm_idx):
        Ai = np.copy(cnm_A[:, idx])
        Ai = Ai[Ai > 0]
        thr = np.percentile(Ai, p_th)
        imt = np.reshape(cnm_A[:, idx], im_sz, order='F')
        im_thr = np.copy(imt)
        im_thr[im_thr < thr] = 0
        im_thr[im_thr >= thr] = 1

        # Determine the color based on clustering or use the default colormap
        if clustering and dff_dat is not None:
            cluster = best_labels[i]
            color = cluster_colors[cluster]
        else:
            color = cmap(norm(i))[:3]  # RGB values from the colormap
            color = np.array(color) * 255  # Convert to 0-255 range

        # Create an RGB overlay for this component and combine with the existing overlay image
        overlay_image[..., 0] = np.maximum(overlay_image[..., 0], im_thr * color[0])
        overlay_image[..., 1] = np.maximum(overlay_image[..., 1], im_thr * color[1])
        overlay_image[..., 2] = np.maximum(overlay_image[..., 2], im_thr * color[2])
    
    # Plot the overlay image if show_plots is True
    if show_plots:
        plt.figure(figsize=(10, 10))
        plt.imshow(im_min, cmap='gray')
        plt.imshow(overlay_image, alpha=0.6)  # Overlay with transparency
        if clustering and dff_dat is not None:
            plt.title('Overlay of Clustered Neuron Activities')
        else:
            plt.title('Overlay of Neuron Activities')
        plt.axis('off')
        plt.show()
    else:
        plt.close()
    
    # Save the overlay image if save_files is True
    if save_files:
        # Expand the user directory if it exists in the output_dir path
        output_dir = os.path.expanduser(output_dir)
        output_dir_pngs = os.path.join(output_dir, 'cluster_activity_maps')
        
        # Create the output directory if it does not exist
        os.makedirs(output_dir_pngs, exist_ok=True)
        
        # Define the file path
        if clustering and dff_dat is not None:
            overlay_image_path = os.path.join(output_dir_pngs, f'{sample_name}_clustered-activity-overlay.png')
        else:
            overlay_image_path = os.path.join(output_dir_pngs, f'{sample_name}_activity-overlay.png')
        
        # Create a new figure for saving
        plt.figure(figsize=(10, 10))
        plt.imshow(im_min, cmap='gray')  # Plot the min projection image in grayscale
        plt.imshow(overlay_image, alpha=0.6)  # Overlay the spatial activity map with transparency
        plt.axis('off')  # Turn off axis labels

        # Save the overlay image
        plt.savefig(overlay_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    

def plot_kmeans_heatmap(dff_dat: np.ndarray, filtered_idx: np.ndarray, sample_name: str, 
                        output_dir: str='wizard_staff_outputs', min_clusters: int=2, 
                        max_clusters: int=10, random_seed: int=1111111, show_plots: bool=True, 
                        save_files: bool=True) -> Tuple[Optional[float], Optional[int]]:
    """
    Plot K-means clustering of the given data and outputs synchronization metrics and clustering information to a spreadsheet.

    Args:
        dff_dat: The dF/F data array.
        filter_idx: The indices of the data to be filtered.
        sample_name: The name of the sample.
        output_dir: Directory where output files will be saved.
        min_clusters: The minimum number of clusters to try. Default is 2.
        max_clusters: The maximum number of clusters to try. Default is 10.
        random_seed: The seed for random number generation in K-means. Default is 1111111.
        show_plots: If True, shows the plots. Default is True.
        save_files: If True, saves the plot and clustering information to the output directory. Default is True.

    Returns:
        best_silhouette_score: The silhouette score of the best clustering.
        best_num_clusters: The number of clusters in the best clustering.
    """
    # Filter the data
    data_t = dff_dat[filtered_idx]

    # If empty, return None  # TODO: check with Jesse about this
    if data_t.shape[0] == 0:
        return None, None

    # Init params
    best_silhouette_score = -1
    best_num_clusters = min_clusters
    best_labels = None
    best_sorted_data_t = None

    # Try K-means with different numbers of clusters
    for num_clusters in range(min_clusters, max_clusters + 1):
        # Suppress warnings of empty clusters from k-means clustering
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            # Perform K-means clustering
            centroids, labels = kmeans2(data_t, num_clusters, seed=random_seed)
        
        # Calculate silhouette score as a measure of clustering quality
        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(data_t, labels)
        else:
            silhouette_avg = -1

        # If the sillhouette score is better, update the best values
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = num_clusters
            best_labels = labels
            sorted_indices = np.argsort(labels)
            best_sorted_data_t = data_t[sorted_indices, :]

    # If just one cluster is found, return None
    if best_sorted_data_t is None:
        print('Only one cluster found. No kmeans heatmap plot will be generated.', file=sys.stderr)
        return None, None

    # Plotting the best result
    plt.figure(figsize=(14, 14))
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(data_t, cmap='plasma')
    ax1.set_aspect(14)
    plt.title('Original Traces')

    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(best_sorted_data_t, cmap='plasma')
    ax2.set_aspect(14)
    plt.title(f'Traces sorted by K-means (Best Num Clusters: {best_num_clusters})')

    # Get a colormap for the clusters
    cmap = plt.colormaps['terrain'](np.linspace(0, 1, best_num_clusters))

    # Add rectangle patches to label specific clusters
    cluster_offset = 0
    for cluster_idx in range(best_num_clusters):
        num_cells_in_cluster = np.sum(best_labels == cluster_idx)
        rect = patches.Rectangle(
            (0, cluster_offset), 50, num_cells_in_cluster, 
            linewidth=1, 
            edgecolor='none', 
            facecolor=cmap[cluster_idx]
        )
        # axis2 
        ax2.text(
            10, cluster_offset + num_cells_in_cluster / 2, str(cluster_idx), 
            color='k', 
            weight='bold'
        )
        ax2.add_patch(rect)
        ax2.text(10, -5, '↓ Cluster ID', fontsize=10)
        cluster_offset += num_cells_in_cluster

    # Set axis labels
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('ROI')
    if best_sorted_data_t is not None:
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('ROI')

    # Save the plot
    if save_files == True:
        # Expand the user directory if it exists in the output_dir path
        output_dir = os.path.expanduser(output_dir)
        output_dir_pngs = os.path.join(output_dir, 'kmeans_heatmap_plots')

        # Create the output directory if it does not exist
        os.makedirs(output_dir_pngs, exist_ok=True)

        # Save the figure
        fig_path = os.path.join(output_dir_pngs, f'{sample_name}_kmeans-clustering-plot.png')
        plt.savefig(fig_path, bbox_inches='tight')

    # Display the plot
    if show_plots == True:
        plt.show()
    else:
        plt.close()

    # Save clustering information to a CSV file
    clustering_info = {
        'Original Data': [list(row) for row in data_t],
        'Clustered Data': [list(row) for row in best_sorted_data_t],
        'Labels': best_labels.tolist(),
    }

    # Create a DataFrame and save to Excel
    df = pd.DataFrame(clustering_info)

    # Add the best silhouette score and number of clusters to the first row
    df.loc[0, 'Best Silhouette Score'] = best_silhouette_score
    df.loc[0, 'Best Num Clusters'] = best_num_clusters
    
    # Save the DataFrame to a CSV file
    if save_files==True:
        output_dir_csv = os.path.join(output_dir, 'kmeans_heatmap_csv')
        os.makedirs(output_dir_csv, exist_ok=True)
        csv_path = os.path.join(output_dir_csv, f'{sample_name}_clustering-info.csv')
        df.to_csv(csv_path, index=False)

    return best_silhouette_score, best_num_clusters

def plot_cluster_activity(dff_dat: np.ndarray, filtered_idx: np.ndarray, sample_name: str, 
                          min_clusters: int=2, max_clusters: int=10, random_seed: int=1111111, 
                          norm: bool=False, show_plots: bool=True, save_files: bool=True, 
                          output_dir: str='wizard_staff_outputs') -> None:
    """
    Plot the average activity of each cluster and the average activity of a specified cluster with std.

    Parameters:
        dff_dat: The dF/F data array.
        filtered_idx: The indices of the data to be filtered.
        sample_name: The name of the sample.
        min_clusters: The minimum number of clusters to try. Default is 2.
        max_clusters: The maximum number of clusters to try. Default is 10.
        random_seed: The seed for random number generation in K-means. Default is 1111111.
        norm: Whether to normalize the data. Default is False.
        show_plots: If True, shows the plots. Default is True.
        save_files: If True, saves the plot to the output directory. Default is True.
        output_dir: Directory where output files will be saved.
    """
    # Filter the data
    data_t = dff_dat[filtered_idx]

    # If empty, return None  # TODO: check with Jesse about this
    if data_t.shape[0] == 0:
        return None

    # Perform K-means clustering
    best_silhouette_score = -1
    best_num_clusters = min_clusters
    best_labels = None
    best_sorted_data_t = None

    # Try K-means with different numbers of clusters
    for num_clusters in range(min_clusters, max_clusters + 1):
            # Suppress warnings of empty clusters from k-means clustering
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            # Perform K-means clustering
            centroids, labels = kmeans2(data_t, num_clusters, seed=random_seed)
        
        # Calculate silhouette score as a measure of clustering quality
        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(data_t, labels)
        else:
            silhouette_avg = -1
            
        # If the sillhouette score is better, update the best values
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = num_clusters
            best_labels = labels
            sorted_indices = np.argsort(labels)
            best_sorted_data_t = data_t[sorted_indices, :]

    # Select the cluster with the highest mean activity as the cluster of interest
    cluster_means = []
    for Ki in range(best_num_clusters):
        js = np.where(best_labels == Ki)
        cluster_mean = np.mean(np.squeeze(data_t[js, :]), axis=0)
        cluster_means.append(np.mean(cluster_mean))
    
    # Select the cluster with the highest mean activity
    cluster_id = np.argmax(cluster_means)  
    
    def normalize(trace):
        """Normalizes the given trace to be between 0 and 1."""
        return (trace - np.min(trace)) / (np.max(trace) - np.min(trace))

    f = plt.figure(figsize=(20, 5))
    # Plot the average activity of each cluster
    plt.subplot(121)

    for Ki in range(best_num_clusters):
        # Find indices of traces where cluster label equals to Ki
        js = np.where(best_labels == Ki)
        # Calculate average activity trace for cluster Ki
        d = np.mean(np.squeeze(data_t[js, :]), axis=0)
        # Normalize if norm is True
        if norm:
            d = normalize(d)
        plt.plot(d + Ki)
    
    p = plt.gca()
    p.set_ylabel('Cluster ID', fontsize=15)
    p.set_xlabel('Frame', fontsize=15)
    p.set_title('Average activity of each cluster', fontsize=20)

    p.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot the average activity of the selected cluster with std
    plt.subplot(122)

    iis = np.where(best_labels == cluster_id)[0]
    m = np.mean(data_t[iis, :], 0)
    s = np.std(data_t[iis, :], 0)
    ts = range(data_t.shape[1])
    plt.plot(m, '-k')
    
    # Create fill between mean-std and mean+std, fill with grey color
    plt.fill_between(ts, m - s, m + s, alpha=0.4, color=(0.1, 0.1, 0.1))
    p = plt.gca()

    p.set_title(f'Average activity of Cluster #{cluster_id} (mean ± std)', fontsize=20)
    p.set_xlabel('Frame', fontsize=15)
    p.set_ylabel('Activity,  \u0394f/f\u2080', fontsize=15)

    p.legend(('mean', 'std'), fontsize=15)

    # Use tight_layout to adjust spaces between subplots
    plt.tight_layout()

    # Save the plot
    if save_files:
        # Expand the user directory if it exists in the output_dir path
        output_dir = os.path.expanduser(output_dir)
        output_dir_pngs = os.path.join(output_dir, 'cluster_activity_plots')

        # Create the output directory if it does not exist
        os.makedirs(output_dir_pngs, exist_ok=True)

        # Save the figure
        fig_path = os.path.join(output_dir_pngs, f'{sample_name}_cluster-activity-plot.png')
        plt.savefig(fig_path, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def overlay_images(im_avg: np.ndarray, binary_overlay: np.ndarray, overlay_color: list=[255, 255, 0]
                   ) -> np.ndarray:
    """
    Create an overlay image with a specific color for the binary overlay and gray for the background.

    Args:
        im_avg: The average image (grayscale).
        binary_overlay: The binary overlay image.
        overlay_color: The RGB color for the binary overlay.

    Returns:
        overlay_image: The combined overlay image.
    """
    # Normalize the grayscale image to the range [0, 255]
    im_avg_norm = (255 * (im_avg - np.min(im_avg)) / (np.max(im_avg) - np.min(im_avg))).astype(np.uint8)
    
    # Convert grayscale image to 3-channel RGB
    im_avg_rgb = np.stack([im_avg_norm] * 3, axis=-1)
    
    # Create an RGB overlay with the specified color
    overlay_rgb = np.zeros_like(im_avg_rgb)
    overlay_rgb[binary_overlay > 0] = overlay_color
    
    # Combine the two images
    combined_image = np.where(binary_overlay[..., None] > 0, overlay_rgb, im_avg_rgb)
    
    return combined_image

def plot_montage(images: List[np.ndarray], im_avg: np.ndarray, grid_shape: Tuple, 
                 overlay_color: list=[255, 255, 0], rescale_intensity: bool=False
                 ) -> np.ndarray:
    """
    Creates a montage from a list of images arranged in a specified grid shape,
    with an overlay color applied to binary images and gray background.

    Args:
        images: List of binary images to be arranged in a montage.
        im_avg: The average image (grayscale) for background.
        grid_shape: Shape of the grid for arranging the images (rows, columns).
        overlay_color: The RGB color for the binary overlay.
        rescale_intensity: Flag to indicate if image intensity should be rescaled. Default is False.

    Returns:
        montage: Montage image.
    """
    # Calculate the shape of the montage grid
    img_height, img_width = im_avg.shape[:2]
    montage_height = grid_shape[0] * img_height
    montage_width = grid_shape[1] * img_width

    # Create an empty array for the montage
    montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

    # Populate the montage array with overlay images
    for idx, img in enumerate(images):
        y = idx // grid_shape[1]
        x = idx % grid_shape[1]
        overlay_img = overlay_images(im_avg, img, overlay_color)
        montage[y * img_height:(y + 1) * img_height, x * img_width:(x + 1) * img_width] = overlay_img

    return montage

def plot_dff_activity(dff_dat: np.ndarray, cnm_idx: np.array, frate: int, sample_name: str, 
                      sz_per_neuron: float=0.5, max_z: float=0.45, begin_tp: int=0, 
                      end_tp: int=-1, n_start: int=0, n_stop: int=-1, dff_bar: float=1, 
                      lw: float=0.5, show_plots: bool=True, save_files: bool=True, 
                      output_dir: str='wizard_staff_outputs') -> None:
    """
    Plots the activity data of neurons within a specified time range.
    
    Parameters:
        dff_dat: Activity data matrix with neurons as rows and time points as columns.
        cnm_idx: Array of neuron IDs corresponding to the rows of dff_dat.
        frate: Frames per second of the data.
        sample_name: The filename of the data. Needed for saving the plot.
        sz_per_neuron: Size of each neuron in the plot.
        max_z: Maximum ΔF/F₀ intensity for scaling the plot.
        begin_tp: Starting time point for the plot.
        end_tp: Ending time point for the plot.
        n_start: Index of the first cell to plot.
        n_stop: Index of the last cell to plot.
        dff_bar: Height of the ΔF/F₀ scale bar.
        lw: Line width of the lines drawn in the plot.
        show_plots: If True, shows the plots. Default is True.
        save_files: If True, saves the plot to the output directory. Default is True.
        output_dir: Directory where the plot will be saved.
    """
    # Ensure valid end_tp
    end_tp = end_tp if end_tp >= 0 else dff_dat.shape[1]
    
    # Sort the data by neuron IDs and select the time range
    sorted_indices = np.argsort(cnm_idx)
    dff_dat_sorted = dff_dat[sorted_indices, begin_tp:end_tp]
    
    # Determine the number of neurons to plot
    if n_stop is None or n_stop < 0:
        n_stop = dff_dat_sorted.shape[0]
    n_neurons = min(n_stop, dff_dat_sorted.shape[0])
    
    # Scale the maximum ΔF/F₀ value
    max_dff_int = max(max_z / 2, 0.25)
    
    # Create a time vector
    time_vector = np.arange(dff_dat_sorted.shape[1]) / frate

    # Calculate plot height
    gr_ht = np.maximum(1, int(dff_dat_sorted.shape[0] * sz_per_neuron))

    # Create a figure for the plot
    _, ax = plt.subplots(1, 1, figsize=(7 / 2, gr_ht / 2), sharey=True)

    # Plot the activity data for each neuron
    for i in range(n_start, n_neurons):
        color_str = f'C{i % 9}'  # Cycle through 9 colors
        ax.plot(time_vector, dff_dat_sorted[i, :] + (n_neurons - i - 1) * max_dff_int, linewidth=lw, color=color_str)
    
    # Draw a vertical line indicating the ΔF/F₀ scale in black
    ax.vlines(x=-1., ymin=0, ymax = dff_bar, lw=2, color='black')
    ax.text(-1.5, dff_bar / 2, f'{dff_bar} ΔF/F₀', ha='center', va='center', rotation='vertical')
    
    # Label the x-axis
    ax.set_xlabel('Time(s)')

    # Hide the top, right, and left spines (borders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Hide the y-axis ticks and labels
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])

    # Adjust the margins of the plot
    ax.margins(0.008)
    
    # Use tight_layout to adjust spaces between subplots
    plt.tight_layout()

    # Save the plot
    if save_files:
        # Expand the user directory if it exists in the output_dir path
        output_dir = os.path.expanduser(output_dir)
        output_dir_pngs = os.path.join(output_dir, 'dff_activity_plots')

        # Create the output directory if it does not exist
        os.makedirs(output_dir_pngs, exist_ok=True)

        # Save the figure
        fig_path = os.path.join(output_dir_pngs, f'{sample_name}_dff-activity-plot.png')
        plt.savefig(fig_path, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()


# Metric configuration for plot_metric_by_group
METRIC_CONFIG = {
    "frpm": {
        "data_attr": "frpm_data",
        "value_col": "Firing Rate Per Min",
        "ylabel": "Firing Rate (spikes/min)",
        "title": "Neuronal Firing Rates",
        "neuron_col": "Neuron Index",
        "time_convertible": False,
    },
    "firing_rate": {
        "data_attr": "frpm_data",
        "value_col": "Firing Rate Per Min",
        "ylabel": "Firing Rate (spikes/min)",
        "title": "Neuronal Firing Rates",
        "neuron_col": "Neuron Index",
        "time_convertible": False,
    },
    "rise_time": {
        "data_attr": "rise_time_data",
        "value_col": "Rise Times",
        "ylabel_frames": "Rise Time (frames)",
        "ylabel_s": "Rise Time (s)",
        "ylabel_ms": "Rise Time (ms)",
        "title": "Calcium Signal Rise Times",
        "neuron_col": "Neuron",
        "time_convertible": True,
    },
    "fwhm": {
        "data_attr": "fwhm_data",
        "value_col": "FWHM Values",
        "ylabel_frames": "FWHM (frames)",
        "ylabel_s": "FWHM (s)",
        "ylabel_ms": "FWHM (ms)",
        "title": "Event Duration (FWHM)",
        "neuron_col": "Neuron",
        "time_convertible": True,
    },
    "spike_counts": {
        "data_attr": "fwhm_data",
        "value_col": "Spike Counts",
        "ylabel": "Spike Count",
        "title": "Spike Counts per Neuron",
        "neuron_col": "Neuron",
        "time_convertible": False,
    },
    "roundness": {
        "data_attr": "mask_metrics_data",
        "value_col": "Roundness",
        "ylabel": "Roundness",
        "title": "Cell/Organoid Roundness",
        "neuron_col": None,  # Already sample-level
        "time_convertible": False,
    },
    "diameter": {
        "data_attr": "mask_metrics_data",
        "value_col": "Diameter",
        "ylabel": "Diameter (pixels)",
        "title": "Cell/Organoid Diameter",
        "neuron_col": None,
        "time_convertible": False,
    },
    "area": {
        "data_attr": "mask_metrics_data",
        "value_col": "Area",
        "ylabel": "Area (pixels²)",
        "title": "Cell/Organoid Area",
        "neuron_col": None,
        "time_convertible": False,
    },
    "silhouette": {
        "data_attr": "silhouette_scores_data",
        "value_col": "Silhouette Score",
        "ylabel": "Silhouette Score",
        "title": "Clustering Quality (Silhouette)",
        "neuron_col": None,
        "time_convertible": False,
    },
}

# Recommended color palettes for different use cases
COLOR_PALETTES = {
    # Categorical palettes (good for distinct groups)
    "categorical": ["tab10", "Set2", "Dark2", "Paired", "Set1"],
    # Sequential palettes (good for ordered groups)  
    "sequential": ["Blues", "Greens", "Oranges", "Purples", "Reds"],
    # Diverging palettes (good for showing deviation from center)
    "diverging": ["RdBu", "PiYG", "PRGn", "BrBG"],
    # Perceptually uniform (good for accessibility)
    "uniform": ["viridis", "plasma", "cividis", "mako"],
}


def plot_metric_by_group(
        data,
        metric: str = "frpm",
        group_col: str = "Well",
        plot_type: str = "bar",
        aggregate: bool = True,
        agg_func: str = "mean",
        frame_rate: float = None,
        time_unit: str = "ms",
        figsize: Tuple[int, int] = (10, 6),
        palette: str = "Set2",
        colors: List = None,
        title: str = None,
        ylabel: str = None,
        xlabel: str = None,
        show_individual_points: bool = True,
        show_plots: bool = True,
        save_files: bool = False,
        output_dir: str = 'wizard_staff_outputs',
        filename: str = None,
    ) -> Optional[plt.Figure]:
    """
    Create publication-ready plots of metrics grouped by experimental conditions.
    
    This is a unified plotting function that works with all Wizards Staff metrics:
    FRPM, Rise Time, FWHM, Spike Counts, and Mask Metrics.
    
    Parameters
    ----------
    data : Orb or pd.DataFrame
        Either a Wizards Staff Orb object or a DataFrame containing the metric data.
        If an Orb is provided, the metric data and frame rate will be extracted automatically.
    metric : str
        Which metric to plot. Options:
        - "frpm" or "firing_rate": Firing Rate Per Minute
        - "rise_time": Calcium signal rise times (convertible to seconds/ms)
        - "fwhm": Full Width at Half Maximum (convertible to seconds/ms)
        - "spike_counts": Number of spikes per neuron
        - "roundness", "diameter", "area": Mask/morphology metrics
        - "silhouette": Clustering quality scores
    group_col : str
        Column name to group data by (e.g., "Well", "Treatment", "Genotype").
        Default is "Well".
    plot_type : str
        Type of plot to create: "bar", "box", or "violin". Default is "bar".
    aggregate : bool
        If True (default), aggregates neuron-level data to sample-level means.
        Set to False to show all individual data points (not recommended for bar plots).
    agg_func : str
        Aggregation function when aggregate=True. Default is "mean".
        Options: "mean", "median", "sum".
    frame_rate : float, optional
        Frame rate in Hz (frames per second) for converting frame-based metrics
        (rise_time, fwhm) to time units. If None and data is an Orb, will attempt
        to extract from metadata 'Frate' column. If not available, keeps frames.
    time_unit : str
        Time unit for converted metrics: "s" for seconds, "ms" for milliseconds.
        Default is "ms" (milliseconds are typically more intuitive for transients).
    figsize : tuple
        Figure size as (width, height). Default is (10, 6).
    palette : str
        Matplotlib colormap name for coloring groups. Default is "Set2".
        Recommended categorical palettes: "Set2", "Dark2", "tab10", "Paired"
        Recommended sequential palettes: "Blues", "Greens", "viridis"
    colors : list, optional
        Explicit list of colors for each group (overrides palette).
        Can be color names ('steelblue'), hex codes ('#1f77b4'), or RGB tuples.
    title : str, optional
        Custom plot title. If None, uses default based on metric.
    ylabel : str, optional
        Custom y-axis label. If None, uses default based on metric and time unit.
    xlabel : str, optional
        Custom x-axis label. If None, uses "Group/Condition".
    show_individual_points : bool
        If True (default), overlays individual sample points on bar/box/violin plots.
    show_plots : bool
        If True (default), displays the plot.
    save_files : bool
        If True, saves the plot to output_dir. Default is False.
    output_dir : str
        Directory for saving plots. Default is 'wizard_staff_outputs'.
    filename : str, optional
        Custom filename for saved plot. If None, generates from metric and group.
    
    Returns
    -------
    matplotlib.figure.Figure or None
        The matplotlib Figure object if successful, None if data is unavailable.
    
    Examples
    --------
    >>> # Basic usage with Orb (auto-extracts frame rate from metadata)
    >>> plot_metric_by_group(orb, metric="frpm", group_col="Treatment")
    
    >>> # FWHM in milliseconds with box plot
    >>> plot_metric_by_group(orb, metric="fwhm", group_col="Genotype", 
    ...                      plot_type="box", time_unit="ms")
    
    >>> # Rise time in seconds with explicit frame rate
    >>> plot_metric_by_group(orb, metric="rise_time", frame_rate=30, time_unit="s")
    
    >>> # Custom colors for specific groups
    >>> plot_metric_by_group(orb, metric="frpm", group_col="Treatment",
    ...                      colors=['#2ecc71', '#e74c3c', '#3498db'])
    
    >>> # Using different palettes
    >>> plot_metric_by_group(orb, metric="fwhm", palette="Dark2")  # Categorical
    >>> plot_metric_by_group(orb, metric="fwhm", palette="Blues")  # Sequential
    
    See Also
    --------
    COLOR_PALETTES : dict
        Dictionary of recommended palettes by use case (categorical, sequential, etc.)
    """
    # Normalize metric name
    metric_key = metric.lower().replace(" ", "_").replace("-", "_")
    
    if metric_key not in METRIC_CONFIG:
        available = list(METRIC_CONFIG.keys())
        raise ValueError(f"Unknown metric '{metric}'. Available: {available}")
    
    config = METRIC_CONFIG[metric_key]
    is_orb = hasattr(data, config["data_attr"])
    
    # Get data - either from Orb or use DataFrame directly
    if is_orb:
        # It's an Orb object
        orb_obj = data
        df = getattr(data, config["data_attr"])
        if df is None:
            print(f"❌ No {metric} data available. Make sure you've run orb.run_all() first.")
            return None
        df = df.copy()
        
        # Try to extract frame rate from metadata if not provided
        if frame_rate is None and hasattr(orb_obj, 'metadata'):
            metadata = orb_obj.metadata
            if 'Frate' in metadata.columns:
                # Get the most common frame rate (they should all be the same)
                frame_rate = metadata['Frate'].mode().iloc[0] if len(metadata['Frate'].mode()) > 0 else None
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        # Try to get frame rate from DataFrame if present
        if frame_rate is None and 'Frate' in df.columns:
            frame_rate = df['Frate'].mode().iloc[0] if len(df['Frate'].mode()) > 0 else None
    else:
        raise TypeError(
            f"'data' must be an Orb object or pandas DataFrame, got {type(data).__name__}"
        )
    
    value_col = config["value_col"]
    neuron_col = config["neuron_col"]
    is_time_convertible = config.get("time_convertible", False)
    
    # Validate columns exist
    if value_col not in df.columns:
        print(f"❌ Column '{value_col}' not found in data.")
        print(f"   Available columns: {list(df.columns)}")
        return None
    
    if group_col not in df.columns:
        print(f"❌ Column '{group_col}' not found in data.")
        print(f"   Available columns: {list(df.columns)}")
        return None
    
    # Drop rows with NaN in key columns
    df = df.dropna(subset=[value_col, group_col])
    
    if len(df) == 0:
        print(f"❌ No data remaining after dropping NaN values.")
        return None
    
    # Convert frame-based metrics to time units if frame_rate is available
    time_converted = False
    if is_time_convertible and frame_rate is not None and frame_rate > 0:
        # Convert frames to time
        if time_unit == "ms":
            df[value_col] = df[value_col] / frame_rate * 1000  # frames -> ms
            time_converted = True
        elif time_unit == "s":
            df[value_col] = df[value_col] / frame_rate  # frames -> seconds
            time_converted = True
        else:
            print(f"⚠️ Unknown time_unit '{time_unit}'. Using frames. Valid options: 's', 'ms'")
    
    # Determine y-axis label based on conversion
    if ylabel is None:
        if is_time_convertible:
            if time_converted and time_unit == "ms":
                auto_ylabel = config.get("ylabel_ms", config.get("ylabel", value_col))
            elif time_converted and time_unit == "s":
                auto_ylabel = config.get("ylabel_s", config.get("ylabel", value_col))
            else:
                auto_ylabel = config.get("ylabel_frames", config.get("ylabel", value_col))
        else:
            auto_ylabel = config.get("ylabel", value_col)
    else:
        auto_ylabel = ylabel
    
    # Aggregate to sample level if requested and data is neuron-level
    if aggregate and neuron_col is not None and neuron_col in df.columns:
        # Aggregate by Sample and group_col
        agg_cols = ['Sample']
        if group_col != 'Sample':
            agg_cols.append(group_col)
        
        # Keep only unique combinations for grouping
        group_df = df[agg_cols].drop_duplicates()
        
        # Perform aggregation
        if agg_func == "mean":
            agg_values = df.groupby('Sample')[value_col].mean()
        elif agg_func == "median":
            agg_values = df.groupby('Sample')[value_col].median()
        elif agg_func == "sum":
            agg_values = df.groupby('Sample')[value_col].sum()
        else:
            raise ValueError(f"Unknown agg_func '{agg_func}'. Use 'mean', 'median', or 'sum'.")
        
        # Merge back with group info
        df = group_df.set_index('Sample').join(agg_values).reset_index()
        df = df.rename(columns={value_col: value_col})  # Ensure column name preserved
    
    # Calculate group statistics
    group_means = df.groupby(group_col)[value_col].mean().sort_values()
    group_stds = df.groupby(group_col)[value_col].std()
    group_order = group_means.index.tolist()
    
    # Set up colors
    n_groups = len(group_order)
    
    if colors is not None:
        # Use explicitly provided colors
        if len(colors) < n_groups:
            # Cycle through provided colors if not enough
            colors_list = [colors[i % len(colors)] for i in range(n_groups)]
        else:
            colors_list = colors[:n_groups]
    else:
        # Use colormap/palette
        try:
            cmap = plt.get_cmap(palette)
            # For qualitative colormaps, sample evenly; for continuous, spread across range
            if hasattr(cmap, 'colors'):
                # Qualitative colormap (like Set2, tab10)
                colors_list = [cmap(i % len(cmap.colors)) for i in range(n_groups)]
            else:
                # Continuous colormap
                colors_list = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]
        except ValueError:
            print(f"⚠️ Unknown palette '{palette}'. Using 'Set2'.")
            cmap = plt.get_cmap('Set2')
            colors_list = [cmap(i % 8) for i in range(n_groups)]
    
    color_map = dict(zip(group_order, colors_list))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type == "bar":
        # Bar plot with error bars
        x_pos = range(len(group_order))
        bars = ax.bar(
            x_pos, 
            group_means.values,
            yerr=group_stds[group_order].values,
            color=[color_map[g] for g in group_order],
            capsize=5,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.8
        )
        
        # Overlay individual points
        if show_individual_points:
            for i, group in enumerate(group_order):
                group_data = df[df[group_col] == group][value_col]
                jitter = np.random.uniform(-0.15, 0.15, size=len(group_data))
                ax.scatter(
                    i + jitter, 
                    group_data, 
                    color='black', 
                    alpha=0.5, 
                    s=30, 
                    zorder=5,
                    edgecolor='white',
                    linewidth=0.5
                )
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_order, rotation=45, ha='right')
        
    elif plot_type == "box":
        # Box plot
        box_data = [df[df[group_col] == g][value_col].dropna().values for g in group_order]
        bp = ax.boxplot(
            box_data,
            labels=group_order,
            patch_artist=True,
            notch=False,
            showfliers=not show_individual_points  # Hide outliers if showing points
        )
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Overlay individual points
        if show_individual_points:
            for i, group in enumerate(group_order):
                group_data = df[df[group_col] == group][value_col]
                jitter = np.random.uniform(-0.15, 0.15, size=len(group_data))
                ax.scatter(
                    i + 1 + jitter, 
                    group_data, 
                    color='black', 
                    alpha=0.5, 
                    s=30, 
                    zorder=5,
                    edgecolor='white',
                    linewidth=0.5
                )
        
        ax.set_xticklabels(group_order, rotation=45, ha='right')
        
    elif plot_type == "violin":
        # Violin plot
        violin_data = [df[df[group_col] == g][value_col].dropna().values for g in group_order]
        
        # Filter out empty arrays
        valid_indices = [i for i, d in enumerate(violin_data) if len(d) > 0]
        violin_data_filtered = [violin_data[i] for i in valid_indices]
        group_order_filtered = [group_order[i] for i in valid_indices]
        colors_filtered = [colors_list[i] for i in valid_indices]
        
        if len(violin_data_filtered) == 0:
            print("❌ No valid data for violin plot.")
            return None
        
        parts = ax.violinplot(
            violin_data_filtered,
            positions=range(1, len(group_order_filtered) + 1),
            showmeans=True,
            showmedians=False,
            showextrema=True
        )
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_filtered[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
        
        # Style the lines
        for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1)
        
        # Overlay individual points
        if show_individual_points:
            for i, group in enumerate(group_order_filtered):
                group_data = df[df[group_col] == group][value_col]
                jitter = np.random.uniform(-0.15, 0.15, size=len(group_data))
                ax.scatter(
                    i + 1 + jitter, 
                    group_data, 
                    color='black', 
                    alpha=0.5, 
                    s=30, 
                    zorder=5,
                    edgecolor='white',
                    linewidth=0.5
                )
        
        ax.set_xticks(range(1, len(group_order_filtered) + 1))
        ax.set_xticklabels(group_order_filtered, rotation=45, ha='right')
        
    else:
        raise ValueError(f"Unknown plot_type '{plot_type}'. Use 'bar', 'box', or 'violin'.")
    
    # Labels and title
    ax.set_xlabel(xlabel if xlabel else 'Group/Condition', fontsize=12)
    ax.set_ylabel(auto_ylabel, fontsize=12)
    ax.set_title(title if title else f"{config['title']} by {group_col}", fontsize=14)
    
    # Clean up appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save if requested
    if save_files:
        output_dir = os.path.expanduser(output_dir)
        output_dir_plots = os.path.join(output_dir, 'metric_plots')
        os.makedirs(output_dir_plots, exist_ok=True)
        
        if filename is None:
            filename = f'{metric_key}_by_{group_col}.png'
        
        fig_path = os.path.join(output_dir_plots, filename)
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"✅ Plot saved to: {fig_path}")
    
    if show_plots:
        plt.show()
        plt.close(fig)  # Close to prevent Jupyter from displaying it twice
        return None  # Already displayed
    else:
        # Don't show, return figure for programmatic use (saving, further modification)
        return fig

def plot_paired_lines(
    data: pd.DataFrame,
    metric: str,
    baseline_col: str,
    dosing_col: str,
    group_col: str = None,
    pair_id_col: str = "pair_id",
    title: str = None,
    ylabel: str = None,
    figsize: Tuple[int, int] = (8, 6),
    palette: str = "Set2",
    colors: List = None,
    line_alpha: float = 0.6,
    line_width: float = 1.5,
    marker_size: float = 80,
    show_plots: bool = True,
    save_files: bool = False,
    output_dir: str = "drug_response_outputs",
    filename: str = None,
) -> Optional[plt.Figure]:
    """
    Create paired line plot showing baseline to dosing changes for each sample.
    
    Lines connect each sample's baseline and dosing values, making it easy to
    visualize the direction and magnitude of drug effects for individual samples.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame from compare_baseline_dosing() containing paired data.
    metric : str
        Name of the metric (used for labeling and filename).
    baseline_col : str
        Column name containing baseline values.
    dosing_col : str
        Column name containing dosing values.
    group_col : str, optional
        Column for coloring lines by group (e.g., "Treatment").
    pair_id_col : str, default "pair_id"
        Column identifying each sample pair.
    title : str, optional
        Plot title. If None, auto-generated from metric.
    ylabel : str, optional
        Y-axis label. If None, uses metric name.
    figsize : tuple, default (8, 6)
        Figure size as (width, height).
    palette : str, default "Set2"
        Matplotlib colormap for coloring groups.
    colors : list, optional
        Explicit list of colors (overrides palette).
    line_alpha : float, default 0.6
        Transparency of connecting lines.
    line_width : float, default 1.5
        Width of connecting lines.
    marker_size : float, default 80
        Size of data point markers.
    show_plots : bool, default True
        If True, display the plot.
    save_files : bool, default False
        If True, save the plot to output_dir.
    output_dir : str, default "drug_response_outputs"
        Directory for saving plots.
    filename : str, optional
        Custom filename. If None, auto-generated.
    
    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show_plots=False, else None.
    
    Examples
    --------
    >>> plot_paired_lines(
    ...     data=results["frpm"],
    ...     metric="frpm",
    ...     baseline_col="baseline_Firing Rate Per Min",
    ...     dosing_col="dosing_Firing Rate Per Min",
    ...     group_col="Treatment"
    ... )
    """
    if data is None or len(data) == 0:
        print(f"❌ No data available for paired line plot.")
        return None
    
    # Validate columns
    for col in [baseline_col, dosing_col]:
        if col not in data.columns:
            print(f"❌ Column '{col}' not found in data.")
            return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up colors
    if group_col is not None and group_col in data.columns:
        groups = data[group_col].unique()
        n_groups = len(groups)
        
        if colors is not None:
            colors_list = colors[:n_groups] if len(colors) >= n_groups else colors * (n_groups // len(colors) + 1)
        else:
            cmap = plt.get_cmap(palette)
            if hasattr(cmap, 'colors'):
                colors_list = [cmap(i % len(cmap.colors)) for i in range(n_groups)]
            else:
                colors_list = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]
        
        color_map = dict(zip(groups, colors_list))
        
        # Plot each group
        for group in groups:
            group_data = data[data[group_col] == group]
            color = color_map[group]
            
            for _, row in group_data.iterrows():
                ax.plot(
                    [0, 1], 
                    [row[baseline_col], row[dosing_col]],
                    color=color, 
                    alpha=line_alpha, 
                    linewidth=line_width,
                    zorder=1
                )
            
            # Add scatter points
            ax.scatter(
                [0] * len(group_data), 
                group_data[baseline_col],
                color=color, 
                s=marker_size, 
                label=f"{group} (n={len(group_data)})",
                edgecolor='white',
                linewidth=0.5,
                zorder=2
            )
            ax.scatter(
                [1] * len(group_data), 
                group_data[dosing_col],
                color=color, 
                s=marker_size,
                edgecolor='white',
                linewidth=0.5,
                zorder=2
            )
        
        ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        # No grouping - use single color
        default_color = '#3498db'
        
        for _, row in data.iterrows():
            ax.plot(
                [0, 1], 
                [row[baseline_col], row[dosing_col]],
                color=default_color, 
                alpha=line_alpha, 
                linewidth=line_width,
                zorder=1
            )
        
        ax.scatter(
            [0] * len(data), 
            data[baseline_col],
            color=default_color, 
            s=marker_size,
            edgecolor='white',
            linewidth=0.5,
            zorder=2
        )
        ax.scatter(
            [1] * len(data), 
            data[dosing_col],
            color=default_color, 
            s=marker_size,
            edgecolor='white',
            linewidth=0.5,
            zorder=2
        )
    
    # Customize axes
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline', 'Dosing'], fontsize=12)
    ax.set_xlim(-0.3, 1.3)
    
    # Labels
    ax.set_ylabel(ylabel if ylabel else metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title if title else f"Paired Comparison: {metric.replace('_', ' ').title()}", fontsize=14)
    
    # Clean up appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save if requested
    if save_files:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            filename = f"drug_response_{metric}_paired_lines.png"
        
        fig_path = os.path.join(output_dir, filename)
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"  ✅ Plot saved: {fig_path}")
    
    if show_plots:
        plt.show()
        plt.close(fig)
        return None
    else:
        return fig


def plot_fold_change_distribution(
    data: pd.DataFrame,
    metric: str,
    value_col: str,
    group_col: str = None,
    normalization: str = "fold_change",
    plot_type: str = "violin",
    title: str = None,
    ylabel: str = None,
    figsize: Tuple[int, int] = (8, 6),
    palette: str = "Set2",
    colors: List = None,
    reference_line: float = None,
    show_individual_points: bool = True,
    show_plots: bool = True,
    save_files: bool = False,
    output_dir: str = "drug_response_outputs",
    filename: str = None,
) -> Optional[plt.Figure]:
    """
    Create violin/box plot showing distribution of fold changes or other normalized values.
    
    Includes a reference line at the "no change" value (1.0 for fold change, 0 for
    percent change/delta) to easily identify samples with increased vs decreased activity.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame from compare_baseline_dosing() containing normalized data.
    metric : str
        Name of the metric (for labeling and filename).
    value_col : str
        Column containing normalized values (e.g., "fold_change").
    group_col : str, optional
        Column for grouping the distribution (e.g., "Treatment").
    normalization : str, default "fold_change"
        Type of normalization (used to determine reference line).
    plot_type : str, default "violin"
        Type of distribution plot: "violin", "box", or "bar".
    title : str, optional
        Plot title.
    ylabel : str, optional
        Y-axis label.
    figsize : tuple, default (8, 6)
        Figure size.
    palette : str, default "Set2"
        Colormap for groups.
    colors : list, optional
        Explicit colors list.
    reference_line : float, optional
        Y-value for reference line. If None, auto-determined from normalization.
    show_individual_points : bool, default True
        If True, overlay individual data points.
    show_plots : bool, default True
        If True, display the plot.
    save_files : bool, default False
        If True, save to output_dir.
    output_dir : str, default "drug_response_outputs"
        Output directory.
    filename : str, optional
        Custom filename.
    
    Returns
    -------
    matplotlib.figure.Figure or None
        Figure if show_plots=False, else None.
    """
    if data is None or len(data) == 0:
        print(f"❌ No data available for fold change distribution plot.")
        return None
    
    if value_col not in data.columns:
        print(f"❌ Column '{value_col}' not found in data.")
        return None
    
    # Determine reference line value
    if reference_line is None:
        reference_values = {
            "fold_change": 1.0,
            "percent_change": 0.0,
            "delta": 0.0,
            "log2_fold_change": 0.0,
        }
        reference_line = reference_values.get(normalization, 1.0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for plotting
    if group_col is not None and group_col in data.columns:
        groups = sorted(data[group_col].unique())
        n_groups = len(groups)
        plot_data = [data[data[group_col] == g][value_col].dropna().values for g in groups]
        x_labels = groups
    else:
        plot_data = [data[value_col].dropna().values]
        groups = ["All Samples"]
        n_groups = 1
        x_labels = ["All"]
    
    # Set up colors
    if colors is not None:
        colors_list = colors[:n_groups] if len(colors) >= n_groups else colors * (n_groups // len(colors) + 1)
    else:
        cmap = plt.get_cmap(palette)
        if hasattr(cmap, 'colors'):
            colors_list = [cmap(i % len(cmap.colors)) for i in range(n_groups)]
        else:
            colors_list = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]
    
    # Filter empty groups
    valid_indices = [i for i, d in enumerate(plot_data) if len(d) > 0]
    plot_data = [plot_data[i] for i in valid_indices]
    x_labels = [x_labels[i] for i in valid_indices]
    colors_list = [colors_list[i] for i in valid_indices]
    
    if len(plot_data) == 0:
        print("❌ No valid data for plotting.")
        return None
    
    positions = range(1, len(plot_data) + 1)
    
    if plot_type == "violin":
        parts = ax.violinplot(
            plot_data,
            positions=positions,
            showmeans=True,
            showmedians=False,
            showextrema=True
        )
        
        # Color violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_list[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
        
        for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1)
    
    elif plot_type == "box":
        bp = ax.boxplot(
            plot_data,
            positions=positions,
            patch_artist=True,
            notch=False,
            showfliers=not show_individual_points
        )
        
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    elif plot_type == "bar":
        means = [np.nanmean(d) for d in plot_data]
        stds = [np.nanstd(d) for d in plot_data]
        
        ax.bar(
            positions, means, 
            yerr=stds,
            color=colors_list,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.8,
            capsize=5
        )
    
    else:
        raise ValueError(f"Unknown plot_type '{plot_type}'. Use 'violin', 'box', or 'bar'.")
    
    # Overlay individual points
    if show_individual_points:
        for i, (pos, vals) in enumerate(zip(positions, plot_data)):
            jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(
                pos + jitter, vals,
                color='black', alpha=0.5, s=30, zorder=5,
                edgecolor='white', linewidth=0.5
            )
    
    # Add reference line
    ax.axhline(y=reference_line, color='red', linestyle='--', linewidth=1.5, 
               label=f'No change ({reference_line})', alpha=0.8)
    
    # Customize axes
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Determine y-axis label
    if ylabel is None:
        ylabel_map = {
            "fold_change": "Fold Change",
            "percent_change": "Percent Change (%)",
            "delta": "Delta (Dosing - Baseline)",
            "log2_fold_change": "Log₂ Fold Change",
        }
        ylabel = ylabel_map.get(normalization, value_col)
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Group' if group_col else '', fontsize=12)
    ax.set_title(
        title if title else f"{metric.replace('_', ' ').title()}: {normalization.replace('_', ' ').title()}", 
        fontsize=14
    )
    
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save if requested
    if save_files:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            filename = f"drug_response_{metric}_{normalization}_distribution.png"
        
        fig_path = os.path.join(output_dir, filename)
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"  ✅ Plot saved: {fig_path}")
    
    if show_plots:
        plt.show()
        plt.close(fig)
        return None
    else:
        return fig


def plot_baseline_vs_dosing_scatter(
    data: pd.DataFrame,
    metric: str,
    baseline_col: str,
    dosing_col: str,
    group_col: str = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: Tuple[int, int] = (8, 8),
    palette: str = "Set2",
    colors: List = None,
    marker_size: float = 80,
    alpha: float = 0.7,
    show_identity_line: bool = True,
    show_plots: bool = True,
    save_files: bool = False,
    output_dir: str = "drug_response_outputs",
    filename: str = None,
) -> Optional[plt.Figure]:
    """
    Create scatter plot with baseline on X-axis and dosing on Y-axis.
    
    Includes a diagonal y=x identity line. Points above the line indicate 
    increase with drug treatment; points below indicate decrease.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame from compare_baseline_dosing().
    metric : str
        Name of the metric.
    baseline_col : str
        Column with baseline values.
    dosing_col : str
        Column with dosing values.
    group_col : str, optional
        Column for coloring points by group.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    figsize : tuple, default (8, 8)
        Figure size (square recommended).
    palette : str, default "Set2"
        Colormap.
    colors : list, optional
        Explicit colors.
    marker_size : float, default 80
        Marker size.
    alpha : float, default 0.7
        Marker transparency.
    show_identity_line : bool, default True
        If True, show y=x diagonal line.
    show_plots : bool, default True
        Display plot.
    save_files : bool, default False
        Save plot.
    output_dir : str, default "drug_response_outputs"
        Output directory.
    filename : str, optional
        Custom filename.
    
    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if data is None or len(data) == 0:
        print(f"❌ No data available for scatter plot.")
        return None
    
    for col in [baseline_col, dosing_col]:
        if col not in data.columns:
            print(f"❌ Column '{col}' not found in data.")
            return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get data ranges for identity line
    all_vals = pd.concat([data[baseline_col], data[dosing_col]]).dropna()
    min_val = all_vals.min()
    max_val = all_vals.max()
    padding = (max_val - min_val) * 0.05
    
    # Plot identity line first (background)
    if show_identity_line:
        line_range = [min_val - padding, max_val + padding]
        ax.plot(line_range, line_range, 'k--', linewidth=1.5, alpha=0.5, 
                label='No change (y=x)', zorder=1)
    
    # Plot scatter points
    if group_col is not None and group_col in data.columns:
        groups = sorted(data[group_col].unique())
        n_groups = len(groups)
        
        if colors is not None:
            colors_list = colors[:n_groups] if len(colors) >= n_groups else colors * (n_groups // len(colors) + 1)
        else:
            cmap = plt.get_cmap(palette)
            if hasattr(cmap, 'colors'):
                colors_list = [cmap(i % len(cmap.colors)) for i in range(n_groups)]
            else:
                colors_list = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]
        
        for i, group in enumerate(groups):
            group_data = data[data[group_col] == group]
            ax.scatter(
                group_data[baseline_col], 
                group_data[dosing_col],
                c=[colors_list[i]], 
                s=marker_size, 
                alpha=alpha,
                label=f"{group} (n={len(group_data)})",
                edgecolor='white',
                linewidth=0.5,
                zorder=2
            )
        
        ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        ax.scatter(
            data[baseline_col], 
            data[dosing_col],
            c='#3498db', 
            s=marker_size, 
            alpha=alpha,
            edgecolor='white',
            linewidth=0.5,
            zorder=2
        )
    
    # Set equal aspect ratio and limits
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)
    ax.set_aspect('equal')
    
    # Labels
    metric_label = metric.replace("_", " ").title()
    ax.set_xlabel(xlabel if xlabel else f"Baseline {metric_label}", fontsize=12)
    ax.set_ylabel(ylabel if ylabel else f"Dosing {metric_label}", fontsize=12)
    ax.set_title(title if title else f"{metric_label}: Baseline vs Dosing", fontsize=14)
    
    # Add annotation for interpretation
    ax.text(
        0.02, 0.98, "↑ Points above line = increase with drug",
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        color='green', alpha=0.8
    )
    ax.text(
        0.02, 0.94, "↓ Points below line = decrease with drug",
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        color='red', alpha=0.8
    )
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save if requested
    if save_files:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            filename = f"drug_response_{metric}_scatter.png"
        
        fig_path = os.path.join(output_dir, filename)
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
        print(f"  ✅ Plot saved: {fig_path}")
    
    if show_plots:
        plt.show()
        plt.close(fig)
        return None
    else:
        return fig