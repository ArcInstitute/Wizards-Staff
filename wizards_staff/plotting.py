# import
## batteries
import os
import random
import logging
import warnings
from typing import List, Tuple
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
                              output_dir: str='wizard_staff_outputs') -> np.ndarray:
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
            logging.warning('No clusters found. No overlay image will be generated.')
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
            overlay_image_path = os.path.join(output_dir_pngs, f'{sample_Name}_activity-overlay.png')
        
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
                        save_files: bool=True) -> Tuple[float, int]:
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
        logging.warning('Only one cluster found. No kmeans heatmap plot will be generated.')
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