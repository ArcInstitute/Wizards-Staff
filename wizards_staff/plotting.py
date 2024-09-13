from skimage.io import imread
from sklearn.metrics import silhouette_score
from scipy.cluster.vq import kmeans2
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator 
from matplotlib import colors
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import os
import numpy as np
import random

def plot_spatial_activity_map(im_min, cnm_A, cnm_idx, raw_filename, p_th=75, 
                        min_clusters=2, max_clusters=10, random_seed = 1111111,
                        show_plots=True, save_files=False,
                        clustering = False, dff_data = None, output_dir='./wizard_staff_outputs'):
    """
    Plots the activity of neurons by overlaying the spatial footprints on a single image.
    
    Parameters:
    im_min (ndarray): Minimum intensity image for overlay.
    cnm_A (ndarray): Spatial footprint matrix of neurons.
    cnm_idx (ndarray): Indices of accepted components.
    raw_filename (str): The raw filename of the image.
    p_th (float): Percentile threshold for image processing.
    min_clusters (int): The minimum number of clusters to try. Default is 2.
    max_clusters (int): The maximum number of clusters to try. Default is 10.
    random_seed (int): The seed for random number generation in K-means. Default is 1111111.
    show_plots (bool): If True, shows the plots. Default is True.
    save_files (bool): If True, saves the overlay image to the output directory. Default is True.
    clustering (bool): If True, perform K-means clustering and color ROIs by cluster.
    dff_data (ndarray): The dF/F data array, required if clustering=True.
    output_dir (str): Directory where output files will be saved.
    
    Returns:
    overlay_image (ndarray): The combined overlay image.
    """
    # Load the minimum intensity image
    im_min = imread(im_min)
    im_shape = im_min.shape
    im_sz = [im_shape[0], im_shape[1]]
    
    # Initialize an image to store the overlay of all neuron activities
    overlay_image = np.zeros((im_sz[0], im_sz[1], 3), dtype=np.uint8)
    
    # Generate color map using nipyspectral
    cmap = plt.get_cmap('nipy_spectral')
    norm = colors.Normalize(vmin=0, vmax=len(cnm_idx))

    if clustering and dff_data is not None:
        # Perform clustering on the filtered data
        data_t = dff_data[cnm_idx]

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
            silhouette_avg = silhouette_score(data_t, labels)

            # Update if this is the best silhouette score
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_num_clusters = num_clusters
                best_labels = labels
        print(f"Best number of clusters: {best_num_clusters}")
        print(f"Best silhouette score: {best_silhouette_score}")
        
        # Assign colors to clusters
        unique_clusters = np.unique(best_labels)
        cluster_colors = {}
        for cluster in unique_clusters:
            color_idx = random.uniform(0.0, 1.0)  # Random float between 0 and 1
            color = np.array(cmap(color_idx)[:3]) * 255  # Select a random color from the full colormap
            cluster_colors[cluster] = color

        print(f"Best labels: {best_labels}")
        print(f"Cluster colors: {cluster_colors}")

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
        if clustering and dff_data is not None:
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
        if clustering and dff_data is not None:
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
        if clustering and dff_data is not None:
            overlay_image_path = os.path.join(output_dir_pngs, f'{raw_filename}_clustered_activity_overlay.png')
        else:
            overlay_image_path = os.path.join(output_dir_pngs, f'{raw_filename}_activity_overlay.png')
        
        # Create a new figure for saving
        plt.figure(figsize=(10, 10))
        plt.imshow(im_min, cmap='gray')  # Plot the min projection image in grayscale
        plt.imshow(overlay_image, alpha=0.6)  # Overlay the spatial activity map with transparency
        plt.axis('off')  # Turn off axis labels

        # Save the overlay image
        plt.savefig(overlay_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        # print(f'Overlay image saved to {overlay_image_path}')
    

def plot_kmeans_heatmap(dff_data, filtered_idx, raw_filename,  output_dir='./wizard_staff_outputs', 
                        min_clusters=2, max_clusters=10, random_seed=1111111, show_plots=True, save_files = True):
    """
    Plots K-means clustering of the given data and outputs synchronization metrics and clustering information to a spreadsheet.

    Parameters:
    dff_data (np.ndarray): The dF/F data array.
    filtered_idx (np.ndarray): The indices of the data to be filtered.
    min_clusters (int): The minimum number of clusters to try. Default is 2.
    max_clusters (int): The maximum number of clusters to try. Default is 10.
    random_seed (int): The seed for random number generation in K-means. Default is 1111111.
    raw_filename (str): The raw filename of the data.
    output_dir (str): Directory where output files will be saved.
    """
    # Filter the data
    data_t = dff_data[filtered_idx]

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
        silhouette_avg = silhouette_score(data_t, labels)
            
        # print(f'Clusters: {num_clusters}, Silhouette Score: {silhouette_avg}')

        # If the sillhouette score is better, update the best values
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = num_clusters
            best_labels = labels
            sorted_indices = np.argsort(labels)
            best_sorted_data_t = data_t[sorted_indices, :]

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
        rect = patches.Rectangle((0, cluster_offset), 50, num_cells_in_cluster, linewidth=1, edgecolor='none', facecolor=cmap[cluster_idx])
        ax2.text(10, cluster_offset + num_cells_in_cluster / 2, str(cluster_idx), color='k', weight='bold')
        ax2.add_patch(rect)
        cluster_offset += num_cells_in_cluster

    ax2.text(10, -5, '↓ Cluster ID', fontsize=10)

    # Set axis labels
    ax1.set_xlabel('Frame')
    ax2.set_xlabel('Frame')
    ax1.set_ylabel('ROI')
    ax2.set_ylabel('ROI')

    # Save the plot
    if save_files==True:
        # Expand the user directory if it exists in the output_dir path
        output_dir = os.path.expanduser(output_dir)
        output_dir_pngs = os.path.join(output_dir, 'kmeans_heatmap_plots')

        # Create the output directory if it does not exist
        os.makedirs(output_dir_pngs, exist_ok=True)

        # Save the figure
        fig_path = os.path.join(output_dir_pngs, f'{raw_filename}_kmeans_clustering_plot.png')
        plt.savefig(fig_path, bbox_inches='tight')
    # print(f'Plot saved as {raw_filename}_kmeans_clustering_plot.png')

    # Display the plot
    if show_plots==True:
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
        csv_path = os.path.join(output_dir_csv, f'{raw_filename}_clustering_info.csv')
        df.to_csv(csv_path, index=False)

    return best_silhouette_score, best_num_clusters

def plot_cluster_activity(dff_data, filtered_idx, raw_filename, min_clusters=2, max_clusters=10, random_seed=1111111, 
                            norm=False, show_plots=True, save_files = True, output_dir='./wizard_staff_outputs'):
    """
    Plots the average activity of each cluster and the average activity of a specified cluster with std.

    Parameters:
    dff_data (np.ndarray): The dF/F data array.
    filtered_idx (np.ndarray): The indices of the data to be filtered.
    num_clusters (int): The number of clusters. Default is 5.
    cluster_id (int): The specific cluster ID to plot the average activity and std. Default is 2.
    random_seed (int): The seed for random number generation in K-means. Default is 1111111.
    norm (bool): Whether to normalize the data. Default is False.
    raw_filename (str): The raw filename of the data. 
    output_dir (str): Directory where output files will be saved.
    """
    # Filter the data
    data_t = dff_data[filtered_idx]

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
        silhouette_avg = silhouette_score(data_t, labels)
            
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
    
    cluster_id = np.argmax(cluster_means)  # Select the cluster with the highest mean activity
    
    def normalize(trace):
        """Normalizes the given trace to be between 0 and 1."""
        return (trace - np.min(trace)) / (np.max(trace) - np.min(trace))

    f = plt.figure(figsize=(20, 5))
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
    if save_files==True:
        # Expand the user directory if it exists in the output_dir path
        output_dir = os.path.expanduser(output_dir)
        output_dir_pngs = os.path.join(output_dir, 'cluster_activity_plots')

        # Create the output directory if it does not exist
        os.makedirs(output_dir_pngs, exist_ok=True)

        # Save the figure
        fig_path = os.path.join(output_dir_pngs, f'{raw_filename}_cluster_activity_plot.png')
        plt.savefig(fig_path, bbox_inches='tight')
    
    if show_plots==True:
        plt.show()
    else:
        plt.close()

def overlay_images(im_avg, binary_overlay, overlay_color=[255, 255, 0]):
    """
    Create an overlay image with a specific color for the binary overlay and gray for the background.

    Parameters:
    im_avg (ndarray): The average image (grayscale).
    binary_overlay (ndarray): The binary overlay image.
    overlay_color (list): The RGB color for the binary overlay.

    Returns:
    overlay_image (ndarray): The combined overlay image.
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

def plot_montage(images, im_avg, grid_shape, overlay_color=[255, 255, 0], rescale_intensity=False):
    """
    Creates a montage from a list of images arranged in a specified grid shape,
    with an overlay color applied to binary images and gray background.

    Parameters:
    images (list of ndarray): List of binary images to be arranged in a montage.
    im_avg (ndarray): The average image (grayscale) for background.
    grid_shape (tuple): Shape of the grid for arranging the images (rows, columns).
    overlay_color (list): The RGB color for the binary overlay.
    rescale_intensity (bool): Flag to indicate if image intensity should be rescaled. Default is False.

    Returns:
    montage (ndarray): Montage image.
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

def plot_dff_activity(act_dat, act_filt_nsp_ids, max_dff_int, begin_tp, end_tp, sz_per_neuron, analysis_dir, base_fname, n_start=0, n_stop=-1, dff_bar=1, frate=30, lw=.2):
    """
    Plots the activity data of neurons within a specified time range.
    
    Parameters:
    act_dat (ndarray): Activity data matrix with neurons as rows and time points as columns.
    act_filt_nsp_ids (array): Array of neuron IDs corresponding to the rows of act_dat.
    max_dff_int (float): Maximum ΔF/F intensity for scaling the plot.
    begin_tp (int): Starting time point for the plot.
    end_tp (int): Ending time point for the plot.
    n_start (int): Index of the first cell to plot.
    n_stop (int): Index of the last cell to plot.
    dff_bar (float): Height of the ΔF/F scale bar.
    frate (int): Frames per second of the data.
    lw (float): Line width of the plot.
    analysis_dir (str): Directory where the plot will be saved.
    base_fname (str): Base filename for the plot.
    """
    # Ensure valid end_tp
    end_tp = end_tp if end_tp >= 0 else act_dat.shape[1]
    
    # Sort the data by neuron IDs and select the time range
    sorted_indices = np.argsort(act_filt_nsp_ids)
    act_dat_sorted = act_dat[sorted_indices, begin_tp:end_tp]
    act_filt_nsp_ids_sorted = act_filt_nsp_ids[sorted_indices]
    
    # Determine the number of neurons to plot
    if n_stop is None or n_stop < 0:
        n_stop = act_dat_sorted.shape[0]
    n_neurons = min(n_stop, act_dat_sorted.shape[0])
    
    # Scale the maximum ΔF/F value
    max_dff_int = max(max_dff_int / 2, 0.25)
    
    # Create a time vector
    time_vector = np.arange(act_dat_sorted.shape[1]) / frate

    # Calculate plot height
    gr_ht = np.maximum(1, int(act_dat_sorted.shape[0] * sz_per_neuron))

    # Create a figure for the plot
    fig, ax = plt.subplots(1, 1, figsize=(7 / 2, gr_ht / 2), sharey=True)

    # Plot the activity data for each neuron
    for i in range(n_start, n_neurons):
        color_str = f'C{i % 9}'  # Cycle through 9 colors
        ax.plot(time_vector, act_dat_sorted[i, :] + (n_neurons - i - 1) * max_dff_int, linewidth=lw, color=color_str)
    
    # Draw a vertical line indicating the ΔF/F scale in black
    ax.vlines(x=-1., ymin=0, ymax=dff_bar, lw=2, color='black')
    ax.text(-1.5, dff_bar / 2, f'{dff_bar} ΔF/F', ha='center', va='center', rotation='vertical')
    
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
    if save_files==True:
        # Expand the user directory if it exists in the output_dir path
        output_dir = os.path.expanduser(output_dir)
        output_dir_pngs = os.path.join(output_dir, 'cluster_activity_plots')

        # Create the output directory if it does not exist
        os.makedirs(output_dir_pngs, exist_ok=True)

        # Save the figure
        fig_path = os.path.join(output_dir_pngs, f'{raw_filename}_cluster_activity_plot.png')
        plt.savefig(fig_path, bbox_inches='tight')
    
    if show_plots==True:
        plt.show()
    else:
        plt.close()