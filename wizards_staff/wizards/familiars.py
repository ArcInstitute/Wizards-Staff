# import
## batteries
import os
import sys
import warnings
from typing import List, Tuple
from collections import defaultdict
## 3rd party
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
## package
from wizards_staff.plotting import plot_montage

# Suppress RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# functions    
def load_and_filter_files(categorized_files: dict, p_th: int=75, size_threshold: int=20000
                          ) -> Tuple[dict, dict]:
    """
    Loads dF/F0 output data and filter cell IDs based on spatial filtering criteria.

    Args:
        categorized_files: Dictionary mapping filenames to their corresponding file paths.
        p_th: Percentile threshold for image processing. Default is 75.
        size_threshold: Size threshold for filtering out noise events. Default is 20000.

    Returns:
        d_dff: Dictionary where each key is a filename and the value is the loaded dF/F data matrix.
        d_nspIDs: Dictionary where each key is a filename and the value is the list of filtered neuron IDs.
    """
    # TODO: update for Orb data structure
    d_dff = {}
    d_nspIDs = {}

    for raw_filename, _ in categorized_files.items():
        file_data = load_required_files(categorized_files, raw_filename)
        if not file_data:
            continue
        
        # Load dF/F0 data
        d_dff[raw_filename] = file_data['dff_f_mean']  # dff_dat

        filtered_idx = spatial_filtering(
            cn_filter= file_data['cn_filter_img'], p_th=p_th, size_threshold=size_threshold, 
            cnm_A=file_data['cnm_A'], cnm_idx=file_data['cnm_idx'], im_min = file_data['im_min'], plot=False, silence=True
        )
        d_nspIDs[raw_filename] = filtered_idx

    return d_dff, d_nspIDs

def spatial_filtering(p_th: float, size_threshold: int, cnm_A: np.ndarray, cnm_idx: np.ndarray, 
                      im_min: np.ndarray, plot: bool=True, silence: bool=False) -> List[int]:
    """
    Applies spatial filtering to components, generates montages, and optionally plots the results.
    
    Args:
        p_th: Percentile threshold for image processing.
        size_threshold: Size threshold for filtering out noise events.
        cnm_A: Spatial footprint matrix of neurons.
        im_min: Minimum intensity image for overlay.
        cnm_idx: Indices of accepted components.
        plot: If True, shows the plots. Default is True.
        silence: If True, suppresses print statements. Default is False.
 
    Returns:
        filtered_idx (list): List of indices of the filtered components.
    """
    assert len(cnm_A.shape) == 2, f"cnm_A should be 2D array, got shape {cnm_A.shape}"
    assert len(cnm_idx.shape) == 1, f"cnm_idx should be 1D array, got shape {cnm_idx.shape}"
    assert cnm_idx.max() < cnm_A.shape[1], (
        f"cnm_idx contains invalid indices. Max index {cnm_idx.max()} "
        f"exceeds cnm_A dimensions {cnm_A.shape[1]}"
    )
    assert len(im_min.shape) == 2, f"im_min should be 2D array, got shape {im_min.shape}"

    # Load the mask image and get its shape
    im_shape = im_min.shape
    im_sz = [im_shape[0], im_shape[1]]
    
    # Initialize storage for processed images
    im_st = np.zeros((cnm_A.shape[1], im_sz[0], im_sz[1]), dtype='uint16')
    dict_mask = {}

    # Generate im_st by thresholding the components in A
    for i in range(cnm_A.shape[1]):
        Ai = np.copy(cnm_A[:, i])
        Ai = Ai[Ai > 0]
        thr = np.percentile(Ai, p_th)
        imt = np.reshape(cnm_A[:, i], im_sz, order='F')
        im_thr = np.copy(imt)
        im_thr[im_thr < thr] = 0
        im_thr[im_thr >= thr] = i + 1
        im_st[i, :, :] = im_thr
        dict_mask[i] = im_thr > 0

    # Calculate the grid shape for all components
    n_images = len(im_st)
    grid_shape = (np.ceil(np.sqrt(n_images)).astype(int), np.ceil(np.sqrt(n_images)).astype(int))

    # Optionally plot the montage for all components
    if plot:
        montage_image = plot_montage(im_st, im_min, grid_shape)
        plt.figure(figsize=(10, 10))
        plt.imshow(montage_image, cmap='cividis')
        plt.title('Montage of All df/f0 Spatial Components')
        plt.axis('off')
        plt.show()

    # Compute the size of each neuron's footprint
    footprint_sizes = np.sum(cnm_A > 0, axis=0)

    # Find indices of neurons with footprints larger than the threshold
    large_footprint_indices = np.where(footprint_sizes > size_threshold)[0]

    # Filter out these indices from the idx array
    filtered_idx = [i for i in cnm_idx if i not in large_footprint_indices]

    # Generate im_st by thresholding the components in A for filtered indices
    for i in filtered_idx:
        Ai = np.copy(cnm_A[:, i])
        Ai = Ai[Ai > 0]
        thr = np.percentile(Ai, p_th)
        imt = np.reshape(cnm_A[:, i], im_sz, order='F')
        im_thr = np.copy(imt)
        im_thr[im_thr < thr] = 0
        im_thr[im_thr >= thr] = i + 1
        im_st[i, :, :] = im_thr
        dict_mask[i] = im_thr > 0

    # Calculate the grid shape for filtered components
    n_images = len(filtered_idx)
    grid_shape = (np.ceil(np.sqrt(n_images)).astype(int), np.ceil(np.sqrt(n_images)).astype(int)) 

    # Optionally plot the montage for filtered components
    if plot:
        montage_image = plot_montage(im_st[filtered_idx], im_min, grid_shape)
        plt.figure(figsize=(10, 10))
        plt.imshow(montage_image, cmap='cividis')
        plt.title('Montage of Spatial Filtered df/f0 Spatial Components')
        plt.axis('off')
        plt.show()
    
    if silence == False:
        # Print the number of components before and after filtering
        print(f'Total Number of Components: {len(cnm_idx)}')
        print(f'Number of Components after Spatial filtering: {len(filtered_idx)}')

    return filtered_idx