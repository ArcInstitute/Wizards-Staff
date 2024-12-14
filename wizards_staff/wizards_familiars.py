import os
from pathlib import Path
import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from collections import defaultdict
from wizards_staff.plotting import plot_montage


def categorize_files(results_folder, metadata_csv):
    """
    Categorizes files in the results folder and its subfolders based on specified suffix patterns
    and uses a metadata CSV file to create dictionary keys.

    Args:
        results_folder (str or Path): Path to the folder containing result files.
        metadata_csv (str or Path): Path to the CSV file containing a `Sample` column.
        
    Returns:
        dict: Dictionary where keys are from the `Sample` column and values are sorted lists of file paths.
              If no files are matched for a key, the value is [None].
    """
    results_folder = Path(results_folder)  # Ensure it's a Path object

    filters = [
        'cn-filter.npy',
        'cnm-A.npy',
        'cnm-C.npy',
        'cnm-S.npy',
        'cnm-idx.npy',
        'dff-dat.npy',
        'f-dat.npy',
        'pnr-filter.npy',
        'pnr-cn-filter.png',
        'minprojection.tif',
        'masks.tif',
    ]
    
    # Load metadata CSV
    metadata = pd.read_csv(metadata_csv)
    
    # Extract filenames from the metadata
    filenames = metadata['Sample'].tolist()
    
    # Dictionary to hold categorized files
    categorized_files = defaultdict(list)

    # Recursively find and filter files
    for path in results_folder.rglob("*"):
        if path.is_file():
            # Check if the file matches any of the filters and metadata filenames
            for suffix in filters:
                if path.name.endswith(suffix):
                    # Match to a key from the metadata
                    for key in filenames:
                        if path.name.startswith(key):
                            categorized_files[key].append(str(path))
                            break

    # Ensure that every key from metadata is in the dictionary
    for key in filenames:
        if key not in categorized_files:
            categorized_files[key] = [None]  # If no files, set to [None]
        else:
            # Ensure a mask is included; append None if missing
            if not any('mask' in file for file in categorized_files[key]):
                categorized_files[key].append(None)

    # Sort the file paths within each key and return an ordered dictionary
    ordered_categorized_files = {
        key: sorted(categorized_files[key]) for key in sorted(categorized_files)
    }

    return ordered_categorized_files

def load_and_filter_files(categorized_files, p_th=75, size_threshold=20000):
    """
    Loads dF/F0 output data and filter cell IDs based on spatial filtering criteria.

    Args:
        categorized_files (dict): Dictionary mapping filenames to their corresponding file paths.
        p_th (float): Percentile threshold for image processing. Default is 75.
        size_threshold (int): Size threshold for filtering out noise events. Default is 20000.

    Returns:
        d_dff (dict): Dictionary where each key is a filename and the value is the loaded dF/F data matrix.
        d_nspIDs (dict): Dictionary where each key is a filename and the value is the list of filtered neuron IDs.
    """
    d_dff = {}
    d_nspIDs = {}

    for raw_filename, _ in categorized_files.items():
        file_data = load_required_files(categorized_files, raw_filename)
        if not file_data:
            continue
        
        # Load dF/F0 data
        d_dff[raw_filename] = file_data['dff_dat']

        filtered_idx = spatial_filtering(
            im_min= file_data['im_min'], p_th=p_th, size_threshold=size_threshold, 
            cnm_A=file_data['cnm_A'], cnm_idx=file_data['cnm_idx'], plot=False, silence=True
        )
        d_nspIDs[raw_filename] = filtered_idx

    return d_dff, d_nspIDs

from skimage.io import imread

def load_required_files(categorized_files, raw_filename):
    """
    Loads necessary files for a given raw filename from the categorized_files dictionary.
    
    Args:
        categorized_files (dict): Dictionary mapping filenames to their corresponding file paths.
        raw_filename (str): The filename to load the files for.
    
    Returns:
        dict: Dictionary containing loaded files with keys corresponding to the file type.
    """
    try:
        return {
            'cn_filter': np.load(categorized_files[raw_filename][0], allow_pickle=True),
            # 'cn_filter_img': categorized_files[raw_filename][1],
            'cnm_A': np.load(categorized_files[raw_filename][1], allow_pickle=True),
            'cnm_C': np.load(categorized_files[raw_filename][2], allow_pickle=True),
            'cnm_S': np.load(categorized_files[raw_filename][3], allow_pickle=True),
            'cnm_idx': np.load(categorized_files[raw_filename][4], allow_pickle=True),
            'pnr_hist': imread(categorized_files[raw_filename][5]),
            'pnr_filter': np.load(categorized_files[raw_filename][6], allow_pickle=True),
            'dff_dat': np.load(categorized_files[raw_filename][7], allow_pickle=True),
            'dat': np.load(categorized_files[raw_filename][9], allow_pickle=True),
            'im_min': categorized_files[raw_filename][11],
            'mask': imread(categorized_files[raw_filename][10]) if 10 in categorized_files[raw_filename] else None
        }
    except Exception as e:
        print(f"Error loading files for {raw_filename}: {e}")
        return None
    
def spatial_filtering(im_min, p_th, size_threshold, cnm_A, cnm_idx, plot=True, silence = False):
    """
    Applies spatial filtering to components, generates montages, and optionally plots the results.
    
    Parameters:
    im_min (ndarray): Minimum intensity image for overlay.
    p_th (float): Percentile threshold for image processing.
    size_threshold (int): Size threshold for filtering out noise events.
    cnm_A (ndarray): Spatial footprint matrix of neurons.
    cnm_idx (ndarray): Indices of accepted components.
    plot (bool): If True, shows the plots. Default is True.
    
    Returns:
    filtered_idx (list): List of indices of the filtered components.
    """
    # Load the image and get its shape
    im_min = imread(im_min)
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