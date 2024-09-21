# import
## batteries
import os
import sys
import logging
from typing import List, Tuple
from collections import defaultdict
## 3rd party
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
## package
from wizards_staff.plotting import plot_montage


# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# def list_files(indir):
#     files = []
#     for dirpath, dirnames, filenames in os.walk(indir):
#         for filename in filenames:
#             files.append(os.path.join(dirpath, filename))
#     return files

# def categorize_files(results_folder, metadata_path):
#     """
#     Categorizes files in the results folder based on their prefixes and extensions.
    
#     Args:
#         results_folder (str): Path to the folder containing result files.
        
#     Returns:
#         dict: Dictionary where keys are raw filenames and values are lists of categorized file paths.
#     """
#     # Define prefixes and extensions to filter
#     filters = [
#         ('cn_filter', '.npy'),
#         ('cn_filter', '.tif'),
#         ('cnm_A', '.npy'),
#         ('cnm_C', '.npy'),
#         ('cnm_S', '.npy'),
#         ('cnm_idx', '.npy'),
#         ('corr_pnr_histograms', '.tif'),
#         ('df_f0_graph', '.tif'),
#         ('dff_f_mean', '.npy'),
#         ('f_mean', '.npy'),
#         ('pnr_filter', '.npy'),
#         ('pnr_filter', '.tif'),
#         ('minprojection', '.tif'),
#         ('mask', '.tif'),
#     ]

#     # Filter files and populate the dictionary
#     categorized_files = defaultdict(dict)
#     for file in list_files(results_folder):
#         for prefix, extension in filters:
#             if not file.endswith(prefix + extension):
#                 continue
#             basename = os.path.splitext(os.path.basename(file))[0]
#             categorized_files[basename][prefix] = file

#     # Ensure that each entry has a mask, set to None if not present
#     for base_filename, prefix in categorized_files.items():
#         if categorized_files[base_filename].get('mask') is None:
#             categorized_files[base_filename]['mask'] = [None]

#     return categorized_files
    

# def load_required_files(raw_filename, cat_files):
#     """
#     Loads necessary files for a given raw filename from the categorized_files dictionary.
    
#     Args:
#         raw_filename (str): Raw filename to load files for.
#         cat_files (dict): Dictionary mapping filenames to their corresponding file paths.    
#     Returns:
#         dict: Dictionary containing loaded files with keys corresponding to the file type.
#     """
#     data = dict()
#     for prefix in cat_files.keys():
#         try:
#             if prefix in ('cn_filter', 'cnm_A', 'cnm_C', 'cnm_idx', 'dff_dat', 'pnr_filter'):
#                 data[prefix] = np.load(cat_files[prefix], allow_pickle=True)
#             elif prefix in ('pnr_hist', 'df_f0_graph'):
#                 data[prefix] = imread(cat_files[prefix])
#             elif prefix in ('im_min', 'mask'):
#                 data[prefix] = imread(cat_files[prefix]) if cat_files[prefix] is not None else None
#         except Exception as e:
#             print(f"Error loading {prefix} for {raw_filename}: {e}")
#             exit(1)
#     return data

#     # res = {
#     #         'cn_filter': np_load(cat_files, 'cn_filter', allow_pickle=True),
#     #         'cn_filter_img': np_load(cat_files, 'cn_filter_img'),
#     #         'cnm_A': np_load(cat_files, 'cnm_A', allow_pickle=True),
#     #         'cnm_C': np_load(cat_files, 'cnm_C', allow_pickle=True),
#     #         'cnm_S': np_load(cat_files, 'cnm_S', allow_pickle=True),
#     #         'cnm_idx': np_load(cat_files, 'cnm_idx', allow_pickle=True),
#     #         'pnr_hist': imread(cat_files['pnr_hist']),
#     #         'df_f0_graph': imread(cat_files['df_f0_graph']),
#     #         'dff_dat': np_load(cat_files, 'dff_dat', allow_pickle=True),
#     #         'dat': np_load(cat_files, 'dat', allow_pickle=True),
#     #         'pnr_filter': np_load(cat_files, 'pnr_filter', allow_pickle=True),
#     #         #'im_min': categorized_files[raw_filename][11] if 12 not in categorized_files[raw_filename] else categorized_files[raw_filename][12],
#     #         #'mask': imread(categorized_files[raw_filename][13]) if 13 in categorized_files[raw_filename] else None
#     # }
#     # #except Exception as e:
#     # #    print(f"Error loading files for {raw_filename}: {e}")
#     # #    return None
    
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
        d_dff[raw_filename] = file_data['dff_f_mean']

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
    logger.info('Applying spatial filtering to components...')

    # Load the mask image and get its shape
    if plot:
        im_min = imread(im_min)
    #mask_image = imread(cn_filter)
    #im_shape = mask_image.shape
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