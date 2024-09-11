import os
from collections import defaultdict

def categorize_files(results_folder):
    """
    Categorizes files in the results folder based on their prefixes and extensions.
    
    Args:
        results_folder (str): Path to the folder containing result files.
        
    Returns:
        dict: Dictionary where keys are raw filenames and values are lists of categorized file paths.
    """
    # List all files in the results folder
    file_list = os.listdir(results_folder)

    # Helper function to filter files
    def filter_files(prefix, extension):
        return [f for f in file_list if f.startswith(prefix) and f.endswith(extension)]

    # Dictionary to hold categorized files
    categorized_files = defaultdict(list)

    # Define prefixes and extensions to filter
    filters = [
        ('cn_filter', '.npy'),
        ('cn_filter', '.tif'),
        ('cnm_A', '.npy'),
        ('cnm_C', '.npy'),
        ('cnm_S', '.npy'),
        ('cnm_idx', '.npy'),
        ('corr_pnr_histograms', '.tif'),
        ('df_f0_graph', '.tif'),
        ('dff_f_mean', '.npy'),
        ('f_mean', '.npy'),
        ('mask', '.tif'),
        ('pnr_filter', '.npy'),
        ('pnr_filter', '.tif'),
        ('minprojection', '.tif')
    ]

    # Filter files and populate the dictionary
    for prefix, extension in filters:
        filtered_files = filter_files(prefix, extension)
        for file in filtered_files:
            # Extract the base filename by removing prefix and extension
            base_filename = file.replace(prefix + "_", "").rsplit(extension, 1)[0]
            # Add the full path of the file to the categorized dictionary
            categorized_files[base_filename].append(os.path.join(results_folder, file))
    
    # Ensure that each entry has a mask, set to None if not present
    for base_filename, files in categorized_files.items():
        if not any('mask' in f for f in files):
            categorized_files[base_filename].append(None)

    return categorized_files
