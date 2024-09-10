import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm
from collections import defaultdict
from wizards_staff.metrics import calc_rise_tm, calc_fwhm_spikes, calc_frpm, calc_mask_shape_metrics, convert_f_to_cs
from wizards_staff.plotting import plot_kmeans_heatmap, plot_cluster_activity, spatial_filter_and_plot, plot_activity_map
from wizards_staff.pwc import lizard_wizard_pwc
from wizards_staff.metadata import append_metadata_to_dfs

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


def wizards_staff_all(results_folder, metadata_path, group_name = None, poly = False, size_threshold = 20000, show_plots=True, save_files=True, output_dir='./lizard_wizard_outputs'):
    """
    Processes the results folder, computes metrics, and stores them in DataFrames.
    
    Args:
        results_folder (str): Path to the results folder.
        size_threshold (int): Size threshold for filtering out noise events.    
        metadata_path (str): Path to the metadata CSV file.
        group_name (str): Name of the group to which the data belongs. Used for PWC analysis.
        poly (bool): Flag to control whether to perform polynomial fitting during PWC analysis. Default is False.
        show_plots (bool): Flag to control whether plots are displayed. Default is True.
        save_files (bool): Flag to control whether files are saved. Default is True.
        output_dir (str): Directory where output files will be saved.
        
    Returns:
        rise_time_df (pd.DataFrame): DataFrame containing rise time metrics.
            - rise_tm: dict where keys are neuron indices and values are lists of rise times for each neuron.
            - rise_tm_pos: dict where keys are neuron indices and values are lists of time points corresponding to end of rise times
        fwhm_df (pd.DataFrame): DataFrame containing FWHM metrics.
            - fwhm_pos_back: dict keys are neuron indices and values are lists of backward positions of FWHM
            - fwhm_pos_fwd: dict keys are neuron indices and values are lists of forward positions of FWHM
            - fwhm: dict keys are neuron indices and values are lists of FWHM
            - spike_counts: dictionary where keys are neuron indices and values are lists of the number of spikes within the FWHM for each neuron.
        frpm_df (pd.DataFrame): DataFrame containing FRPM metrics.
            - frpm: dict keys are neuron indices and values are lists of average frpm for each neuron
            - frpm_avg: float value of average frpm for the dataset
        mask_metrics_df (pd.DataFrame): DataFrame containing mask metrics.
            - file: The filename of the processed file.
            - roundness: The roundness of the masked object.
            - diameter: The diameter of the masked object.
            - area: The area of the masked object.
        silhouette_scores_df (pd.DataFrame): DataFrame containing silhouette scores for K-means clustering.
    """
    categorized_files = categorize_files(results_folder)

    # Initialize lists to store data for each metric type
    rise_time_data = []
    fwhm_data = []
    frpm_data = []
    mask_metrics_data = []
    silhouette_scores_data = []

    for raw_filename, file_paths in tqdm(categorized_files.items(), desc="Processing files"):
        try:
            # Load the necessary files
            cn_filter = np.load(categorized_files[raw_filename][0], allow_pickle=True)
            cn_filter_img = categorized_files[raw_filename][1]
            cnm_A = np.load(categorized_files[raw_filename][2], allow_pickle=True)
            cnm_C = np.load(categorized_files[raw_filename][3], allow_pickle=True)
            cnm_S = np.load(categorized_files[raw_filename][4], allow_pickle=True)
            cnm_idx = np.load(categorized_files[raw_filename][5], allow_pickle=True)
            dff_dat = np.load(categorized_files[raw_filename][8], allow_pickle=True)
            dat = np.load(categorized_files[raw_filename][8], allow_pickle=True)

            # im_min = categorized_files[raw_filename][13]
            im_min = categorized_files[raw_filename][12] # this is the PNR filter image, use only for testing

            p_th = 75  # Threshold percentile for image processing
            
            # Apply spatial filtering and plot the results
            filtered_idx = spatial_filter_and_plot(cn_filter_img, p_th, size_threshold, 
                cnm_A, cnm_idx, im_min, plot = False, silence = True)

            dff = np.copy(dff_dat)  # Copy the ΔF/F₀ data
            dff += 0.0001  # Small constant added to avoid division by zero

            # Convert ΔF/F₀ to calcium signals and spike events
            calcium_signals, spike_events = convert_f_to_cs(dff, p = 2)

            # Z-score the spike events
            zscored_spike_events = zscore(np.copy(spike_events), axis = 1)

            # Filter the calcium signals and z-scored spike events based on the spatial filtering
            zscored_spike_events_filtered = zscored_spike_events[filtered_idx, :]
            calcium_signals_filtered = calcium_signals[filtered_idx, :]

            # Calculate rise time and positions:
            rise_tm, rise_tm_pos = calc_rise_tm(calcium_signals_filtered, zscored_spike_events_filtered, zscore_threshold = 3)

            # Calculate FWHM and related metrics
            fwhm_pos_back, fwhm_pos_fwd, fwhm, spike_counts = calc_fwhm_spikes(calcium_signals_filtered, zscored_spike_events_filtered,
                                                                                        zscore_threshold = 3, percentage_threshold = 0.2)

            # Calculate FRPM:
            frpm_avg, frpm  = calc_frpm(zscored_spike_events, filtered_idx, zscore_threshold = 5)

            # Store the results in the respective lists
            for neuron_idx, rise_times in rise_tm.items():
                rise_time_data.append({
                    'Filename': raw_filename,
                    'Neuron': neuron_idx,
                    'Rise Times': rise_times,
                    'Rise Positions': rise_tm_pos[neuron_idx]
                })

            for neuron_idx, fwhm_values in fwhm.items():
                fwhm_data.append({
                    'Filename': raw_filename,
                    'Neuron': neuron_idx,
                    'FWHM Backward Positions': fwhm_pos_back[neuron_idx],
                    'FWHM Forward Positions': fwhm_pos_fwd[neuron_idx],
                    'FWHM Values': fwhm_values,
                    'Spike Counts': spike_counts[neuron_idx]
                })

            for neuron_idx, frpm_value in frpm.items():
                frpm_data.append({
                    'Filename': raw_filename,
                    'Neuron Index': neuron_idx,
                    'Firing Rate Per Min.': frpm_value
                })

            # Calculate mask metrics and store them
            mask = categorized_files[raw_filename][10]
                    
            if mask:
                mask_metrics = calc_mask_shape_metrics(mask)
                
                mask_metrics_ordered = {
                    'Filename': raw_filename,
                    'Roundness': mask_metrics.get('roundness'),
                    'Diameter': mask_metrics.get('diameter'),
                    'Area (Pixels)': mask_metrics.get('area')
                }
                mask_metrics_data.append(mask_metrics_ordered)
            else:
                mask_metrics_data.append({
                    'Filename': raw_filename,
                    'Roundness': None,
                    'Diameter': None,
                    'Area': None
                })

            # Perform K-means clustering and plot
            silhouette_score, num_clusters = plot_kmeans_heatmap(
                dff_data = dff_dat, filtered_idx = filtered_idx, raw_filename = raw_filename, 
                min_clusters = 2, max_clusters = 10, show_plots = show_plots, save_files = save_files, output_dir= output_dir)

            # Append silhouette score to the list
            silhouette_scores_data.append({
                'Filename': raw_filename,
                'Silhouette Score': silhouette_score,
                'Number of Clusters': num_clusters
            })

            # Plot cluster activity
            plot_cluster_activity(dff_data = dff_dat, filtered_idx = filtered_idx, raw_filename = raw_filename, 
                min_clusters=2, max_clusters=10, random_seed=1111111, norm=False, show_plots = show_plots, 
                save_files = save_files, output_dir= output_dir)

            base_act_img = plot_activity_map(im_min, cnm_A, cnm_idx, raw_filename, show_plots = show_plots, save_files = save_files)
            clust_act_img = plot_activity_map(im_min, cnm_A, cnm_idx, raw_filename, clustering = True, dff_data = dff_dat, show_plots = show_plots, save_files = save_files)

        except Exception as e:
            print(f"Error processing file {raw_filename}: {e}")
            continue

    # Convert the lists to DataFrames
    rise_time_df = pd.DataFrame(rise_time_data)
    fwhm_df = pd.DataFrame(fwhm_data)
    frpm_df = pd.DataFrame(frpm_data)
    mask_metrics_df = pd.DataFrame(mask_metrics_data)
    silhouette_scores_df = pd.DataFrame(silhouette_scores_data)
    
    # Use explode to handle lists in DataFrames
    rise_time_df = rise_time_df.explode(['Rise Times', 'Rise Positions'])
    fwhm_df = fwhm_df.explode(['FWHM Backward Positions', 'FWHM Forward Positions', 'FWHM Values', 'Spike Counts'])
    
    # # Append metadata to dataframes
    rise_time_df, fwhm_df, frpm_df, mask_metrics_df, silhouette_scores_df = append_metadata_to_dfs(
        rise_time_df, fwhm_df, frpm_df, mask_metrics_df, silhouette_scores_df, metadata_path
    )
    
    # Save DataFrames if required
    if save_files:
        # Expand the user directory if it exists in the output_dir path
        output_dir = os.path.expanduser(output_dir)
        
        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Get the base name of the results folder
        fname = os.path.splitext(os.path.basename(results_folder))[0]

        # Define the file paths
        rise_time_path = os.path.join(output_dir, f'{fname}_rise_time_df.csv')
        fwhm_path = os.path.join(output_dir, f'{fname}_fwhm_df.csv')
        frpm_path = os.path.join(output_dir, f'{fname}_frpm_df.csv')
        mask_metrics_path = os.path.join(output_dir, f'{fname}_mask_metrics_df.csv')
        silhouette_scores_path = os.path.join(output_dir, f'{fname}_silhouette_scores_df.csv')

        # Save each DataFrame to a CSV file
        rise_time_df.to_csv(rise_time_path, index=False)
        fwhm_df.to_csv(fwhm_path, index=False)
        frpm_df.to_csv(frpm_path, index=False)
        mask_metrics_df.to_csv(mask_metrics_path, index=False)
        silhouette_scores_df.to_csv(silhouette_scores_path, index=False)

        print(f'Data saved to {output_dir}')
    
    
    if group_name:
        df_mn_pwc, df_mn_pwc_inter, df_mn_pwc_intra = lizard_wizard_pwc(
                            group_name, metadata_path, results_folder, poly = poly, show_plots = show_plots, save_files = save_files, output_dir = output_dir)
        return rise_time_df, fwhm_df, frpm_df, mask_metrics_df, silhouette_scores_df, df_mn_pwc, df_mn_pwc_inter, df_mn_pwc_intra

    return rise_time_df, fwhm_df, frpm_df, mask_metrics_df, silhouette_scores_df
