import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm.notebook import tqdm
from wizards_staff.wizards_spellbook import calc_rise_tm, calc_fwhm_spikes, calc_frpm, calc_mask_shape_metrics, convert_f_to_cs
from wizards_staff.plotting import plot_kmeans_heatmap, plot_cluster_activity, plot_spatial_activity_map, plot_dff_activity
from wizards_staff.pwc import run_pwc
from wizards_staff.metadata import append_metadata_to_dfs
from wizards_staff.wizards_familiars import categorize_files, load_required_files, spatial_filtering

def run_all(results_folder, metadata_path, frate, zscore_threshold = 3, percentage_threshold = 0.2, p_th = 75,  
            min_clusters=2, max_clusters=10, random_seed = 1111111, group_name = None, poly = False,  
            size_threshold = 20000, show_plots=True, save_files=True, output_dir='./wizard_staff_outputs'):
    """
    Processes the results folder, computes metrics, and stores them in DataFrames.
    
    Args:
        results_folder (str): Path to the results folder.
        metadata_path (str): Path to the metadata CSV file.
        frate (int): Frames per second of the imaging session.
        zscore_threshold (int): Z-score threshold for spike detection. Default is 3.
        percentage_threshold (float): Percentage threshold for FWHM calculation. Default is 0.2.
        p_th (float): Percentile threshold for image processing. Default is 75.
        min_clusters (int): The minimum number of clusters to try. Default is 2.
        max_clusters (int): The maximum number of clusters to try. Default is 10.
        random_seed (int): The seed for random number generation in K-means. Default is 1111111.
        group_name (str): Name of the group to which the data belongs. Required for PWC analysis.
        poly (bool): Flag to control whether to perform polynomial fitting during PWC analysis. Default is False.
        size_threshold (int): Size threshold for filtering out noise events.    
        show_plots (bool): Flag to control whether plots are displayed. Default is True.
        save_files (bool): Flag to control whether files are saved. Default is True.
        output_dir (str): Directory where output files will be saved. Default is './wizard_staff_outputs'.
        
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
    # Scan output folder for files necessary for processing based on their prefixes and extensions
    categorized_files = categorize_files(results_folder)

    # Initialize lists to store data for each metric type
    rise_time_data = []
    fwhm_data = []
    frpm_data = []
    mask_metrics_data = []
    silhouette_scores_data = []

    for raw_filename, _ in tqdm(categorized_files.items(), desc="Processing files"):
        try:
            # Load the necessary Lizard_Wizard Output files for the current image file
            file_data = load_required_files(categorized_files, raw_filename)
            
            # Apply spatial filtering to the data to remove noise
            filtered_idx = spatial_filtering(
                cn_filter= file_data['cn_filter_img'], p_th = p_th, size_threshold=size_threshold, 
                cnm_A=file_data['cnm_A'], cnm_idx=file_data['cnm_idx'], im_min = file_data['im_min'], plot=False, silence=True
            )
            
            # Load the ΔF/F₀ data for the given image file and add a small constant to avoid division by zero``
            dff_dat = file_data['dff_dat']
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
            rise_tm, rise_tm_pos = calc_rise_tm(calcium_signals_filtered, zscored_spike_events_filtered, zscore_threshold = zscore_threshold)

            # Calculate FWHM and related metrics
            fwhm_pos_back, fwhm_pos_fwd, fwhm, spike_counts = calc_fwhm_spikes(calcium_signals_filtered, zscored_spike_events_filtered,
                                                                                        zscore_threshold = zscore_threshold, percentage_threshold = percentage_threshold)

            # Calculate FRPM:
            _, frpm  = calc_frpm(zscored_spike_events, filtered_idx, frate, zscore_threshold = zscore_threshold)

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

            # Check if mask exists and calculate mask metrics if so
            mask = file_data['mask'] if file_data['mask'] else None

            # Calculate mask metrics and store them      
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

            # Create ΔF/F₀ graph
            plot_dff_activity(dff_dat, filtered_idx, frate, raw_filename, sz_per_neuron = 0.5, 
                    show_plots=show_plots, save_files = save_files, output_dir = output_dir)

            # Perform K-means clustering and plot
            silhouette_score, num_clusters = plot_kmeans_heatmap(
                dff_data = dff_dat, filtered_idx = filtered_idx, raw_filename = raw_filename, 
                min_clusters = min_clusters, max_clusters = max_clusters, random_seed= random_seed, 
                show_plots = show_plots, save_files = save_files, output_dir= output_dir)

            # Append silhouette score to the list
            silhouette_scores_data.append({
                'Filename': raw_filename,
                'Silhouette Score': silhouette_score,
                'Number of Clusters': num_clusters
            })

            # Plot cluster activity
            plot_cluster_activity(dff_data = dff_dat, filtered_idx = filtered_idx, raw_filename = raw_filename, 
                min_clusters = min_clusters, max_clusters = max_clusters, random_seed= random_seed, norm=False, show_plots = show_plots, 
                save_files = save_files, output_dir= output_dir)

            # Plot spatial activity map
            plot_spatial_activity_map(file_data['im_min'], file_data['cnm_A'], filtered_idx, raw_filename, 
                min_clusters = min_clusters, max_clusters = max_clusters, random_seed= random_seed,
                show_plots = show_plots, save_files = save_files)
            
            # Plot spatial activity map with clustering
            plot_spatial_activity_map(file_data['im_min'], file_data['cnm_A'],filtered_idx, raw_filename,
                min_clusters = min_clusters, max_clusters = max_clusters, random_seed= random_seed,
                clustering = True, dff_data = dff_dat, show_plots = show_plots, save_files = save_files)

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
    
    # Append metadata to dataframes
    updated_dfs = append_metadata_to_dfs(metadata_path,
                                        rise_time = rise_time_df,
                                        frpm = frpm_df, 
                                        fwhm = fwhm_df, 
                                        mask_metrics = mask_metrics_df, 
                                        sil_scores = silhouette_scores_df)
    
    # Save DataFrames as CSV files if required
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
        updated_dfs['rise_time'].to_csv(rise_time_path, index=False)
        updated_dfs['frpm'].to_csv(frpm_path, index=False)
        updated_dfs['fwhm'].to_csv(fwhm_path, index=False)
        updated_dfs['mask_metrics'].to_csv(mask_metrics_path, index=False)
        updated_dfs['sil_scores'].to_csv(silhouette_scores_path, index=False)

        print(f'Data saved to {output_dir}')
    
    # Run PWC analysis if group_name is provided
    if group_name:
        df_mn_pwc, df_mn_pwc_inter, df_mn_pwc_intra = run_pwc(group_name, metadata_path, results_folder, poly = poly,
                                                    show_plots = show_plots, save_files = save_files, output_dir = output_dir)
        
        return rise_time_df, fwhm_df, frpm_df, mask_metrics_df, silhouette_scores_df, df_mn_pwc, df_mn_pwc_inter, df_mn_pwc_intra

    return rise_time_df, fwhm_df, frpm_df, mask_metrics_df, silhouette_scores_df
