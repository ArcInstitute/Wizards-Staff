# import
## batteries
import os
import logging
from typing import Tuple
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
## 3rd party
import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm.notebook import tqdm
## package
from wizards_staff.wizards.spellbook import calc_rise_tm, calc_fwhm_spikes, calc_frpm, calc_mask_shape_metrics, convert_f_to_cs
from wizards_staff.plotting import plot_kmeans_heatmap, plot_cluster_activity, plot_spatial_activity_map, plot_dff_activity
from wizards_staff.pwc import run_pwc
from wizards_staff.metadata import append_metadata_to_dfs
from wizards_staff.wizards.familiars import spatial_filtering
from wizards_staff.wizards.shard import Shard

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# functions
def _run_all(shard: Shard, frate: int, zscore_threshold: int, percentage_threshold: float, 
             p_th: float, min_clusters: int, max_clusters: int, random_seed: int, 
             size_threshold: int, group_name: str = None, poly: bool = False,
             show_plots: bool = True, save_files: bool = True, 
             output_dir: str = 'wizard_staff_outputs') -> Shard:
    """
    Process each shard of the Orb and compute metrics.
    Args:
        See run_all function.
    Returns:
        shard: The updated shard
    """    
    logger.info(f'Processing shard: {shard.sample_name}')

    # Apply spatial filtering to the data to remove noise
    filtered_idx = spatial_filtering(
        #cn_filter=shard.get('cn_filter'), 
        p_th=p_th, 
        size_threshold=size_threshold, 
        cnm_A=shard.get('cnm_A', req=True), 
        cnm_idx=shard.get('cnm_idx', req=True), 
        im_min=shard.get('minprojection', req=True),   # ['im_min'], 
        plot=False, 
        silence=True
    )
    
    # Load the ΔF/F₀ data for the given image file and add a small constant to avoid division by zero``
    dff_dat = np.copy(shard.get('dff_dat', req=True))  # Copy the ΔF/F₀ data
    dff_dat += 0.0001  # Small constant added to avoid division by zero

    # Convert ΔF/F₀ to calcium signals and spike events
    calcium_signals, spike_events = convert_f_to_cs(dff_dat, p=2)

    # Z-score the spike events
    zscored_spike_events = zscore(np.copy(spike_events), axis=1)

    # Filter the calcium signals and z-scored spike events based on the spatial filtering
    zscored_spike_events_filtered = zscored_spike_events[filtered_idx, :]
    calcium_signals_filtered = calcium_signals[filtered_idx, :]

    # Calculate rise time and positions:
    rise_tm, rise_tm_pos = calc_rise_tm(
        calcium_signals_filtered, zscored_spike_events_filtered, 
        zscore_threshold=zscore_threshold
    )

    # Calculate FWHM and related metrics
    fwhm_pos_back, fwhm_pos_fwd, fwhm, spike_counts = calc_fwhm_spikes(
        calcium_signals_filtered, zscored_spike_events_filtered,
        zscore_threshold=zscore_threshold, 
        percentage_threshold=percentage_threshold
    )

    # Calculate FRPM:
    _, frpm  = calc_frpm(
        zscored_spike_events, filtered_idx, frate, 
        zscore_threshold=zscore_threshold
    )

    # Store the results in the respective lists
    for neuron_idx, rise_times in rise_tm.items():
        shard._rise_time_data.append({
            'Sample': shard.sample_name,
            'Neuron': neuron_idx,
            'Rise Times': rise_times,
            'Rise Positions': rise_tm_pos[neuron_idx]
        })

    for neuron_idx, fwhm_values in fwhm.items():
        shard._fwhm_data.append({
            'Sample': shard.sample_name,
            'Neuron': neuron_idx,
            'FWHM Backward Positions': fwhm_pos_back[neuron_idx],
            'FWHM Forward Positions': fwhm_pos_fwd[neuron_idx],
            'FWHM Values': fwhm_values,
            'Spike Counts': spike_counts[neuron_idx]
        })

    for neuron_idx, frpm_value in frpm.items():
        shard._frpm_data.append({
            'Sample': shard.sample_name,
            'Neuron Index': neuron_idx,
            'Firing Rate Per Min.': frpm_value
        })

    # Calculate mask metrics and store them      
    if shard.get('mask') is not None:
        mask_metrics = calc_mask_shape_metrics(shard.get('mask'))
    shard._mask_metrics_data.append({
        'Sample': shard.sample_name,
        'Roundness':  mask_metrics.get('roundness'),
        'Diameter': mask_metrics.get('diameter'),
        'Area': mask_metrics.get('area')
    })

    # Create ΔF/F₀ graph
    plot_dff_activity(
        shard.get('dff_dat', req=True), filtered_idx, frate, shard.sample_name,
        sz_per_neuron=0.5, 
        show_plots=show_plots, 
        save_files=save_files, 
        output_dir=output_dir
    )

    # Perform K-means clustering and plot
    silhouette_score, num_clusters = plot_kmeans_heatmap(
        dff_dat=shard.get('dff_dat', req=True), 
        filtered_idx=filtered_idx, 
        sample_name=shard.sample_name,
        min_clusters=min_clusters, 
        max_clusters=max_clusters, 
        random_seed=random_seed, 
        show_plots=show_plots, 
        save_files=save_files, 
        output_dir=output_dir
    )

    # Append silhouette score to the list
    shard._silhouette_scores_data.append({
        'Sample': shard.sample_name,
        'Silhouette Score': silhouette_score,
        'Number of Clusters': num_clusters
    })

    # Plot cluster activity
    plot_cluster_activity(
        dff_dat = shard.get('dff_dat', req=True), 
        filtered_idx = filtered_idx, 
        min_clusters = min_clusters, 
        max_clusters = max_clusters, 
        random_seed = random_seed, 
        norm = False, 
        show_plots = show_plots, 
        save_files = save_files, 
        sample_name = shard.sample_name,
        output_dir= output_dir
    )

    # Plot spatial activity map
    plot_spatial_activity_map(
        shard.get('minprojection', req=True),  #['im_min'], 
        shard.get('cnm_A', req=True), 
        filtered_idx, 
        shard.sample_name,
        min_clusters = min_clusters, 
        max_clusters = max_clusters, 
        random_seed = random_seed,
        show_plots = show_plots, 
        save_files = save_files,
        dff_dat = shard.get('dff_dat')
    )
    
    # Plot spatial activity map with clustering
    plot_spatial_activity_map(
        shard.get('minprojection', req=True),  #['im_min'], 
        shard.get('cnm_A', req=True), 
        filtered_idx, 
        shard.sample_name,
        min_clusters = min_clusters, 
        max_clusters = max_clusters, 
        random_seed= random_seed,
        clustering = True, 
        dff_dat = shard.get('dff_dat', req=True), 
        show_plots = show_plots, 
        save_files = save_files
    )

    return shard