# import
## batteries
import os
import sys
import logging
import warnings
from typing import Tuple
from functools import partial
# joblib is imported locally in run_all when threads > 1
## 3rd party
import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm.notebook import tqdm
## package
from wizards_staff.wizards.spellbook import (
    calc_rise_tm, calc_fwhm_spikes, calc_frpm, calc_fall_tm, 
    calc_peak_amplitude, calc_peak_to_peak, calc_mask_shape_metrics, convert_f_to_cs
)
from wizards_staff.plotting import plot_kmeans_heatmap, plot_cluster_activity, plot_spatial_activity_map, plot_dff_activity
from wizards_staff.pwc import run_pwc
from wizards_staff.metadata import append_metadata_to_dfs
from wizards_staff.wizards.familiars import spatial_filtering
from wizards_staff.wizards.shard import Shard

# Suppress RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# functions
def run_all(orb: "Orb", frate: int=None, zscore_threshold: int=3, 
            percentage_threshold: float=0.2, p_th: float=75, min_clusters: int=2, 
            max_clusters: int=10, random_seed: int=1111111, group_name: str=None, 
            poly: bool=False, size_threshold: int=20000, show_plots: bool=False, 
            save_files: bool=False, output_dir: str='wizards_staff_outputs', 
            threads: int=1, debug: bool=False, **kwargs) -> None:
    """
    Process the results folder, computes metrics, and stores them in DataFrames.

    Args:
        results_folder: Path to the results folder.
        metadata_path: Path to the metadata CSV file.
        frate: Frames per second of the imaging session. If None (default), reads from 
               the 'Frate' column in metadata for each sample, supporting per-sample frame rates.
        zscore_threshold: Z-score threshold for spike detection.
        percentage_threshold: Percentage threshold for FWHM calculation.
        p_th: Percentile threshold for image processing.
        min_clusters: The minimum number of clusters to try.
        max_clusters: The maximum number of clusters to try.
        random_seed: The seed for random number generation in K-means.
        group_name: Name of the group to which the data belongs. Required for PWC analysis.
        poly: Flag to control whether to perform polynomial fitting during PWC analysis.
        size_threshold: Size threshold for filtering out noise events.
        show_plots: Flag to control whether plots are displayed.
        save_files: Flag to control whether files are saved.
        output_dir: Directory where output files will be saved.
        threads: Number of threads to use for processing.
        kwargs: Additional keyword arguments that will be passed to run_pwc
    """
    # Check if the output directory exists
    if save_files:
        orb._logger.info(f'Saving output to: {output_dir}')
        # Expand the user directory if it exists in the output_dir path
        output_dir = os.path.expanduser(output_dir)
        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

    # Process each sample (shard) in parallel
    func = partial(
        _run_all, 
        frate=frate, 
        zscore_threshold=zscore_threshold, 
        percentage_threshold=percentage_threshold,
        p_th=p_th,
        min_clusters=min_clusters,
        max_clusters=max_clusters, 
        random_seed=random_seed,
        group_name=group_name,
        poly=poly,
        size_threshold=size_threshold,
        show_plots=show_plots,
        save_files=save_files,
        output_dir=output_dir
    )
    desc = 'Shattering the orb and processing each shard...'
    shards = list(orb.shatter())
    
    if debug or threads == 1:
        # Sequential processing (more memory efficient)
        for shard in tqdm(shards, desc=desc):
            try:
                updated_shard = func(shard)
                orb._shards[updated_shard.sample_name] = updated_shard
            except Exception as e:
                print(f'WARNING: {e}', file=sys.stderr)
    else:
        # Parallel processing using joblib for better numpy array handling
        from joblib import Parallel, delayed
        logging.disable(logging.CRITICAL)
        try:
            results = Parallel(n_jobs=threads, prefer="processes")(
                delayed(func)(shard) for shard in tqdm(shards, desc=desc)
            )
            for updated_shard in results:
                if updated_shard is not None:
                    orb._shards[updated_shard.sample_name] = updated_shard
        except Exception as e:
            print(f'WARNING: {e}', file=sys.stderr)
        finally:
            logging.disable(logging.NOTSET)
    
    # Run PWC analysis if group_name is provided
    if group_name:
        print('Running pairwise correlation analysis...', flush=True)
        orb.run_pwc(
            group_name,
            poly = poly,
            p_th = p_th,
            size_threshold = size_threshold,
            show_plots = show_plots, 
            **kwargs
        )
        print('Pairwise correlation analysis complete.', flush=True)
    else:
        orb._logger.warning('Skipping PWC analysis as group_name is not provided.')

    # Save DataFrames as CSV files if required
    if save_files:
        print(f'Saving results to {output_dir}...', flush=True)
        orb.save_results(output_dir)
        print('Results saved.', flush=True)

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
        shard: The updated shard object
    """    
    shard._logger.info(f'Processing shard: {shard.sample_name}')

    # Get frame rate from metadata if not explicitly provided
    if frate is None:
        frate = int(shard.metadata['Frate'].iloc[0])
        shard._logger.info(f'Using frame rate from metadata: {frate} fps')

    # Check for required inputs
    for key in ['dff_dat', 'minprojection', 'cnm_A']:
        if not shard.has_file(key):
            shard._logger.warning(f'Missing required input: {key}; skipping {shard.sample_name}')
            return shard

    # Apply spatial filtering to the data to remove noise
    filtered_idx = shard.spatial_filtering(
        p_th=p_th, 
        size_threshold=size_threshold,
        plot=False, 
        silence=True,
    )
    
    # Convert ΔF/F₀ to calcium signals and spike events
    calcium_signals, spike_events = shard.convert_f_to_cs(p=2)

    # Z-score the spike events
    zscored_spike_events = zscore(np.copy(spike_events), axis=1)

    # Filter the calcium signals and z-scored spike events based on the spatial filtering
    zscored_spike_events_filtered = zscored_spike_events[filtered_idx, :]
    calcium_signals_filtered = calcium_signals[filtered_idx, :]

    # Calculate rise time and positions:
    rise_tm, rise_tm_pos = shard.calc_rise_tm(
        calcium_signals_filtered, 
        zscored_spike_events_filtered, 
        zscore_threshold=zscore_threshold
    )

    # Calculate FWHM and related metrics
    fwhm_pos_back, fwhm_pos_fwd, fwhm, spike_counts = shard.calc_fwhm_spikes(
        calcium_signals_filtered, 
        zscored_spike_events_filtered,
        zscore_threshold=zscore_threshold, 
        percentage_threshold=percentage_threshold
    )

    # Calculate FRPM:
    _, frpm  = shard.calc_frpm(
        zscored_spike_events, filtered_idx, frate, 
        zscore_threshold=zscore_threshold
    )

    # Calculate fall time
    fall_tm, fall_tm_pos = shard.calc_fall_tm(
        calcium_signals_filtered,
        zscored_spike_events_filtered,
        zscore_threshold=zscore_threshold
    )

    # Get raw ΔF/F₀ data filtered by spatial filtering for amplitude measurement
    dff_data_raw = shard.get_input('dff_dat', req=True)
    dff_data_filtered = dff_data_raw[filtered_idx, :]

    # Calculate peak amplitude (using raw ΔF/F₀ for interpretable units)
    peak_amp, peak_pos = shard.calc_peak_amplitude(
        calcium_signals_filtered,
        zscored_spike_events_filtered,
        zscore_threshold=zscore_threshold,
        dff_data=dff_data_filtered
    )

    # Calculate peak-to-peak intervals (inter-spike intervals)
    peak_to_peak = shard.calc_peak_to_peak(
        calcium_signals_filtered,
        zscored_spike_events_filtered,
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
            'Firing Rate Per Min': frpm_value
        })

    for neuron_idx, fall_times in fall_tm.items():
        shard._fall_time_data.append({
            'Sample': shard.sample_name,
            'Neuron': neuron_idx,
            'Fall Times': fall_times,
            'Fall Positions': fall_tm_pos[neuron_idx]
        })

    for neuron_idx, amplitudes in peak_amp.items():
        shard._peak_amplitude_data.append({
            'Sample': shard.sample_name,
            'Neuron': neuron_idx,
            'Peak Amplitudes': amplitudes,
            'Peak Positions': peak_pos[neuron_idx]
        })

    for neuron_idx, intervals in peak_to_peak.items():
        shard._peak_to_peak_data.append({
            'Sample': shard.sample_name,
            'Neuron': neuron_idx,
            'Inter-Spike Intervals': intervals
        })

    # Calculate mask metrics and store them      
    if shard.has_file('mask'):
        mask_metrics = shard.calc_mask_shape_metrics()
        shard._mask_metrics_data.append({
            'Sample': shard.sample_name,
            'Roundness':  mask_metrics.get('roundness'),
            'Diameter': mask_metrics.get('diameter'),
            'Area': mask_metrics.get('area')
        })

    # Create ΔF/F₀ graph
    plot_dff_activity(
        shard.get_input('dff_dat', req=True), 
        filtered_idx, frate, shard.sample_name,
        sz_per_neuron=0.5, 
        show_plots=show_plots, 
        save_files=save_files, 
        output_dir=output_dir
    )

    # Perform K-means clustering and plot
    silhouette_score, num_clusters = plot_kmeans_heatmap(
        dff_dat=shard.get_input('dff_dat', req=True), 
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
        dff_dat = shard.get_input('dff_dat', req=True), 
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
        shard.get_input('minprojection', req=True),  #['im_min'], 
        shard.get_input('cnm_A', req=True), 
        filtered_idx, 
        shard.sample_name,
        min_clusters = min_clusters, 
        max_clusters = max_clusters, 
        random_seed = random_seed,
        show_plots = show_plots, 
        save_files = save_files,
        dff_dat = shard.get_input('dff_dat'),
        output_dir = output_dir
    )
    
    # Plot spatial activity map with clustering
    plot_spatial_activity_map(
        shard.get_input('minprojection', req=True),  #['im_min'], 
        shard.get_input('cnm_A', req=True), 
        filtered_idx, 
        shard.sample_name,
        min_clusters = min_clusters, 
        max_clusters = max_clusters, 
        random_seed= random_seed,
        clustering = True, 
        dff_dat = shard.get_input('dff_dat', req=True), 
        show_plots = show_plots, 
        save_files = save_files,
        output_dir = output_dir
    )

    return shard