"""
Time-series analysis for calcium transient data.

These functions analyze the temporal dynamics of calcium signals, including
transient kinetics, event frequency changes, burst detection, and network
synchronization.

Functions
---------
analyze_transient_kinetics : Comprehensive analysis of calcium transient timing
calculate_event_frequency_over_time : Track how firing frequency changes
detect_bursting : Identify and characterize burst events
calculate_synchrony_metrics : Measure network synchronization
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def analyze_transient_kinetics(
    rise_time_df: pd.DataFrame,
    fwhm_df: pd.DataFrame,
    group_col: Optional[str] = None,
    frame_rate: float = 30.0
) -> pd.DataFrame:
    """
    Comprehensive analysis of calcium transient kinetics.
    
    Combines rise time and FWHM data to characterize the temporal properties
    of calcium transients for each neuron or sample.
    
    Parameters
    ----------
    rise_time_df : pd.DataFrame
        DataFrame containing rise time data from Orb.rise_time_data.
        Expected columns: Sample, Neuron Index, Rise Times, Rise Positions.
    fwhm_df : pd.DataFrame
        DataFrame containing FWHM data from Orb.fwhm_data.
        Expected columns: Sample, Neuron Index, FWHM Values, Spike Counts.
    group_col : str, optional
        Column name for grouping (e.g., "Well", "Treatment").
        If provided, includes group-level summaries.
    frame_rate : float
        Recording frame rate in Hz (frames per second). Default is 30.0.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with kinetic metrics for each neuron/sample:
        - mean_rise_time_frames: Average rise time in frames
        - mean_rise_time_sec: Average rise time in seconds
        - mean_fwhm_frames: Average FWHM in frames
        - mean_fwhm_sec: Average FWHM in seconds
        - rise_fwhm_ratio: Ratio of rise time to FWHM (transient shape)
        - cv_rise_time: Coefficient of variation of rise times
        - cv_fwhm: Coefficient of variation of FWHM
        - n_events: Number of events
    
    Notes
    -----
    Transient kinetics can reveal important biological differences:
    
    - **Faster rise times** may indicate more efficient calcium influx
    - **Longer FWHM** may indicate slower calcium clearance
    - **Rise/FWHM ratio** characterizes the shape of transients
    - **High CV** suggests variable transient properties
    
    Examples
    --------
    >>> kinetics = analyze_transient_kinetics(
    ...     orb.rise_time_data,
    ...     orb.fwhm_data,
    ...     group_col="Treatment",
    ...     frame_rate=30.0
    ... )
    >>> print(kinetics.head())
    """
    # Aggregate rise time data by sample/neuron
    rise_agg = rise_time_df.groupby(['Sample']).agg({
        'Rise Times': ['mean', 'std', 'count']
    }).reset_index()
    rise_agg.columns = ['Sample', 'mean_rise_time', 'std_rise_time', 'n_rise_events']
    
    # Aggregate FWHM data by sample
    fwhm_agg = fwhm_df.groupby(['Sample']).agg({
        'FWHM Values': ['mean', 'std', 'count']
    }).reset_index()
    fwhm_agg.columns = ['Sample', 'mean_fwhm', 'std_fwhm', 'n_fwhm_events']
    
    # Merge
    kinetics = rise_agg.merge(fwhm_agg, on='Sample', how='outer')
    
    # Convert to seconds
    kinetics['mean_rise_time_sec'] = kinetics['mean_rise_time'] / frame_rate
    kinetics['mean_fwhm_sec'] = kinetics['mean_fwhm'] / frame_rate
    
    # Calculate coefficient of variation
    kinetics['cv_rise_time'] = kinetics['std_rise_time'] / kinetics['mean_rise_time']
    kinetics['cv_fwhm'] = kinetics['std_fwhm'] / kinetics['mean_fwhm']
    
    # Calculate rise/FWHM ratio (shape index)
    kinetics['rise_fwhm_ratio'] = kinetics['mean_rise_time'] / kinetics['mean_fwhm']
    
    # Rename for clarity
    kinetics = kinetics.rename(columns={
        'mean_rise_time': 'mean_rise_time_frames',
        'mean_fwhm': 'mean_fwhm_frames',
    })
    
    # Add group information if provided
    if group_col is not None:
        # Get group info from either dataframe
        if group_col in rise_time_df.columns:
            group_info = rise_time_df[['Sample', group_col]].drop_duplicates()
            kinetics = kinetics.merge(group_info, on='Sample', how='left')
        elif group_col in fwhm_df.columns:
            group_info = fwhm_df[['Sample', group_col]].drop_duplicates()
            kinetics = kinetics.merge(group_info, on='Sample', how='left')
    
    return kinetics


def calculate_event_frequency_over_time(
    spike_data: np.ndarray,
    window_size: int = 300,
    frame_rate: float = 30.0
) -> pd.DataFrame:
    """
    Calculate how firing frequency changes over the recording.
    
    Divides the recording into time windows and calculates the event
    frequency in each window. Useful for detecting:
    - Adaptation (decreasing activity over time)
    - Sensitization (increasing activity)
    - Baseline drift
    - Periodic patterns
    
    Parameters
    ----------
    spike_data : np.ndarray
        Binary spike matrix (neurons × frames) where 1 indicates a spike.
    window_size : int
        Size of sliding window in frames. Default is 300 (10 sec at 30 Hz).
    frame_rate : float
        Recording frame rate in Hz. Default is 30.0.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - window_start: Start frame of each window
        - window_end: End frame of each window
        - time_sec: Time in seconds (center of window)
        - event_rate: Events per minute across all neurons
        - active_neurons: Number of neurons with at least one event
        - mean_neuron_rate: Mean events per neuron
    
    Examples
    --------
    >>> # Detect if activity decreases over time (adaptation)
    >>> freq_over_time = calculate_event_frequency_over_time(
    ...     spike_data, window_size=600, frame_rate=30.0
    ... )
    >>> plt.plot(freq_over_time['time_sec'], freq_over_time['event_rate'])
    >>> plt.xlabel('Time (s)')
    >>> plt.ylabel('Events per minute')
    """
    spike_data = np.atleast_2d(spike_data)
    n_neurons, n_frames = spike_data.shape
    
    results = []
    
    for start in range(0, n_frames - window_size + 1, window_size // 2):
        end = start + window_size
        window_data = spike_data[:, start:end]
        
        # Count events per neuron in this window
        events_per_neuron = window_data.sum(axis=1)
        total_events = events_per_neuron.sum()
        active_neurons = (events_per_neuron > 0).sum()
        
        # Convert to events per minute
        window_duration_min = window_size / frame_rate / 60
        event_rate = total_events / window_duration_min
        mean_neuron_rate = (events_per_neuron.mean() / window_duration_min) if n_neurons > 0 else 0
        
        results.append({
            'window_start': start,
            'window_end': end,
            'time_sec': (start + end) / 2 / frame_rate,
            'event_rate': event_rate,
            'active_neurons': active_neurons,
            'mean_neuron_rate': mean_neuron_rate,
            'total_events': total_events,
        })
    
    return pd.DataFrame(results)


def detect_bursting(
    spike_times: List[float],
    frame_rate: float = 30.0,
    min_spikes: int = 3,
    max_isi: float = 0.5
) -> Dict:
    """
    Detect and characterize burst events in spike trains.
    
    A burst is defined as a cluster of spikes occurring in rapid succession
    (with inter-spike intervals less than max_isi).
    
    Parameters
    ----------
    spike_times : list of float
        Times of spikes in frames or seconds (will be converted using frame_rate).
    frame_rate : float
        Recording frame rate in Hz. Default is 30.0.
    min_spikes : int
        Minimum number of spikes to constitute a burst. Default is 3.
    max_isi : float
        Maximum inter-spike interval in seconds within a burst. Default is 0.5.
    
    Returns
    -------
    dict
        Dictionary containing:
        - n_bursts: Number of bursts detected
        - n_spikes_in_bursts: Total spikes occurring within bursts
        - burst_fraction: Fraction of spikes occurring in bursts
        - bursts_per_minute: Burst frequency
        - mean_spikes_per_burst: Average spikes in each burst
        - mean_burst_duration: Average burst length in seconds
        - burst_details: List of dicts with info about each burst
    
    Notes
    -----
    Bursting behavior can indicate:
    - Network synchronization
    - Specific cell types (bursting vs. regular spiking)
    - Pathological activity patterns
    
    Examples
    --------
    >>> spike_times = [0, 2, 5, 6, 7, 8, 15, 20, 21, 22]  # in seconds
    >>> bursts = detect_bursting(spike_times, min_spikes=3, max_isi=1.0)
    >>> print(f"Found {bursts['n_bursts']} bursts")
    >>> print(f"Burst fraction: {bursts['burst_fraction']:.1%}")
    """
    if len(spike_times) < min_spikes:
        return {
            'n_bursts': 0,
            'n_spikes_in_bursts': 0,
            'burst_fraction': 0.0,
            'bursts_per_minute': 0.0,
            'mean_spikes_per_burst': 0.0,
            'mean_burst_duration': 0.0,
            'burst_details': [],
        }
    
    # Convert to seconds if in frames
    spike_times = np.array(spike_times, dtype=float)
    if spike_times.max() > 1000:  # Probably in frames
        spike_times = spike_times / frame_rate
    
    spike_times = np.sort(spike_times)
    
    # Find bursts
    bursts = []
    current_burst = [spike_times[0]]
    
    for i in range(1, len(spike_times)):
        isi = spike_times[i] - spike_times[i-1]
        
        if isi <= max_isi:
            current_burst.append(spike_times[i])
        else:
            if len(current_burst) >= min_spikes:
                bursts.append({
                    'start': current_burst[0],
                    'end': current_burst[-1],
                    'duration': current_burst[-1] - current_burst[0],
                    'n_spikes': len(current_burst),
                })
            current_burst = [spike_times[i]]
    
    # Check last burst
    if len(current_burst) >= min_spikes:
        bursts.append({
            'start': current_burst[0],
            'end': current_burst[-1],
            'duration': current_burst[-1] - current_burst[0],
            'n_spikes': len(current_burst),
        })
    
    # Calculate summary statistics
    n_bursts = len(bursts)
    n_spikes_in_bursts = sum(b['n_spikes'] for b in bursts)
    total_spikes = len(spike_times)
    burst_fraction = n_spikes_in_bursts / total_spikes if total_spikes > 0 else 0
    
    # Recording duration
    recording_duration_min = (spike_times[-1] - spike_times[0]) / 60 if len(spike_times) > 1 else 1
    bursts_per_minute = n_bursts / recording_duration_min if recording_duration_min > 0 else 0
    
    mean_spikes_per_burst = n_spikes_in_bursts / n_bursts if n_bursts > 0 else 0
    mean_burst_duration = np.mean([b['duration'] for b in bursts]) if n_bursts > 0 else 0
    
    return {
        'n_bursts': n_bursts,
        'n_spikes_in_bursts': n_spikes_in_bursts,
        'burst_fraction': burst_fraction,
        'bursts_per_minute': bursts_per_minute,
        'mean_spikes_per_burst': mean_spikes_per_burst,
        'mean_burst_duration': mean_burst_duration,
        'burst_details': bursts,
    }


def calculate_synchrony_metrics(
    pwc_df: pd.DataFrame,
    threshold: float = 0.3
) -> Dict:
    """
    Calculate network synchronization metrics from pairwise correlations.
    
    Parameters
    ----------
    pwc_df : pd.DataFrame
        Pairwise correlation DataFrame from Orb.df_mn_pwc.
        Expected to have correlation values between neuron pairs.
    threshold : float
        Correlation threshold for considering neurons as "synchronized".
        Default is 0.3 (moderate correlation).
    
    Returns
    -------
    dict
        Dictionary containing:
        - global_synchrony: Mean pairwise correlation (overall network synchrony)
        - synchrony_index: Fraction of pairs above threshold
        - std_synchrony: Standard deviation of correlations
        - n_pairs: Number of neuron pairs analyzed
        - highly_correlated_pairs: Number of pairs with r > threshold
        - interpretation: Plain-language explanation
    
    Notes
    -----
    Synchronization metrics can reveal:
    - Functional connectivity in neuronal networks
    - Effects of treatments on network organization
    - Differences in network structure between conditions
    
    Interpretation guidelines:
    - Global synchrony > 0.3: Moderate network synchronization
    - Global synchrony > 0.5: Strong network synchronization
    - Synchrony index > 0.5: Most neurons are synchronized
    
    Examples
    --------
    >>> synchrony = calculate_synchrony_metrics(orb.df_mn_pwc, threshold=0.3)
    >>> print(f"Global synchrony: {synchrony['global_synchrony']:.3f}")
    >>> print(synchrony['interpretation'])
    """
    # Extract correlation values
    # Handle different possible column names
    corr_cols = [c for c in pwc_df.columns if 'corr' in c.lower() or 'pwc' in c.lower()]
    
    if len(corr_cols) > 0:
        correlations = pwc_df[corr_cols[0]].dropna().values
    else:
        # Try to find numeric columns that look like correlations
        numeric_cols = pwc_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Assume the last numeric column is correlations
            correlations = pwc_df[numeric_cols[-1]].dropna().values
        else:
            return {
                'global_synchrony': np.nan,
                'synchrony_index': np.nan,
                'std_synchrony': np.nan,
                'n_pairs': 0,
                'highly_correlated_pairs': 0,
                'interpretation': "Could not identify correlation values in DataFrame.",
            }
    
    # Filter valid correlations (between -1 and 1)
    correlations = correlations[(correlations >= -1) & (correlations <= 1)]
    n_pairs = len(correlations)
    
    if n_pairs == 0:
        return {
            'global_synchrony': np.nan,
            'synchrony_index': np.nan,
            'std_synchrony': np.nan,
            'n_pairs': 0,
            'highly_correlated_pairs': 0,
            'interpretation': "No valid correlation values found.",
        }
    
    # Calculate metrics
    global_synchrony = float(np.mean(correlations))
    std_synchrony = float(np.std(correlations))
    highly_correlated = (correlations > threshold).sum()
    synchrony_index = highly_correlated / n_pairs
    
    # Build interpretation
    if global_synchrony > 0.5:
        sync_level = "high"
        interpretation = (
            f"The network shows high synchronization (mean r = {global_synchrony:.3f}). "
            f"Neurons tend to fire together, suggesting strong functional connectivity."
        )
    elif global_synchrony > 0.3:
        sync_level = "moderate"
        interpretation = (
            f"The network shows moderate synchronization (mean r = {global_synchrony:.3f}). "
            f"Some neurons show coordinated activity patterns."
        )
    elif global_synchrony > 0.1:
        sync_level = "low"
        interpretation = (
            f"The network shows low synchronization (mean r = {global_synchrony:.3f}). "
            f"Neuronal activity is largely independent."
        )
    else:
        sync_level = "minimal"
        interpretation = (
            f"Minimal network synchronization detected (mean r = {global_synchrony:.3f}). "
            f"Neurons appear to fire independently."
        )
    
    interpretation += (
        f" {highly_correlated} of {n_pairs} pairs ({synchrony_index:.1%}) "
        f"exceed the correlation threshold of {threshold}."
    )
    
    return {
        'global_synchrony': global_synchrony,
        'synchrony_index': synchrony_index,
        'std_synchrony': std_synchrony,
        'n_pairs': n_pairs,
        'highly_correlated_pairs': int(highly_correlated),
        'sync_level': sync_level,
        'interpretation': interpretation,
    }

