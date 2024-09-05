import numpy as np
from skimage.io import imread 
from skimage.measure import label, regionprops
from caiman.source_extraction.cnmf import deconvolution


def convert_f_to_cs(fluorescence_data, p=2, noise_range=[0.25, 0.5]):
    """
    Converts fluorescence data to calcium and spike signals using deconvolution.
    
    Parameters:
    fluorescence_data (ndarray): Fluorescence data matrix with neurons as rows and time points as columns.
    p (int): Order of the autoregressive process.
    noise_range (list): Range for estimating noise.
    
    Returns:
    calcium_signal (ndarray): Calcium signal matrix.
    spike_signal (ndarray): Spike signal matrix.
    """
    # Initialize arrays for calcium and spike signals
    calcium_signal = np.zeros_like(fluorescence_data)
    spike_signal = np.zeros_like(fluorescence_data)
    
    # Iterate over each neuron's fluorescence data
    for i in range(fluorescence_data.shape[0]):
        fluorescence_trace = np.copy(fluorescence_data[i, :])
        
        # Perform deconvolution to extract calcium and spike signals
        calcium, baseline, c1, g, sn, spikes, lam = deconvolution.constrained_foopsi(
            fluorescence_trace, bl=None, c1=None, g=None, sn=None, p=p,
            method_deconvolution='oasis', bas_nonneg=True, noise_range=noise_range,
            noise_method='logmexp', lags=5, fudge_factor=1.0, verbosity=True,
            solvers=None, optimize_g=0
        )
        # Store the results in the respective arrays
        calcium_signal[i, :] = calcium
        spike_signal[i, :] = spikes
    
    return calcium_signal, spike_signal

def calc_rise_tm(calcium_signals, spike_zscores, zscore_threshold=3):
    """
    Calculates the rise time of calcium signals based on spike detection.
    
    Parameters:
    calcium_signals (ndarray): Calcium signal matrix with neurons as rows and time points as columns.
    spike_zscores (ndarray): Z-scored spike signal matrix.
    zscore_threshold (float): Z-score threshold for spike detection.
    
    Returns:
    rise_times (dict): Dictionary of rise times for each neuron.
    rise_positions (dict): Dictionary of positions corresponding to the rise times for each neuron.
    """
    rise_times = {}
    rise_positions = {}
    
    # Iterate over each neuron
    for neuron_idx in range(calcium_signals.shape[0]):
        # Identify spike events based on z-score threshold
        spikes_above_threshold = spike_zscores[neuron_idx] >= zscore_threshold
        calcium_trace = calcium_signals[neuron_idx]
        
        neuron_rise_times = []
        neuron_rise_positions = []
        
        index = 0
        # Loop through the calcium signal to find rise times
        while index < len(spikes_above_threshold) - 10:
            if spikes_above_threshold[index]:
                j = index + 1
                prev_calcium = calcium_trace[index]
                rise_time = 0
                
                # Calculate the rise time by comparing subsequent calcium values
                while j < len(calcium_trace) - 2 and calcium_trace[j] >= prev_calcium:
                    rise_time += 1
                    prev_calcium = calcium_trace[j]
                    j += 1
                
                # Record the rise time and position
                neuron_rise_times.append(rise_time)
                neuron_rise_positions.append(j)
                index = j + 1
            else:
                index += 1
        
        # Store rise times and positions for each neuron
        rise_times[neuron_idx] = neuron_rise_times
        rise_positions[neuron_idx] = neuron_rise_positions
    
    return rise_times, rise_positions

def calc_fwhm_spikes(calcium_signals, spike_zscores, zscore_threshold=3, percentage_threshold=0.2):
    """
    Calculates the full width at half maximum (FWHM) of spikes in calcium signals.
    
    Parameters:
    calcium_signals (ndarray): Calcium signal matrix with neurons as rows and time points as columns.
    spike_zscores (ndarray): Z-scored spike signal matrix.
    zscore_threshold (float): Z-score threshold for spike detection.
    percentage_threshold (float): Percentage threshold for determining half maximum.
    
    Returns:
    fwhm_backward_positions (dict): Dictionary of backward positions of FWHM for each neuron.
    fwhm_forward_positions (dict): Dictionary of forward positions of FWHM for each neuron.
    fwhm_values (dict): Dictionary of FWHM values for each neuron.
    spike_counts (dict): Dictionary of the number of spikes within the FWHM for each neuron.
    """
    fwhm_values = {}
    fwhm_backward_positions = {}
    fwhm_forward_positions = {}
    spike_counts = {}
    
    # Iterate over each neuron
    for neuron_idx in range(calcium_signals.shape[0]):
        # Identify spike events based on z-score threshold
        spikes_above_threshold = spike_zscores[neuron_idx] > zscore_threshold
        calcium_trace = calcium_signals[neuron_idx]
        
        neuron_fwhm_backward_positions = []
        neuron_fwhm_forward_positions = []
        neuron_fwhm_values = []
        neuron_spike_counts = []
        
        index = 0
        # Loop through the calcium signal to find FWHM values
        while index < len(spikes_above_threshold) - 10:
            if spikes_above_threshold[index]:
                j = index + 1
                prev_calcium = calcium_trace[index]
                
                # Identify the peak of the spike
                while j < len(calcium_trace) - 2 and calcium_trace[j] >= prev_calcium:
                    prev_calcium = calcium_trace[j]
                    j += 1
                
                j -= 1
                half_max_value = percentage_threshold * (prev_calcium - calcium_trace[index]) + calcium_trace[index]
                
                # Find the backward index where the calcium value falls below half maximum
                backward_index = j - 1
                while backward_index >= 0 and calcium_trace[backward_index] >= half_max_value:
                    backward_index -= 1
                backward_index += 1
                neuron_fwhm_backward_positions.append(backward_index)
                
                # Find the forward index where the calcium value falls below half maximum
                forward_index = j + 1
                while forward_index < len(calcium_trace) - 10 and calcium_trace[forward_index] >= half_max_value:
                    forward_index += 1
                forward_index -= 1
                neuron_fwhm_forward_positions.append(forward_index)
                
                # Calculate the FWHM value
                fwhm_value = forward_index - backward_index + 1
                neuron_fwhm_values.append(fwhm_value)
                
                # Count the number of spikes within the FWHM
                neuron_spike_counts.append(1 + np.sum(spikes_above_threshold[backward_index:(forward_index + 1)]))
                
                index = forward_index + 1
            else:
                index += 1
        
        # Store FWHM positions, values, and spike counts for each neuron
        fwhm_backward_positions[neuron_idx] = neuron_fwhm_backward_positions
        fwhm_forward_positions[neuron_idx] = neuron_fwhm_forward_positions
        fwhm_values[neuron_idx] = neuron_fwhm_values
        spike_counts[neuron_idx] = neuron_spike_counts
    
    return fwhm_backward_positions, fwhm_forward_positions, fwhm_values, spike_counts

def calc_frpm(zscored_spike_events, neuron_ids, zscore_threshold=5):
    """
    Calculates the firing rate per minute (FRPM) for given z-scored spike event data.
    
    Parameters:
    zscored_spike_events (ndarray): Z-scored spike events with neurons as rows and time points as columns.
    neuron_ids (ndarray): Array containing neuron IDs.
    zscore_threshold (int): Z-score threshold for detecting spikes.
    
    Returns:
    frpm (float): Average firing rate per minute for the dataset.
    """
    # Filter z-scored spike events for valid neuron IDs
    valid_spike_zscores = zscored_spike_events[neuron_ids, :]
    
    # Threshold the z-scored spikes to binary events
    spikes_above_threshold = valid_spike_zscores >= zscore_threshold
    valid_spike_zscores[spikes_above_threshold] = 1  # Spikes (above threshold) set to 1
    valid_spike_zscores[~spikes_above_threshold] = 0  # Non-spikes (below threshold) set to 0
    
    # Sum the spikes for each neuron across all time points
    spike_sums = np.sum(valid_spike_zscores, axis=1)
    
    # Normalize to get the firing rate per minute
    # Multiply by 30 (frames per second) and 60 (seconds per minute), and divide by the number of frames
    spike_sums_per_minute = spike_sums * 30 * 60 / valid_spike_zscores.shape[1]

    spike_dict = {}

    for spike in range(len(spike_sums_per_minute)):
        spike_dict[neuron_ids[spike]] = spike_sums_per_minute[spike]
    
    # Calculate the average firing rate per minute
    frpm = np.mean(spike_sums_per_minute)
    
    return frpm, spike_dict

def calc_mask_shape_metrics(mask_path):
    """
    Loads a binary mask image and calculates roundness, diameter, and area of the masked spheroid/organoid.

    Args:
        mask_path (str): Path to the mask image.

    Returns:
        dict: Dictionary containing roundness, diameter, and area of the masked object.
    """
    try:
        mask_image = imread(mask_path)
        labeled_image = label(mask_image)
        properties = regionprops(labeled_image)

        if properties:
            prop = properties[0]  # Assuming there's only one region in the mask
            area = prop.area
            perimeter = prop.perimeter
            diameter = prop.equivalent_diameter
            roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            return {
                'roundness': roundness,
                'diameter': diameter,
                'area': area
            }
        else:
            return {
                'roundness': None,
                'diameter': None,
                'area': None
            }
    except Exception as e:
        print(f"Error calculating mask metrics: {e}")
        return {
            'roundness': None,
            'diameter': None,
            'area': None
        }
