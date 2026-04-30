# import
## batteries
from typing import Dict, List, Tuple
## 3rd party
import logging
import numpy as np
import scipy.linalg
from skimage.io import imread 
from skimage.measure import label, regionprops
import warnings

# Suppress RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Lazy caiman.deconvolution access
#
# Importing ``caiman`` transitively pulls in TensorFlow, which is slow
# (tens of seconds) and leaves non-daemon threads alive at process exit.
# Anything that *imports* spellbook -- the test suite, the CLI's
# ``--help``, notebooks that just use the metric calculators on
# pre-computed traces -- pays that cost even though only
# :func:`convert_f_to_cs` actually calls into deconvolution.
#
# Defer the import (and the scipy>=1.14 monkey-patch) until the first
# caller asks for it. Subsequent calls reuse the cached handle, so the
# patch is applied exactly once and the import cost is amortised.
# ---------------------------------------------------------------------------
_DECONVOLUTION = None  # populated on first call to _get_deconvolution()


def _get_deconvolution():
    """Return ``caiman.source_extraction.cnmf.deconvolution`` (cached, patched).

    The scipy>=1.14 compatibility patch for ``estimate_time_constant`` is
    applied exactly once on first access. The deconvolution logger is
    silenced at the same time.
    """
    global _DECONVOLUTION
    if _DECONVOLUTION is not None:
        return _DECONVOLUTION

    from caiman.source_extraction.cnmf import deconvolution  # noqa: WPS433

    logging.getLogger(
        "caiman.source_extraction.cnmf.deconvolution"
    ).setLevel(logging.CRITICAL)

    # Patch CaImAn's estimate_time_constant for scipy >=1.14 compatibility.
    # Newer scipy refactored toeplitz() to use batched broadcasting, which
    # fails when CaImAn passes 2D column vectors (e.g. shape (7,1) and
    # (2,1)) instead of 1D arrays. We flatten the slices before calling
    # toeplitz. Done at first-use so a no-op import of spellbook (e.g.
    # in tests, in ``--help`` paths, in notebooks that only call the
    # metric calculators) does not pay the caiman/TensorFlow import cost.
    def _patched_estimate_time_constant(
        fluor, p=2, sn=None, lags=5, fudge_factor=1.0
    ):
        if sn is None:
            sn = deconvolution.GetSn(fluor)

        lags += p
        xc = deconvolution.axcov(fluor, lags)
        xc = xc[:, np.newaxis]

        A = scipy.linalg.toeplitz(
            xc[lags + np.arange(lags)].flatten(),
            xc[lags + np.arange(p)].flatten(),
        ) - sn ** 2 * np.eye(lags, p)
        g = np.linalg.lstsq(A, xc[lags + 1:], rcond=None)[0]
        gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
        gr = (gr + gr.conjugate()) / 2.0
        np.random.seed(45)
        gr[gr > 1] = 0.95 + np.random.normal(0, 0.01, np.sum(gr > 1))
        gr[gr < 0] = 0.15 + np.random.normal(0, 0.01, np.sum(gr < 0))
        g = np.poly(fudge_factor * gr)
        g = -g[1:]
        return g.flatten()

    deconvolution.estimate_time_constant = _patched_estimate_time_constant

    _DECONVOLUTION = deconvolution
    return _DECONVOLUTION

# functions
def convert_f_to_cs(fluorescence_data: np.ndarray, p: int=2, noise_range: list=[0.25, 0.5]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts fluorescence data to calcium and spike signals using deconvolution.
    
    Args:
        fluorescence_data: Fluorescence data matrix with neurons as rows and time points as columns.
        p: Order of the autoregressive process.
        noise_range: Range for estimating noise.
    
    Returns:
        calcium_signal: Calcium signal matrix.
        spike_signal: Spike signal matrix.
    """
    deconvolution = _get_deconvolution()

    # Initialize arrays for calcium and spike signals
    calcium_signal = np.zeros_like(fluorescence_data)
    spike_signal = np.zeros_like(fluorescence_data)

    # Iterate over each neuron's fluorescence data
    for i in range(fluorescence_data.shape[0]):
        fluorescence_trace = np.copy(fluorescence_data[i, :])
        
        # Perform deconvolution to extract calcium and spike signals
        calcium, _, _, _, _, spikes, _ = deconvolution.constrained_foopsi(
            fluorescence_trace, bl=None, c1=None, g=None, sn=None, p=p,
            method_deconvolution='oasis', bas_nonneg=True, noise_range=noise_range,
            noise_method='logmexp', lags=5, fudge_factor=1.0, verbosity=False,
            solvers=None, optimize_g=0
        )
        # Store the results in the respective arrays
        calcium_signal[i, :] = calcium
        spike_signal[i, :] = spikes
    
    return calcium_signal, spike_signal

def calc_rise_tm(calcium_signals: np.ndarray, spike_zscores: np.ndarray, 
                 zscore_threshold: float=3) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Calculates the rise time of calcium signals based on spike detection.

    Walks each neuron's spike-z-score trace looking for above-threshold
    crossings, then for each crossing climbs the calcium trace until it
    starts to decrease. Every per-event metric in this module
    (``calc_rise_tm``, ``calc_fall_tm``, ``calc_fwhm_spikes``,
    ``calc_peak_amplitude``, ``calc_peak_to_peak``) shares this same walk
    so that the i-th entry of each output corresponds to the same event
    for a given neuron. The walk is anchored on:

    * ``spikes_above_threshold = spike_zscores >= zscore_threshold``
      (inclusive comparison — must match the other walkers).
    * After finding the post-peak frame ``j`` (first frame where the
      calcium trace stops increasing), the next event search resumes at
      ``index = j + 1``.

    Downstream code in ``cauldron._apply_event_filters`` relies on the
    i-th-event correspondence to apply a single positional keep-mask to
    every per-event metric. Diverging from these two invariants (using
    a different threshold operator or skipping further past the peak)
    will silently break that mask.

    Args:
        calcium_signals: Calcium signal matrix with neurons as rows and time points as columns.
        spike_zscores: Z-scored spike signal matrix.
        zscore_threshold: Z-score threshold for spike detection.
    
    Returns:
        rise_times: Dictionary of rise times for each neuron.
        rise_positions: Dictionary of positions corresponding to the rise times for each neuron.
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

def calc_fwhm_spikes(calcium_signals: np.ndarray, spike_zscores: np.ndarray, 
                     zscore_threshold: float=3, percentage_threshold: float=0.2
                     ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Calculates the full width at half maximum (FWHM) of spikes in calcium signals.
    
    Args:
        calcium_signals: Calcium signal matrix with neurons as rows and time points as columns.
        spike_zscores: Z-scored spike signal matrix.
        zscore_threshold: Z-score threshold for spike detection.
        percentage_threshold: Percentage threshold for determining half maximum.
    
    Returns:
        fwhm_backward_positions: Dictionary of backward positions of FWHM for each neuron.
        fwhm_forward_positions: Dictionary of forward positions of FWHM for each neuron.
        fwhm_values: Dictionary of FWHM values for each neuron.
        spike_counts: Dictionary of the number of spikes within the FWHM for each neuron.
    """
    fwhm_values = {}
    fwhm_backward_positions = {}
    fwhm_forward_positions = {}
    spike_counts = {}
    
    # Iterate over each neuron
    for neuron_idx in range(calcium_signals.shape[0]):
        # Identify spike events based on z-score threshold. Use ``>=`` to
        # match the canonical walk in ``calc_rise_tm`` so the i-th event
        # produced here aligns with the i-th rise/peak/fall/peak-to-peak
        # event for this neuron (see ``calc_rise_tm`` docstring).
        spikes_above_threshold = spike_zscores[neuron_idx] >= zscore_threshold
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
                
                # j is the first non-increasing frame; the peak sits at j-1.
                # We deliberately do NOT decrement j itself — it's reused
                # below to advance ``index`` in lockstep with calc_rise_tm
                # / calc_peak_amplitude.
                peak_pos = j - 1
                half_max_value = percentage_threshold * (prev_calcium - calcium_trace[index]) + calcium_trace[index]
                
                # Find the backward index where the calcium value falls below half maximum
                backward_index = peak_pos - 1
                while backward_index >= 0 and calcium_trace[backward_index] >= half_max_value:
                    backward_index -= 1
                backward_index += 1
                neuron_fwhm_backward_positions.append(backward_index)
                
                # Find the forward index where the calcium value falls below half maximum
                forward_index = peak_pos + 1
                while forward_index < len(calcium_trace) - 10 and calcium_trace[forward_index] >= half_max_value:
                    forward_index += 1
                forward_index -= 1
                neuron_fwhm_forward_positions.append(forward_index)
                
                # Calculate the FWHM value
                fwhm_value = forward_index - backward_index + 1
                neuron_fwhm_values.append(fwhm_value)
                
                # Count the number of spikes within the FWHM
                neuron_spike_counts.append(1 + np.sum(spikes_above_threshold[backward_index:(forward_index + 1)]))
                
                # Advance to ``j + 1`` (one past the peak), matching the
                # canonical walk. Earlier versions advanced past
                # ``forward_index``, which silently dropped any events
                # whose rise fell inside the previous event's FWHM
                # window, breaking the i-th-event correspondence with
                # rise/fall/peak/peak-to-peak.
                index = j + 1
            else:
                index += 1
        
        # Store FWHM positions, values, and spike counts for each neuron
        fwhm_backward_positions[neuron_idx] = neuron_fwhm_backward_positions
        fwhm_forward_positions[neuron_idx] = neuron_fwhm_forward_positions
        fwhm_values[neuron_idx] = neuron_fwhm_values
        spike_counts[neuron_idx] = neuron_spike_counts
    
    return fwhm_backward_positions, fwhm_forward_positions, fwhm_values, spike_counts

def calc_frpm(zscored_spike_events: np.ndarray, neuron_ids: np.ndarray, fps: int, 
              zscore_threshold: int=5) -> float:
    """
    Calculates the firing rate per minute (FRPM) for given z-scored spike event data.
    
    Args:
        zscored_spike_events: Z-scored spike events with neurons as rows and time points as columns.
        neuron_ids: Array containing neuron IDs.
        fps: Frames per second of the recording.
        zscore_threshold: Z-score threshold for detecting spikes.
    
    Returns:
        frpm: Average firing rate per minute for the dataset.
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
    spike_sums_per_minute = spike_sums * fps * 60 / valid_spike_zscores.shape[1]

    spike_dict = {}

    for spike in range(len(spike_sums_per_minute)):
        spike_dict[neuron_ids[spike]] = spike_sums_per_minute[spike]
    
    # Calculate the average firing rate per minute
    frpm = np.mean(spike_sums_per_minute)
    
    return frpm, spike_dict

def calc_fall_tm(calcium_signals: np.ndarray, spike_zscores: np.ndarray,
                 zscore_threshold: float = 3, baseline_fraction: float = 0.1
                 ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Calculates the fall time of calcium signals (time from peak to return to baseline).
    
    Args:
        calcium_signals: Calcium signal matrix with neurons as rows and time points as columns.
        spike_zscores: Z-scored spike signal matrix.
        zscore_threshold: Z-score threshold for spike detection.
        baseline_fraction: Fraction of peak amplitude to consider as returned to baseline (default 0.1 = 10%).
    
    Returns:
        fall_times: Dictionary of fall times (in frames) for each neuron.
        fall_positions: Dictionary of peak positions corresponding to the fall times.
    """
    fall_times = {}
    fall_positions = {}
    
    # Iterate over each neuron
    for neuron_idx in range(calcium_signals.shape[0]):
        # Identify spike events based on z-score threshold
        spikes_above_threshold = spike_zscores[neuron_idx] >= zscore_threshold
        calcium_trace = calcium_signals[neuron_idx]
        
        neuron_fall_times = []
        neuron_fall_positions = []
        
        index = 0
        # Loop through the calcium signal to find fall times
        while index < len(spikes_above_threshold) - 10:
            if spikes_above_threshold[index]:
                # Find the peak (same logic as calc_rise_tm)
                j = index + 1
                prev_calcium = calcium_trace[index]
                
                while j < len(calcium_trace) - 2 and calcium_trace[j] >= prev_calcium:
                    prev_calcium = calcium_trace[j]
                    j += 1
                
                # j-1 is the peak position, prev_calcium is the peak value
                peak_pos = j - 1
                peak_value = prev_calcium
                baseline_value = calcium_trace[index]
                
                # Calculate threshold for "return to baseline" (baseline + fraction of amplitude)
                fall_threshold = baseline_value + baseline_fraction * (peak_value - baseline_value)
                
                # Find when signal falls below threshold
                fall_time = 0
                k = peak_pos + 1
                while k < len(calcium_trace) - 2 and calcium_trace[k] >= fall_threshold:
                    fall_time += 1
                    k += 1
                
                # Record the fall time and peak position
                neuron_fall_times.append(fall_time)
                neuron_fall_positions.append(peak_pos)
                # Advance to ``j + 1`` (one past the peak) — the canonical
                # walk shared with calc_rise_tm / calc_peak_amplitude /
                # calc_peak_to_peak. Earlier versions advanced to ``k + 1``
                # (past the fall), which silently dropped any events whose
                # rise sat inside the previous event's fall window and
                # broke the i-th-event correspondence relied on by
                # ``cauldron._apply_event_filters``.
                index = j + 1
            else:
                index += 1
        
        # Store fall times and positions for each neuron
        fall_times[neuron_idx] = neuron_fall_times
        fall_positions[neuron_idx] = neuron_fall_positions
    
    return fall_times, fall_positions


def calc_peak_amplitude(calcium_signals: np.ndarray, spike_zscores: np.ndarray,
                        zscore_threshold: float = 3, dff_data: np.ndarray = None
                        ) -> Tuple[Dict[int, List[float]], Dict[int, List[int]]]:
    """
    Calculates the amplitude (height) of each calcium transient peak.
    
    Uses deconvolved calcium signals for spike detection/timing, but measures
    amplitudes from raw ΔF/F₀ data if provided (recommended for interpretable units).
    
    Args:
        calcium_signals: Calcium signal matrix with neurons as rows and time points as columns.
                        Used for spike detection and peak finding.
        spike_zscores: Z-scored spike signal matrix.
        zscore_threshold: Z-score threshold for spike detection.
        dff_data: Optional raw ΔF/F₀ data matrix. If provided, amplitudes are measured
                 from this data (in ΔF/F₀ units). If None, amplitudes are measured
                 from deconvolved calcium_signals (arbitrary units).
    
    Returns:
        peak_amplitudes: Dictionary of peak amplitude values (baseline-subtracted) for each neuron.
                        Units are ΔF/F₀ if dff_data provided, otherwise arbitrary deconvolved units.
        peak_positions: Dictionary of positions of each peak.
    """
    peak_amplitudes = {}
    peak_positions = {}
    
    # Use raw ΔF/F₀ for amplitude measurement if provided, otherwise use deconvolved signal
    amplitude_source = dff_data if dff_data is not None else calcium_signals
    
    # Iterate over each neuron
    for neuron_idx in range(calcium_signals.shape[0]):
        # Identify spike events based on z-score threshold
        spikes_above_threshold = spike_zscores[neuron_idx] >= zscore_threshold
        calcium_trace = calcium_signals[neuron_idx]
        amplitude_trace = amplitude_source[neuron_idx]
        
        neuron_peak_amplitudes = []
        neuron_peak_positions = []
        
        index = 0
        # Loop through the calcium signal to find peak amplitudes
        while index < len(spikes_above_threshold) - 10:
            if spikes_above_threshold[index]:
                # Find the peak using deconvolved signal (better for timing)
                j = index + 1
                prev_calcium = calcium_trace[index]
                
                while j < len(calcium_trace) - 2 and calcium_trace[j] >= prev_calcium:
                    prev_calcium = calcium_trace[j]
                    j += 1
                
                # j-1 is the peak position
                peak_pos = j - 1
                
                # Measure amplitude from the amplitude source (raw ΔF/F₀ or deconvolved)
                baseline_value = amplitude_trace[index]
                peak_value = amplitude_trace[peak_pos]
                amplitude = peak_value - baseline_value
                
                # Record the amplitude and position
                neuron_peak_amplitudes.append(amplitude)
                neuron_peak_positions.append(peak_pos)
                index = j + 1
            else:
                index += 1
        
        # Store amplitudes and positions for each neuron
        peak_amplitudes[neuron_idx] = neuron_peak_amplitudes
        peak_positions[neuron_idx] = neuron_peak_positions
    
    return peak_amplitudes, peak_positions


def calc_peak_to_peak(calcium_signals: np.ndarray, spike_zscores: np.ndarray,
                      zscore_threshold: float = 3) -> Dict[int, List[int]]:
    """
    Calculates the inter-spike intervals (peak-to-peak distances) for each neuron.
    
    Args:
        calcium_signals: Calcium signal matrix with neurons as rows and time points as columns.
        spike_zscores: Z-scored spike signal matrix.
        zscore_threshold: Z-score threshold for spike detection.
    
    Returns:
        inter_spike_intervals: Dictionary of intervals (in frames) between consecutive peaks for each neuron.
    """
    inter_spike_intervals = {}
    
    # First, get peak positions using the same logic as other functions
    # Iterate over each neuron
    for neuron_idx in range(calcium_signals.shape[0]):
        # Identify spike events based on z-score threshold
        spikes_above_threshold = spike_zscores[neuron_idx] >= zscore_threshold
        calcium_trace = calcium_signals[neuron_idx]
        
        peak_positions = []
        
        index = 0
        # Loop through the calcium signal to find peak positions
        while index < len(spikes_above_threshold) - 10:
            if spikes_above_threshold[index]:
                # Find the peak (same logic as calc_rise_tm)
                j = index + 1
                prev_calcium = calcium_trace[index]
                
                while j < len(calcium_trace) - 2 and calcium_trace[j] >= prev_calcium:
                    prev_calcium = calcium_trace[j]
                    j += 1
                
                # j-1 is the peak position
                peak_pos = j - 1
                peak_positions.append(peak_pos)
                index = j + 1
            else:
                index += 1
        
        # Calculate inter-spike intervals (differences between consecutive peaks)
        if len(peak_positions) >= 2:
            intervals = [peak_positions[i+1] - peak_positions[i] 
                        for i in range(len(peak_positions) - 1)]
        else:
            intervals = []  # Need at least 2 peaks to calculate intervals
        
        inter_spike_intervals[neuron_idx] = intervals
    
    return inter_spike_intervals


def calc_mask_shape_metrics(mask_image: np.ndarray) -> Dict[str, float]:
    """
    Loads a binary mask image and calculates roundness, diameter, and area of the masked spheroid/organoid.
    Returns None if the input is None.

    Args:
        mask_image: Binary mask image of the spheroid/organoid.

    Returns:
        Dictionary containing roundness, diameter, and area of the masked object.
    """
    if mask_image is None:
        return {}
    
    # Get mask properties
    labeled_image = label(mask_image)
    properties = regionprops(labeled_image)

    # Calculate shape metrics
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
    return {}

