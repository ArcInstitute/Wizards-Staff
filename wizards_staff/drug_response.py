# Baseline-Normalized Drug Response Analysis
"""
Drug response analysis module for paired baseline-dosing experiments.

This module provides functionality to:
1. Link paired samples (baseline + dosed) across two Lizard-Wizard runs
2. Normalize post-drug metrics to each sample's own baseline
3. Compute and visualize drug-induced fold-changes or percent changes

"""

# imports
## batteries
import os
import sys
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

## 3rd party
import numpy as np
import pandas as pd

## package
from wizards_staff.logger import init_custom_logger

if TYPE_CHECKING:
    from wizards_staff.wizards.orb import Orb
    from wizards_staff.wizards.shard import Shard

# Initialize logger
_logger = init_custom_logger(__name__)


# =============================================================================
# Metric Configuration
# =============================================================================

# Maps metric names to their data attributes and value columns
DRUG_RESPONSE_METRICS = {
    "frpm": {
        "data_attr": "frpm_data",
        "value_col": "Firing Rate Per Min",
        "display_name": "Firing Rate (spikes/min)",
        "neuron_col": "Neuron Index",
        "time_convertible": False,
    },
    "rise_time": {
        "data_attr": "rise_time_data",
        "value_col": "Rise Times",
        "display_name": "Rise Time",
        "neuron_col": "Neuron",
        "time_convertible": True,
    },
    "fwhm": {
        "data_attr": "fwhm_data",
        "value_col": "FWHM Values",
        "display_name": "FWHM (Event Duration)",
        "neuron_col": "Neuron",
        "time_convertible": True,
    },
    "pwc": {
        "data_attr": "df_mn_pwc",
        "value_col": None,  # PWC has different structure (groups as columns)
        "display_name": "Pairwise Correlation",
        "neuron_col": None,  # Already sample-level
        "time_convertible": False,
    },
}


# =============================================================================
# Sample Pairing Functions
# =============================================================================

def pair_samples_across_folders(
        baseline_orb: "Orb",
        dosing_orb: "Orb",
        pair_by: str = "Well"
    ) -> Dict[str, Dict[str, "Shard"]]:
    """
    Match samples between baseline and dosing Orbs based on a shared identifier.
    
    Samples are paired by matching values in a metadata column (default: "Well").
    For example, if baseline has sample "A1_baseline" with Well="A1" and dosing
    has sample "A1_drug" with Well="A1", they will be paired.
    
    Parameters
    ----------
    baseline_orb : Orb
        Orb object containing baseline (pre-drug) samples.
    dosing_orb : Orb
        Orb object containing dosing (post-drug) samples.
    pair_by : str, default "Well"
        Column name in metadata used to match samples across folders.
        Both Orbs must have this column in their metadata.
    
    Returns
    -------
    dict
        Dictionary mapping pair identifiers to their baseline and dosing Shards:
        {
            "A1": {"baseline": Shard, "dosing": Shard, "baseline_sample": "A1_pre", "dosing_sample": "A1_post"},
            "A2": {"baseline": Shard, "dosing": Shard, "baseline_sample": "A2_pre", "dosing_sample": "A2_post"},
            ...
        }
    
    Raises
    ------
    ValueError
        If pair_by column is not found in either Orb's metadata.
    
    Warnings
    --------
    Logs warnings for:
    - Samples in baseline without matching dosing sample
    - Samples in dosing without matching baseline sample
    - Duplicate pair_by values within a single Orb
    
    Examples
    --------
    >>> pairs = pair_samples_across_folders(orb_baseline, orb_dosing, pair_by="Well")
    >>> for pair_id, samples in pairs.items():
    ...     print(f"Pair {pair_id}: baseline={samples['baseline_sample']}, dosing={samples['dosing_sample']}")
    """
    # Validate pair_by column exists in both Orbs
    if pair_by not in baseline_orb.metadata.columns:
        raise ValueError(
            f"Column '{pair_by}' not found in baseline Orb metadata. "
            f"Available columns: {list(baseline_orb.metadata.columns)}"
        )
    if pair_by not in dosing_orb.metadata.columns:
        raise ValueError(
            f"Column '{pair_by}' not found in dosing Orb metadata. "
            f"Available columns: {list(dosing_orb.metadata.columns)}"
        )
    
    # Build lookup: pair_by value -> sample name for each orb
    baseline_lookup = {}
    for _, row in baseline_orb.metadata.iterrows():
        pair_key = row[pair_by]
        sample_name = row['Sample']
        if pair_key in baseline_lookup:
            _logger.warning(
                f"Duplicate '{pair_by}' value '{pair_key}' in baseline Orb. "
                f"Sample '{sample_name}' will override '{baseline_lookup[pair_key]}'."
            )
        baseline_lookup[pair_key] = sample_name
    
    dosing_lookup = {}
    for _, row in dosing_orb.metadata.iterrows():
        pair_key = row[pair_by]
        sample_name = row['Sample']
        if pair_key in dosing_lookup:
            _logger.warning(
                f"Duplicate '{pair_by}' value '{pair_key}' in dosing Orb. "
                f"Sample '{sample_name}' will override '{dosing_lookup[pair_key]}'."
            )
        dosing_lookup[pair_key] = sample_name
    
    # Find matching pairs
    paired_samples = {}
    baseline_keys = set(baseline_lookup.keys())
    dosing_keys = set(dosing_lookup.keys())
    
    # Matched pairs
    matched_keys = baseline_keys & dosing_keys
    for pair_key in matched_keys:
        baseline_sample = baseline_lookup[pair_key]
        dosing_sample = dosing_lookup[pair_key]
        
        # Get the Shard objects
        baseline_shard = baseline_orb._shards.get(baseline_sample)
        dosing_shard = dosing_orb._shards.get(dosing_sample)
        
        if baseline_shard is None:
            _logger.warning(f"Baseline sample '{baseline_sample}' not found in shards. Skipping pair '{pair_key}'.")
            continue
        if dosing_shard is None:
            _logger.warning(f"Dosing sample '{dosing_sample}' not found in shards. Skipping pair '{pair_key}'.")
            continue
        
        paired_samples[pair_key] = {
            "baseline": baseline_shard,
            "dosing": dosing_shard,
            "baseline_sample": baseline_sample,
            "dosing_sample": dosing_sample,
        }
    
    # Warn about unpaired samples
    baseline_only = baseline_keys - dosing_keys
    dosing_only = dosing_keys - baseline_keys
    
    if baseline_only:
        _logger.warning(
            f"Unpaired baseline samples (no matching dosing): {sorted(baseline_only)}"
        )
    if dosing_only:
        _logger.warning(
            f"Unpaired dosing samples (no matching baseline): {sorted(dosing_only)}"
        )
    
    # Summary
    _logger.info(
        f"Sample pairing complete: {len(paired_samples)} pairs found, "
        f"{len(baseline_only)} baseline-only, {len(dosing_only)} dosing-only"
    )
    
    return paired_samples


def get_paired_metadata(
        baseline_orb: "Orb",
        dosing_orb: "Orb",
        pair_by: str = "Well"
    ) -> pd.DataFrame:
    """
    Create a metadata DataFrame showing paired samples.
    
    Parameters
    ----------
    baseline_orb : Orb
        Orb object containing baseline samples.
    dosing_orb : Orb
        Orb object containing dosing samples.
    pair_by : str, default "Well"
        Column name used for pairing.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: pair_id, baseline_sample, dosing_sample, 
        plus any shared metadata columns.
    """
    pairs = pair_samples_across_folders(baseline_orb, dosing_orb, pair_by)
    
    rows = []
    for pair_id, pair_info in pairs.items():
        row = {
            "pair_id": pair_id,
            "baseline_sample": pair_info["baseline_sample"],
            "dosing_sample": pair_info["dosing_sample"],
        }
        # Add metadata from baseline (they should match for paired samples)
        baseline_meta = baseline_orb.metadata[
            baseline_orb.metadata["Sample"] == pair_info["baseline_sample"]
        ].iloc[0].to_dict()
        
        # Add non-sample columns from metadata
        for col, val in baseline_meta.items():
            if col not in ["Sample", pair_by]:
                row[col] = val
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# =============================================================================
# Aggregation Functions
# =============================================================================

def aggregate_to_sample_level(
        neuron_df: pd.DataFrame,
        sample_col: str = "Sample",
        value_col: str = None,
        agg_func: str = "mean"
    ) -> pd.DataFrame:
    """
    Aggregate neuron-level metrics to sample-level summary statistics.
    
    This is critical for avoiding pseudoreplication in paired statistics.
    Each sample should be represented by a single value (e.g., mean of all neurons)
    rather than treating each neuron as an independent observation.
    
    Parameters
    ----------
    neuron_df : pd.DataFrame
        DataFrame with neuron-level data. Must contain the sample_col.
    sample_col : str, default "Sample"
        Column name identifying which sample each neuron belongs to.
    value_col : str, optional
        Column name containing the metric values to aggregate.
        If None, aggregates all numeric columns.
    agg_func : str, default "mean"
        Aggregation function: "mean", "median", "sum", "std", "count".
    
    Returns
    -------
    pd.DataFrame
        Sample-level aggregated data with one row per sample.
    
    Examples
    --------
    >>> # Aggregate FRPM data to sample means
    >>> sample_frpm = aggregate_to_sample_level(
    ...     orb.frpm_data, 
    ...     value_col="Firing Rate Per Min",
    ...     agg_func="mean"
    ... )
    """
    if neuron_df is None or len(neuron_df) == 0:
        return pd.DataFrame()
    
    df = neuron_df.copy()
    
    # Determine aggregation function
    agg_funcs = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "sum": np.nansum,
        "std": np.nanstd,
        "count": "count",
    }
    
    if agg_func not in agg_funcs:
        raise ValueError(f"Unknown agg_func '{agg_func}'. Options: {list(agg_funcs.keys())}")
    
    func = agg_funcs[agg_func]
    
    # If specific value column provided, aggregate just that column
    if value_col is not None:
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found in DataFrame")
        
        # Get non-numeric columns to preserve (metadata)
        # Group by sample and aggregate the value column
        result = df.groupby(sample_col).agg({value_col: func}).reset_index()
        
        # Merge back metadata (first row per sample)
        metadata_cols = [col for col in df.columns 
                        if col != value_col and col != sample_col 
                        and not pd.api.types.is_numeric_dtype(df[col])]
        if metadata_cols:
            metadata = df.groupby(sample_col)[metadata_cols].first().reset_index()
            result = result.merge(metadata, on=sample_col, how="left")
        
        return result
    
    # Otherwise aggregate all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if sample_col in numeric_cols:
        numeric_cols.remove(sample_col)
    
    result = df.groupby(sample_col)[numeric_cols].agg(func).reset_index()
    
    # Merge back non-numeric metadata
    metadata_cols = [col for col in df.columns 
                    if col not in numeric_cols and col != sample_col]
    if metadata_cols:
        metadata = df.groupby(sample_col)[metadata_cols].first().reset_index()
        result = result.merge(metadata, on=sample_col, how="left")
    
    return result


# =============================================================================
# Normalization Functions
# =============================================================================

def normalize_metric_to_baseline(
        baseline_values: Union[pd.DataFrame, pd.Series, np.ndarray],
        dosing_values: Union[pd.DataFrame, pd.Series, np.ndarray],
        method: str = "fold_change",
        zero_handling: str = "nan",
        epsilon: float = 1e-10
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Compute normalized values comparing dosing to baseline.
    
    Parameters
    ----------
    baseline_values : array-like
        Baseline (pre-drug) metric values.
    dosing_values : array-like
        Dosing (post-drug) metric values. Must have same shape as baseline_values.
    method : str, default "fold_change"
        Normalization method:
        - "fold_change": dosing / baseline (1.0 = no change, 2.0 = doubled)
        - "percent_change": ((dosing - baseline) / baseline) * 100 (0% = no change)
        - "delta": dosing - baseline (0 = no change)
        - "log2_fold_change": log2(dosing / baseline) (0 = no change)
    zero_handling : str, default "nan"
        How to handle zero baseline values:
        - "nan": Return NaN for those entries
        - "epsilon": Add small epsilon to baseline before division
        - "skip": Return original dosing value (fold_change of infinity)
    epsilon : float, default 1e-10
        Small value added to baseline when zero_handling="epsilon".
    
    Returns
    -------
    Same type as input
        Normalized values.
    
    Raises
    ------
    ValueError
        If shapes don't match or unknown method specified.
    
    Examples
    --------
    >>> baseline = np.array([10, 20, 0, 5])
    >>> dosing = np.array([15, 20, 10, 10])
    >>> normalize_metric_to_baseline(baseline, dosing, method="fold_change")
    array([1.5, 1.0, nan, 2.0])
    >>> normalize_metric_to_baseline(baseline, dosing, method="percent_change")
    array([50.0, 0.0, nan, 100.0])
    """
    # Convert to numpy for calculations
    if isinstance(baseline_values, pd.DataFrame):
        baseline_arr = baseline_values.values
        is_df = True
        original_index = baseline_values.index
        original_columns = baseline_values.columns
    elif isinstance(baseline_values, pd.Series):
        baseline_arr = baseline_values.values
        is_series = True
        is_df = False
        original_index = baseline_values.index
        original_name = baseline_values.name
    else:
        baseline_arr = np.asarray(baseline_values)
        is_df = False
        is_series = False
    
    if isinstance(dosing_values, (pd.DataFrame, pd.Series)):
        dosing_arr = dosing_values.values
    else:
        dosing_arr = np.asarray(dosing_values)
    
    # Validate shapes
    if baseline_arr.shape != dosing_arr.shape:
        raise ValueError(
            f"Shape mismatch: baseline {baseline_arr.shape} vs dosing {dosing_arr.shape}"
        )
    
    # Handle zeros in baseline
    baseline_safe = baseline_arr.astype(float).copy()
    if zero_handling == "epsilon":
        baseline_safe = np.where(baseline_safe == 0, epsilon, baseline_safe)
    elif zero_handling == "nan":
        # Will naturally produce inf, we'll convert to nan after
        pass
    elif zero_handling != "skip":
        raise ValueError(f"Unknown zero_handling '{zero_handling}'. Options: 'nan', 'epsilon', 'skip'")
    
    # Calculate normalized values
    with np.errstate(divide='ignore', invalid='ignore'):
        if method == "fold_change":
            result = dosing_arr / baseline_safe
        elif method == "percent_change":
            result = ((dosing_arr - baseline_arr) / baseline_safe) * 100
        elif method == "delta":
            result = dosing_arr - baseline_arr
        elif method == "log2_fold_change":
            result = np.log2(dosing_arr / baseline_safe)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Options: 'fold_change', 'percent_change', 'delta', 'log2_fold_change'"
            )
    
    # Handle infinities from zero division
    if zero_handling == "nan":
        result = np.where(np.isinf(result), np.nan, result)
    
    # Convert back to original type
    if is_df:
        return pd.DataFrame(result, index=original_index, columns=original_columns)
    elif is_series:
        return pd.Series(result, index=original_index, name=original_name)
    else:
        return result


# =============================================================================
# Main Drug Response Analysis Function
# =============================================================================

def compare_baseline_dosing(
        baseline_orb: "Orb",
        dosing_orb: "Orb",
        pair_by: str = "Well",
        metrics: List[str] = None,
        normalization: str = "fold_change",
        group_col: str = None,
        aggregate: bool = True,
        agg_func: str = "mean",
        frame_rate: float = None,
        time_unit: str = "ms",
        show_plots: bool = True,
        save_files: bool = False,
        output_dir: str = "drug_response_outputs"
    ) -> Dict[str, pd.DataFrame]:
    """
    Main entry point for baseline-normalized drug response analysis.
    
    This function:
    1. Pairs samples between baseline and dosing Orbs
    2. Aggregates neuron-level metrics to sample-level (if aggregate=True)
    3. Computes normalized values (fold change, percent change, or delta)
    4. Optionally creates visualization plots
    5. Returns results as DataFrames
    
    Parameters
    ----------
    baseline_orb : Orb
        Orb object with analyzed baseline data. Must have run run_all() first.
    dosing_orb : Orb
        Orb object with analyzed dosing data. Must have run run_all() first.
    pair_by : str, default "Well"
        Metadata column used to match baseline and dosing samples.
    metrics : list of str, optional
        Which metrics to analyze. Default: ["frpm", "rise_time", "fwhm"]
        Options: "frpm", "rise_time", "fwhm", "pwc"
    normalization : str, default "fold_change"
        How to normalize dosing to baseline:
        - "fold_change": dosing / baseline (1.0 = no change)
        - "percent_change": ((dosing - baseline) / baseline) * 100
        - "delta": dosing - baseline
        - "log2_fold_change": log2(dosing / baseline)
    group_col : str, optional
        Metadata column for grouping results (e.g., "Treatment", "Genotype").
        Used for grouped visualizations.
    aggregate : bool, default True
        If True, aggregate neuron-level data to sample means before normalization.
        This is recommended to avoid pseudoreplication.
    agg_func : str, default "mean"
        Aggregation function when aggregate=True: "mean", "median", "sum".
    frame_rate : float, optional
        Recording frame rate in Hz (frames per second). If provided, converts
        time-based metrics (rise_time, fwhm) from frames to real time units.
        If None, attempts to auto-detect from metadata 'Frate' column.
    time_unit : str, default "ms"
        Time unit for converted values: "ms" (milliseconds) or "s" (seconds).
        Only used if frame_rate is provided.
    show_plots : bool, default True
        If True, display visualization plots.
    save_files : bool, default False
        If True, save result CSVs and plots to output_dir.
    output_dir : str, default "drug_response_outputs"
        Directory for saving output files.
    
    Returns
    -------
    dict
        Dictionary of result DataFrames:
        {
            "frpm": DataFrame with baseline, dosing, and normalized FRPM values,
            "rise_time": DataFrame with baseline, dosing, and normalized rise times,
            "fwhm": DataFrame with baseline, dosing, and normalized FWHM values,
            "pairs": DataFrame summarizing sample pairs,
            "summary": DataFrame with summary statistics per group (if group_col provided)
        }
    
    Examples
    --------
    >>> # Basic usage
    >>> results = compare_baseline_dosing(
    ...     baseline_orb=orb_pre,
    ...     dosing_orb=orb_post,
    ...     pair_by="Well",
    ...     normalization="fold_change"
    ... )
    >>> 
    >>> # Access results
    >>> print(results["frpm"])  # FRPM fold changes
    >>> print(results["summary"])  # Summary by group
    
    >>> # With grouping and file saving
    >>> results = compare_baseline_dosing(
    ...     baseline_orb=orb_pre,
    ...     dosing_orb=orb_post,
    ...     pair_by="Well",
    ...     group_col="Treatment",
    ...     save_files=True,
    ...     output_dir="./my_analysis/"
    ... )
    
    See Also
    --------
    pair_samples_across_folders : For just pairing samples without analysis.
    plot_paired_lines : For custom paired line plots.
    plot_fold_change_distribution : For custom fold change plots.
    """
    # Import plotting functions (delayed to avoid circular imports)
    from wizards_staff.plotting import (
        plot_paired_lines,
        plot_fold_change_distribution,
        plot_baseline_vs_dosing_scatter
    )
    
    # Default metrics
    if metrics is None:
        metrics = ["frpm", "rise_time", "fwhm"]
    
    # Validate metrics
    invalid_metrics = set(metrics) - set(DRUG_RESPONSE_METRICS.keys())
    if invalid_metrics:
        raise ValueError(
            f"Unknown metrics: {invalid_metrics}. "
            f"Available: {list(DRUG_RESPONSE_METRICS.keys())}"
        )
    
    # Create output directory if saving files
    if save_files:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        _logger.info(f"Saving outputs to: {output_dir}")
    
    # Auto-detect frame rate from metadata if not provided
    if frame_rate is None:
        if hasattr(baseline_orb, 'metadata') and 'Frate' in baseline_orb.metadata.columns:
            # Get the most common frame rate (they should all be the same)
            frate_values = baseline_orb.metadata['Frate'].dropna()
            if len(frate_values) > 0:
                frame_rate = frate_values.mode().iloc[0] if len(frate_values.mode()) > 0 else frate_values.iloc[0]
                print(f"Auto-detected frame rate from metadata: {frame_rate} Hz", flush=True)
    
    # Step 1: Pair samples
    print("Pairing samples between baseline and dosing...", flush=True)
    pairs = pair_samples_across_folders(baseline_orb, dosing_orb, pair_by)
    
    if len(pairs) == 0:
        raise ValueError(
            f"No sample pairs found using '{pair_by}' column. "
            "Check that baseline and dosing Orbs have matching values in this column."
        )
    
    print(f"  Found {len(pairs)} matched sample pairs", flush=True)
    
    # Get paired metadata for results
    pairs_df = get_paired_metadata(baseline_orb, dosing_orb, pair_by)
    
    # Initialize results dictionary
    results = {"pairs": pairs_df}
    
    # Step 2: Process each metric
    for metric in metrics:
        print(f"\nProcessing {metric}...", flush=True)
        
        config = DRUG_RESPONSE_METRICS[metric]
        data_attr = config["data_attr"]
        value_col = config["value_col"]
        neuron_col = config["neuron_col"]
        display_name = config["display_name"]
        is_time_convertible = config.get("time_convertible", False)
        
        # Determine time conversion factor
        time_conversion = 1.0
        if is_time_convertible and frame_rate is not None and frame_rate > 0:
            if time_unit == "ms":
                time_conversion = 1000.0 / frame_rate  # frames -> ms
                display_name = f"{display_name} (ms)"
            elif time_unit == "s":
                time_conversion = 1.0 / frame_rate  # frames -> seconds
                display_name = f"{display_name} (s)"
            else:
                _logger.warning(f"Unknown time_unit '{time_unit}'. Using frames.")
        elif is_time_convertible:
            display_name = f"{display_name} (frames)"
        
        # Get data from both Orbs
        baseline_data = getattr(baseline_orb, data_attr)
        dosing_data = getattr(dosing_orb, data_attr)
        
        if baseline_data is None:
            _logger.warning(f"No {metric} data in baseline Orb. Run run_all() first. Skipping.")
            continue
        if dosing_data is None:
            _logger.warning(f"No {metric} data in dosing Orb. Run run_all() first. Skipping.")
            continue
        
        # Handle PWC differently (already sample-level, different structure)
        if metric == "pwc":
            result_df = _process_pwc_metric(
                baseline_data, dosing_data, pairs, pair_by,
                baseline_orb, dosing_orb, normalization
            )
            if result_df is not None:
                results[metric] = result_df
            continue
        
        # Step 2a: Aggregate to sample level if requested
        if aggregate and neuron_col is not None:
            print(f"  Aggregating to sample level using {agg_func}...", flush=True)
            baseline_agg = aggregate_to_sample_level(
                baseline_data, value_col=value_col, agg_func=agg_func
            )
            dosing_agg = aggregate_to_sample_level(
                dosing_data, value_col=value_col, agg_func=agg_func
            )
        else:
            baseline_agg = baseline_data.copy()
            dosing_agg = dosing_data.copy()
        
        # Step 2b: Build paired DataFrame
        paired_rows = []
        for pair_id, pair_info in pairs.items():
            baseline_sample = pair_info["baseline_sample"]
            dosing_sample = pair_info["dosing_sample"]
            
            # Get baseline value(s)
            baseline_row = baseline_agg[baseline_agg["Sample"] == baseline_sample]
            dosing_row = dosing_agg[dosing_agg["Sample"] == dosing_sample]
            
            if len(baseline_row) == 0:
                _logger.warning(f"No {metric} data for baseline sample '{baseline_sample}'")
                continue
            if len(dosing_row) == 0:
                _logger.warning(f"No {metric} data for dosing sample '{dosing_sample}'")
                continue
            
            # Get the metric values (apply time conversion if applicable)
            baseline_val = baseline_row[value_col].values[0] * time_conversion
            dosing_val = dosing_row[value_col].values[0] * time_conversion
            
            # Calculate normalized value
            normalized_val = normalize_metric_to_baseline(
                np.array([baseline_val]),
                np.array([dosing_val]),
                method=normalization
            )[0]
            
            # Build row with metadata
            row = {
                "pair_id": pair_id,
                "baseline_sample": baseline_sample,
                "dosing_sample": dosing_sample,
                f"baseline_{value_col}": baseline_val,
                f"dosing_{value_col}": dosing_val,
                f"{normalization}": normalized_val,
            }
            
            # Add metadata columns from baseline
            for col in baseline_row.columns:
                if col not in ["Sample", value_col, pair_by]:
                    row[col] = baseline_row[col].values[0]
            
            paired_rows.append(row)
        
        if not paired_rows:
            _logger.warning(f"No paired data found for {metric}")
            continue
        
        result_df = pd.DataFrame(paired_rows)
        results[metric] = result_df
        
        print(f"  Computed {normalization} for {len(result_df)} pairs", flush=True)
        
        # Step 3: Create plots if requested
        if show_plots or save_files:
            # Paired line plot
            fig_lines = plot_paired_lines(
                data=result_df,
                metric=metric,
                baseline_col=f"baseline_{value_col}",
                dosing_col=f"dosing_{value_col}",
                group_col=group_col,
                title=f"{display_name}: Baseline vs Dosing",
                show_plots=show_plots,
                save_files=save_files,
                output_dir=output_dir,
            )
            
            # Fold change distribution
            fig_dist = plot_fold_change_distribution(
                data=result_df,
                metric=metric,
                value_col=normalization,
                group_col=group_col,
                normalization=normalization,
                title=f"{display_name}: {normalization.replace('_', ' ').title()}",
                show_plots=show_plots,
                save_files=save_files,
                output_dir=output_dir,
            )
            
            # Baseline vs dosing scatter
            fig_scatter = plot_baseline_vs_dosing_scatter(
                data=result_df,
                metric=metric,
                baseline_col=f"baseline_{value_col}",
                dosing_col=f"dosing_{value_col}",
                group_col=group_col,
                title=f"{display_name}: Baseline vs Dosing",
                show_plots=show_plots,
                save_files=save_files,
                output_dir=output_dir,
            )
        
        # Save CSV if requested
        if save_files:
            csv_path = os.path.join(output_dir, f"drug-response-{metric}.csv")
            result_df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}", flush=True)
    
    # Step 4: Create summary statistics if group_col provided
    if group_col is not None:
        summary_rows = []
        for metric in metrics:
            if metric not in results:
                continue
            
            df = results[metric]
            if group_col not in df.columns:
                continue
            
            norm_col = normalization
            if norm_col not in df.columns:
                continue
            
            for group, group_df in df.groupby(group_col):
                vals = group_df[norm_col].dropna()
                summary_rows.append({
                    "metric": metric,
                    "group": group,
                    "n_samples": len(vals),
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "median": vals.median(),
                    "min": vals.min(),
                    "max": vals.max(),
                })
        
        if summary_rows:
            results["summary"] = pd.DataFrame(summary_rows)
            
            if save_files:
                summary_path = os.path.join(output_dir, "drug-response-summary.csv")
                results["summary"].to_csv(summary_path, index=False)
                print(f"\nSaved summary: {summary_path}", flush=True)
    
    print("\n✅ Drug response analysis complete!", flush=True)
    return results


def _process_pwc_metric(
        baseline_data: pd.DataFrame,
        dosing_data: pd.DataFrame,
        pairs: Dict,
        pair_by: str,
        baseline_orb: "Orb",
        dosing_orb: "Orb",
        normalization: str
    ) -> Optional[pd.DataFrame]:
    """
    Process pairwise correlation data for drug response analysis.
    
    PWC data has a different structure (groups as columns) than other metrics,
    so it requires special handling.
    """
    # PWC is tricky - the structure is groups as columns, values as rows
    # For now, we'll skip PWC and log a message
    _logger.warning(
        "PWC drug response analysis is not yet fully implemented. "
        "PWC data structure requires special handling. Skipping."
    )
    return None


# =============================================================================
# Utility Functions
# =============================================================================

def get_reference_line_value(normalization: str) -> float:
    """
    Get the reference value representing "no change" for a normalization method.
    
    Parameters
    ----------
    normalization : str
        The normalization method.
    
    Returns
    -------
    float
        The reference value (e.g., 1.0 for fold_change, 0.0 for percent_change).
    """
    reference_values = {
        "fold_change": 1.0,
        "percent_change": 0.0,
        "delta": 0.0,
        "log2_fold_change": 0.0,
    }
    return reference_values.get(normalization, 1.0)


def interpret_fold_change(fold_change: float, metric_name: str = "activity") -> str:
    """
    Generate a human-readable interpretation of a fold change value.
    
    Parameters
    ----------
    fold_change : float
        The fold change value.
    metric_name : str
        Name of the metric for the interpretation string.
    
    Returns
    -------
    str
        Human-readable interpretation.
    
    Examples
    --------
    >>> interpret_fold_change(2.0, "firing rate")
    "2.0x increase in firing rate (doubled)"
    >>> interpret_fold_change(0.5, "firing rate")
    "2.0x decrease in firing rate (halved)"
    """
    if np.isnan(fold_change):
        return f"Unable to calculate (baseline was zero)"
    elif fold_change == 1.0:
        return f"No change in {metric_name}"
    elif fold_change > 1.0:
        magnitude = fold_change
        if magnitude == 2.0:
            return f"{magnitude:.1f}x increase in {metric_name} (doubled)"
        elif magnitude == 3.0:
            return f"{magnitude:.1f}x increase in {metric_name} (tripled)"
        else:
            return f"{magnitude:.1f}x increase in {metric_name}"
    else:
        magnitude = 1.0 / fold_change
        if fold_change == 0.5:
            return f"{magnitude:.1f}x decrease in {metric_name} (halved)"
        elif fold_change == 0:
            return f"Complete elimination of {metric_name}"
        else:
            return f"{magnitude:.1f}x decrease in {metric_name}"




