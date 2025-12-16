"""
Core statistical infrastructure for Wizards Staff analysis.

This module provides foundational classes and utilities for statistical analysis
of calcium imaging data, with emphasis on reproducibility and ease of use.

Classes
-------
StatsResult : Container for statistical analysis results
StatsConfig : Configuration for statistical analyses
DataValidator : Validates and prepares data for statistical analysis

Functions
---------
aggregate_to_biological_replicates : Aggregate neuron-level to sample-level data
merge_with_metadata : Merge analysis data with experimental metadata
prepare_for_stats : One-stop function to prepare Orb data for statistical analysis
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from wizards_staff.wizards.orb import Orb


@dataclass
class StatsResult:
    """
    Container for statistical analysis results.
    
    This class stores all relevant information from a statistical test and provides
    methods for interpretation, visualization, and export.
    
    Attributes
    ----------
    test_name : str
        Name of the statistical test performed (e.g., "Mann-Whitney U test")
    statistic : float
        Test statistic value
    p_value : float
        Raw p-value from the test
    p_adjusted : float, optional
        Adjusted p-value after multiple comparison correction
    effect_size : float
        Effect size measure (e.g., Cohen's d, rank-biserial r)
    effect_size_type : str
        Type of effect size (e.g., "Cohen's d", "rank-biserial r", "eta-squared")
    effect_size_magnitude : str
        Interpretation of effect size (e.g., "small", "medium", "large")
    ci_lower : float
        Lower bound of confidence interval for effect size
    ci_upper : float
        Upper bound of confidence interval for effect size
    sample_sizes : dict
        Sample sizes per group (e.g., {"Control": 10, "Treatment": 12})
    group_stats : dict
        Descriptive statistics per group (mean, median, std, etc.)
    assumptions_met : dict
        Results of assumption checks (e.g., {"normality": True, "equal_variance": True})
    interpretation : str
        Plain-language interpretation of the results
    warnings : list
        Any warnings about the analysis
    raw_data : dict, optional
        Original data used for the analysis
    test_details : dict, optional
        Additional test-specific details
    
    Examples
    --------
    >>> result = compare_two_groups(data, group_col="Treatment", metric_col="firing_rate")
    >>> print(result.summary)
    >>> result.plot()
    >>> result.export("my_results.csv")
    """
    
    test_name: str
    statistic: float
    p_value: float
    p_adjusted: Optional[float] = None
    effect_size: float = 0.0
    effect_size_type: str = ""
    effect_size_magnitude: str = ""
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    group_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    assumptions_met: Dict[str, bool] = field(default_factory=dict)
    interpretation: str = ""
    warnings: List[str] = field(default_factory=list)
    raw_data: Optional[Dict[str, np.ndarray]] = None
    test_details: Dict[str, Any] = field(default_factory=dict)
    post_hoc_table: Optional[pd.DataFrame] = None
    
    @property
    def summary(self) -> str:
        """
        Return a formatted, human-readable summary of the statistical results.
        
        Returns
        -------
        str
            Multi-line summary including test results, effect size, and interpretation.
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"Statistical Test: {self.test_name}")
        lines.append("=" * 60)
        lines.append("")
        
        # Sample sizes
        if self.sample_sizes:
            sizes_str = ", ".join([f"{k}: n={v}" for k, v in self.sample_sizes.items()])
            lines.append(f"Sample sizes: {sizes_str}")
            lines.append("")
        
        # Group statistics
        if self.group_stats:
            lines.append("Group Statistics:")
            for group, stats in self.group_stats.items():
                stats_str = ", ".join([f"{k}={v:.3f}" for k, v in stats.items()])
                lines.append(f"  {group}: {stats_str}")
            lines.append("")
        
        # Test results
        lines.append("Test Results:")
        lines.append(f"  Statistic: {self.statistic:.4f}")
        lines.append(f"  P-value: {self.p_value:.4f}")
        if self.p_adjusted is not None:
            lines.append(f"  Adjusted P-value: {self.p_adjusted:.4f}")
        lines.append("")
        
        # Effect size
        if self.effect_size_type:
            lines.append(f"Effect Size ({self.effect_size_type}): {self.effect_size:.3f}")
            if self.effect_size_magnitude:
                lines.append(f"  Magnitude: {self.effect_size_magnitude}")
            if self.ci_lower is not None and self.ci_upper is not None:
                lines.append(f"  95% CI: [{self.ci_lower:.3f}, {self.ci_upper:.3f}]")
            lines.append("")
        
        # Interpretation
        if self.interpretation:
            lines.append("Interpretation:")
            lines.append(f"  {self.interpretation}")
            lines.append("")
        
        # Warnings
        if self.warnings:
            lines.append("⚠️ Warnings:")
            for warning in self.warnings:
                lines.append(f"  • {warning}")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.
        
        Returns
        -------
        dict
            Dictionary containing all result attributes.
        """
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "effect_size": self.effect_size,
            "effect_size_type": self.effect_size_type,
            "effect_size_magnitude": self.effect_size_magnitude,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "sample_sizes": self.sample_sizes,
            "interpretation": self.interpretation,
            "warnings": self.warnings,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a single-row DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with one row containing the results.
        """
        data = self.to_dict()
        # Flatten sample_sizes
        for group, n in self.sample_sizes.items():
            data[f"n_{group}"] = n
        del data["sample_sizes"]
        data["warnings"] = "; ".join(self.warnings) if self.warnings else ""
        return pd.DataFrame([data])
    
    def plot(self, figsize: tuple = (8, 6), **kwargs) -> plt.Figure:
        """
        Generate an appropriate visualization of the results.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (8, 6).
        **kwargs
            Additional arguments passed to the plotting function.
        
        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        from .visualizations import plot_group_comparison
        
        if self.raw_data is None:
            warnings.warn("No raw data available for plotting. "
                         "Run the test with store_data=True to enable plotting.")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available for plotting",
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create DataFrame from raw data
        data_list = []
        for group, values in self.raw_data.items():
            for val in values:
                data_list.append({"Group": group, "Value": val})
        df = pd.DataFrame(data_list)
        
        return plot_group_comparison(
            data=df,
            group_col="Group",
            metric_col="Value",
            stats_result=self,
            figsize=figsize,
            **kwargs
        )
    
    def export(self, filepath: str) -> None:
        """
        Save results to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the results. File type is determined by extension.
            Supported: .csv, .xlsx, .json
        """
        df = self.to_dataframe()
        
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filepath.endswith('.xlsx'):
            df.to_excel(filepath, index=False)
        elif filepath.endswith('.json'):
            import json
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        else:
            # Default to CSV
            df.to_csv(filepath, index=False)
    
    def __str__(self) -> str:
        return self.summary
    
    def __repr__(self) -> str:
        return (f"StatsResult(test='{self.test_name}', "
                f"statistic={self.statistic:.4f}, "
                f"p={self.p_value:.4f}, "
                f"effect_size={self.effect_size:.3f})")


@dataclass
class StatsConfig:
    """
    Configuration for statistical analyses.
    
    This class holds all the settings that control how statistical tests are
    performed. Using a configuration object ensures consistency across analyses
    and makes it easy to document your analysis choices.
    
    Attributes
    ----------
    alpha : float
        Significance level for hypothesis tests. Default is 0.05.
    confidence_level : float
        Confidence level for confidence intervals. Default is 0.95.
    correction_method : str
        Method for multiple comparison correction. Default is "bonferroni".
        Options: "bonferroni", "holm", "fdr_bh", "fdr_by", "none"
    min_samples_per_group : int
        Minimum number of samples required per group. Default is 3.
    handle_ties : str
        How to handle tied values in rank-based tests. Default is "average".
        Options: "average", "min", "max", "dense", "ordinal"
    nan_policy : str
        How to handle NaN values. Default is "omit".
        Options: "omit" (remove), "raise" (error), "propagate" (include)
    random_state : int
        Random seed for reproducibility. Default is 42.
    
    Examples
    --------
    >>> config = StatsConfig(alpha=0.01, correction_method="fdr_bh")
    >>> result = compare_two_groups(data, group_col="Treatment", 
    ...                              metric_col="firing_rate", config=config)
    """
    
    alpha: float = 0.05
    confidence_level: float = 0.95
    correction_method: str = "bonferroni"
    min_samples_per_group: int = 3
    handle_ties: str = "average"
    nan_policy: str = "omit"
    random_state: int = 42
    
    def __post_init__(self):
        """Validate configuration values."""
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")
        if not 0 < self.confidence_level < 1:
            raise ValueError(f"confidence_level must be between 0 and 1, "
                           f"got {self.confidence_level}")
        
        valid_corrections = {"bonferroni", "holm", "fdr_bh", "fdr_by", "none"}
        if self.correction_method not in valid_corrections:
            raise ValueError(f"correction_method must be one of {valid_corrections}, "
                           f"got '{self.correction_method}'")
        
        valid_nan_policies = {"omit", "raise", "propagate"}
        if self.nan_policy not in valid_nan_policies:
            raise ValueError(f"nan_policy must be one of {valid_nan_policies}, "
                           f"got '{self.nan_policy}'")


class DataValidator:
    """
    Validates and prepares data for statistical analysis.
    
    This class provides methods to check data quality, validate structure,
    and identify potential issues before running statistical tests.
    
    Examples
    --------
    >>> validator = DataValidator()
    >>> validator.validate_dataframe(df, required_cols=["Sample", "Treatment", "Value"])
    >>> issues = validator.flag_potential_issues(df)
    >>> if issues:
    ...     print("Data quality concerns:", issues)
    """
    
    def __init__(self, config: Optional[StatsConfig] = None):
        """
        Initialize the validator.
        
        Parameters
        ----------
        config : StatsConfig, optional
            Configuration object with validation parameters.
        """
        self.config = config or StatsConfig()
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        required_cols: List[str],
        raise_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Check DataFrame structure and content.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        required_cols : list of str
            Column names that must be present.
        raise_on_error : bool
            If True, raise ValueError on validation failure.
        
        Returns
        -------
        dict
            Validation results with 'valid' boolean and 'errors' list.
        """
        errors = []
        
        # Check required columns
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for empty DataFrame
        if len(df) == 0:
            errors.append("DataFrame is empty")
        
        # Check for all-NaN columns
        for col in required_cols:
            if col in df.columns and df[col].isna().all():
                errors.append(f"Column '{col}' contains only NaN values")
        
        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "n_rows": len(df),
            "n_cols": len(df.columns),
        }
        
        if raise_on_error and not result["valid"]:
            raise ValueError(f"DataFrame validation failed: {errors}")
        
        return result
    
    def check_group_sizes(
        self,
        df: pd.DataFrame,
        group_col: str,
        min_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Verify adequate sample sizes per group.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data.
        group_col : str
            Column name containing group labels.
        min_n : int, optional
            Minimum required samples per group. Uses config default if None.
        
        Returns
        -------
        dict
            Dictionary with group sizes and adequacy assessment.
        """
        if min_n is None:
            min_n = self.config.min_samples_per_group
        
        group_sizes = df.groupby(group_col).size().to_dict()
        small_groups = {g: n for g, n in group_sizes.items() if n < min_n}
        
        return {
            "group_sizes": group_sizes,
            "min_n_required": min_n,
            "all_adequate": len(small_groups) == 0,
            "small_groups": small_groups,
            "recommendation": (
                None if len(small_groups) == 0 else
                f"Groups {list(small_groups.keys())} have fewer than {min_n} samples. "
                f"Consider collecting more data or using non-parametric tests."
            )
        }
    
    def identify_replicates(
        self,
        df: pd.DataFrame,
        sample_col: str = "Sample",
        neuron_col: str = "Neuron"
    ) -> Dict[str, Any]:
        """
        Identify technical vs biological replicates in the data.
        
        In calcium imaging, individual neurons within a sample are TECHNICAL
        replicates, while different samples are BIOLOGICAL replicates.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with neuron-level data.
        sample_col : str
            Column identifying biological samples.
        neuron_col : str
            Column identifying individual neurons.
        
        Returns
        -------
        dict
            Information about replicate structure.
        """
        if sample_col not in df.columns:
            return {
                "error": f"Sample column '{sample_col}' not found",
                "is_aggregated": None,
            }
        
        n_samples = df[sample_col].nunique()
        
        if neuron_col in df.columns:
            n_neurons = df[neuron_col].nunique()
            neurons_per_sample = df.groupby(sample_col)[neuron_col].nunique()
            is_aggregated = False
        else:
            n_neurons = None
            neurons_per_sample = None
            is_aggregated = True
        
        return {
            "n_biological_replicates": n_samples,
            "n_technical_replicates": n_neurons,
            "neurons_per_sample": neurons_per_sample.to_dict() if neurons_per_sample is not None else None,
            "is_aggregated": is_aggregated,
            "recommendation": (
                "Data appears to be at neuron-level. Consider aggregating to sample-level "
                "using aggregate_to_biological_replicates() before statistical testing "
                "to avoid pseudoreplication."
                if not is_aggregated else
                "Data is already aggregated to sample-level. Ready for statistical testing."
            )
        }
    
    def summarize_by_sample(
        self,
        df: pd.DataFrame,
        metric_cols: List[str],
        sample_col: str = "Sample"
    ) -> pd.DataFrame:
        """
        Aggregate neuron-level data to sample-level summaries.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with neuron-level data.
        metric_cols : list of str
            Columns to aggregate.
        sample_col : str
            Column identifying samples.
        
        Returns
        -------
        pd.DataFrame
            Aggregated data with one row per sample.
        """
        agg_funcs = {col: ['mean', 'std', 'median', 'count'] for col in metric_cols}
        
        result = df.groupby(sample_col).agg(agg_funcs)
        # Flatten column names
        result.columns = ['_'.join(col).strip() for col in result.columns.values]
        return result.reset_index()
    
    def flag_potential_issues(
        self,
        df: pd.DataFrame,
        metric_cols: Optional[List[str]] = None
    ) -> List[str]:
        """
        Return list of data quality concerns.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to check.
        metric_cols : list of str, optional
            Specific columns to check. If None, checks all numeric columns.
        
        Returns
        -------
        list of str
            List of potential issues found.
        """
        issues = []
        
        if metric_cols is None:
            metric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in metric_cols:
            if col not in df.columns:
                continue
            
            values = df[col].dropna()
            
            # Check for missing values
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                pct_missing = (n_missing / len(df)) * 100
                issues.append(f"Column '{col}': {n_missing} missing values ({pct_missing:.1f}%)")
            
            # Check for constant values
            if len(values) > 0 and values.std() == 0:
                issues.append(f"Column '{col}': All values are identical (no variance)")
            
            # Check for potential outliers (using IQR)
            if len(values) >= 4:
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                n_outliers = ((values < lower_bound) | (values > upper_bound)).sum()
                if n_outliers > 0:
                    issues.append(f"Column '{col}': {n_outliers} potential outliers detected")
            
            # Check for negative values in typically positive metrics
            if col.lower() in ['firing_rate', 'frpm', 'fwhm', 'rise_time', 'area', 'diameter']:
                n_negative = (values < 0).sum()
                if n_negative > 0:
                    issues.append(f"Column '{col}': {n_negative} unexpected negative values")
        
        return issues


def aggregate_to_biological_replicates(
    df: pd.DataFrame,
    sample_col: str = "Sample",
    neuron_col: str = "Neuron",
    metric_cols: Optional[List[str]] = None,
    agg_func: str = "mean"
) -> pd.DataFrame:
    """
    Aggregate neuron-level measurements to sample-level for proper statistical inference.
    
    IMPORTANT: In calcium imaging, individual neurons within a sample are TECHNICAL
    replicates. Statistical tests should typically be performed on sample-level
    summaries (BIOLOGICAL replicates), not individual neurons, to avoid pseudoreplication.
    
    What is pseudoreplication?
    --------------------------
    If you have 5 samples with 100 neurons each, you have 500 data points but only
    5 independent observations. Treating neurons as independent would artificially
    inflate your statistical power and increase false positive rates.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with neuron-level data.
    sample_col : str
        Column identifying biological samples. Default is "Sample".
    neuron_col : str
        Column identifying individual neurons. Default is "Neuron".
    metric_cols : list of str, optional
        Columns to aggregate. If None, all numeric columns are aggregated.
    agg_func : str
        Aggregation function to use. Default is "mean".
        Options: "mean", "median", "sum", "std", "var", "min", "max"
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per sample, containing aggregated metrics.
        New column names are prefixed with the aggregation function
        (e.g., "mean_firing_rate").
    
    Examples
    --------
    >>> # Convert neuron-level FRPM to sample-level
    >>> sample_frpm = aggregate_to_biological_replicates(
    ...     orb.frpm_data,
    ...     metric_cols=["Firing Rate Per Min"]
    ... )
    >>> print(f"Aggregated {len(orb.frpm_data)} neurons to {len(sample_frpm)} samples")
    """
    if sample_col not in df.columns:
        raise ValueError(f"Sample column '{sample_col}' not found in DataFrame")
    
    # Identify metric columns if not specified
    if metric_cols is None:
        # Get all numeric columns except the sample/neuron identifiers
        exclude_cols = {sample_col, neuron_col, 'Neuron Index'}
        metric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]
    
    if not metric_cols:
        raise ValueError("No metric columns found to aggregate")
    
    # Define aggregation
    valid_agg_funcs = {"mean", "median", "sum", "std", "var", "min", "max"}
    if agg_func not in valid_agg_funcs:
        raise ValueError(f"agg_func must be one of {valid_agg_funcs}, got '{agg_func}'")
    
    # Get non-metric columns to preserve (metadata)
    id_cols = [sample_col]
    metadata_cols = [
        col for col in df.columns
        if col not in metric_cols and col != neuron_col and col != 'Neuron Index'
    ]
    
    # Aggregate metrics
    agg_dict = {col: agg_func for col in metric_cols if col in df.columns}
    
    if not agg_dict:
        missing_cols = [col for col in metric_cols if col not in df.columns]
        raise ValueError(
            f"None of the specified metric columns were found in the DataFrame. "
            f"Expected: {metric_cols}, but DataFrame columns are: {list(df.columns)}. "
            f"Missing columns: {missing_cols}"
        )
    
    aggregated = df.groupby(sample_col).agg(agg_dict).reset_index()
    
    # Rename columns to include aggregation function
    rename_dict = {col: f"{agg_func}_{col}" for col in metric_cols if col in aggregated.columns}
    aggregated = aggregated.rename(columns=rename_dict)
    
    # Add count of neurons per sample
    neuron_counts = df.groupby(sample_col).size().reset_index(name='n_neurons')
    aggregated = aggregated.merge(neuron_counts, on=sample_col)
    
    # Merge back metadata (take first value per sample)
    metadata_to_keep = [col for col in metadata_cols if col in df.columns and col != sample_col]
    if metadata_to_keep:
        metadata = df.groupby(sample_col)[metadata_to_keep].first().reset_index()
        aggregated = aggregated.merge(metadata, on=sample_col, how='left')
    
    return aggregated


def merge_with_metadata(
    data_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    on: str = "Sample"
) -> pd.DataFrame:
    """
    Merge analysis data with experimental metadata.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing analysis results (e.g., from Orb).
    metadata_df : pd.DataFrame
        DataFrame containing experimental metadata.
    on : str
        Column name to join on. Default is "Sample".
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all columns from both inputs.
    
    Examples
    --------
    >>> merged = merge_with_metadata(orb.frpm_data, orb.metadata)
    """
    if on not in data_df.columns:
        raise ValueError(f"Column '{on}' not found in data DataFrame")
    if on not in metadata_df.columns:
        raise ValueError(f"Column '{on}' not found in metadata DataFrame")
    
    return data_df.merge(metadata_df, on=on, how='left')


def prepare_for_stats(
    orb: "Orb",
    metric: str,
    aggregate: bool = True,
    metadata_cols: Optional[List[str]] = None,
    agg_func: str = "mean"
) -> pd.DataFrame:
    """
    One-stop function to prepare Orb data for statistical analysis.
    
    This is the RECOMMENDED entry point for most users. It handles:
    1. Extracting the requested metric from the Orb
    2. Optionally aggregating from neuron-level to sample-level
    3. Merging with metadata columns
    
    Parameters
    ----------
    orb : Orb
        Wizards Staff Orb object with completed analysis.
    metric : str
        Which metric to prepare. Options:
        - "frpm" : Firing Rate Per Minute
        - "rise_time" : Calcium signal rise times
        - "fwhm" : Full Width at Half Maximum (event duration)
        - "mask_metrics" : Cell shape metrics (roundness, diameter, area)
        - "silhouette" : Clustering quality scores
        - "pwc" : Pairwise correlations (returns different format)
    aggregate : bool
        If True (recommended), aggregate to biological replicates (sample-level).
        Set to False only if you specifically need neuron-level data.
    metadata_cols : list of str, optional
        Additional metadata columns to include in the output.
        If None, all metadata columns are included.
    agg_func : str
        Aggregation function when aggregate=True. Default is "mean".
    
    Returns
    -------
    pd.DataFrame
        Analysis-ready DataFrame with metrics and metadata.
    
    Examples
    --------
    >>> # Basic usage - prepare FRPM data for treatment comparison
    >>> df = prepare_for_stats(orb, metric="frpm", metadata_cols=["Treatment", "Genotype"])
    >>> result = compare_two_groups(df, group_col="Treatment", metric_col="mean_Firing Rate Per Min")
    
    >>> # Get neuron-level data (for specialized analyses)
    >>> df_neurons = prepare_for_stats(orb, metric="frpm", aggregate=False)
    """
    # Map metric names to Orb properties
    metric_mapping = {
        "frpm": "frpm_data",
        "firing_rate": "frpm_data",
        "rise_time": "rise_time_data",
        "fwhm": "fwhm_data",
        "mask_metrics": "mask_metrics_data",
        "mask": "mask_metrics_data",
        "silhouette": "silhouette_scores_data",
        "silhouette_scores": "silhouette_scores_data",
    }
    
    # Normalize metric name
    metric_lower = metric.lower().replace(" ", "_").replace("-", "_")
    if metric_lower not in metric_mapping:
        available = list(metric_mapping.keys())
        raise ValueError(f"Unknown metric '{metric}'. Available: {available}")
    
    # Get data from Orb
    data_attr = metric_mapping[metric_lower]
    data = getattr(orb, data_attr, None)
    
    if data is None:
        raise ValueError(
            f"No {metric} data found in Orb. "
            f"Make sure you've run orb.run_all() first."
        )
    
    # Make a copy to avoid modifying original
    df = data.copy()
    
    # Handle special case for silhouette scores (already sample-level)
    if metric_lower in ["silhouette", "silhouette_scores"]:
        aggregate = False  # Already at sample level
    
    # Aggregate if requested
    if aggregate:
        # Determine metric columns based on metric type
        if metric_lower in ["frpm", "firing_rate"]:
            metric_cols = ["Firing Rate Per Min"]
        elif metric_lower == "rise_time":
            metric_cols = ["Rise Times"]
        elif metric_lower == "fwhm":
            metric_cols = ["FWHM Values", "Spike Counts"]
        elif metric_lower in ["mask_metrics", "mask"]:
            metric_cols = ["roundness", "diameter", "area"]
        else:
            metric_cols = None
        
        df = aggregate_to_biological_replicates(
            df,
            sample_col="Sample",
            neuron_col="Neuron Index" if "Neuron Index" in df.columns else "Neuron",
            metric_cols=metric_cols,
            agg_func=agg_func
        )
    
    # Filter metadata columns if specified
    if metadata_cols is not None:
        # Keep only specified metadata columns plus the aggregated metrics and Sample
        keep_cols = ["Sample"] + [col for col in df.columns if col.startswith(f"{agg_func}_") or col == "n_neurons"]
        keep_cols.extend([col for col in metadata_cols if col in df.columns])
        keep_cols = list(dict.fromkeys(keep_cols))  # Remove duplicates while preserving order
        df = df[[col for col in keep_cols if col in df.columns]]
    
    return df

