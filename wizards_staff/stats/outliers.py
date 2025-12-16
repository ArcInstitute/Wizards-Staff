"""
Outlier detection and handling for calcium imaging data.

Provides multiple methods for identifying outliers with clear guidance
on when to use each. Important: Always investigate outliers before
removing them - they may represent real biological phenomena!

Functions
---------
detect_outliers : Identify potential outliers using various methods
visualize_outliers : Create visualization showing outliers
handle_outliers : Apply various outlier handling strategies
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def detect_outliers(
    data: pd.DataFrame,
    metric_col: str,
    method: str = "iqr",
    threshold: Optional[float] = None,
    group_col: Optional[str] = None
) -> Dict:
    """
    Detect potential outliers in the data.
    
    IMPORTANT: Outliers in biological data may represent real phenomena!
    Always investigate outliers before deciding how to handle them.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to analyze.
    metric_col : str
        Column name containing the metric to check for outliers.
    method : str
        Detection method to use:
        
        - "iqr": Interquartile range method (1.5*IQR rule).
          Best for: Most calcium imaging data. Robust to non-normality.
          
        - "zscore": Z-score > threshold.
          Best for: Normally distributed data.
          
        - "mad": Median Absolute Deviation.
          Best for: Data with non-normal distributions or heavy tails.
          
        - "isolation_forest": ML-based detection.
          Best for: Multivariate outliers (not yet implemented).
          
    threshold : float, optional
        Method-specific threshold:
        - For "iqr": Multiplier for IQR (default 1.5)
        - For "zscore": Z-score cutoff (default 3.0)
        - For "mad": Number of MADs from median (default 3.5)
        
    group_col : str, optional
        Column for group-wise outlier detection. If provided, outliers
        are detected within each group separately.
    
    Returns
    -------
    dict
        Dictionary containing:
        - outlier_mask: Boolean mask of outliers (True = outlier)
        - outlier_indices: DataFrame indices of outlier rows
        - outlier_values: The outlier values
        - n_outliers: Number of outliers detected
        - pct_outliers: Percentage of data that are outliers
        - bounds: Lower and upper bounds used (for IQR/MAD)
        - summary: Description of outliers found
        - recommendation: Whether to remove or investigate
    
    Examples
    --------
    >>> # Detect outliers in firing rate data
    >>> result = detect_outliers(
    ...     data=frpm_df,
    ...     metric_col="mean_Firing Rate Per Min",
    ...     method="iqr"
    ... )
    >>> print(result["summary"])
    >>> print(f"Found {result['n_outliers']} outliers")
    
    >>> # Detect outliers within each treatment group
    >>> result = detect_outliers(
    ...     data=frpm_df,
    ...     metric_col="mean_Firing Rate Per Min",
    ...     group_col="Treatment"
    ... )
    """
    df = data.copy()
    values = df[metric_col].values
    
    # Set default thresholds
    default_thresholds = {
        "iqr": 1.5,
        "zscore": 3.0,
        "mad": 3.5,
        "isolation_forest": 0.1,
    }
    if threshold is None:
        threshold = default_thresholds.get(method, 1.5)
    
    # Detect outliers based on method
    if group_col is not None:
        # Detect within groups
        outlier_mask = np.zeros(len(df), dtype=bool)
        bounds = {}
        
        for group_name in df[group_col].unique():
            group_mask = df[group_col] == group_name
            group_values = values[group_mask]
            group_result = _detect_outliers_single(
                group_values, method, threshold
            )
            # Map back to original indices
            outlier_mask[group_mask] = group_result["outlier_mask"]
            bounds[group_name] = group_result.get("bounds")
    else:
        result = _detect_outliers_single(values, method, threshold)
        outlier_mask = result["outlier_mask"]
        bounds = result.get("bounds")
    
    # Extract outlier information
    outlier_indices = df.index[outlier_mask].tolist()
    outlier_values = values[outlier_mask]
    n_outliers = int(outlier_mask.sum())
    pct_outliers = (n_outliers / len(df)) * 100
    
    # Build summary
    if n_outliers == 0:
        summary = f"No outliers detected using {method} method (threshold: {threshold})."
        recommendation = "No action needed."
    elif n_outliers < 5:
        summary = (
            f"Detected {n_outliers} outlier(s) ({pct_outliers:.1f}% of data) "
            f"using {method} method. Values: {outlier_values.tolist()}"
        )
        recommendation = (
            "Review these cases individually to determine if they represent "
            "real biological variation or technical artifacts."
        )
    else:
        summary = (
            f"Detected {n_outliers} outliers ({pct_outliers:.1f}% of data) "
            f"using {method} method. Range: {outlier_values.min():.2f} to {outlier_values.max():.2f}"
        )
        if pct_outliers > 10:
            recommendation = (
                "High outlier rate suggests possible data quality issues or "
                "non-normal distribution. Consider using robust statistics."
            )
        else:
            recommendation = (
                "Consider flagging outliers for sensitivity analysis rather "
                "than removing them outright."
            )
    
    return {
        "outlier_mask": outlier_mask,
        "outlier_indices": outlier_indices,
        "outlier_values": outlier_values,
        "n_outliers": n_outliers,
        "pct_outliers": pct_outliers,
        "bounds": bounds,
        "method": method,
        "threshold": threshold,
        "summary": summary,
        "recommendation": recommendation,
    }


def _detect_outliers_single(
    values: np.ndarray,
    method: str,
    threshold: float
) -> Dict:
    """Detect outliers for a single array of values."""
    values = np.asarray(values)
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    
    if len(valid_values) < 4:
        return {
            "outlier_mask": np.zeros(len(values), dtype=bool),
            "bounds": None,
        }
    
    if method == "iqr":
        q1 = np.percentile(valid_values, 25)
        q3 = np.percentile(valid_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        bounds = {"lower": lower_bound, "upper": upper_bound, "q1": q1, "q3": q3}
        
    elif method == "zscore":
        mean = np.mean(valid_values)
        std = np.std(valid_values, ddof=1)
        if std == 0:
            outlier_mask = np.zeros(len(values), dtype=bool)
            bounds = None
        else:
            z_scores = (values - mean) / std
            outlier_mask = np.abs(z_scores) > threshold
            bounds = {
                "lower": mean - threshold * std,
                "upper": mean + threshold * std,
                "mean": mean,
                "std": std,
            }
    
    elif method == "mad":
        median = np.median(valid_values)
        mad = np.median(np.abs(valid_values - median))
        if mad == 0:
            # Use a small value to avoid division by zero
            mad = 1e-10
        modified_z = 0.6745 * (values - median) / mad
        outlier_mask = np.abs(modified_z) > threshold
        bounds = {
            "lower": median - threshold * mad / 0.6745,
            "upper": median + threshold * mad / 0.6745,
            "median": median,
            "mad": mad,
        }
    
    elif method == "isolation_forest":
        try:
            from sklearn.ensemble import IsolationForest
            
            # Reshape for sklearn
            X = valid_values.reshape(-1, 1)
            clf = IsolationForest(contamination=threshold, random_state=42)
            predictions = clf.fit_predict(X)
            
            # Map back to original array
            outlier_mask = np.zeros(len(values), dtype=bool)
            outlier_mask[valid_mask] = predictions == -1
            bounds = None
            
        except ImportError:
            raise ImportError(
                "Isolation Forest requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )
    
    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Valid options: 'iqr', 'zscore', 'mad', 'isolation_forest'"
        )
    
    return {"outlier_mask": outlier_mask, "bounds": bounds}


def visualize_outliers(
    data: pd.DataFrame,
    metric_col: str,
    outlier_result: Dict,
    group_col: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Create visualization showing data distribution and outliers.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    metric_col : str
        Column name containing the metric.
    outlier_result : dict
        Result dictionary from detect_outliers().
    group_col : str, optional
        Column for grouping the visualization.
    figsize : tuple
        Figure size (width, height) in inches.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    
    Examples
    --------
    >>> outliers = detect_outliers(df, "firing_rate")
    >>> fig = visualize_outliers(df, "firing_rate", outliers)
    >>> plt.show()
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    values = data[metric_col].values
    outlier_mask = outlier_result["outlier_mask"]
    
    # Left plot: Box plot with outliers highlighted
    ax1 = axes[0]
    if group_col is not None:
        groups = data[group_col].unique()
        positions = []
        labels = []
        for i, group in enumerate(groups):
            group_mask = data[group_col] == group
            group_values = values[group_mask]
            positions.append(i + 1)
            labels.append(str(group))
            
            # Box plot
            bp = ax1.boxplot([group_values], positions=[i + 1], widths=0.6)
            
            # Highlight outliers
            group_outliers = group_values[outlier_mask[group_mask]]
            ax1.scatter(
                [i + 1] * len(group_outliers), group_outliers,
                c='red', marker='x', s=100, zorder=5, label='Outlier' if i == 0 else None
            )
        
        ax1.set_xticks(positions)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
    else:
        ax1.boxplot([values], widths=0.6)
        outlier_values = values[outlier_mask]
        ax1.scatter(
            [1] * len(outlier_values), outlier_values,
            c='red', marker='x', s=100, zorder=5, label='Outlier'
        )
    
    ax1.set_ylabel(metric_col)
    ax1.set_title("Box Plot with Outliers")
    if outlier_result["n_outliers"] > 0:
        ax1.legend()
    
    # Right plot: Histogram with bounds
    ax2 = axes[1]
    ax2.hist(values[~np.isnan(values)], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(np.median(values[~np.isnan(values)]), color='green', linestyle='--', 
                label='Median', linewidth=2)
    
    # Show bounds if available
    bounds = outlier_result.get("bounds")
    if bounds is not None and isinstance(bounds, dict):
        if "lower" in bounds:
            ax2.axvline(bounds["lower"], color='red', linestyle=':', 
                       label=f'Lower bound ({bounds["lower"]:.2f})', linewidth=2)
        if "upper" in bounds:
            ax2.axvline(bounds["upper"], color='red', linestyle=':', 
                       label=f'Upper bound ({bounds["upper"]:.2f})', linewidth=2)
    
    ax2.set_xlabel(metric_col)
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution with Outlier Bounds")
    ax2.legend(fontsize=8)
    
    # Add summary text
    method = outlier_result.get("method", "unknown")
    n_outliers = outlier_result.get("n_outliers", 0)
    pct = outlier_result.get("pct_outliers", 0)
    fig.suptitle(
        f"Outlier Detection: {n_outliers} outliers ({pct:.1f}%) using {method} method",
        fontsize=12, fontweight='bold'
    )
    
    plt.tight_layout()
    return fig


def handle_outliers(
    data: pd.DataFrame,
    outlier_result: Dict,
    method: str = "flag"
) -> pd.DataFrame:
    """
    Handle outliers according to specified method.
    
    ALWAYS returns a copy of the data - never modifies the original.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    outlier_result : dict
        Result dictionary from detect_outliers().
    method : str
        How to handle outliers:
        
        - "flag": Add a column marking outliers (recommended for transparency)
        - "remove": Remove outlier rows
        - "winsorize": Replace outliers with boundary values
        - "impute_median": Replace outliers with group median
        - "impute_mean": Replace outliers with group mean
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame (always a copy, original is unchanged).
    
    Notes
    -----
    Recommendations by use case:
    
    - **Exploratory analysis**: Use "flag" to see impact of outliers
    - **Robust analysis**: Use "winsorize" to reduce outlier influence
    - **Final analysis**: Consider running with both "flag" and "remove"
      to assess sensitivity
    
    Examples
    --------
    >>> outliers = detect_outliers(df, "firing_rate")
    >>> 
    >>> # Option 1: Flag outliers (recommended)
    >>> df_flagged = handle_outliers(df, outliers, method="flag")
    >>> 
    >>> # Option 2: Remove outliers
    >>> df_clean = handle_outliers(df, outliers, method="remove")
    >>> 
    >>> # Option 3: Winsorize (cap at boundaries)
    >>> df_winsorized = handle_outliers(df, outliers, method="winsorize")
    """
    df = data.copy()
    outlier_mask = outlier_result["outlier_mask"]
    
    if method == "flag":
        df["is_outlier"] = outlier_mask
        df["outlier_method"] = outlier_result.get("method", "unknown")
        
    elif method == "remove":
        df = df[~outlier_mask].reset_index(drop=True)
        
    elif method == "winsorize":
        bounds = outlier_result.get("bounds")
        if bounds is None:
            raise ValueError("Winsorization requires bounds from outlier detection")
        
        # Get the metric column (assume it's stored or infer from bounds)
        # Find numeric columns that might have been analyzed
        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col].values
            if isinstance(bounds, dict) and "lower" in bounds and "upper" in bounds:
                df.loc[values < bounds["lower"], col] = bounds["lower"]
                df.loc[values > bounds["upper"], col] = bounds["upper"]
                break
    
    elif method in ["impute_median", "impute_mean"]:
        agg_func = "median" if method == "impute_median" else "mean"
        
        # Find the metric column
        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col].values
            non_outlier_values = values[~outlier_mask]
            
            if len(non_outlier_values) > 0:
                if agg_func == "median":
                    replacement = np.median(non_outlier_values)
                else:
                    replacement = np.mean(non_outlier_values)
                
                df.loc[outlier_mask, col] = replacement
            break
    
    else:
        valid_methods = ["flag", "remove", "winsorize", "impute_median", "impute_mean"]
        raise ValueError(f"Unknown method: '{method}'. Valid options: {valid_methods}")
    
    return df

