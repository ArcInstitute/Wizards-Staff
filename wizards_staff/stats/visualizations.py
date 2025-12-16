"""
Publication-quality statistical visualizations for calcium imaging data.

All functions return matplotlib Figure objects that can be further customized.
These visualizations are designed to clearly communicate statistical results
to both experts and non-specialists.

Functions
---------
plot_group_comparison : Box/violin plot with statistical annotations
plot_paired_comparison : Paired data visualization with connecting lines
plot_distribution_check : Histogram and Q-Q plot for normality assessment
plot_effect_sizes : Forest plot of effect sizes
plot_power_curve : Power as function of effect size or sample size
plot_multiple_metrics_summary : Summary panel of multiple metrics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from .core import StatsResult


def plot_group_comparison(
    data: pd.DataFrame,
    group_col: str,
    metric_col: str,
    stats_result: Optional["StatsResult"] = None,
    plot_type: str = "auto",
    show_points: bool = True,
    show_stats: bool = True,
    colors: Optional[List[str]] = None,
    figsize: tuple = (6, 5),
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None
) -> plt.Figure:
    """
    Create comparison plot with optional statistical annotations.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    group_col : str
        Column name containing group labels.
    metric_col : str
        Column name containing metric values.
    stats_result : StatsResult, optional
        Result from statistical test. If provided, adds significance annotation.
    plot_type : str
        Type of plot: "auto", "box", "violin", "strip", "bar".
        "auto" chooses based on sample size.
    show_points : bool
        Whether to overlay individual data points.
    show_stats : bool
        Whether to show statistical annotations.
    colors : list of str, optional
        Colors for each group.
    figsize : tuple
        Figure size (width, height).
    title : str, optional
        Plot title.
    ylabel : str, optional
        Custom y-axis label. If None, auto-generates from metric_col.
    xlabel : str, optional
        Custom x-axis label. If None, no x-axis label is shown.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    
    Examples
    --------
    >>> fig = plot_group_comparison(
    ...     data=df,
    ...     group_col="Treatment",
    ...     metric_col="firing_rate",
    ...     stats_result=result
    ... )
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    groups = data[group_col].unique()
    n_groups = len(groups)
    
    # Default colors
    if colors is None:
        colors = plt.cm.Set2(np.linspace(0, 1, max(n_groups, 3)))[:n_groups]
    
    # Choose plot type based on sample size
    if plot_type == "auto":
        min_n = min(len(data[data[group_col] == g]) for g in groups)
        if min_n < 10:
            plot_type = "box"
        else:
            plot_type = "violin"
    
    # Prepare data
    group_data = [data[data[group_col] == g][metric_col].dropna().values for g in groups]
    positions = np.arange(n_groups)
    
    # Create the plot
    if plot_type == "violin":
        parts = ax.violinplot(group_data, positions=positions, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
    
    elif plot_type == "box":
        bp = ax.boxplot(group_data, positions=positions, patch_artist=True, widths=0.6)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)
    
    elif plot_type == "bar":
        means = [np.mean(d) for d in group_data]
        sems = [np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0 for d in group_data]
        ax.bar(positions, means, yerr=sems, color=colors, alpha=0.7, capsize=5)
    
    elif plot_type == "strip":
        for i, (pos, vals) in enumerate(zip(positions, group_data)):
            jitter = np.random.uniform(-0.15, 0.15, len(vals))
            ax.scatter(pos + jitter, vals, c=[colors[i]], alpha=0.6, s=50)
            ax.hlines(np.median(vals), pos - 0.25, pos + 0.25, colors='black', linewidth=2)
    
    # Overlay individual points
    if show_points and plot_type in ["box", "violin"]:
        for i, (pos, vals) in enumerate(zip(positions, group_data)):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(pos + jitter, vals, c='black', alpha=0.4, s=20, zorder=5)
    
    # Add statistical annotation
    if show_stats and stats_result is not None:
        p_value = stats_result.p_value
        
        # Format p-value
        if p_value < 0.001:
            p_text = "p < 0.001"
        elif p_value < 0.01:
            p_text = f"p = {p_value:.3f}"
        else:
            p_text = f"p = {p_value:.2f}"
        
        # Significance stars
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"
        else:
            stars = "ns"
        
        # Add bracket between groups
        y_max = max(max(d) for d in group_data if len(d) > 0)
        y_range = y_max - min(min(d) for d in group_data if len(d) > 0)
        bracket_height = y_max + 0.05 * y_range
        
        if n_groups == 2:
            ax.plot([0, 0, 1, 1], 
                   [bracket_height, bracket_height + 0.02 * y_range, 
                    bracket_height + 0.02 * y_range, bracket_height],
                   'k-', linewidth=1.5)
            ax.text(0.5, bracket_height + 0.04 * y_range, f"{stars}\n{p_text}",
                   ha='center', va='bottom', fontsize=10)
    
    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, fontsize=11)
    
    # Clean up y-axis label (remove "mean_" prefix and underscores)
    if ylabel is None:
        ylabel = metric_col.replace("mean_", "").replace("_", " ")
    ax.set_ylabel(ylabel, fontsize=12)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    
    # Add title with padding for stats annotation
    if title:
        pad = 25 if (show_stats and stats_result is not None) else 10
        ax.set_title(title, fontsize=14, fontweight='bold', pad=pad)
    
    # Add sample sizes below x-axis labels
    for i, (pos, vals) in enumerate(zip(positions, group_data)):
        ax.annotate(f'n={len(vals)}', xy=(pos, 0), xycoords=('data', 'axes fraction'),
                   xytext=(0, -25), textcoords='offset points',
                   ha='center', va='top', fontsize=9, color='gray')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    # Adjust bottom margin for sample size labels
    plt.subplots_adjust(bottom=0.15)
    return fig


def plot_paired_comparison(
    data: pd.DataFrame,
    group_col: str,
    metric_col: str,
    subject_col: str,
    stats_result: Optional["StatsResult"] = None,
    figsize: tuple = (6, 5)
) -> plt.Figure:
    """
    Create paired comparison plot with connecting lines.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with paired data.
    group_col : str
        Column containing group/condition labels.
    metric_col : str
        Column containing metric values.
    subject_col : str
        Column identifying paired subjects.
    stats_result : StatsResult, optional
        Result from paired statistical test.
    figsize : tuple
        Figure size.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    groups = data[group_col].unique()
    if len(groups) != 2:
        raise ValueError("Paired plot requires exactly 2 groups")
    
    g1, g2 = groups[0], groups[1]
    
    # Get paired data
    subjects = data[subject_col].unique()
    
    for subject in subjects:
        subj_data = data[data[subject_col] == subject]
        if len(subj_data) == 2:
            y1 = subj_data[subj_data[group_col] == g1][metric_col].values[0]
            y2 = subj_data[subj_data[group_col] == g2][metric_col].values[0]
            
            color = 'green' if y2 > y1 else 'red' if y2 < y1 else 'gray'
            ax.plot([0, 1], [y1, y2], 'o-', color=color, alpha=0.5, markersize=8)
    
    # Add group means
    mean1 = data[data[group_col] == g1][metric_col].mean()
    mean2 = data[data[group_col] == g2][metric_col].mean()
    ax.scatter([0, 1], [mean1, mean2], s=200, c='black', marker='D', zorder=10)
    
    # Add statistical annotation
    if stats_result is not None:
        p_value = stats_result.p_value
        stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        y_max = data[metric_col].max()
        ax.text(0.5, y_max * 1.1, f"{stars}\np = {p_value:.3f}",
               ha='center', fontsize=12)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g1, g2], fontsize=12)
    ax.set_ylabel(metric_col, fontsize=12)
    ax.set_xlim(-0.5, 1.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_distribution_check(
    data: Union[np.ndarray, pd.Series],
    group_data: Optional[Dict[str, np.ndarray]] = None,
    figsize: tuple = (10, 4)
) -> plt.Figure:
    """
    Create histogram + Q-Q plot for distribution assessment.
    
    Parameters
    ----------
    data : np.ndarray or pd.Series
        Data to check.
    group_data : dict, optional
        Dictionary of {group_name: array} for multiple groups.
    figsize : tuple
        Figure size.
    
    Returns
    -------
    plt.Figure
        Figure with histogram and Q-Q plot.
    """
    if group_data is not None:
        n_groups = len(group_data)
        fig, axes = plt.subplots(n_groups, 2, figsize=(figsize[0], figsize[1] * n_groups))
        if n_groups == 1:
            axes = axes.reshape(1, -1)
        
        for i, (name, values) in enumerate(group_data.items()):
            values = np.asarray(values)
            values = values[~np.isnan(values)]
            
            # Histogram
            axes[i, 0].hist(values, bins='auto', density=True, alpha=0.7, 
                           color='steelblue', edgecolor='black')
            
            # Overlay normal curve
            mu, sigma = np.mean(values), np.std(values)
            x = np.linspace(values.min(), values.max(), 100)
            axes[i, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                           label='Normal fit')
            axes[i, 0].set_title(f'{name}: Histogram')
            axes[i, 0].legend()
            
            # Q-Q plot
            stats.probplot(values, dist="norm", plot=axes[i, 1])
            axes[i, 1].set_title(f'{name}: Q-Q Plot')
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        values = np.asarray(data)
        values = values[~np.isnan(values)]
        
        # Histogram
        axes[0].hist(values, bins='auto', density=True, alpha=0.7,
                    color='steelblue', edgecolor='black')
        
        mu, sigma = np.mean(values), np.std(values)
        x = np.linspace(values.min(), values.max(), 100)
        axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                    label='Normal fit')
        axes[0].set_title('Histogram with Normal Fit')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        
        # Q-Q plot
        stats.probplot(values, dist="norm", plot=axes[1])
        axes[1].set_title('Normal Q-Q Plot')
    
    plt.tight_layout()
    return fig


def plot_effect_sizes(
    results: List["StatsResult"],
    labels: Optional[List[str]] = None,
    figsize: tuple = (8, 5)
) -> plt.Figure:
    """
    Forest plot of effect sizes across multiple comparisons.
    
    Parameters
    ----------
    results : list of StatsResult
        List of statistical results to plot.
    labels : list of str, optional
        Labels for each comparison.
    figsize : tuple
        Figure size.
    
    Returns
    -------
    plt.Figure
        Forest plot figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_results = len(results)
    positions = np.arange(n_results)
    
    if labels is None:
        labels = [f"Comparison {i+1}" for i in range(n_results)]
    
    effect_sizes = []
    ci_lowers = []
    ci_uppers = []
    colors = []
    
    for result in results:
        effect_sizes.append(result.effect_size)
        ci_lowers.append(result.ci_lower if result.ci_lower else result.effect_size - 0.2)
        ci_uppers.append(result.ci_upper if result.ci_upper else result.effect_size + 0.2)
        colors.append('green' if result.p_value < 0.05 else 'gray')
    
    # Plot effect sizes with CI
    for i, (pos, es, ci_l, ci_u, color) in enumerate(zip(positions, effect_sizes, 
                                                          ci_lowers, ci_uppers, colors)):
        ax.errorbar(es, pos, xerr=[[es - ci_l], [ci_u - es]], 
                   fmt='o', markersize=10, color=color, capsize=5, linewidth=2)
    
    # Add reference line at 0
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add effect size magnitude regions
    ax.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Negligible')
    ax.axvspan(-0.5, -0.2, alpha=0.1, color='yellow')
    ax.axvspan(0.2, 0.5, alpha=0.1, color='yellow', label='Small')
    ax.axvspan(-0.8, -0.5, alpha=0.1, color='orange')
    ax.axvspan(0.5, 0.8, alpha=0.1, color='orange', label='Medium')
    
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Effect Size', fontsize=12)
    ax.set_title('Effect Size Forest Plot', fontsize=14, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_power_curve(
    effect_sizes: Optional[np.ndarray] = None,
    n_per_group: Optional[int] = None,
    test_type: str = "two_sample_t",
    figsize: tuple = (8, 5)
) -> plt.Figure:
    """
    Plot power as a function of effect size or sample size.
    
    Parameters
    ----------
    effect_sizes : np.ndarray, optional
        Effect sizes to plot. If None, uses default range.
    n_per_group : int, optional
        Sample size per group. If provided, plots power vs effect size.
        If None, plots power vs sample size for medium effect.
    test_type : str
        Type of statistical test.
    figsize : tuple
        Figure size.
    
    Returns
    -------
    plt.Figure
        Power curve figure.
    """
    from .power import calculate_achieved_power
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if n_per_group is not None:
        # Plot power vs effect size for fixed n
        if effect_sizes is None:
            effect_sizes = np.linspace(0.1, 2.0, 50)
        
        powers = [
            calculate_achieved_power(n_per_group, es, test_type=test_type)["power"]
            for es in effect_sizes
        ]
        
        ax.plot(effect_sizes, powers, 'b-', linewidth=2)
        ax.axhline(0.8, color='red', linestyle='--', label='80% power')
        ax.axhline(0.5, color='orange', linestyle=':', alpha=0.7, label='50% power')
        
        # Mark effect size categories
        ax.axvline(0.2, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0.8, color='gray', linestyle=':', alpha=0.5)
        
        ax.text(0.2, 0.05, 'Small', ha='center', fontsize=9, color='gray')
        ax.text(0.5, 0.05, 'Medium', ha='center', fontsize=9, color='gray')
        ax.text(0.8, 0.05, 'Large', ha='center', fontsize=9, color='gray')
        
        ax.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
        ax.set_ylabel('Statistical Power', fontsize=12)
        ax.set_title(f'Power Curve (n = {n_per_group} per group)', fontsize=14)
        
    else:
        # Plot power vs sample size for different effect sizes
        sample_sizes = np.arange(5, 101, 5)
        effect_sizes_to_show = [0.2, 0.5, 0.8]
        colors = ['red', 'orange', 'green']
        labels = ['Small (d=0.2)', 'Medium (d=0.5)', 'Large (d=0.8)']
        
        for es, color, label in zip(effect_sizes_to_show, colors, labels):
            powers = [
                calculate_achieved_power(n, es, test_type=test_type)["power"]
                for n in sample_sizes
            ]
            ax.plot(sample_sizes, powers, color=color, linewidth=2, label=label)
        
        ax.axhline(0.8, color='black', linestyle='--', alpha=0.5)
        ax.text(sample_sizes[-1], 0.82, '80% power', fontsize=9)
        
        ax.set_xlabel('Sample Size (per group)', fontsize=12)
        ax.set_ylabel('Statistical Power', fontsize=12)
        ax.set_title('Power vs Sample Size', fontsize=14)
        ax.legend(loc='lower right')
    
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_multiple_metrics_summary(
    results_dict: Dict[str, "StatsResult"],
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Summary visualization of multiple metric comparisons.
    
    Creates panel with effect sizes, p-values, and sample sizes.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping metric names to StatsResult objects.
    figsize : tuple
        Figure size.
    
    Returns
    -------
    plt.Figure
        Summary panel figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    metrics = list(results_dict.keys())
    n_metrics = len(metrics)
    positions = np.arange(n_metrics)
    
    effect_sizes = [results_dict[m].effect_size for m in metrics]
    p_values = [results_dict[m].p_value for m in metrics]
    
    # Colors based on significance
    colors = ['green' if p < 0.05 else 'gray' for p in p_values]
    
    # Panel 1: Effect sizes
    ax1 = axes[0]
    bars1 = ax1.barh(positions, effect_sizes, color=colors, alpha=0.7)
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.axvline(0.2, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(0.8, color='gray', linestyle=':', alpha=0.5)
    ax1.set_yticks(positions)
    ax1.set_yticklabels(metrics)
    ax1.set_xlabel('Effect Size')
    ax1.set_title('Effect Sizes', fontweight='bold')
    
    # Panel 2: P-values
    ax2 = axes[1]
    bars2 = ax2.barh(positions, -np.log10(p_values), color=colors, alpha=0.7)
    ax2.axvline(-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
    ax2.axvline(-np.log10(0.01), color='red', linestyle=':', alpha=0.7, label='p = 0.01')
    ax2.set_yticks(positions)
    ax2.set_yticklabels([])
    ax2.set_xlabel('-log₁₀(p-value)')
    ax2.set_title('Statistical Significance', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    
    # Panel 3: Sample sizes
    ax3 = axes[2]
    
    # Collect sample sizes
    sample_data = []
    for m in metrics:
        result = results_dict[m]
        total_n = sum(result.sample_sizes.values())
        sample_data.append(total_n)
    
    bars3 = ax3.barh(positions, sample_data, color='steelblue', alpha=0.7)
    ax3.set_yticks(positions)
    ax3.set_yticklabels([])
    ax3.set_xlabel('Total N')
    ax3.set_title('Sample Sizes', fontweight='bold')
    
    # Add values on bars
    for i, (bar, n) in enumerate(zip(bars3, sample_data)):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'n={n}', va='center', fontsize=9)
    
    plt.suptitle('Multi-Metric Statistical Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

