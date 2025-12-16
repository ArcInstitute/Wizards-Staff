"""
Statistical tests for comparing calcium imaging metrics between experimental groups.

All functions return StatsResult objects with interpretations appropriate for
neuroscientists analyzing calcium imaging data.

Functions
---------
compare_two_groups : Compare a metric between two experimental groups
compare_multiple_groups : Compare a metric across three or more groups
compare_two_factors : Two-way analysis with two categorical factors
test_correlation : Test correlation between two continuous variables
test_proportion : Compare proportions between groups
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .core import StatsConfig, StatsResult
from .effect_size import cliffs_delta, cohens_d, eta_squared, rank_biserial
from .assumptions import check_normality, check_homogeneity_of_variance

if TYPE_CHECKING:
    pass


def _get_config(config: Optional[StatsConfig] = None) -> StatsConfig:
    """Get configuration, using defaults if not provided."""
    return config if config is not None else StatsConfig()


def _clean_data(
    data: pd.DataFrame,
    group_col: str,
    metric_col: str,
    config: StatsConfig
) -> pd.DataFrame:
    """Clean data according to nan_policy."""
    df = data.copy()
    
    if config.nan_policy == "omit":
        df = df.dropna(subset=[group_col, metric_col])
    elif config.nan_policy == "raise":
        if df[[group_col, metric_col]].isna().any().any():
            raise ValueError("Data contains NaN values and nan_policy='raise'")
    # propagate: do nothing
    
    return df


def _get_group_stats(values: np.ndarray) -> dict:
    """Calculate descriptive statistics for a group."""
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _should_use_parametric(
    group_data: dict,
    parametric: str,
    config: StatsConfig
) -> tuple:
    """
    Determine whether to use parametric tests.
    
    Returns
    -------
    tuple
        (use_parametric: bool, assumptions_met: dict)
    """
    if parametric == "yes":
        return True, {"forced": True}
    if parametric == "no":
        return False, {"forced": True}
    
    # Auto-detect based on assumptions
    assumptions_met = {}
    all_normal = True
    
    for group_name, values in group_data.items():
        if len(values) < 3:
            assumptions_met[f"normality_{group_name}"] = None
            continue
        
        norm_result = check_normality(values, alpha=config.alpha)
        assumptions_met[f"normality_{group_name}"] = norm_result["is_normal"]
        if not norm_result["is_normal"]:
            all_normal = False
    
    # Check variance homogeneity if we have enough data
    if len(group_data) >= 2:
        df_temp = pd.DataFrame([
            {"group": g, "value": v}
            for g, vals in group_data.items()
            for v in vals
        ])
        if len(df_temp) >= 4:
            var_result = check_homogeneity_of_variance(
                df_temp, "group", "value"
            )
            assumptions_met["equal_variance"] = var_result["equal_variance"]
        else:
            assumptions_met["equal_variance"] = None
    
    use_parametric = all_normal and assumptions_met.get("equal_variance", True)
    return use_parametric, assumptions_met


def compare_two_groups(
    data: pd.DataFrame,
    group_col: str,
    metric_col: str,
    group_a: Optional[str] = None,
    group_b: Optional[str] = None,
    paired: bool = False,
    parametric: str = "auto",
    config: Optional[StatsConfig] = None
) -> StatsResult:
    """
    Compare a metric between two experimental groups.
    
    Automatically selects the appropriate statistical test based on data
    characteristics and the paired parameter:
    
    - Parametric + unpaired: Independent samples t-test (or Welch's t-test)
    - Parametric + paired: Paired samples t-test
    - Non-parametric + unpaired: Mann-Whitney U test
    - Non-parametric + paired: Wilcoxon signed-rank test
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to analyze.
    group_col : str
        Column name containing group labels (e.g., "Treatment").
    metric_col : str
        Column name containing the metric to compare (e.g., "mean_frpm").
    group_a : str, optional
        Name of first group. Auto-detected if None.
    group_b : str, optional
        Name of second group. Auto-detected if None.
    paired : bool
        Whether samples are paired/matched. Default is False.
    parametric : str
        Force parametric ("yes"), non-parametric ("no"), or auto-detect ("auto").
        Default is "auto".
    config : StatsConfig, optional
        Configuration for the analysis.
    
    Returns
    -------
    StatsResult
        Object containing test results, effect size, and interpretation.
    
    Examples
    --------
    >>> # Compare firing rates between Treatment and Control
    >>> result = compare_two_groups(
    ...     data=sample_frpm,
    ...     group_col="Treatment",
    ...     metric_col="mean_frpm"
    ... )
    >>> print(result.summary)
    
    >>> # Force non-parametric test
    >>> result = compare_two_groups(
    ...     data=sample_frpm,
    ...     group_col="Treatment",
    ...     metric_col="mean_frpm",
    ...     parametric="no"
    ... )
    """
    config = _get_config(config)
    df = _clean_data(data, group_col, metric_col, config)
    
    # Get unique groups
    groups = df[group_col].unique()
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups, found {len(groups)}")
    if len(groups) > 2:
        if group_a is None or group_b is None:
            raise ValueError(
                f"Found {len(groups)} groups. Please specify group_a and group_b, "
                f"or use compare_multiple_groups() for multi-group comparisons."
            )
    
    # Determine which groups to compare
    if group_a is None:
        group_a = groups[0]
    if group_b is None:
        group_b = groups[1] if groups[1] != group_a else groups[0]
    
    # Extract data for each group
    values_a = df[df[group_col] == group_a][metric_col].values
    values_b = df[df[group_col] == group_b][metric_col].values
    
    # Validate sample sizes
    warnings_list = []
    if len(values_a) < config.min_samples_per_group:
        warnings_list.append(
            f"Group '{group_a}' has only {len(values_a)} samples "
            f"(minimum recommended: {config.min_samples_per_group})"
        )
    if len(values_b) < config.min_samples_per_group:
        warnings_list.append(
            f"Group '{group_b}' has only {len(values_b)} samples "
            f"(minimum recommended: {config.min_samples_per_group})"
        )
    
    # Check for paired data requirements
    if paired and len(values_a) != len(values_b):
        raise ValueError(
            f"Paired test requires equal sample sizes. "
            f"Group '{group_a}' has {len(values_a)}, "
            f"Group '{group_b}' has {len(values_b)}."
        )
    
    # Determine parametric vs non-parametric
    group_data = {group_a: values_a, group_b: values_b}
    use_parametric, assumptions_met = _should_use_parametric(
        group_data, parametric, config
    )
    
    # Run appropriate test
    if paired:
        if use_parametric:
            # Paired t-test
            stat, p_value = stats.ttest_rel(values_a, values_b)
            test_name = "Paired t-test"
            # Effect size: Cohen's d for paired data
            diff = values_a - values_b
            effect_result = cohens_d(diff, np.zeros_like(diff))
        else:
            # Wilcoxon signed-rank test
            stat, p_value = stats.wilcoxon(values_a, values_b)
            test_name = "Wilcoxon signed-rank test"
            # Effect size: matched-pairs rank biserial
            n = len(values_a)
            effect_result = {
                "d": stat / (n * (n + 1) / 2),  # Approximate r
                "magnitude": "not calculated",
                "interpretation": "Effect size for paired non-parametric test"
            }
    else:
        if use_parametric:
            # Check variance equality for t-test variant selection
            _, p_levene = stats.levene(values_a, values_b)
            if p_levene < config.alpha:
                # Unequal variances: use Welch's t-test
                stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
                test_name = "Welch's t-test"
            else:
                # Equal variances: use standard t-test
                stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=True)
                test_name = "Independent t-test"
            effect_result = cohens_d(values_a, values_b)
        else:
            # Mann-Whitney U test
            stat, p_value = stats.mannwhitneyu(
                values_a, values_b, alternative='two-sided'
            )
            test_name = "Mann-Whitney U test"
            effect_result = rank_biserial(stat, len(values_a), len(values_b))
    
    # Calculate group statistics
    group_stats = {
        group_a: _get_group_stats(values_a),
        group_b: _get_group_stats(values_b),
    }
    
    # Determine which group is higher
    higher_group = group_a if np.median(values_a) > np.median(values_b) else group_b
    lower_group = group_b if higher_group == group_a else group_a
    higher_median = group_stats[higher_group]["median"]
    lower_median = group_stats[lower_group]["median"]
    
    # Build interpretation
    significant = p_value < config.alpha
    if significant:
        interpretation = (
            f"There is a statistically significant difference between groups "
            f"({test_name}, p = {p_value:.4f}). "
            f"'{higher_group}' shows higher values than '{lower_group}' "
            f"(median: {higher_median:.2f} vs {lower_median:.2f}). "
            f"Effect size is {effect_result['magnitude']} "
            f"({effect_result.get('type', 'effect')} = {effect_result['d']:.3f})."
        )
    else:
        interpretation = (
            f"No statistically significant difference was found between groups "
            f"({test_name}, p = {p_value:.4f}). "
            f"'{higher_group}' median: {higher_median:.2f}, "
            f"'{lower_group}' median: {lower_median:.2f}. "
            f"Effect size: {effect_result['d']:.3f} ({effect_result['magnitude']})."
        )
    
    return StatsResult(
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p_value),
        effect_size=effect_result["d"],
        effect_size_type=effect_result.get("type", "Cohen's d" if use_parametric else "rank-biserial r"),
        effect_size_magnitude=effect_result["magnitude"],
        ci_lower=effect_result.get("ci_lower"),
        ci_upper=effect_result.get("ci_upper"),
        sample_sizes={group_a: len(values_a), group_b: len(values_b)},
        group_stats=group_stats,
        assumptions_met=assumptions_met,
        interpretation=interpretation,
        warnings=warnings_list,
        raw_data={group_a: values_a, group_b: values_b},
    )


def compare_multiple_groups(
    data: pd.DataFrame,
    group_col: str,
    metric_col: str,
    parametric: str = "auto",
    post_hoc: bool = True,
    config: Optional[StatsConfig] = None
) -> StatsResult:
    """
    Compare a metric across three or more experimental groups.
    
    Automatically selects the appropriate test:
    - Parametric: One-way ANOVA with Tukey HSD post-hoc
    - Non-parametric: Kruskal-Wallis H test with Dunn's post-hoc
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to analyze.
    group_col : str
        Column name containing group labels.
    metric_col : str
        Column name containing the metric to compare.
    parametric : str
        Force parametric ("yes"), non-parametric ("no"), or auto-detect ("auto").
    post_hoc : bool
        Whether to perform post-hoc pairwise comparisons. Default is True.
    config : StatsConfig, optional
        Configuration for the analysis.
    
    Returns
    -------
    StatsResult
        Object containing omnibus test results AND post-hoc comparison table.
    
    Examples
    --------
    >>> result = compare_multiple_groups(
    ...     data=sample_data,
    ...     group_col="Timepoint",
    ...     metric_col="mean_fwhm"
    ... )
    >>> print(result.summary)
    >>> print(result.post_hoc_table)  # Pairwise comparisons
    """
    config = _get_config(config)
    df = _clean_data(data, group_col, metric_col, config)
    
    # Get unique groups
    groups = df[group_col].unique()
    n_groups = len(groups)
    
    if n_groups < 2:
        raise ValueError(f"Need at least 2 groups, found {n_groups}")
    if n_groups == 2:
        warnings.warn(
            "Only 2 groups found. Using compare_two_groups() would be more appropriate.",
            UserWarning
        )
    
    # Extract data for each group
    group_data = {}
    group_stats = {}
    warnings_list = []
    
    for group in groups:
        values = df[df[group_col] == group][metric_col].values
        group_data[group] = values
        group_stats[group] = _get_group_stats(values)
        
        if len(values) < config.min_samples_per_group:
            warnings_list.append(
                f"Group '{group}' has only {len(values)} samples "
                f"(minimum recommended: {config.min_samples_per_group})"
            )
    
    # Determine parametric vs non-parametric
    use_parametric, assumptions_met = _should_use_parametric(
        group_data, parametric, config
    )
    
    # Run omnibus test
    group_values = [group_data[g] for g in groups]
    
    if use_parametric:
        # One-way ANOVA
        stat, p_value = stats.f_oneway(*group_values)
        test_name = "One-way ANOVA"
        
        # Calculate eta-squared effect size
        # Total sum of squares
        all_values = np.concatenate(group_values)
        grand_mean = np.mean(all_values)
        ss_total = np.sum((all_values - grand_mean) ** 2)
        
        # Between-group sum of squares
        ss_between = sum(
            len(vals) * (np.mean(vals) - grand_mean) ** 2
            for vals in group_values
        )
        
        df_between = n_groups - 1
        df_within = len(all_values) - n_groups
        effect_result = eta_squared(stat, df_between, df_within)
        
    else:
        # Kruskal-Wallis H test
        stat, p_value = stats.kruskal(*group_values)
        test_name = "Kruskal-Wallis H test"
        
        # Effect size: epsilon-squared
        n = len(np.concatenate(group_values))
        epsilon_sq = (stat - n_groups + 1) / (n - n_groups)
        effect_result = {
            "d": max(0, epsilon_sq),  # Can be negative in theory
            "magnitude": (
                "negligible" if epsilon_sq < 0.01 else
                "small" if epsilon_sq < 0.06 else
                "medium" if epsilon_sq < 0.14 else
                "large"
            ),
            "type": "epsilon-squared",
        }
    
    # Post-hoc tests
    post_hoc_df = None
    if post_hoc and p_value < config.alpha:
        post_hoc_results = []
        
        from itertools import combinations
        for g1, g2 in combinations(groups, 2):
            if use_parametric:
                # Tukey HSD approximation using t-tests with Bonferroni
                t_stat, t_p = stats.ttest_ind(group_data[g1], group_data[g2])
                eff = cohens_d(group_data[g1], group_data[g2])
            else:
                # Mann-Whitney for each pair
                t_stat, t_p = stats.mannwhitneyu(
                    group_data[g1], group_data[g2], alternative='two-sided'
                )
                eff = rank_biserial(t_stat, len(group_data[g1]), len(group_data[g2]))
            
            post_hoc_results.append({
                "Group 1": g1,
                "Group 2": g2,
                "Statistic": t_stat,
                "P-value": t_p,
                "Effect Size": eff["d"],
                "Magnitude": eff["magnitude"],
            })
        
        post_hoc_df = pd.DataFrame(post_hoc_results)
        
        # Apply multiple comparison correction
        from .corrections import apply_correction
        corrected = apply_correction(
            post_hoc_df["P-value"].values,
            method=config.correction_method,
            alpha=config.alpha
        )
        post_hoc_df["P-adjusted"] = corrected["adjusted_p"]
        post_hoc_df["Significant"] = corrected["significant"]
    
    # Build interpretation
    significant = p_value < config.alpha
    sample_sizes = {g: len(v) for g, v in group_data.items()}
    
    if significant:
        # Find group with highest median
        medians = {g: np.median(v) for g, v in group_data.items()}
        best_group = max(medians, key=medians.get)
        worst_group = min(medians, key=medians.get)
        
        interpretation = (
            f"Significant differences found among groups ({test_name}, "
            f"p = {p_value:.4f}, {effect_result.get('type', 'effect size')} = "
            f"{effect_result['d']:.3f} [{effect_result['magnitude']}]). "
            f"'{best_group}' shows the highest values (median: {medians[best_group]:.2f}), "
            f"'{worst_group}' shows the lowest (median: {medians[worst_group]:.2f})."
        )
        if post_hoc_df is not None:
            n_sig = post_hoc_df["Significant"].sum()
            interpretation += f" {n_sig} of {len(post_hoc_df)} pairwise comparisons are significant after correction."
    else:
        interpretation = (
            f"No significant differences found among groups ({test_name}, "
            f"p = {p_value:.4f}). Effect size: {effect_result['d']:.3f} ({effect_result['magnitude']})."
        )
    
    return StatsResult(
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p_value),
        effect_size=effect_result["d"],
        effect_size_type=effect_result.get("type", "eta-squared"),
        effect_size_magnitude=effect_result["magnitude"],
        sample_sizes=sample_sizes,
        group_stats=group_stats,
        assumptions_met=assumptions_met,
        interpretation=interpretation,
        warnings=warnings_list,
        raw_data=group_data,
        post_hoc_table=post_hoc_df,
    )


def compare_two_factors(
    data: pd.DataFrame,
    factor1_col: str,
    factor2_col: str,
    metric_col: str,
    config: Optional[StatsConfig] = None
) -> StatsResult:
    """
    Two-way analysis for experiments with two categorical factors.
    
    Tests main effects of each factor AND their interaction.
    Uses two-way ANOVA (requires statsmodels for full implementation).
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to analyze.
    factor1_col : str
        Column name for the first factor (e.g., "Treatment").
    factor2_col : str
        Column name for the second factor (e.g., "Genotype").
    metric_col : str
        Column name containing the metric to compare.
    config : StatsConfig, optional
        Configuration for the analysis.
    
    Returns
    -------
    StatsResult
        Object containing main effects, interaction, and interpretation.
    
    Examples
    --------
    >>> # Does firing rate depend on Treatment, Genotype, or their interaction?
    >>> result = compare_two_factors(
    ...     data=sample_frpm,
    ...     factor1_col="Treatment",
    ...     factor2_col="Genotype",
    ...     metric_col="mean_frpm"
    ... )
    """
    config = _get_config(config)
    
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        raise ImportError(
            "Two-way ANOVA requires statsmodels. "
            "Install with: pip install statsmodels"
        )
    
    df = _clean_data(data, factor1_col, metric_col, config)
    df = _clean_data(df, factor2_col, metric_col, config)
    
    # Build formula for two-way ANOVA
    # Need to handle column names with spaces
    safe_metric = metric_col.replace(" ", "_").replace("-", "_")
    safe_f1 = factor1_col.replace(" ", "_").replace("-", "_")
    safe_f2 = factor2_col.replace(" ", "_").replace("-", "_")
    
    df_safe = df.rename(columns={
        metric_col: safe_metric,
        factor1_col: safe_f1,
        factor2_col: safe_f2,
    })
    
    formula = f"{safe_metric} ~ C({safe_f1}) + C({safe_f2}) + C({safe_f1}):C({safe_f2})"
    
    model = ols(formula, data=df_safe).fit()
    anova_table = anova_lm(model, typ=2)
    
    # Extract results
    f1_f = anova_table.loc[f"C({safe_f1})", "F"]
    f1_p = anova_table.loc[f"C({safe_f1})", "PR(>F)"]
    f2_f = anova_table.loc[f"C({safe_f2})", "F"]
    f2_p = anova_table.loc[f"C({safe_f2})", "PR(>F)"]
    int_f = anova_table.loc[f"C({safe_f1}):C({safe_f2})", "F"]
    int_p = anova_table.loc[f"C({safe_f1}):C({safe_f2})", "PR(>F)"]
    
    # Calculate effect sizes (partial eta-squared)
    ss_total = anova_table["sum_sq"].sum()
    f1_eta = anova_table.loc[f"C({safe_f1})", "sum_sq"] / ss_total
    f2_eta = anova_table.loc[f"C({safe_f2})", "sum_sq"] / ss_total
    int_eta = anova_table.loc[f"C({safe_f1}):C({safe_f2})", "sum_sq"] / ss_total
    
    # Build interpretation
    interpretation_parts = []
    
    if f1_p < config.alpha:
        interpretation_parts.append(
            f"Main effect of {factor1_col} is significant (F = {f1_f:.2f}, "
            f"p = {f1_p:.4f}, η² = {f1_eta:.3f})"
        )
    else:
        interpretation_parts.append(f"No main effect of {factor1_col} (p = {f1_p:.4f})")
    
    if f2_p < config.alpha:
        interpretation_parts.append(
            f"Main effect of {factor2_col} is significant (F = {f2_f:.2f}, "
            f"p = {f2_p:.4f}, η² = {f2_eta:.3f})"
        )
    else:
        interpretation_parts.append(f"No main effect of {factor2_col} (p = {f2_p:.4f})")
    
    if int_p < config.alpha:
        interpretation_parts.append(
            f"INTERACTION between {factor1_col} and {factor2_col} is significant "
            f"(F = {int_f:.2f}, p = {int_p:.4f}, η² = {int_eta:.3f}). "
            f"The effect of one factor depends on the level of the other."
        )
    else:
        interpretation_parts.append(
            f"No interaction between factors (p = {int_p:.4f})"
        )
    
    interpretation = ". ".join(interpretation_parts)
    
    # Sample sizes
    sample_sizes = {}
    for f1_val in df[factor1_col].unique():
        for f2_val in df[factor2_col].unique():
            key = f"{f1_val} x {f2_val}"
            n = len(df[(df[factor1_col] == f1_val) & (df[factor2_col] == f2_val)])
            sample_sizes[key] = n
    
    return StatsResult(
        test_name="Two-way ANOVA",
        statistic=float(f1_f),  # Report factor1 F as main statistic
        p_value=float(f1_p),
        effect_size=float(f1_eta),
        effect_size_type="partial eta-squared",
        effect_size_magnitude=(
            "negligible" if f1_eta < 0.01 else
            "small" if f1_eta < 0.06 else
            "medium" if f1_eta < 0.14 else
            "large"
        ),
        sample_sizes=sample_sizes,
        interpretation=interpretation,
        test_details={
            "factor1": {
                "name": factor1_col,
                "F": float(f1_f),
                "p": float(f1_p),
                "eta_squared": float(f1_eta),
            },
            "factor2": {
                "name": factor2_col,
                "F": float(f2_f),
                "p": float(f2_p),
                "eta_squared": float(f2_eta),
            },
            "interaction": {
                "F": float(int_f),
                "p": float(int_p),
                "eta_squared": float(int_eta),
            },
            "anova_table": anova_table.to_dict(),
        },
    )


def test_correlation(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    method: str = "auto",
    config: Optional[StatsConfig] = None
) -> StatsResult:
    """
    Test correlation between two continuous variables.
    
    Automatically selects appropriate method based on data distribution:
    - Pearson: For normally distributed data (measures linear relationship)
    - Spearman: For non-normal data (measures monotonic relationship)
    - Kendall: Alternative to Spearman, better for small samples or many ties
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    x_col : str
        Column name for the first variable.
    y_col : str
        Column name for the second variable.
    method : str
        Correlation method: "auto", "pearson", "spearman", or "kendall".
    config : StatsConfig, optional
        Configuration for the analysis.
    
    Returns
    -------
    StatsResult
        Object containing correlation coefficient, p-value, and interpretation.
    
    Examples
    --------
    >>> # Is firing rate correlated with cell size?
    >>> result = test_correlation(
    ...     data=merged_df,
    ...     x_col="mean_frpm",
    ...     y_col="mean_area"
    ... )
    """
    config = _get_config(config)
    
    # Clean data
    df = data[[x_col, y_col]].dropna()
    x = df[x_col].values
    y = df[y_col].values
    
    if len(x) < 3:
        raise ValueError(f"Need at least 3 data points, got {len(x)}")
    
    # Determine method
    if method == "auto":
        # Check normality of both variables
        x_normal = check_normality(x, alpha=config.alpha)["is_normal"]
        y_normal = check_normality(y, alpha=config.alpha)["is_normal"]
        
        if x_normal and y_normal:
            method = "pearson"
        else:
            method = "spearman"
    
    # Calculate correlation
    if method == "pearson":
        r, p_value = stats.pearsonr(x, y)
        test_name = "Pearson correlation"
    elif method == "spearman":
        r, p_value = stats.spearmanr(x, y)
        test_name = "Spearman correlation"
    elif method == "kendall":
        r, p_value = stats.kendalltau(x, y)
        test_name = "Kendall's tau correlation"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Effect size interpretation (using Cohen's conventions for r)
    abs_r = abs(r)
    if abs_r < 0.1:
        magnitude = "negligible"
    elif abs_r < 0.3:
        magnitude = "small"
    elif abs_r < 0.5:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    # Calculate confidence interval for r
    n = len(x)
    if n > 3:
        # Fisher z-transformation for CI
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - config.alpha / 2)
        ci_lower = np.tanh(z - z_crit * se)
        ci_upper = np.tanh(z + z_crit * se)
    else:
        ci_lower, ci_upper = None, None
    
    # Build interpretation
    direction = "positive" if r > 0 else "negative"
    significant = p_value < config.alpha
    
    if significant:
        interpretation = (
            f"There is a significant {magnitude} {direction} correlation between "
            f"{x_col} and {y_col} ({test_name}, r = {r:.3f}, p = {p_value:.4f}). "
            f"As {x_col} increases, {y_col} tends to "
            f"{'increase' if r > 0 else 'decrease'}."
        )
    else:
        interpretation = (
            f"No significant correlation found between {x_col} and {y_col} "
            f"({test_name}, r = {r:.3f}, p = {p_value:.4f})."
        )
    
    return StatsResult(
        test_name=test_name,
        statistic=float(r),
        p_value=float(p_value),
        effect_size=float(r),
        effect_size_type="correlation coefficient",
        effect_size_magnitude=magnitude,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        sample_sizes={"total": n},
        interpretation=interpretation,
        raw_data={"x": x, "y": y},
        test_details={
            "x_col": x_col,
            "y_col": y_col,
            "method": method,
        },
    )


def test_proportion(
    data: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    success_value: any = True,
    config: Optional[StatsConfig] = None
) -> StatsResult:
    """
    Compare proportions between groups.
    
    Useful for comparing the percentage of "active" neurons, neurons above
    a threshold, or any binary outcome between experimental groups.
    
    Uses chi-square test for large samples or Fisher's exact test for small samples.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    group_col : str
        Column name containing group labels.
    outcome_col : str
        Column name containing the binary outcome.
    success_value : any
        Value in outcome_col that counts as "success". Default is True.
    config : StatsConfig, optional
        Configuration for the analysis.
    
    Returns
    -------
    StatsResult
        Object containing test results and interpretation.
    
    Examples
    --------
    >>> # Compare proportion of "active" neurons (firing rate > 1) between groups
    >>> df["is_active"] = df["firing_rate"] > 1
    >>> result = test_proportion(
    ...     data=df,
    ...     group_col="Treatment",
    ...     outcome_col="is_active"
    ... )
    """
    config = _get_config(config)
    
    groups = data[group_col].unique()
    if len(groups) != 2:
        raise ValueError(
            f"Proportion test currently supports exactly 2 groups, found {len(groups)}. "
            f"For multiple groups, consider chi-square test of independence."
        )
    
    g1, g2 = groups[0], groups[1]
    
    # Create contingency table
    g1_data = data[data[group_col] == g1][outcome_col]
    g2_data = data[data[group_col] == g2][outcome_col]
    
    n1_success = (g1_data == success_value).sum()
    n1_total = len(g1_data)
    n2_success = (g2_data == success_value).sum()
    n2_total = len(g2_data)
    
    contingency_table = np.array([
        [n1_success, n1_total - n1_success],
        [n2_success, n2_total - n2_success]
    ])
    
    # Choose test based on expected frequencies
    expected = np.outer(
        contingency_table.sum(axis=1),
        contingency_table.sum(axis=0)
    ) / contingency_table.sum()
    
    use_fisher = (expected < 5).any()
    
    if use_fisher:
        # Fisher's exact test
        odds_ratio, p_value = stats.fisher_exact(contingency_table)
        test_name = "Fisher's exact test"
        stat = odds_ratio
    else:
        # Chi-square test
        stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        test_name = "Chi-square test"
    
    # Calculate proportions
    prop1 = n1_success / n1_total
    prop2 = n2_success / n2_total
    
    # Effect size: odds ratio or relative risk
    # Using odds ratio
    if n1_success > 0 and n2_success > 0 and (n1_total - n1_success) > 0 and (n2_total - n2_success) > 0:
        odds1 = n1_success / (n1_total - n1_success)
        odds2 = n2_success / (n2_total - n2_success)
        odds_ratio = odds1 / odds2
    else:
        odds_ratio = np.nan
    
    # Build interpretation
    significant = p_value < config.alpha
    higher_group = g1 if prop1 > prop2 else g2
    lower_group = g2 if prop1 > prop2 else g1
    higher_prop = max(prop1, prop2)
    lower_prop = min(prop1, prop2)
    
    if significant:
        interpretation = (
            f"Significant difference in proportions between groups "
            f"({test_name}, p = {p_value:.4f}). "
            f"'{higher_group}' has a higher rate ({higher_prop:.1%}) compared to "
            f"'{lower_group}' ({lower_prop:.1%})."
        )
    else:
        interpretation = (
            f"No significant difference in proportions ({test_name}, p = {p_value:.4f}). "
            f"'{g1}': {prop1:.1%}, '{g2}': {prop2:.1%}."
        )
    
    return StatsResult(
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p_value),
        effect_size=float(odds_ratio) if not np.isnan(odds_ratio) else 0,
        effect_size_type="odds ratio",
        sample_sizes={g1: n1_total, g2: n2_total},
        group_stats={
            g1: {"proportion": prop1, "n_success": n1_success, "n_total": n1_total},
            g2: {"proportion": prop2, "n_success": n2_success, "n_total": n2_total},
        },
        interpretation=interpretation,
        test_details={
            "contingency_table": contingency_table.tolist(),
            "test_used": "fisher" if use_fisher else "chi_square",
        },
    )

