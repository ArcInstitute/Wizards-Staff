"""
Effect size calculations for calcium imaging comparisons.

Effect sizes are essential for understanding biological significance beyond
statistical significance. A p < 0.05 doesn't tell you if the effect matters;
effect sizes quantify HOW MUCH groups differ.

Why Effect Sizes Matter
-----------------------
- With large samples, tiny (biologically meaningless) differences can be "significant"
- With small samples, large (biologically meaningful) differences may not be "significant"
- Effect sizes are comparable across studies regardless of sample size
- They help with power analysis and planning future experiments

Functions
---------
cohens_d : Effect size for two-group parametric comparisons
eta_squared : Effect size for ANOVA
rank_biserial : Effect size for Mann-Whitney U test
cliffs_delta : Robust non-parametric effect size
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled: bool = True
) -> Dict:
    """
    Calculate Cohen's d for two-group comparisons.
    
    Cohen's d measures the standardized difference between two means.
    It tells you how many standard deviations apart the groups are.
    
    Parameters
    ----------
    group1 : np.ndarray
        Values from the first group.
    group2 : np.ndarray
        Values from the second group.
    pooled : bool
        If True (default), use pooled standard deviation (assumes equal variances).
        If False, use the standard deviation of group1 only (Glass's delta).
    
    Returns
    -------
    dict
        Dictionary containing:
        - d: Cohen's d value
        - ci_lower, ci_upper: 95% confidence interval
        - magnitude: "negligible", "small", "medium", or "large"
        - interpretation: Plain-language interpretation
        - type: "Cohen's d"
    
    Notes
    -----
    Interpretation guidelines for calcium imaging:
    
    - |d| < 0.2: Negligible - likely not biologically meaningful
    - 0.2 ≤ |d| < 0.5: Small - subtle effect, may need larger samples to confirm
    - 0.5 ≤ |d| < 0.8: Medium - meaningful biological effect
    - |d| ≥ 0.8: Large - substantial biological effect
    
    Examples
    --------
    >>> control = np.array([10, 12, 11, 13, 10, 12])
    >>> treatment = np.array([15, 17, 16, 14, 18, 16])
    >>> result = cohens_d(control, treatment)
    >>> print(f"Cohen's d = {result['d']:.2f} ({result['magnitude']})")
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    if pooled:
        # Pooled standard deviation
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    else:
        # Glass's delta (uses group1 SD only)
        std1 = np.std(group1, ddof=1)
        d = (mean1 - mean2) / std1 if std1 > 0 else 0
    
    # Confidence interval using non-central t distribution approximation
    # Simplified approach using normal approximation
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    ci_lower = d - 1.96 * se_d
    ci_upper = d + 1.96 * se_d
    
    # Interpret magnitude
    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
        interpretation = (
            "The difference between groups is negligible. "
            "This is unlikely to be biologically meaningful."
        )
    elif abs_d < 0.5:
        magnitude = "small"
        interpretation = (
            "There is a small difference between groups. "
            "This may be biologically meaningful but is subtle."
        )
    elif abs_d < 0.8:
        magnitude = "medium"
        interpretation = (
            "There is a medium-sized difference between groups. "
            "This represents a meaningful biological effect."
        )
    else:
        magnitude = "large"
        interpretation = (
            "There is a large difference between groups. "
            "This represents a substantial biological effect."
        )
    
    # Add direction to interpretation
    if d > 0:
        interpretation += " Group 1 has higher values than Group 2."
    elif d < 0:
        interpretation += " Group 2 has higher values than Group 1."
    
    return {
        "d": float(d),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "magnitude": magnitude,
        "interpretation": interpretation,
        "type": "Cohen's d",
    }


def eta_squared(
    f_statistic: float,
    df_between: int,
    df_within: int
) -> Dict:
    """
    Calculate eta-squared (η²) for ANOVA results.
    
    Eta-squared measures the proportion of variance in the dependent variable
    that is explained by the independent variable (group membership).
    
    Parameters
    ----------
    f_statistic : float
        F-statistic from ANOVA.
    df_between : int
        Degrees of freedom between groups (number of groups - 1).
    df_within : int
        Degrees of freedom within groups (total N - number of groups).
    
    Returns
    -------
    dict
        Dictionary containing:
        - eta_sq: Eta-squared value
        - magnitude: Interpretation of effect size
        - interpretation: Plain-language explanation
        - type: "eta-squared"
    
    Notes
    -----
    Interpretation guidelines:
    
    - η² < 0.01: Negligible
    - 0.01 ≤ η² < 0.06: Small
    - 0.06 ≤ η² < 0.14: Medium
    - η² ≥ 0.14: Large
    
    Examples
    --------
    >>> # From ANOVA with F(2, 27) = 5.5
    >>> result = eta_squared(f_statistic=5.5, df_between=2, df_within=27)
    >>> print(f"η² = {result['eta_sq']:.3f} ({result['magnitude']})")
    """
    # Calculate eta-squared from F-statistic
    eta_sq = (f_statistic * df_between) / (f_statistic * df_between + df_within)
    
    # Interpret magnitude
    if eta_sq < 0.01:
        magnitude = "negligible"
        interpretation = (
            f"Group membership explains less than 1% of the variance. "
            f"Differences between groups are negligible."
        )
    elif eta_sq < 0.06:
        magnitude = "small"
        interpretation = (
            f"Group membership explains {eta_sq*100:.1f}% of the variance. "
            f"This is a small effect."
        )
    elif eta_sq < 0.14:
        magnitude = "medium"
        interpretation = (
            f"Group membership explains {eta_sq*100:.1f}% of the variance. "
            f"This is a medium effect, indicating meaningful group differences."
        )
    else:
        magnitude = "large"
        interpretation = (
            f"Group membership explains {eta_sq*100:.1f}% of the variance. "
            f"This is a large effect, indicating substantial group differences."
        )
    
    return {
        "d": float(eta_sq),  # Use 'd' key for consistency with other effect sizes
        "eta_sq": float(eta_sq),
        "magnitude": magnitude,
        "interpretation": interpretation,
        "type": "eta-squared",
    }


def rank_biserial(
    u_statistic: float,
    n1: int,
    n2: int
) -> Dict:
    """
    Calculate rank-biserial correlation for Mann-Whitney U test.
    
    The rank-biserial correlation is the preferred effect size measure for
    non-parametric two-group comparisons. It ranges from -1 to +1.
    
    Parameters
    ----------
    u_statistic : float
        U statistic from Mann-Whitney test.
    n1 : int
        Sample size of group 1.
    n2 : int
        Sample size of group 2.
    
    Returns
    -------
    dict
        Dictionary containing:
        - r: Rank-biserial correlation coefficient
        - magnitude: Interpretation of effect size
        - interpretation: Plain-language explanation
        - type: "rank-biserial r"
    
    Notes
    -----
    Interpretation (uses same thresholds as Pearson correlation):
    
    - |r| < 0.1: Negligible
    - 0.1 ≤ |r| < 0.3: Small
    - 0.3 ≤ |r| < 0.5: Medium
    - |r| ≥ 0.5: Large
    
    Examples
    --------
    >>> # From Mann-Whitney U test with U = 45, n1 = 10, n2 = 12
    >>> result = rank_biserial(u_statistic=45, n1=10, n2=12)
    >>> print(f"r = {result['r']:.3f} ({result['magnitude']})")
    """
    # Calculate rank-biserial correlation
    # r = 1 - (2U) / (n1 * n2)
    r = 1 - (2 * u_statistic) / (n1 * n2)
    
    # Interpret magnitude
    abs_r = abs(r)
    if abs_r < 0.1:
        magnitude = "negligible"
        interpretation = (
            "There is negligible separation between groups in terms of ranks. "
            "The distributions largely overlap."
        )
    elif abs_r < 0.3:
        magnitude = "small"
        interpretation = (
            "There is small separation between groups. "
            "One group tends to have slightly higher ranks than the other."
        )
    elif abs_r < 0.5:
        magnitude = "medium"
        interpretation = (
            "There is medium separation between groups. "
            "One group clearly tends to have higher values than the other."
        )
    else:
        magnitude = "large"
        interpretation = (
            "There is large separation between groups. "
            "The distributions show substantial non-overlap."
        )
    
    return {
        "d": float(r),  # Use 'd' key for consistency
        "r": float(r),
        "magnitude": magnitude,
        "interpretation": interpretation,
        "type": "rank-biserial r",
    }


def cliffs_delta(
    group1: np.ndarray,
    group2: np.ndarray
) -> Dict:
    """
    Calculate Cliff's Delta - robust non-parametric effect size.
    
    Cliff's Delta measures the probability that a randomly selected value
    from group1 is greater than a randomly selected value from group2,
    minus the reverse probability.
    
    This is a very intuitive measure: δ = 0.5 means that values from group1
    are greater than values from group2 75% of the time.
    
    Parameters
    ----------
    group1 : np.ndarray
        Values from the first group.
    group2 : np.ndarray
        Values from the second group.
    
    Returns
    -------
    dict
        Dictionary containing:
        - delta: Cliff's Delta value (range: -1 to +1)
        - magnitude: Interpretation of effect size
        - interpretation: Plain-language explanation
        - type: "Cliff's delta"
    
    Notes
    -----
    Interpretation guidelines:
    
    - |δ| < 0.147: Negligible
    - 0.147 ≤ |δ| < 0.33: Small
    - 0.33 ≤ |δ| < 0.474: Medium
    - |δ| ≥ 0.474: Large
    
    The advantage of Cliff's Delta is that it's purely ordinal - it only
    considers which values are greater, not by how much. This makes it
    robust to outliers and non-normal distributions.
    
    Examples
    --------
    >>> control = np.array([1, 2, 3, 4, 5])
    >>> treatment = np.array([4, 5, 6, 7, 8])
    >>> result = cliffs_delta(control, treatment)
    >>> print(f"δ = {result['delta']:.3f}")
    >>> print(result['interpretation'])
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    n1, n2 = len(group1), len(group2)
    
    # Count comparisons
    # For each pair (x1, x2), count whether x1 > x2, x1 < x2, or x1 == x2
    greater = 0
    less = 0
    
    for x1 in group1:
        for x2 in group2:
            if x1 > x2:
                greater += 1
            elif x1 < x2:
                less += 1
    
    # Cliff's delta = (n_greater - n_less) / (n1 * n2)
    delta = (greater - less) / (n1 * n2)
    
    # Interpret magnitude
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        magnitude = "negligible"
        interpretation = (
            "Values from the two groups are nearly equally likely to be larger. "
            "There is no meaningful difference in the distributions."
        )
    elif abs_delta < 0.33:
        magnitude = "small"
        pct = 50 + abs_delta * 50
        interpretation = (
            f"Values from {'group 1' if delta > 0 else 'group 2'} are larger "
            f"about {pct:.0f}% of the time (vs 50% if no difference). "
            f"This is a small effect."
        )
    elif abs_delta < 0.474:
        magnitude = "medium"
        pct = 50 + abs_delta * 50
        interpretation = (
            f"Values from {'group 1' if delta > 0 else 'group 2'} are larger "
            f"about {pct:.0f}% of the time. This is a medium effect."
        )
    else:
        magnitude = "large"
        pct = 50 + abs_delta * 50
        interpretation = (
            f"Values from {'group 1' if delta > 0 else 'group 2'} are larger "
            f"about {pct:.0f}% of the time. This is a large effect with "
            f"substantial separation between groups."
        )
    
    return {
        "d": float(delta),  # Use 'd' key for consistency
        "delta": float(delta),
        "magnitude": magnitude,
        "interpretation": interpretation,
        "type": "Cliff's delta",
        "n_greater": greater,
        "n_less": less,
        "n_comparisons": n1 * n2,
    }

