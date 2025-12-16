"""
Power analysis utilities for calcium imaging experiments.

Use these functions BEFORE your experiment to determine needed sample sizes,
or AFTER to understand what effect sizes your study could detect.

Why Power Analysis Matters
--------------------------
- **Before experiment**: Know how many samples you need to detect meaningful effects
- **After experiment**: Understand what effects you had the power to detect
- **Null results**: Distinguish "no effect" from "insufficient power"

Key Concepts
------------
- **Power**: Probability of detecting an effect if it exists (typically aim for 80%)
- **Effect size**: How large the difference between groups is (Cohen's d)
- **Alpha**: Significance threshold (typically 0.05)
- **Sample size**: Number of biological replicates per group

Functions
---------
calculate_required_n : Sample size needed to detect a given effect
calculate_achieved_power : Power achieved with current sample size
calculate_detectable_effect : Minimum effect size detectable with current samples
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats


def calculate_required_n(
    effect_size: float,
    power: float = 0.80,
    alpha: float = 0.05,
    test_type: str = "two_sample_t",
    n_groups: int = 2,
    alternative: str = "two-sided"
) -> Dict:
    """
    Calculate required sample size to detect a given effect.
    
    This is the MOST IMPORTANT function to use BEFORE starting your experiment.
    It tells you how many biological replicates (samples, wells, animals) you
    need per group to have a good chance of detecting the effect you expect.
    
    Parameters
    ----------
    effect_size : float
        Expected effect size:
        - For t-tests: Cohen's d (standardized mean difference)
        - For ANOVA: Cohen's f (sqrt(η² / (1 - η²)))
        - For correlation: Pearson r
        
        Typical values for calcium imaging:
        - Small: d = 0.2 (subtle changes in activity)
        - Medium: d = 0.5 (noticeable changes)
        - Large: d = 0.8 (dramatic changes, e.g., drug effects)
        
    power : float
        Desired statistical power (probability of detecting a true effect).
        Default is 0.80 (80%), which is conventional.
        Use 0.90 (90%) for more important confirmatory studies.
        
    alpha : float
        Significance level (Type I error rate). Default is 0.05.
        
    test_type : str
        Type of statistical test planned:
        - "two_sample_t": Independent samples t-test (most common)
        - "paired_t": Paired samples t-test
        - "one_sample_t": One-sample t-test
        - "anova": One-way ANOVA
        - "correlation": Correlation test
        
    n_groups : int
        Number of groups (for ANOVA). Default is 2.
        
    alternative : str
        "two-sided" (default) or "one-sided".
        Use one-sided only if you have strong prior expectation of direction.
    
    Returns
    -------
    dict
        Dictionary containing:
        - n_per_group: Required samples per group (rounded up)
        - total_n: Total samples needed
        - interpretation: Explanation with practical guidance
        - recommendations: List of practical suggestions
        - power: Requested power
        - effect_size: Requested effect size
        - alpha: Significance level used
    
    Examples
    --------
    >>> # How many samples to detect a medium effect?
    >>> result = calculate_required_n(effect_size=0.5)
    >>> print(result["interpretation"])
    
    >>> # For a smaller effect
    >>> result = calculate_required_n(effect_size=0.3, power=0.90)
    >>> print(f"Need {result['n_per_group']} samples per group")
    """
    # Validate inputs
    if not 0 < effect_size:
        raise ValueError("effect_size must be positive")
    if not 0 < power < 1:
        raise ValueError("power must be between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    
    # Calculate critical value
    if alternative == "two-sided":
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_power = stats.norm.ppf(power)
    
    # Calculate sample size based on test type
    if test_type == "two_sample_t":
        # Two-sample t-test: n = 2 * ((z_α + z_β) / d)²
        n_per_group = 2 * ((z_alpha + z_power) / effect_size) ** 2
        
    elif test_type == "paired_t":
        # Paired t-test: n = ((z_α + z_β) / d)²
        n_per_group = ((z_alpha + z_power) / effect_size) ** 2
        
    elif test_type == "one_sample_t":
        # One-sample t-test
        n_per_group = ((z_alpha + z_power) / effect_size) ** 2
        
    elif test_type == "anova":
        # One-way ANOVA approximation
        # Using approximation: n ≈ (n_groups * (z_α + z_β)²) / (f² * (n_groups - 1))
        # where f = effect_size (Cohen's f)
        f = effect_size
        n_per_group = (n_groups * (z_alpha + z_power) ** 2) / (f ** 2 * (n_groups - 1))
        
    elif test_type == "correlation":
        # Correlation: use Fisher's z transformation
        # n = ((z_α + z_β) / arctanh(r))² + 3
        r = effect_size
        n_per_group = ((z_alpha + z_power) / np.arctanh(r)) ** 2 + 3
        
    else:
        raise ValueError(f"Unknown test_type: {test_type}")
    
    # Round up
    n_per_group = int(np.ceil(n_per_group))
    total_n = n_per_group * n_groups if test_type != "correlation" else n_per_group
    
    # Describe effect size
    if effect_size < 0.2:
        effect_desc = "very small"
    elif effect_size < 0.5:
        effect_desc = "small"
    elif effect_size < 0.8:
        effect_desc = "medium"
    else:
        effect_desc = "large"
    
    # Build interpretation
    interpretation = (
        f"To detect a {effect_desc} effect (d = {effect_size}) with {power*100:.0f}% power "
        f"at α = {alpha}, you need at least {n_per_group} biological replicates per group "
        f"({total_n} total). "
        f"\n\nThis means {n_per_group} independent samples (e.g., different wells, "
        f"organoids, animals) per experimental condition. Individual neurons "
        f"within a sample do NOT count as separate replicates."
    )
    
    # Build recommendations
    recommendations = []
    
    if n_per_group > 30:
        recommendations.append(
            f"Sample size ({n_per_group}) is large. Consider whether you're trying "
            f"to detect effects that are too small to be biologically meaningful."
        )
    
    if effect_size < 0.3:
        recommendations.append(
            "Detecting small effects requires many samples. Consider focusing "
            "on larger, more biologically meaningful effects."
        )
    
    if power < 0.80:
        recommendations.append(
            f"Power of {power*100:.0f}% is below conventional (80%). "
            "Consider increasing power to reduce false negative risk."
        )
    
    recommendations.append(
        "Always aim for a few extra samples to account for technical failures "
        "or quality control exclusions."
    )
    
    return {
        "n_per_group": n_per_group,
        "total_n": total_n,
        "interpretation": interpretation,
        "recommendations": recommendations,
        "power": power,
        "effect_size": effect_size,
        "alpha": alpha,
        "test_type": test_type,
    }


def calculate_achieved_power(
    n_per_group: int,
    effect_size: float,
    alpha: float = 0.05,
    test_type: str = "two_sample_t",
    n_groups: int = 2,
    alternative: str = "two-sided"
) -> Dict:
    """
    Calculate the power achieved with your current sample size.
    
    Use this AFTER collecting data to understand what effects you could detect,
    especially useful for interpreting null (non-significant) results.
    
    Parameters
    ----------
    n_per_group : int
        Number of samples per group.
    effect_size : float
        Effect size to detect (Cohen's d for t-tests).
    alpha : float
        Significance level. Default is 0.05.
    test_type : str
        Type of statistical test. Default is "two_sample_t".
    n_groups : int
        Number of groups (for ANOVA). Default is 2.
    alternative : str
        "two-sided" (default) or "one-sided".
    
    Returns
    -------
    dict
        Dictionary containing:
        - power: Achieved statistical power (0 to 1)
        - interpretation: Explanation of what this means
        - is_adequate: Boolean indicating if power ≥ 0.80
    
    Examples
    --------
    >>> result = calculate_achieved_power(n_per_group=10, effect_size=0.5)
    >>> print(f"Power: {result['power']:.2%}")
    >>> print(result["interpretation"])
    """
    # Calculate critical value
    if alternative == "two-sided":
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    # Calculate power based on test type
    if test_type == "two_sample_t":
        # z_β = d * sqrt(n/2) - z_α
        z_beta = effect_size * np.sqrt(n_per_group / 2) - z_alpha
        
    elif test_type == "paired_t":
        z_beta = effect_size * np.sqrt(n_per_group) - z_alpha
        
    elif test_type == "one_sample_t":
        z_beta = effect_size * np.sqrt(n_per_group) - z_alpha
        
    elif test_type == "anova":
        f = effect_size
        z_beta = f * np.sqrt(n_per_group * (n_groups - 1) / n_groups) - z_alpha
        
    elif test_type == "correlation":
        r = effect_size
        z_beta = np.arctanh(r) * np.sqrt(n_per_group - 3) - z_alpha
        
    else:
        raise ValueError(f"Unknown test_type: {test_type}")
    
    # Convert z to power
    power = float(stats.norm.cdf(z_beta))
    power = max(0, min(1, power))  # Clamp to [0, 1]
    
    # Build interpretation
    is_adequate = power >= 0.80
    
    if power >= 0.80:
        interpretation = (
            f"With {n_per_group} samples per group, you have {power*100:.0f}% power "
            f"to detect an effect of d = {effect_size}. This is adequate power "
            f"for most research purposes."
        )
    elif power >= 0.50:
        interpretation = (
            f"With {n_per_group} samples per group, you have only {power*100:.0f}% power "
            f"to detect an effect of d = {effect_size}. This means you have about a "
            f"{(1-power)*100:.0f}% chance of missing a real effect of this size. "
            f"Consider increasing sample size."
        )
    else:
        interpretation = (
            f"With {n_per_group} samples per group, you have very low power "
            f"({power*100:.0f}%) to detect an effect of d = {effect_size}. "
            f"A null result would be difficult to interpret - you likely "
            f"wouldn't detect even a real effect. More samples are needed."
        )
    
    return {
        "power": power,
        "interpretation": interpretation,
        "is_adequate": is_adequate,
        "n_per_group": n_per_group,
        "effect_size": effect_size,
        "alpha": alpha,
    }


def calculate_detectable_effect(
    n_per_group: int,
    power: float = 0.80,
    alpha: float = 0.05,
    test_type: str = "two_sample_t",
    n_groups: int = 2,
    alternative: str = "two-sided"
) -> Dict:
    """
    Calculate the minimum detectable effect size with current samples.
    
    This helps you understand what size of biological effect your study
    is capable of detecting. Useful for interpreting results and
    comparing to effects reported in the literature.
    
    Parameters
    ----------
    n_per_group : int
        Number of samples per group.
    power : float
        Desired power level. Default is 0.80.
    alpha : float
        Significance level. Default is 0.05.
    test_type : str
        Type of statistical test. Default is "two_sample_t".
    n_groups : int
        Number of groups (for ANOVA). Default is 2.
    alternative : str
        "two-sided" (default) or "one-sided".
    
    Returns
    -------
    dict
        Dictionary containing:
        - min_effect_size: Minimum detectable effect (Cohen's d)
        - magnitude: Description of effect size category
        - interpretation: Explanation of what this means
    
    Examples
    --------
    >>> result = calculate_detectable_effect(n_per_group=10)
    >>> print(result["interpretation"])
    """
    # Calculate critical values
    if alternative == "two-sided":
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_power = stats.norm.ppf(power)
    
    # Solve for effect size based on test type
    if test_type == "two_sample_t":
        # d = (z_α + z_β) / sqrt(n/2)
        min_d = (z_alpha + z_power) / np.sqrt(n_per_group / 2)
        
    elif test_type == "paired_t":
        min_d = (z_alpha + z_power) / np.sqrt(n_per_group)
        
    elif test_type == "one_sample_t":
        min_d = (z_alpha + z_power) / np.sqrt(n_per_group)
        
    elif test_type == "anova":
        min_d = (z_alpha + z_power) / np.sqrt(n_per_group * (n_groups - 1) / n_groups)
        
    elif test_type == "correlation":
        z = (z_alpha + z_power) / np.sqrt(n_per_group - 3)
        min_d = np.tanh(z)  # r value
        
    else:
        raise ValueError(f"Unknown test_type: {test_type}")
    
    min_d = float(min_d)
    
    # Categorize effect size
    if min_d < 0.2:
        magnitude = "very small"
    elif min_d < 0.5:
        magnitude = "small"
    elif min_d < 0.8:
        magnitude = "medium"
    elif min_d < 1.2:
        magnitude = "large"
    else:
        magnitude = "very large"
    
    # Build interpretation
    if min_d <= 0.5:
        interpretation = (
            f"With {n_per_group} samples per group, you can detect effects of "
            f"d ≥ {min_d:.2f} ({magnitude}) with {power*100:.0f}% power. "
            f"This is good sensitivity for detecting medium to large effects."
        )
    elif min_d <= 0.8:
        interpretation = (
            f"With {n_per_group} samples per group, you can detect effects of "
            f"d ≥ {min_d:.2f} ({magnitude}) with {power*100:.0f}% power. "
            f"You may miss smaller effects. Consider whether this is acceptable "
            f"for your research question."
        )
    else:
        interpretation = (
            f"With {n_per_group} samples per group, you can only detect very large "
            f"effects (d ≥ {min_d:.2f}) with {power*100:.0f}% power. "
            f"This is low sensitivity - you would miss most biologically "
            f"meaningful effects. Consider increasing sample size."
        )
    
    return {
        "min_effect_size": min_d,
        "magnitude": magnitude,
        "interpretation": interpretation,
        "n_per_group": n_per_group,
        "power": power,
        "alpha": alpha,
        "test_type": test_type,
    }

