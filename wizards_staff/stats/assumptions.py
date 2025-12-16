"""
Statistical assumption checking for calcium imaging analyses.

ALWAYS check assumptions before interpreting results! This module provides
functions to verify that your data meets the requirements for various
statistical tests.

Key Assumptions
---------------
- **Normality**: Data should follow a normal distribution (for parametric tests)
- **Homogeneity of variance**: Groups should have similar variances
- **Independence**: Observations should be independent (avoid pseudoreplication!)

Functions
---------
check_normality : Test if data follows a normal distribution
check_homogeneity_of_variance : Test if groups have equal variances
check_all_assumptions : Comprehensive assumption checking
generate_assumption_report : Generate formatted report
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


def check_normality(
    data: Union[np.ndarray, pd.Series],
    alpha: float = 0.05,
    method: str = "auto"
) -> Dict:
    """
    Test if data follows a normal distribution.
    
    Normality is required for parametric tests (t-test, ANOVA). If data
    is not normal, consider using non-parametric alternatives.
    
    Parameters
    ----------
    data : np.ndarray or pd.Series
        Data to test for normality.
    alpha : float
        Significance level for the test. Default is 0.05.
    method : str
        Which normality test to use:
        
        - "auto": Automatically choose based on sample size (recommended)
        - "shapiro": Shapiro-Wilk test (best for n < 50)
        - "dagostino": D'Agostino-Pearson test (best for n ≥ 50)
        - "lilliefors": Lilliefors test (modification of K-S test)
    
    Returns
    -------
    dict
        Dictionary containing:
        - is_normal: Boolean indicating if normality assumption is met
        - test_used: Name of the test performed
        - statistic: Test statistic value
        - p_value: P-value from the test
        - interpretation: What this means for your analysis
        - recommendation: Which tests to use based on this result
    
    Notes
    -----
    - Small samples (n < 20) often pass normality tests even if non-normal
    - Large samples (n > 100) often fail even with minor deviations
    - Visual inspection (histogram, Q-Q plot) is also recommended
    
    Examples
    --------
    >>> result = check_normality(firing_rates)
    >>> if result["is_normal"]:
    ...     print("Data is approximately normal, can use t-test")
    ... else:
    ...     print("Data is non-normal, use Mann-Whitney U test")
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    n = len(data)
    
    if n < 3:
        return {
            "is_normal": None,
            "test_used": "none",
            "statistic": np.nan,
            "p_value": np.nan,
            "interpretation": "Insufficient data for normality testing (n < 3).",
            "recommendation": "Use non-parametric tests with small samples.",
        }
    
    # Choose test based on sample size
    if method == "auto":
        if n < 50:
            method = "shapiro"
        else:
            method = "dagostino"
    
    # Perform the test
    if method == "shapiro":
        if n > 5000:
            # Shapiro-Wilk has sample size limit
            data_sample = np.random.choice(data, 5000, replace=False)
            stat, p_value = stats.shapiro(data_sample)
            test_name = "Shapiro-Wilk (sampled)"
        else:
            stat, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk"
            
    elif method == "dagostino":
        if n < 8:
            # D'Agostino requires at least 8 samples
            stat, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk (D'Agostino requires n≥8)"
        else:
            stat, p_value = stats.normaltest(data)
            test_name = "D'Agostino-Pearson"
            
    elif method == "lilliefors":
        try:
            from statsmodels.stats.diagnostic import lilliefors
            stat, p_value = lilliefors(data)
            test_name = "Lilliefors"
        except ImportError:
            # Fall back to Shapiro-Wilk
            stat, p_value = stats.shapiro(data if n <= 5000 else np.random.choice(data, 5000, replace=False))
            test_name = "Shapiro-Wilk (Lilliefors unavailable)"
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    is_normal = p_value >= alpha
    
    # Build interpretation
    if is_normal:
        interpretation = (
            f"The data appears to follow a normal distribution "
            f"({test_name} test, p = {p_value:.4f} ≥ {alpha}). "
            f"Parametric tests (t-test, ANOVA) are appropriate."
        )
        recommendation = "You can use parametric tests (t-test, ANOVA)."
    else:
        interpretation = (
            f"The data significantly deviates from normality "
            f"({test_name} test, p = {p_value:.4f} < {alpha}). "
            f"Consider non-parametric alternatives or data transformation."
        )
        recommendation = (
            "Use non-parametric tests (Mann-Whitney U, Kruskal-Wallis) "
            "or consider log-transforming the data."
        )
    
    # Add note about sample size effects
    if n > 100:
        interpretation += (
            " Note: With large samples, even minor deviations from normality "
            "can be statistically significant. Visual inspection is recommended."
        )
    elif n < 20:
        interpretation += (
            " Note: With small samples, normality tests have low power. "
            "Consider using non-parametric tests to be safe."
        )
    
    return {
        "is_normal": is_normal,
        "test_used": test_name,
        "statistic": float(stat),
        "p_value": float(p_value),
        "n": n,
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


def check_homogeneity_of_variance(
    data: pd.DataFrame,
    group_col: str,
    metric_col: str,
    method: str = "levene"
) -> Dict:
    """
    Test if groups have equal variances (homoscedasticity).
    
    Equal variances is an assumption of independent samples t-test and ANOVA.
    If variances are unequal, use Welch's t-test or robust alternatives.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    group_col : str
        Column name containing group labels.
    metric_col : str
        Column name containing the metric values.
    method : str
        Which test to use:
        
        - "levene": Levene's test (default, robust to non-normality)
        - "bartlett": Bartlett's test (more powerful but assumes normality)
        - "brown_forsythe": Brown-Forsythe test (uses median instead of mean)
    
    Returns
    -------
    dict
        Dictionary containing:
        - equal_variance: Boolean indicating if homogeneity is met
        - test_used: Name of the test
        - statistic: Test statistic
        - p_value: P-value
        - group_variances: Variance for each group
        - interpretation: What this means
        - recommendation: How to proceed
    
    Examples
    --------
    >>> result = check_homogeneity_of_variance(
    ...     data=df,
    ...     group_col="Treatment",
    ...     metric_col="firing_rate"
    ... )
    >>> print(result["interpretation"])
    """
    groups = data[group_col].unique()
    group_data = [data[data[group_col] == g][metric_col].dropna().values for g in groups]
    
    # Filter out empty groups
    group_data = [g for g in group_data if len(g) > 0]
    
    if len(group_data) < 2:
        return {
            "equal_variance": None,
            "test_used": "none",
            "statistic": np.nan,
            "p_value": np.nan,
            "interpretation": "Need at least 2 groups with data.",
            "recommendation": "Cannot assess variance homogeneity.",
        }
    
    # Calculate variances
    group_variances = {str(g): float(np.var(d, ddof=1)) for g, d in zip(groups, group_data) if len(d) > 0}
    
    # Perform test
    if method == "levene":
        stat, p_value = stats.levene(*group_data, center='mean')
        test_name = "Levene's test"
    elif method == "bartlett":
        stat, p_value = stats.bartlett(*group_data)
        test_name = "Bartlett's test"
    elif method == "brown_forsythe":
        stat, p_value = stats.levene(*group_data, center='median')
        test_name = "Brown-Forsythe test"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    equal_variance = p_value >= 0.05
    
    # Calculate variance ratio
    var_values = list(group_variances.values())
    if len(var_values) >= 2 and min(var_values) > 0:
        variance_ratio = max(var_values) / min(var_values)
    else:
        variance_ratio = np.nan
    
    # Build interpretation
    if equal_variance:
        interpretation = (
            f"Variances appear homogeneous across groups ({test_name}, p = {p_value:.4f}). "
            f"Variance ratio: {variance_ratio:.2f}. Standard t-test/ANOVA assumptions are met."
        )
        recommendation = "Standard parametric tests are appropriate."
    else:
        interpretation = (
            f"Variances differ significantly between groups ({test_name}, p = {p_value:.4f}). "
            f"Variance ratio: {variance_ratio:.2f}."
        )
        if variance_ratio > 4:
            recommendation = (
                "Large variance difference. Use Welch's t-test (for 2 groups) or "
                "Welch's ANOVA. Consider log-transformation if variances scale with means."
            )
        else:
            recommendation = (
                "Use Welch's t-test instead of Student's t-test, or Welch's ANOVA."
            )
    
    return {
        "equal_variance": equal_variance,
        "test_used": test_name,
        "statistic": float(stat),
        "p_value": float(p_value),
        "group_variances": group_variances,
        "variance_ratio": variance_ratio,
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


def check_all_assumptions(
    data: pd.DataFrame,
    group_col: str,
    metric_col: str,
    test_type: str = "two_group"
) -> Dict:
    """
    Comprehensive assumption checking for common statistical tests.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    group_col : str
        Column containing group labels.
    metric_col : str
        Column containing metric values.
    test_type : str
        Type of test planned:
        - "two_group": Two-sample comparison
        - "multiple_group": ANOVA-type comparison
        - "correlation": Correlation analysis
    
    Returns
    -------
    dict
        Comprehensive results including:
        - normality_tests: Results for each group
        - variance_test: Homogeneity of variance result
        - sample_sizes: Size of each group
        - recommendation: Parametric vs non-parametric
        - report: Formatted text report
    
    Examples
    --------
    >>> assumptions = check_all_assumptions(
    ...     data=df,
    ...     group_col="Treatment",
    ...     metric_col="mean_frpm",
    ...     test_type="two_group"
    ... )
    >>> print(assumptions["report"])
    >>> print(f"Recommendation: {assumptions['recommendation']}")
    """
    results = {
        "normality_tests": {},
        "variance_test": None,
        "sample_sizes": {},
        "overall_normality": True,
        "overall_variance": True,
        "recommendation": "",
        "report": "",
    }
    
    groups = data[group_col].unique()
    
    # Check normality for each group
    for group in groups:
        group_data = data[data[group_col] == group][metric_col].dropna().values
        results["sample_sizes"][str(group)] = len(group_data)
        
        if len(group_data) >= 3:
            norm_result = check_normality(group_data)
            results["normality_tests"][str(group)] = norm_result
            if not norm_result["is_normal"]:
                results["overall_normality"] = False
        else:
            results["normality_tests"][str(group)] = {
                "is_normal": None,
                "interpretation": f"Insufficient data (n={len(group_data)})"
            }
            results["overall_normality"] = False  # Be conservative
    
    # Check variance homogeneity
    if len(groups) >= 2:
        results["variance_test"] = check_homogeneity_of_variance(
            data, group_col, metric_col
        )
        if results["variance_test"]["equal_variance"] is False:
            results["overall_variance"] = False
    
    # Make recommendation
    if results["overall_normality"] and results["overall_variance"]:
        results["recommendation"] = "parametric"
        rec_text = (
            "Data meets assumptions for parametric tests. "
            "Use t-test (2 groups) or ANOVA (3+ groups)."
        )
    elif results["overall_normality"] and not results["overall_variance"]:
        results["recommendation"] = "welch"
        rec_text = (
            "Data is normal but variances are unequal. "
            "Use Welch's t-test or Welch's ANOVA."
        )
    else:
        results["recommendation"] = "non-parametric"
        rec_text = (
            "Data does not meet normality assumptions. "
            "Use Mann-Whitney U (2 groups) or Kruskal-Wallis (3+ groups)."
        )
    
    # Generate report
    report_lines = [
        "=" * 60,
        "STATISTICAL ASSUMPTION CHECK",
        "=" * 60,
        "",
        "Sample Sizes:",
    ]
    for group, n in results["sample_sizes"].items():
        status = "✓" if n >= 10 else "⚠️" if n >= 3 else "❌"
        report_lines.append(f"  {status} {group}: n = {n}")
    
    report_lines.extend(["", "Normality Tests:"])
    for group, norm in results["normality_tests"].items():
        is_normal = norm.get("is_normal")
        if is_normal is None:
            status = "⚠️ Unknown"
        elif is_normal:
            status = "✓ Normal"
        else:
            status = "❌ Non-normal"
        
        p = norm.get("p_value", np.nan)
        p_str = f"p = {p:.4f}" if not np.isnan(p) else "insufficient data"
        report_lines.append(f"  {status} {group}: {p_str}")
    
    if results["variance_test"]:
        var = results["variance_test"]
        report_lines.extend(["", "Variance Homogeneity:"])
        equal_var = var["equal_variance"]
        if equal_var is None:
            status = "⚠️ Unknown"
        elif equal_var:
            status = "✓ Equal variances"
        else:
            status = "❌ Unequal variances"
        report_lines.append(f"  {status} ({var['test_used']}, p = {var['p_value']:.4f})")
    
    report_lines.extend([
        "",
        "Recommendation:",
        f"  {rec_text}",
        "",
        "=" * 60,
    ])
    
    results["report"] = "\n".join(report_lines)
    
    return results


def generate_assumption_report(
    assumption_results: Dict,
    format: str = "markdown"
) -> str:
    """
    Generate formatted report of assumption check results.
    
    Parameters
    ----------
    assumption_results : dict
        Results from check_all_assumptions() or individual tests.
    format : str
        Output format: "markdown", "html", or "text".
    
    Returns
    -------
    str
        Formatted report string.
    """
    # If the result already has a report, return it
    if "report" in assumption_results:
        report = assumption_results["report"]
        
        if format == "text":
            return report
        
        elif format == "markdown":
            lines = report.split("\n")
            md_lines = []
            for line in lines:
                if line.startswith("="):
                    continue  # Skip separator lines
                elif line.isupper() and line.strip():
                    md_lines.append(f"## {line.strip()}")
                elif line.strip().startswith("✓") or line.strip().startswith("❌") or line.strip().startswith("⚠️"):
                    md_lines.append(f"- {line.strip()}")
                elif line.strip().endswith(":"):
                    md_lines.append(f"### {line.strip()[:-1]}")
                else:
                    md_lines.append(line)
            return "\n".join(md_lines)
        
        elif format == "html":
            html = "<div class='assumption-report'>\n"
            lines = report.split("\n")
            for line in lines:
                if line.startswith("="):
                    html += "<hr>\n"
                elif line.isupper() and line.strip():
                    html += f"<h2>{line.strip()}</h2>\n"
                elif line.strip().startswith("✓"):
                    html += f"<p class='pass'>{line.strip()}</p>\n"
                elif line.strip().startswith("❌"):
                    html += f"<p class='fail'>{line.strip()}</p>\n"
                elif line.strip().startswith("⚠️"):
                    html += f"<p class='warn'>{line.strip()}</p>\n"
                elif line.strip().endswith(":"):
                    html += f"<h3>{line.strip()}</h3>\n"
                elif line.strip():
                    html += f"<p>{line.strip()}</p>\n"
            html += "</div>"
            return html
    
    # Generate from raw results
    return str(assumption_results)

