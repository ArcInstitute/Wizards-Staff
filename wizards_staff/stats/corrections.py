"""
Multiple comparison correction methods for calcium imaging analyses.

When comparing many neurons or many experimental conditions, p-values must be
corrected to control for false discoveries. This module provides various
correction methods with guidance on when to use each.

Functions
---------
apply_correction : Apply multiple comparison correction to p-values
recommend_correction_method : Get recommendation for appropriate correction method
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np


def apply_correction(
    p_values: Union[float, List[float], np.ndarray],
    method: str = "bonferroni",
    alpha: float = 0.05
) -> Dict:
    """
    Apply multiple comparison correction to p-values.
    
    When you perform multiple statistical tests, the probability of finding
    at least one false positive increases. Correction methods adjust p-values
    to account for this.
    
    Parameters
    ----------
    p_values : float, list, or np.ndarray
        Single p-value or array of p-values from multiple tests.
    method : str
        Correction method to use:
        
        - "bonferroni": Conservative, controls family-wise error rate (FWER).
          Best for: Confirmatory analyses where false positives are costly.
          
        - "holm": Step-down Bonferroni, more powerful than Bonferroni while
          still controlling FWER.
          Best for: When you want FWER control but more power than Bonferroni.
          
        - "fdr_bh": Benjamini-Hochberg False Discovery Rate. Controls the
          expected proportion of false discoveries.
          Best for: Exploratory analyses, large numbers of tests.
          
        - "fdr_by": Benjamini-Yekutieli FDR. More conservative FDR control
          that works under dependency.
          Best for: When tests are not independent.
          
        - "none": No correction (not recommended for multiple tests).
        
    alpha : float
        Significance level. Default is 0.05.
    
    Returns
    -------
    dict
        Dictionary containing:
        - adjusted_p: Corrected p-values (same shape as input)
        - significant: Boolean array indicating significance after correction
        - method: Method used
        - n_tests: Number of tests corrected for
        - n_significant: Number of significant results after correction
        - interpretation: Plain-language explanation
    
    Examples
    --------
    >>> # After running 10 pairwise comparisons
    >>> p_values = [0.001, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
    >>> corrected = apply_correction(p_values, method="fdr_bh")
    >>> print(corrected["interpretation"])
    
    >>> # Check which comparisons remain significant
    >>> print(corrected["significant"])
    
    Notes
    -----
    Choosing a correction method:
    
    1. **Bonferroni/Holm**: Use when false positives are very costly
       (e.g., clinical decisions, confirmatory studies)
    
    2. **FDR (Benjamini-Hochberg)**: Use for exploratory analyses where
       you're screening many comparisons and will validate hits later
    
    3. **No correction**: Only appropriate for a single, pre-planned comparison
    """
    # Convert to numpy array
    single_value = not hasattr(p_values, '__len__')
    p_array = np.atleast_1d(np.asarray(p_values, dtype=float))
    n_tests = len(p_array)
    
    if n_tests == 0:
        return {
            "adjusted_p": np.array([]),
            "significant": np.array([], dtype=bool),
            "method": method,
            "n_tests": 0,
            "n_significant": 0,
            "interpretation": "No p-values provided."
        }
    
    # Apply correction
    if method == "none":
        adjusted_p = p_array.copy()
        method_description = "no correction"
        
    elif method == "bonferroni":
        adjusted_p = np.minimum(p_array * n_tests, 1.0)
        method_description = "Bonferroni correction"
        
    elif method == "holm":
        # Holm step-down method
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]
        
        adjusted_p = np.zeros_like(p_array)
        cummax = 0
        for i, (orig_idx, p) in enumerate(zip(sorted_indices, sorted_p)):
            adjusted = p * (n_tests - i)
            cummax = max(cummax, adjusted)
            adjusted_p[orig_idx] = min(cummax, 1.0)
            
        method_description = "Holm step-down correction"
        
    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]
        
        adjusted_p = np.zeros_like(p_array)
        cummin = 1.0
        for i in range(n_tests - 1, -1, -1):
            adjusted = sorted_p[i] * n_tests / (i + 1)
            cummin = min(cummin, adjusted)
            adjusted_p[sorted_indices[i]] = min(cummin, 1.0)
            
        method_description = "Benjamini-Hochberg FDR correction"
        
    elif method == "fdr_by":
        # Benjamini-Yekutieli FDR
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]
        
        # Harmonic sum correction factor
        c_m = np.sum(1.0 / np.arange(1, n_tests + 1))
        
        adjusted_p = np.zeros_like(p_array)
        cummin = 1.0
        for i in range(n_tests - 1, -1, -1):
            adjusted = sorted_p[i] * n_tests * c_m / (i + 1)
            cummin = min(cummin, adjusted)
            adjusted_p[sorted_indices[i]] = min(cummin, 1.0)
            
        method_description = "Benjamini-Yekutieli FDR correction"
        
    else:
        valid_methods = ["bonferroni", "holm", "fdr_bh", "fdr_by", "none"]
        raise ValueError(
            f"Unknown correction method: '{method}'. "
            f"Valid methods: {valid_methods}"
        )
    
    # Determine significance
    significant = adjusted_p < alpha
    n_significant = int(np.sum(significant))
    n_original_sig = int(np.sum(p_array < alpha))
    
    # Build interpretation
    if n_tests == 1:
        if significant[0]:
            interpretation = (
                f"Result is significant (adjusted p = {adjusted_p[0]:.4f})."
            )
        else:
            interpretation = (
                f"Result is not significant (adjusted p = {adjusted_p[0]:.4f})."
            )
    else:
        if method == "none":
            interpretation = (
                f"No correction applied. {n_original_sig} of {n_tests} tests "
                f"are significant at α = {alpha}. Warning: Multiple testing "
                f"increases false positive rate."
            )
        elif method in ["bonferroni", "holm"]:
            interpretation = (
                f"After {method_description} for {n_tests} tests, "
                f"{n_significant} of {n_tests} comparisons remain significant "
                f"(α = {alpha}). This controls the family-wise error rate "
                f"(probability of ANY false positive)."
            )
        else:  # FDR methods
            interpretation = (
                f"After {method_description} for {n_tests} tests, "
                f"{n_significant} of {n_tests} comparisons are significant "
                f"(q < {alpha}). This controls the expected proportion "
                f"of false discoveries among significant results."
            )
        
        if n_original_sig > n_significant:
            interpretation += (
                f" {n_original_sig - n_significant} results that were "
                f"significant before correction are no longer significant."
            )
    
    # Return appropriate format
    result = {
        "adjusted_p": adjusted_p[0] if single_value else adjusted_p,
        "significant": significant[0] if single_value else significant,
        "method": method,
        "n_tests": n_tests,
        "n_significant": n_significant,
        "n_original_significant": n_original_sig,
        "interpretation": interpretation,
    }
    
    return result


def recommend_correction_method(
    n_comparisons: int,
    analysis_type: str = "exploratory",
    dependent_tests: bool = False
) -> Dict:
    """
    Recommend appropriate correction method based on analysis context.
    
    Parameters
    ----------
    n_comparisons : int
        Number of statistical comparisons being made.
    analysis_type : str
        Type of analysis:
        - "exploratory": Screening for interesting effects, will validate later
        - "confirmatory": Testing specific hypotheses, need strong evidence
    dependent_tests : bool
        Whether the tests are statistically dependent on each other.
        Set to True if comparing the same neurons across conditions,
        or if tests share data in some way.
    
    Returns
    -------
    dict
        Dictionary containing:
        - recommended_method: Suggested correction method
        - explanation: Why this method is recommended
        - alternatives: Other reasonable choices
        - considerations: Things to keep in mind
    
    Examples
    --------
    >>> recommendation = recommend_correction_method(
    ...     n_comparisons=15,
    ...     analysis_type="exploratory"
    ... )
    >>> print(recommendation["recommended_method"])
    'fdr_bh'
    >>> print(recommendation["explanation"])
    """
    if n_comparisons < 1:
        return {
            "recommended_method": "none",
            "explanation": "No multiple comparisons to correct for.",
            "alternatives": [],
            "considerations": [],
        }
    
    if n_comparisons == 1:
        return {
            "recommended_method": "none",
            "explanation": (
                "With a single comparison, no multiple testing correction is needed."
            ),
            "alternatives": [],
            "considerations": [
                "Ensure this was truly a pre-planned, single hypothesis test.",
                "Post-hoc selection of 'the most interesting' comparison requires correction."
            ],
        }
    
    considerations = []
    
    if analysis_type == "confirmatory":
        if dependent_tests:
            recommended = "holm"
            explanation = (
                f"For {n_comparisons} confirmatory comparisons with dependent tests, "
                f"Holm's step-down method is recommended. It controls the family-wise "
                f"error rate (probability of any false positive) while being more "
                f"powerful than Bonferroni."
            )
            alternatives = ["bonferroni", "fdr_by"]
        else:
            recommended = "holm"
            explanation = (
                f"For {n_comparisons} confirmatory comparisons, Holm's step-down "
                f"method provides strong family-wise error rate control with "
                f"more power than Bonferroni."
            )
            alternatives = ["bonferroni"]
        
        considerations = [
            "Family-wise error rate control is conservative - you may miss real effects.",
            "Consider increasing sample size to maintain power.",
            f"With {n_comparisons} tests and α=0.05, Bonferroni requires p < {0.05/n_comparisons:.4f}."
        ]
        
    else:  # exploratory
        if dependent_tests:
            recommended = "fdr_by"
            explanation = (
                f"For {n_comparisons} exploratory comparisons with dependent tests, "
                f"Benjamini-Yekutieli FDR is recommended. It controls false discovery "
                f"rate while accounting for dependencies between tests."
            )
            alternatives = ["fdr_bh", "holm"]
        else:
            recommended = "fdr_bh"
            explanation = (
                f"For {n_comparisons} exploratory comparisons, Benjamini-Hochberg "
                f"FDR is recommended. It controls the expected proportion of false "
                f"discoveries among significant results, providing good power for "
                f"discovering interesting effects."
            )
            alternatives = ["fdr_by", "holm"]
        
        considerations = [
            "FDR control means some 'significant' results may be false positives.",
            "Plan to validate key findings with independent experiments.",
            "Consider reporting both raw and adjusted p-values.",
        ]
    
    # Add consideration about number of tests
    if n_comparisons > 20:
        considerations.append(
            f"With {n_comparisons} tests, be aware that even with correction, "
            f"interpretation requires careful consideration of effect sizes."
        )
    
    return {
        "recommended_method": recommended,
        "explanation": explanation,
        "alternatives": alternatives,
        "considerations": considerations,
    }

