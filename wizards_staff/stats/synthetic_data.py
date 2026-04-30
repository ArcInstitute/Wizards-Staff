"""
Test utilities for validating statistical functions.

These functions generate synthetic data with KNOWN statistical properties,
allowing us to verify that our statistical functions return correct results.

Functions
---------
generate_two_group_data : Generate two-group data with known effect size
generate_multi_group_data : Generate multi-group data for ANOVA testing
generate_correlated_data : Generate correlated data with known r
generate_non_normal_data : Generate clearly non-normal data
generate_calcium_imaging_like_data : Generate realistic calcium imaging structure
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def generate_two_group_data(
    n_per_group: int = 20,
    effect_size: float = 0.8,
    mean_control: float = 10.0,
    sd: float = 2.0,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate two-group data with known effect size.
    
    Parameters
    ----------
    n_per_group : int
        Number of samples per group. Default is 20.
    effect_size : float
        Cohen's d effect size. Default is 0.8 (large effect).
    mean_control : float
        Mean of control group. Default is 10.0.
    sd : float
        Standard deviation for both groups. Default is 2.0.
    seed : int
        Random seed for reproducibility. Default is 42.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with 'Sample', 'Group', and 'Value' columns.
    expected : dict
        Dictionary with expected statistical properties.
    
    Examples
    --------
    >>> df, expected = generate_two_group_data(n_per_group=30, effect_size=0.5)
    >>> print(f"True effect size: {expected['true_effect_size']}")
    """
    np.random.seed(seed)
    
    # Calculate treatment mean based on desired effect size
    mean_treatment = mean_control + (effect_size * sd)
    
    control = np.random.normal(mean_control, sd, n_per_group)
    treatment = np.random.normal(mean_treatment, sd, n_per_group)
    
    df = pd.DataFrame({
        'Sample': [f'S{i}' for i in range(n_per_group * 2)],
        'Group': ['Control'] * n_per_group + ['Treatment'] * n_per_group,
        'Value': np.concatenate([control, treatment])
    })
    
    expected = {
        'true_effect_size': effect_size,
        'true_mean_diff': mean_treatment - mean_control,
        'mean_control': mean_control,
        'mean_treatment': mean_treatment,
        'sd': sd,
        'n_per_group': n_per_group,
        'should_be_significant': effect_size > 0.5 and n_per_group >= 15
    }
    
    return df, expected


def generate_multi_group_data(
    n_groups: int = 4,
    n_per_group: int = 15,
    group_means: Optional[List[float]] = None,
    sd: float = 2.0,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate multi-group data for ANOVA testing.
    
    Parameters
    ----------
    n_groups : int
        Number of groups. Default is 4.
    n_per_group : int
        Samples per group. Default is 15.
    group_means : list of float, optional
        Mean for each group. If None, uses [10, 10, 10, 10] (no effect).
    sd : float
        Standard deviation for all groups. Default is 2.0.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with 'Sample', 'Group', and 'Value' columns.
    expected : dict
        Dictionary with expected statistical properties.
    
    Examples
    --------
    >>> # Create data with no effect (all groups have same mean)
    >>> df, expected = generate_multi_group_data()
    >>> 
    >>> # Create data with effect
    >>> df, expected = generate_multi_group_data(group_means=[10, 12, 14, 16])
    """
    np.random.seed(seed)
    
    if group_means is None:
        group_means = [10.0] * n_groups
    
    if len(group_means) != n_groups:
        raise ValueError(f"group_means length ({len(group_means)}) must match n_groups ({n_groups})")
    
    data = []
    for i, mean in enumerate(group_means):
        group_data = np.random.normal(mean, sd, n_per_group)
        for j, val in enumerate(group_data):
            data.append({
                'Sample': f'G{i}_S{j}',
                'Group': f'Group_{i+1}',
                'Value': val
            })
    
    df = pd.DataFrame(data)
    
    # Calculate expected eta-squared
    grand_mean = np.mean(group_means)
    ss_between = n_per_group * sum((m - grand_mean)**2 for m in group_means)
    ss_within = n_groups * (n_per_group - 1) * sd**2
    ss_total = ss_between + ss_within
    expected_eta_sq = ss_between / ss_total if ss_total > 0 else 0
    
    expected = {
        'n_groups': n_groups,
        'n_per_group': n_per_group,
        'group_means': group_means,
        'sd': sd,
        'expected_eta_squared': expected_eta_sq,
        'should_be_significant': max(group_means) - min(group_means) > sd
    }
    
    return df, expected


def generate_correlated_data(
    n: int = 50,
    true_r: float = 0.6,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate two correlated variables with known correlation coefficient.
    
    Uses Cholesky decomposition to create variables with exact theoretical
    correlation.
    
    Parameters
    ----------
    n : int
        Number of observations. Default is 50.
    true_r : float
        Target correlation coefficient. Default is 0.6.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with 'Sample', 'X', and 'Y' columns.
    expected : dict
        Dictionary with expected statistical properties.
    
    Examples
    --------
    >>> df, expected = generate_correlated_data(n=100, true_r=0.7)
    >>> print(f"True correlation: {expected['true_correlation']}")
    """
    np.random.seed(seed)
    
    # Generate correlated data using Cholesky decomposition
    x = np.random.normal(0, 1, n)
    y = true_r * x + np.sqrt(1 - true_r**2) * np.random.normal(0, 1, n)
    
    df = pd.DataFrame({
        'Sample': [f'S{i}' for i in range(n)],
        'X': x,
        'Y': y
    })
    
    expected = {
        'true_correlation': true_r,
        'n': n,
        'should_be_significant': abs(true_r) > 0.3 and n >= 20
    }
    
    return df, expected


def generate_non_normal_data(
    n_per_group: int = 30,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate clearly non-normal (skewed) data for assumption testing.
    
    Uses exponential distribution which is clearly non-normal.
    
    Parameters
    ----------
    n_per_group : int
        Number of samples per group. Default is 30.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with 'Sample', 'Group', and 'Value' columns.
    expected : dict
        Dictionary with expected properties.
    
    Examples
    --------
    >>> df, expected = generate_non_normal_data()
    >>> assert expected['is_normal'] == False
    """
    np.random.seed(seed)
    
    # Exponential distribution is clearly non-normal (right-skewed)
    control = np.random.exponential(scale=2.0, size=n_per_group)
    treatment = np.random.exponential(scale=4.0, size=n_per_group)
    
    df = pd.DataFrame({
        'Sample': [f'S{i}' for i in range(n_per_group * 2)],
        'Group': ['Control'] * n_per_group + ['Treatment'] * n_per_group,
        'Value': np.concatenate([control, treatment])
    })
    
    expected = {
        'is_normal': False,
        'recommended_test': 'non-parametric',
        'n_per_group': n_per_group
    }
    
    return df, expected


def generate_calcium_imaging_like_data(
    n_samples: int = 8,
    neurons_per_sample: int = 50,
    n_groups: int = 2,
    group_effect_frpm: float = 0.7,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Generate realistic calcium imaging data structure.
    
    Creates data that mimics real Wizards Staff output with proper
    hierarchical structure (neurons nested within samples).
    
    Parameters
    ----------
    n_samples : int
        Total number of samples (biological replicates). Default is 8.
    neurons_per_sample : int
        Number of neurons per sample (technical replicates). Default is 50.
    n_groups : int
        Number of experimental groups. Default is 2.
    group_effect_frpm : float
        Cohen's d effect size for FRPM between groups. Default is 0.7.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    neuron_df : pd.DataFrame
        Neuron-level data (technical replicates).
    sample_df : pd.DataFrame
        Sample-level aggregated data (biological replicates).
    expected : dict
        Dictionary with expected statistical properties.
    
    Examples
    --------
    >>> neuron_df, sample_df, expected = generate_calcium_imaging_like_data(
    ...     n_samples=12, neurons_per_sample=100, group_effect_frpm=1.0
    ... )
    >>> # Use sample_df for statistical testing (biological replicates)
    >>> print(f"N for stats: {len(sample_df)}")
    """
    np.random.seed(seed)
    
    samples_per_group = n_samples // n_groups
    
    neuron_data = []
    
    for group_idx in range(n_groups):
        group_name = f"Group_{group_idx + 1}"
        # Shift mean FRPM based on group (effect size)
        base_frpm = 8.0 + (group_idx * group_effect_frpm * 3.0)
        
        for sample_idx in range(samples_per_group):
            sample_name = f"{group_name}_Sample_{sample_idx + 1}"
            # Sample-level variation (between-subject)
            sample_mean = base_frpm + np.random.normal(0, 1.5)
            
            for neuron_idx in range(neurons_per_sample):
                # Neuron-level variation (within-subject, technical replicate)
                frpm = max(0, sample_mean + np.random.normal(0, 2.5))
                fwhm = max(0.1, np.random.normal(1.5, 0.3))
                rise_time = max(0.05, np.random.normal(0.3, 0.08))
                
                neuron_data.append({
                    'Sample': sample_name,
                    'Group': group_name,
                    'Neuron Index': neuron_idx,
                    'Firing Rate Per Min': frpm,
                    'FWHM Values': fwhm,
                    'Rise Times': rise_time
                })
    
    neuron_df = pd.DataFrame(neuron_data)
    
    # Aggregate to sample level (biological replicates)
    sample_df = neuron_df.groupby(['Sample', 'Group']).agg({
        'Firing Rate Per Min': ['mean', 'std', 'count'],
        'FWHM Values': ['mean', 'std'],
        'Rise Times': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    sample_df.columns = [
        'Sample', 'Group',
        'mean_Firing Rate Per Min', 'std_FRPM', 'n_neurons',
        'mean_FWHM Values', 'std_FWHM',
        'mean_Rise Times', 'std_Rise_Time'
    ]
    
    expected = {
        'n_biological_replicates': n_samples,
        'n_per_group': samples_per_group,
        'n_technical_replicates': n_samples * neurons_per_sample,
        'true_effect_frpm': group_effect_frpm,
        'correct_n_for_stats': samples_per_group,
        'groups': [f"Group_{i+1}" for i in range(n_groups)]
    }
    
    return neuron_df, sample_df, expected


def generate_paired_data(
    n_pairs: int = 15,
    effect_size: float = 0.5,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate paired/repeated measures data with known effect.
    
    Parameters
    ----------
    n_pairs : int
        Number of paired observations (subjects). Default is 15.
    effect_size : float
        Cohen's d for paired comparison. Default is 0.5.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame in long format with Subject, Condition, Value.
    expected : dict
        Dictionary with expected properties.
    
    Examples
    --------
    >>> df, expected = generate_paired_data(n_pairs=20, effect_size=0.8)
    """
    np.random.seed(seed)
    
    # Generate baseline values
    baseline = np.random.normal(10, 2, n_pairs)
    
    # Generate treatment values with correlation to baseline
    correlation = 0.7  # Typical within-subject correlation
    treatment = baseline + effect_size * 2 + np.random.normal(0, 2 * np.sqrt(1 - correlation**2), n_pairs)
    
    df = pd.DataFrame({
        'Subject': [f'S{i}' for i in range(n_pairs)] * 2,
        'Condition': ['Baseline'] * n_pairs + ['Treatment'] * n_pairs,
        'Value': np.concatenate([baseline, treatment])
    })
    
    expected = {
        'n_pairs': n_pairs,
        'true_effect_size': effect_size,
        'should_be_significant': effect_size > 0.4 and n_pairs >= 10
    }
    
    return df, expected


def generate_proportion_data(
    n_per_group: int = 50,
    prop_control: float = 0.3,
    prop_treatment: float = 0.6,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate binary outcome data for proportion testing.
    
    Parameters
    ----------
    n_per_group : int
        Number of observations per group. Default is 50.
    prop_control : float
        Proportion of "successes" in control. Default is 0.3.
    prop_treatment : float
        Proportion of "successes" in treatment. Default is 0.6.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with Group and Outcome columns.
    expected : dict
        Dictionary with expected properties.
    
    Examples
    --------
    >>> df, expected = generate_proportion_data(prop_control=0.2, prop_treatment=0.5)
    """
    np.random.seed(seed)
    
    control_outcomes = np.random.binomial(1, prop_control, n_per_group)
    treatment_outcomes = np.random.binomial(1, prop_treatment, n_per_group)
    
    df = pd.DataFrame({
        'Sample': [f'S{i}' for i in range(n_per_group * 2)],
        'Group': ['Control'] * n_per_group + ['Treatment'] * n_per_group,
        'Outcome': np.concatenate([control_outcomes, treatment_outcomes]).astype(bool)
    })
    
    expected = {
        'true_prop_control': prop_control,
        'true_prop_treatment': prop_treatment,
        'n_per_group': n_per_group,
        'should_be_significant': abs(prop_treatment - prop_control) > 0.2 and n_per_group >= 30
    }
    
    return df, expected

