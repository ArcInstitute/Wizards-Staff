"""
Statistical Analysis Module for Wizards Staff.

This module provides a comprehensive statistical analysis framework for calcium imaging data,
designed for neuroscientists and cell biologists with varying levels of statistical expertise.

Quick Start
-----------
>>> from wizards_staff.stats import prepare_for_stats, compare_two_groups, quick_report
>>> 
>>> # Prepare your data for analysis
>>> df = prepare_for_stats(orb, metric="frpm", metadata_cols=["Treatment"])
>>> 
>>> # Compare treatment groups
>>> result = compare_two_groups(df, group_col="Treatment", metric_col="mean_frpm")
>>> print(result.summary)
>>> 
>>> # Generate a complete report
>>> quick_report(orb, group_col="Treatment", output_path="my_report.html")

Modules
-------
core : Core infrastructure (StatsResult, StatsConfig, DataValidator)
tests : Statistical test implementations
corrections : Multiple comparison corrections
effect_size : Effect size calculations
power : Power analysis utilities
timeseries : Time-series specific analyses
outliers : Outlier detection and handling
assumptions : Statistical assumption checking
reports : Report generation utilities
visualizations : Statistical visualization templates
"""

# Core infrastructure
from .core import (
    StatsResult,
    StatsConfig,
    DataValidator,
    aggregate_to_biological_replicates,
    merge_with_metadata,
    prepare_for_stats,
)

# Statistical tests
from .tests import (
    compare_two_groups,
    compare_multiple_groups,
    compare_two_factors,
    test_correlation,
    test_proportion,
)

# Multiple comparisons
from .corrections import (
    apply_correction,
    recommend_correction_method,
)

# Effect sizes
from .effect_size import (
    cohens_d,
    eta_squared,
    rank_biserial,
    cliffs_delta,
)

# Power analysis
from .power import (
    calculate_required_n,
    calculate_achieved_power,
    calculate_detectable_effect,
)

# Time-series analysis
from .timeseries import (
    analyze_transient_kinetics,
    calculate_event_frequency_over_time,
    detect_bursting,
    calculate_synchrony_metrics,
)

# Outlier detection
from .outliers import (
    detect_outliers,
    visualize_outliers,
    handle_outliers,
)

# Assumption checking
from .assumptions import (
    check_normality,
    check_homogeneity_of_variance,
    check_all_assumptions,
    generate_assumption_report,
)

# Report generation
from .reports import (
    StatsReport,
    quick_report,
    generate_methods_text,
)

# Visualizations
from .visualizations import (
    plot_group_comparison,
    plot_paired_comparison,
    plot_distribution_check,
    plot_effect_sizes,
    plot_power_curve,
    plot_multiple_metrics_summary,
)

# Test utilities (for generating synthetic data)
from .test_utils import (
    generate_two_group_data,
    generate_multi_group_data,
    generate_correlated_data,
    generate_non_normal_data,
    generate_calcium_imaging_like_data,
    generate_paired_data,
    generate_proportion_data,
)

__all__ = [
    # Core
    "StatsResult",
    "StatsConfig",
    "DataValidator",
    "aggregate_to_biological_replicates",
    "merge_with_metadata",
    "prepare_for_stats",
    # Tests
    "compare_two_groups",
    "compare_multiple_groups",
    "compare_two_factors",
    "test_correlation",
    "test_proportion",
    # Corrections
    "apply_correction",
    "recommend_correction_method",
    # Effect sizes
    "cohens_d",
    "eta_squared",
    "rank_biserial",
    "cliffs_delta",
    # Power
    "calculate_required_n",
    "calculate_achieved_power",
    "calculate_detectable_effect",
    # Time-series
    "analyze_transient_kinetics",
    "calculate_event_frequency_over_time",
    "detect_bursting",
    "calculate_synchrony_metrics",
    # Outliers
    "detect_outliers",
    "visualize_outliers",
    "handle_outliers",
    # Assumptions
    "check_normality",
    "check_homogeneity_of_variance",
    "check_all_assumptions",
    "generate_assumption_report",
    # Reports
    "StatsReport",
    "quick_report",
    "generate_methods_text",
    # Visualizations
    "plot_group_comparison",
    "plot_paired_comparison",
    "plot_distribution_check",
    "plot_effect_sizes",
    "plot_power_curve",
    "plot_multiple_metrics_summary",
    # Test utilities
    "generate_two_group_data",
    "generate_multi_group_data",
    "generate_correlated_data",
    "generate_non_normal_data",
    "generate_calcium_imaging_like_data",
    "generate_paired_data",
    "generate_proportion_data",
]

