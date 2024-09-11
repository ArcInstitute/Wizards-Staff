# wizards_staff/__init__.py

# From king_gizzard.py
from .king_gizzard import run_all

# From pwc.py
from .pwc import lizard_wizard_pwc, calc_pwc_mn, plot_pwc_means

# From metadata.py
from .metadata import load_and_process_metadata, append_metadata_to_dfs

# From metrics.py
from .metrics import calc_rise_tm, calc_fwhm_spikes, calc_frpm, convert_f_to_cs, calc_mask_shape_metrics

# From plotting.py
from .plotting import plot_activity_map, plot_kmeans_heatmap, plot_cluster_activity, spatial_filter_and_plot

# From utils.py
from .utils import categorize_files

# Define what is accessible when doing `from wizards_staff import *`
__all__ = [
    "run_all",
    "lizard_wizard_pwc", "calc_pwc_mn", "plot_pwc_means",
    "load_and_process_metadata", "append_metadata_to_dfs",
    "calc_rise_tm", "calc_fwhm_spikes", "calc_frpm", "convert_f_to_cs", "calc_mask_shape_metrics",
    "plot_activity_map", "plot_kmeans_heatmap", "plot_cluster_activity", "spatial_filter_and_plot",
    "categorize_files"
]