Wizards Staff
=============

<img src="./img/wizards_staff.png" alt="drawing" width="400"/>


Calcium imaging analysis toolkit for processing outputs from calcium imaging pipelines (like [Lizard-Wizard](https://github.com/ArcInstitute/Lizard-Wizard)) and extracting advanced metrics, correlations, and visualizations to characterize neural activity.

## Features

- **Comprehensive Metrics Analysis**: Extract rise time, FWHM (Full Width at Half Maximum), and Firing Rate Per Minute (FRPM) metrics from calcium imaging data
- **Advanced Correlation Analysis**: Perform pairwise correlation (PWC) analysis within and between neuron populations
- **Spatial Activity Mapping**: Generate spatial activity maps to visualize active neurons and their clustering
- **K-means Clustering**: Apply clustering algorithms to identify synchronously active neurons
- **Versatile Visualization Tools**: Create publication-quality visualizations for activity traces, spatial components, and clustering results
- **Modular Architecture**: Utilize the `Orb` and `Shard` classes for organized, scalable data processing

## Table of Contents

- [Installation](#installation)
- [Brief Functions Overview](#wizards-staff-python-package---brief-function-overview)

## Installation

### Clone the Repo

To download the latest version of this package, clone it from the repository:

```bash
git clone git@github.com:ArcInstitute/Wizards-Staff.git
cd Wizards-Staff
```

### Create a Virtual Environment (Optional but Recommended)

Its recommended to create a virtual environment for Wizards-Staff as this ensures that your project dependencies are isolated. You can use mamba to create one:

```bash
mamba create -n wizards_staff python=3.11 -y
mamba activate wizards_staff
```

### Install the Package

To install the wizards_staff package within your virtual environment, from the command line run:

```bash
pip install .
```

---

## Quick Start

```python
from wizards_staff import Orb

# Initialize an Orb with your results folder and metadata file
orb = Orb(
    results_folder="/path/to/calcium_imaging_results", 
    metadata_file_path="/path/to/metadata.csv"
)

# Run comprehensive analysis (all metrics)
orb.run_all(
    group_name=None,  # Group samples by this metadata column
    frate=30,           # Frame rate of recording
    show_plots=True,    # Display plots during analysis
    save_files=True     # Save results to disk
)

# Access results as pandas DataFrames
rise_time_df = orb.rise_time_data
fwhm_df = orb.fwhm_data
frpm_df = orb.frpm_data
```

## Data Requirements

### Input Data

Wizards Staff is designed to process outputs from calcium imaging pipelines such as [Lizard-Wizard](https://github.com/ArcInstitute/Lizard-Wizard). The main input data includes:

- Delta F/F0 (dF/F0) matrices
- Spatial footprints of neurons (cnm_A)
- Indices of accepted components (cnm_idx)
- Minimum projection images
- Masks (optional, for shape metrics)

### Metadata Format

A metadata CSV file with the following required columns:
- `Sample`: Unique identifier for each sample, matching filenames
- `Well`: Well identifier (or other grouping variable)
- `Frate`: Frame rate of the recording in frames per second

## Examples

### Pairwise Correlation Analysis

```python
# Run pairwise correlation analysis
orb.run_pwc(
    group_name="Well",  # Group by this metadata column
    poly=True,          # Apply polynomial fitting
    show_plots=True     # Display correlation plots
)

# Access pairwise correlation results
pwc_df = orb.df_mn_pwc  # Overall pairwise correlations
intra_df = orb.df_mn_pwc_intra  # Intra-group correlations
inter_df = orb.df_mn_pwc_inter  # Inter-group correlations
```

## Documentation

For detailed usage instructions and examples, please refer to:

- [Jupyter Notebook Tutorial](notebooks/tutorial-notebook.ipynb)
- [API Reference](docs/api.md)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This software was developed at the [Arc Institute](https://arcinstitute.org/)