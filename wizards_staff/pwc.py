# import
## batteries
import os
import logging
from typing import Tuple
## third party
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
## package
from wizards_staff.metadata import load_and_process_metadata
from wizards_staff.wizards.familiars import spatial_filtering #categorize_files, load_and_filter_files, 

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# functions
def run_pwc(orb: "Orb", group_name: str, poly: bool=False, 
            pdeg: int=4, lw: float=1, lwp: float=0.5, psz: float=2, show_plots: bool=False, 
            save_files: bool=False, output_dir: str='wizard_staff_outputs'
            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Processes data, computes metrics, generates plots, and stores them in DataFrames.

    Args:
        group_name: Column name to group metadata by.
        metadata_path: Path to the metadata CSV file.
        results_folder: Path to the results folder.
        poly: Flag to control whether polynomial fitting is applied. Default is False.
        pdeg: Degree for polynomial fitting. Default is 4.
        lw: Line width for plots. Default is 1.
        lwp: Line width for points. Default is 0.5.
        psz: Point size for plots. Default is 2.
        show_plots: Flag to control whether plots are displayed. Default is False.
        save_files: Flag to control whether plots and dataframes are saved to files. Default is False.
        output_dir: Directory where output files will be saved. Default is 'wizard_staff_outputs'.

    Returns:
        df_mn_pwc: DataFrame containing overall pairwise correlation metrics.
        df_mn_pwc_inter: DataFrame containing inter-group pairwise correlation metrics.
        df_mn_pwc_intra: DataFrame containing intra-group pairwise correlation metrics.
    """
    #print(orb); exit();

    # Load and preprocess the metadata
    #metadata_df = load_and_process_metadata(metadata_path)

    # Group the metadata by the specified column
    d_k_in_groups = orb.metadata.groupby(group_name)['Filename'].apply(list).to_dict()

    # Load and filter data for each file
    #categorized_files = categorize_files(results_folder)

    # Load and filter necessary files
    #d_dff, d_nspIDs = load_and_filter_files(categorized_files)


    # Initialize dictionaries to store pairwise correlation means
    filtered_d_k_in_groups = filter_group_keys(d_k_in_groups, d_dff, d_nspIDs)

    d_mn_pwc, d_mn_pwc_inter, d_mn_pwc_intra = calc_pwc_mn(filtered_d_k_in_groups, d_dff, d_nspIDs, dff_cut=0.0, norm_corr=False)

    # Convert dictionaries to DataFrames
    df_mn_pwc = pd.DataFrame.from_dict(d_mn_pwc, orient='index').transpose()
    df_mn_pwc_inter = pd.DataFrame.from_dict(d_mn_pwc_inter, orient='index').transpose()
    df_mn_pwc_intra = pd.DataFrame.from_dict(d_mn_pwc_intra, orient='index').transpose()

    # Pull the filename from the results folder name
    fname = os.path.splitext(os.path.basename(results_folder))[0]

    # Plotting and saving figures
    plot_pwc_means(
        d_mn_pwc_inter, title = 'Inter_Group_Mean_PWC', xlabel = 'Groups', 
        ylabel = 'Mean Pairwise Correlation', poly = poly, lwp = lwp, psz = psz, pdeg = 4,
        show_plots = show_plots, save_files = save_files, fname = fname, output_dir = output_dir
    )

    plot_pwc_means(
        d_mn_pwc_intra, title = 'Intra_Group_Mean_PWC', xlabel = 'Groups',
         ylabel = 'Mean Pairwise Correlation', poly = poly, lwp = lwp, psz = psz, pdeg = 4,
        show_plots = show_plots, save_files = save_files, fname = fname, output_dir = output_dir
    )

    plot_pwc_means(
        d_mn_pwc, title = 'PWC', xlabel = 'Groups', ylabel = 'Mean Pairwise Correlation', 
        poly = poly, lwp = lwp, psz = psz, pdeg = 4, show_plots=show_plots, 
        save_files = save_files, fname = fname, output_dir = output_dir
    )

    # Save DataFrames if required
    if save_files:
        # Expand the user directory if it exists in the output_dir path
        output_dir = os.path.expanduser(output_dir)
        
        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the file paths
        df_mn_pwc = os.path.join(output_dir, f'{fname}_df_mn_pwc.csv')
        df_mn_pwc_inter_path = os.path.join(output_dir, f'{fname}_df_mn_pwc_intra.csv')
        df_mn_pwc_intra_path = os.path.join(output_dir, f'{fname}_df_mn_pwc_inter.csv')

        # Save each DataFrame to a CSV file
        df_mn_pwc.to_csv(df_mn_pwc, index=False)
        df_mn_pwc_intra.to_csv(df_mn_pwc_inter_path, index=False)
        df_mn_pwc_inter.to_csv(df_mn_pwc_intra_path, index=False)

        print(f'DataFrames saved to {output_dir}')

    return df_mn_pwc, df_mn_pwc_inter, df_mn_pwc_intra

def calc_pwc_mn(d_k_in_groups: dict, d_dff: dict, d_nspIDs: dict, dff_cut: float=0.1, 
                norm_corr: bool=False) -> Tuple[dict, dict, dict]:
    """
    Calculates pairwise correlation means for groups
    
    Args:
        d_k_in_groups: Dictionary where each key is a group identifier and the value is a list of keys.
        d_dff: Dictionary where each key corresponds to a key in d_k_in_groups and the value is a dF/F matrix.
        d_nspIDs: Dictionary where each key corresponds to a key in d_k_in_groups and the value is a neuron ID array.
        dff_cut: Threshold for filtering dF/F values.
        norm_corr: Whether to normalize the correlation using Fisher's z-transformation.
    
    Returns:
        d_mn_pwc: Dictionary of mean pairwise correlations for each group.
        d_mn_pwc_inter: Dictionary of mean inter-group correlations for each group.
        d_mn_pwc_intra: Dictionary of mean intra-group correlations for each group.
    """
    # Initialize dictionaries to store results
    d_mn_pwc = {}
    d_mn_pwc_intra = {}
    d_mn_pwc_inter = {}

    # Iterate over each group
    for group_id in d_k_in_groups.keys():
        # print(f'Group: {group_id}')
        keys_in_group = d_k_in_groups[group_id]  # List of keys for the current group
        list_vals = []
        list_vals_intra = []
        list_vals_inter = []

        # Iterate over each key in the current group
        for key in keys_in_group:
            tmp_dat = np.copy(d_dff[key])  # Copy dF/F data for the current key
            nsp_ids = d_nspIDs[key]  # Neuron ID array for the current key
            nsp_ids_uniq = np.unique(nsp_ids)  # Unique neuron IDs

            # Filter to include only data from neurons that pass QC
            tmp_dat_filt = tmp_dat[nsp_ids, :]
            nsp_ids_filt = nsp_ids

            # Apply dF/F threshold
            tmp_dat_filt[tmp_dat_filt < dff_cut] = 0

            # Calculate the correlation matrix
            r = np.corrcoef(tmp_dat_filt)
            r = np.nan_to_num(r)  # Replace NaNs with zeros
            np.fill_diagonal(r, 0)  # Set diagonal to zero

            # Apply Fisher's z-transformation if normalization is requested
            if norm_corr:
                r = np.arctanh(r)

            r[np.isinf(r)] = 0  # Replace infinities with zeros

            # Extract intra- and inter-group correlations
            r_intra, r_inter = extract_intra_inter_nsp_neurons(r, nsp_ids_filt)

            # Get upper triangle of the correlation matrix
            r = r[np.triu_indices(r.shape[0], k=1)]

            # Uncomment to set negative correlations to zero
            # r[r < 0] = 0
            # r_inter[r_inter < 0] = 0
            # r_intra[r_intra < 0] = 0

            # Calculate means and append to respective lists
            list_vals.append(r.mean())
            list_vals_inter.append(r_inter.mean())
            list_vals_intra.append(r_intra.mean())

        # Store the mean correlation values in the result dictionaries
        d_mn_pwc[group_id] = np.array(list_vals)
        d_mn_pwc_inter[group_id] = np.array(list_vals_inter)
        d_mn_pwc_intra[group_id] = np.array(list_vals_intra)

    # Return the result dictionaries
    return d_mn_pwc, d_mn_pwc_inter, d_mn_pwc_intra

def extract_intra_inter_nsp_neurons(conn_matrix: np.ndarray, nsp_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts intra-subpopulation and inter-subpopulation connections from a connectivity matrix.
    
    Args:
        conn_matrix: A square matrix representing the connectivity between neurons.
        nsp_ids: An array containing the subpopulation IDs for each neuron.
    
    Returns:
        intra_conn: The upper triangular values of the connectivity matrix for intra-subpopulation connections.
        inter_conn: The upper triangular values of the connectivity matrix for inter-subpopulation connections.
    """
    # Make a copy of the input connectivity matrix
    conn_copy = np.copy(conn_matrix)
    
    # Create a list of indices corresponding to rows/columns of the matrix
    neuron_idx = list(range(0, conn_copy.shape[0]))
    
    # Get the unique neuron subpopulation (NSP) IDs
    unique_nsp_ids = np.unique(nsp_ids)
    
    # Initialize a mask matrix of the same size as the connectivity matrix with zeros
    mask_matrix = np.zeros_like(conn_copy)
    
    # Iterate over each unique NSP ID
    for nsp_id in unique_nsp_ids:
        # Create a boolean mask for neurons belonging to the current NSP ID
        mask = nsp_ids == nsp_id
        
        # Get the indices of neurons belonging to the current NSP
        nsp_idx = [neuron_idx[i] for i in range(len(neuron_idx)) if mask[i]]
        
        # Mark the positions in the mask matrix corresponding to intra-NSP connections
        for i in nsp_idx:
            for j in nsp_idx:
                mask_matrix[i, j] = 1
    
    # Extract the upper triangle of the mask matrix (excluding the diagonal)
    intra_mask_upper = mask_matrix[np.triu_indices(mask_matrix.shape[0], k=1)]
    
    # Extract the upper triangle of the connectivity matrix (excluding the diagonal)
    conn_upper = conn_copy[np.triu_indices(conn_copy.shape[0], k=1)]
    
    # Extract inter-NSP connections where the mask is 0
    inter_conn = conn_upper[intra_mask_upper < 1]
    
    # Extract intra-NSP connections where the mask is 1
    intra_conn = conn_upper[intra_mask_upper == 1]
    
    return intra_conn, inter_conn

def gen_mn_std_means(mean_pwc_dict: dict) -> Tuple[dict, dict]:
    """
    Calculates the mean and standard deviation of the means for each key in the input dictionary.
    
    Args:
        mean_pwc_dict: Dictionary where each key is associated with an array of mean pairwise correlations.
    
    Returns:
        Two dictionaries containing the mean of means and standard deviation of means for each key.
    """
    mean_of_means_dict = {}
    std_of_means_dict = {}

    # Iterate over each key in the input dictionary
    for key in mean_pwc_dict:
        # Calculate the mean of the means for the current key, ignoring NaNs
        mean_of_means_dict[key] = np.nanmean(mean_pwc_dict[key])
        
        # Calculate the standard deviation of the means for the current key, ignoring NaNs
        std_of_means_dict[key] = np.nanstd(mean_pwc_dict[key])

    return mean_of_means_dict, std_of_means_dict

def gen_polynomial_fit(data_dict: dict, degree: int=4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a polynomial fit for the given data.

    Args:
        data_dict: Dictionary where keys are the independent variable (x) and values are the dependent variable (y).
        degree: The degree of the polynomial fit.

    Returns:
        tuple: Arrays of x values and corresponding predicted y values from the polynomial fit.
    """
    # Extract keys and values from the dictionary
    x_values = np.array(list(data_dict.keys()))
    y_values = np.array(list(data_dict.values()))

    # Remove NaN values from y_values and corresponding x_values
    mask = np.isnan(y_values)
    y_values = y_values[~mask]
    x_values = x_values[~mask]

    # Reshape x_values for polynomial features
    x_values = x_values[:, np.newaxis]

    # Generate polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x_values)

    # Create and fit the polynomial regression model
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(x_poly, y_values)

    # Predict y values using the polynomial model
    y_predicted = poly_reg_model.predict(poly_features.fit_transform(x_values))

    return x_values, y_predicted

def filter_group_keys(d_k_in_groups: dict, d_dff: dict, d_nspIDs: dict) -> dict:
    """
    Filters group keys to ensure that only those with valid dF/F data and neuron IDs are retained.

    Args:
        d_k_in_groups: Dictionary mapping group IDs to lists of filenames.
        d_dff: Dictionary containing dF/F data matrices for each filename.
        d_nspIDs: Dictionary containing lists of filtered neuron IDs for each filename.

    Returns:
        filtered_d_k_in_groups: Filtered dictionary where each group ID maps to a list of valid filenames.
    """
    filtered_d_k_in_groups = {}
    for group_id, keys_in_group in d_k_in_groups.items():
        filtered_keys_in_group = [key for key in keys_in_group if key in d_dff and key in d_nspIDs]
        filtered_d_k_in_groups[group_id] = filtered_keys_in_group
    return filtered_d_k_in_groups

def plot_pwc_means(d_mn_pwc: dict, title: str, fname: str, output_dir: str, xlabel: str='Groups', 
                   ylabel: str='Mean Pairwise Correlation', poly: bool=False, lwp: float=1, 
                   psz: float=5, pdeg: int=4, show_plots: bool=True, save_files: bool=False
                   ) -> None:
    """
    Generates plots of mean pairwise correlations with error bars and optionally saves the plots.

    Args:
        d_mn_pwc (dict): Dictionary containing mean pairwise correlation data.
        title (str): Title of the plot.
        fname (str): Filename for saving the results (without extension).
        output_dir (str): Directory where output files will be saved.
        xlabel (str): Label for the x-axis. Default is 'Groups'.
        ylabel (str): Label for the y-axis. Default is 'Mean Pairwise Correlation'.
        lwp (float): Line width for the plot. Default is 1.
        psz (float): Point size for the plot. Default is 5.
        pdeg (int): Degree of the polynomial fit, if applied. Default is 4.
        show_plots (bool): Flag to control whether plots are displayed. Default is True.
        save_files (bool): Flag to control whether plots are saved to files. Default is False.
    """
    # Generate and sort means and standard deviations
    d, d_std = gen_mn_std_means(d_mn_pwc)
    d = dict(sorted(d.items()))
    d_std = dict(sorted(d_std.items()))

    # Convert dictionary keys and values to lists
    keys = list(d.keys())
    values = list(d.values())
    errors = list(d_std.values())

    # Create blank figure
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.margins(0.008)
    ax.errorbar(keys, values, yerr=errors, fmt='o', color='gray', 
                label='Cell Pairs', linewidth=lwp, markersize=psz)

    # Polynomial fit can be added if needed
    if poly:
        x, y = gen_polynomial_fit(d, degree=pdeg)
        ax.plot(x, y, color='gray', linewidth=lwp)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    
    if save_files==True:
        # Expand the user directory if it exists in the output_dir path
        output_dir = os.path.expanduser(output_dir)

        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the figure
        plt.savefig(f'{output_dir}{fname}_{title}.png', bbox_inches='tight')
        
    if show_plots:
        plt.show()
    else:
        plt.close()