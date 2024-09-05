import pandas as pd

def load_and_process_metadata(metadata_path):
    """
    Loads and preprocesses the metadata CSV file.

    Args:
        metadata_path (str): Path to the metadata CSV file.

    Returns:
        pd.DataFrame: Preprocessed metadata DataFrame.
    """
    metadata_df = pd.read_csv(metadata_path)
    metadata_df['Filename'] = metadata_df['Filename'].str.replace('.czi', '').str.replace('.tif', '')

    return metadata_df

def append_metadata_to_dfs(rise_time_df, fwhm_df, frpm_df, mask_metrics_df, silhouette_scores_df, metadata_path):
    """
    Appends metadata to the given dataframes based on the filename match.

    Args:
        rise_time_df (pd.DataFrame): DataFrame containing rise time metrics.
        fwhm_df (pd.DataFrame): DataFrame containing FWHM metrics.
        frpm_df (pd.DataFrame): DataFrame containing FRPM metrics.
        mask_metrics_df (pd.DataFrame): DataFrame containing mask metrics.
        metadata_path (str): Path to the metadata CSV file.

    Returns:
        tuple: DataFrames with appended metadata.
    """
    # Load the metadata CSV file
    metadata_df = load_and_process_metadata(metadata_path)

    # Merge each dataframe with the metadata dataframe based on the 'file' column
    rise_time_df = rise_time_df.merge(metadata_df, on='Filename', how='left')
    fwhm_df = fwhm_df.merge(metadata_df, on='Filename', how='left')
    frpm_df = frpm_df.merge(metadata_df, on='Filename', how='left')
    mask_metrics_df = mask_metrics_df.merge(metadata_df, on='Filename', how='left')
    silhouette_scores_df = silhouette_scores_df.merge(metadata_df, on='Filename', how='left')

    return rise_time_df, fwhm_df, frpm_df, mask_metrics_df, silhouette_scores_df
