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
    metadata_df['Sample'] = metadata_df['Sample'].str.replace('.czi', '').str.replace('.tif', '')

    return metadata_df

def append_metadata_to_dfs(metadata_path, **dataframes):
    """
    Appends metadata to the given dataframes based on the filename match.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        **dataframes: Dictionary of DataFrames to append metadata to. Each key should be a string describing the metric 
                      (e.g., 'frpm', 'fwhm'), and each value should be the corresponding DataFrame.

    Returns:
        dict: A dictionary of DataFrames with appended metadata.
    """
    # Load the metadata CSV file
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df.rename(columns={'Sample': 'Filename'})

    # Dictionary to store updated DataFrames
    updated_dfs = {}

    # Loop through each dataframe and merge with the metadata
    for name, df in dataframes.items():
        if df is not None:
            updated_dfs[name] = df.merge(metadata_df, on='Filename', how='left')

    return updated_dfs

