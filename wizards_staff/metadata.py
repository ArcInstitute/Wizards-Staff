# import
## 3rd party
import pandas as pd

# functions
def load_and_process_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the metadata CSV file.

    Args:
        metadata_path: Path to the metadata CSV file.

    Returns:
        Preprocessed metadata DataFrame.
    """
    metadata_df = pd.read_csv(metadata_path)
    metadata_df['Filename'] = metadata_df['Filename'].str.replace('.czi', '').str.replace('.tif', '')

    return metadata_df

def append_metadata_to_dfs(metadata_path: str, **dataframes) -> dict:
    """
    Appends metadata to the given dataframes based on the filename match.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        **dataframes: Dictionary of DataFrames to append metadata to. Each key should be a string describing the metric 
                      (e.g., 'frpm', 'fwhm'), and each value should be the corresponding DataFrame.

    Returns:
        A dictionary of DataFrames with appended metadata.
    """
    # Load the metadata CSV file
    metadata_df = load_and_process_metadata(metadata_path)

    # Dictionary to store updated DataFrames
    updated_dfs = {}

    # Loop through each dataframe and merge with the metadata
    for name, df in dataframes.items():
        if df is not None:
            updated_dfs[name] = df.merge(metadata_df, on='Filename', how='left')

    return updated_dfs

