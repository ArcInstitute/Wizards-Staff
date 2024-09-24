# import
## batteries
import os
import sys
import logging
import pickle
from typing import Callable, Dict, Any, Generator, Tuple, List, Optional
from dataclasses import dataclass, field
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
## 3rd party
import numpy as np
import pandas as pd
from tifffile import imread
from tqdm.notebook import tqdm
## package
from wizards_staff.logger import init_custom_logger
from wizards_staff.pwc import run_pwc
from wizards_staff.wizards.shard import Shard
from wizards_staff.wizards.cauldron import _run_all

# Functions
def npy_loader(infile, allow_pickle=True):
    return np.load(infile, allow_pickle=allow_pickle)

# Data item mapping (how to load each data item)
DATA_ITEM_MAPPING = {
    'cnm_A': {
        'suffixes': ['_cnm-A.npy'],
        'loader': npy_loader
    },
    'cnm_C': {
        'suffixes': ['_cnm-C.npy'],
        'loader': npy_loader
    },
    'cnm_S': {
        'suffixes': ['_cnm-S.npy'],
        'loader': npy_loader
    },
    'cnm_idx': {
        'suffixes': ['_cnm-idx.npy'],
        'loader': npy_loader
    },
    'df_f0_graph': {
        'suffixes': ['_df-f0-graph.tif'],
        'loader': imread
    },
    'dff_dat': { 
        'suffixes': ['_dff-dat.npy'],
        'loader': npy_loader
    },
    'f_dat': { 
        'suffixes': ['_f-dat.npy'],
        'loader': npy_loader
    },
    'minprojection': {    # aka: im_min
        'suffixes': ['_minprojection.tif'],
        'loader': imread
    },
    'mask': {  
        'suffixes': ['_masks.tif'],
        'loader': imread
    }
}

# classes
@dataclass
class Orb:
    """
    Represents a collection of samples and their associated data shards.
    """
    results_folder: str
    metadata_file_path: str
    metadata: pd.DataFrame = field(init=False)
    _logger: Optional[logging.Logger] = field(default=None, init=False)
    _rise_time_data: pd.DataFrame = field(default=None, init=False)
    _fwhm_data: pd.DataFrame = field(default=None, init=False)
    _frpm_data: pd.DataFrame = field(default=None, init=False)
    _mask_metrics_data: pd.DataFrame = field(default=None, init=False)
    _silhouette_scores_data: pd.DataFrame = field(default=None, init=False)
    _shards: Dict[str, Shard] = field(default_factory=dict, init=False)   # loaded data
    _data_mapping: Dict[str, Any] = field(default_factory=lambda: DATA_ITEM_MAPPING, init=False)  # data item mapping
    _input_files: pd.DataFrame = field(default=None, init=False)  # file paths
    _input: pd.DataFrame = field(default=None, init=False)  # all input data
    _samples: set = field(default=None, init=False)  # samples
    
    def __post_init__(self):
        # Configure logging
        self._logger = init_custom_logger(__name__)
        # load metadata
        self._load_metadata(self.metadata_file_path)
        self._samples = set(self.metadata['Sample'])
        # run categorization upon initialization
        self._categorize_files()   

    def _categorize_files(self):
        """
        Categorizes files into corresponding data items for each sample.
        """
        self._logger.info("Categorizing files...")
        # load files 
        for file_path in self._list_files(self.results_folder):
            # get file info 
            ## basename
            file_basename = os.path.basename(file_path)
            ## sample name
            sample_name = None
            for sample in self.metadata['Sample']:
                if file_basename.startswith(sample):
                    sample_name = sample
                    break
            ### filter out samples not in metadata
            if sample_name is None or sample_name not in self._samples:
                continue
            ## suffix
            file_suffix = file_basename[len(sample_name):]
            ## categorize file based on suffix
            for item_name, data_info in self._data_mapping.items():
                # file suffix matches data item suffix?
                if any(file_suffix.endswith(suffix) for suffix in data_info['suffixes']):
                    shard = self._shards.setdefault(
                        sample_name,
                        Shard(
                            sample_name,
                            metadata=self.metadata[self.metadata['Sample'] == sample_name],
                            files={}
                        )
                    )
                    shard.files[item_name] = (file_path, data_info['loader'])
                    break

        # check for missing files
        for item_name in self._data_mapping.keys():
            missing_samples = []
            for sample in self._samples:
                try:
                    # sample has file?
                    if self._shards[sample].has_file(item_name) is False:
                        missing_samples.append(sample)
                except KeyError:
                    # no sample?
                    missing_samples.append(sample)
            if len(missing_samples) > 0:
                missing_samples = ', '.join(missing_samples)
                msg = f"WARNING: No '{item_name}' files found for samples: {missing_samples}"
                print(msg, file=sys.stderr)
    
    def _load_metadata(self, metadata_file_path: str):
        """
        Loads metadata from a CSV file.
        Args:
            metadata_file_path: Path to the metadata CSV file.
        """
        # status
        self._logger.info(f"Loading metadata from: {metadata_file_path}")
        # check if metadata file exists
        if not os.path.exists(metadata_file_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file_path}")
        # load metadata
        self.metadata = pd.read_csv(metadata_file_path)
        # check for required columns
        required_columns = {"Sample", "Well", "Frate"}
        missing_columns = required_columns - set(self.metadata.columns)
        if missing_columns:
            cols_str = ', '.join(missing_columns)
            raise ValueError(f"Missing columns in metadata file: {cols_str}")

    #-- get items --#
    def list_data_items(self, sample: str) -> List[str]:
        """
        Lists all data items for a given sample.
        Args:
            sample: Sample name.
        Returns:
            List of data items.
        """
        shard = self._shards.get(sample)
        if shard:
            return list(shard.get_data_items())
        else:
            self._logger.warning(f"Sample '{sample}' not found")
            return []

    def shatter(self) -> Generator[Shard, None, None]:
        """
        Yields each Shard (sample's data) in the Orb.
        """
        yield from self._shards.values()

    def items(self) -> Generator[Tuple[str, Shard], None, None]:
        """
        Yields tuples of sample names and their Shard objects.
        """
        yield from self._shards.items()

    def _get_shard_data(self, attr_name: str) -> pd.DataFrame:
        """Dynamically generate a DataFrame for the given attribute from shards."""
        attr = getattr(self, attr_name)
        if attr is None:
            # Create DataFrame if it doesn't exist
            DF = []
            for shard in self.shatter():
                shard_data = getattr(shard, attr_name, None)
                if shard_data is not None:
                    DF += shard_data
            if len(DF) == 0:
                return None
            attr = pd.DataFrame(DF)
            # Cache the result
            setattr(self, attr_name, attr)  
        return attr

    #-- save data --#
    def save_data(self, outdir: str, data_items: 
                  list=["rise_time_data", "fwhm_data", "frpm_data", 
                        "mask_metrics_data", "silhouette_scores_data"]):
        """
        Saves data items to disk.
        Args:
            outdir: Output directory.
            data_items: List of data items to save.
        Returns:
            List of saved file paths.
        """
        self._logger.info(f"Saving data items to: {outdir}")
        # output directory
        if outdir != "" and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        # save each data item
        outfiles = []
        for data_item_name in data_items:
            # get property
            data = getattr(self, data_item_name)
            if data is None:
                self._logger.warning(f"Data item '{data_item_name}' not found")
                continue
            # write to disk
            outfile = os.path.join(outdir, data_item_name.replace("_", "-") + ".csv")
            data.to_csv(outfile, index=False)
            self._logger.info(f"'{data_item_name}' saved to: {outfile}")
            outfiles.append(outfile)

    #-- data processing --#
    def run_all(self, frate: int=30, zscore_threshold: int=3, 
                percentage_threshold: float=0.2, p_th: float=75, min_clusters: int=2, 
                max_clusters: int=10, random_seed: int=1111111, group_name: str=None, 
                poly: bool=False, size_threshold: int=20000, show_plots: bool=True, 
                save_files: bool=True, output_dir: str='wizard_staff_outputs', 
                threads: int=2, debug: bool=False) -> None:
        """
        Process the results folder, computes metrics, and stores them in DataFrames.
    
        Args:
            results_folder (str): Path to the results folder.
            metadata_path (str): Path to the metadata CSV file.
            frate (int): Frames per second of the imaging session.
            zscore_threshold (int): Z-score threshold for spike detection.
            percentage_threshold (float): Percentage threshold for FWHM calculation.
            p_th (float): Percentile threshold for image processing.
            min_clusters (int): The minimum number of clusters to try.
            max_clusters (int): The maximum number of clusters to try.
            random_seed (int): The seed for random number generation in K-means.
            group_name (str): Name of the group to which the data belongs. Required for PWC analysis.
            poly (bool): Flag to control whether to perform polynomial fitting during PWC analysis.
            size_threshold (int): Size threshold for filtering out noise events.    
            show_plots (bool): Flag to control whether plots are displayed. 
            save_files (bool): Flag to control whether files are saved.
            output_dir (str): Directory where output files will be saved.
            threads (int): Number of threads to use for processing. 
        """
        # Check if the output directory exists
        if save_files:
            # Expand the user directory if it exists in the output_dir path
            output_dir = os.path.expanduser(output_dir)
            # Create the output directory if it does not exist
            os.makedirs(output_dir, exist_ok=True)
    
        # Process each sample (shard) in parallel
        func = partial(
            _run_all, 
            frate=frate, 
            zscore_threshold=zscore_threshold, 
            percentage_threshold=percentage_threshold,
            p_th=p_th,
            min_clusters=min_clusters,
            max_clusters=max_clusters, 
            random_seed=random_seed,
            group_name=group_name,
            poly=poly,
            size_threshold=size_threshold,
            show_plots=show_plots,
            save_files=save_files,
            output_dir=output_dir
        )
        if debug or threads == 1:
            for shard in self.shatter():
                func(shard)
        else:
            with ProcessPoolExecutor() as executor:
                logging.disable(logging.INFO)
                desc = 'Processing shards of the Wizard Orb'
                # Submit the function to the executor for each shard
                futures = {executor.submit(func, shard) for shard in self.shatter()}
                # Use as_completed to get the results as they are completed
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    try:
                        # Get the result from each completed future
                        updated_shard = future.result()
                        self._shards[updated_shard.sample_name] = updated_shard
                    except Exception as e:
                        # Handle any exception that occurred during the execution
                        print(f'Exception occurred: {e}')
                # Re-enable logging
                logging.disable(logging.NOTSET)
    
        # Save DataFrames as CSV files if required
        if save_files:
            self.save_data(output_dir)
        
        # Run PWC analysis if group_name is provided
        # if group_name:
        #     orb.run_pwc(
        #         group_name, metadata_path, results_folder, 
        #         poly = poly,
        #         show_plots = show_plots, 
        #         save_files = save_files, 
        #         output_dir = output_dir
        #     )

    def run_pwc(self, **kwargs) -> None:
        """
        Runs pairwise correlation analysis on all samples.
        Args:
            **kwargs: Keyword arguments to pass to `run_pwc
        """
        # run PWC on each sample
        run_pwc(self, **kwargs)

    def save(self, outfile: str) -> None:
        """
        Saves the Orb object to disk.
        Args:
            outfile: Output file path.
        """
        outdir = os.path.dirname(outfile)
        if outdir != "" and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        # save object
        with open(outfile, 'wb') as f:
            pickle.dump(self, f)
        self._logger.info(f"Orb saved to: {outfile}")

    #-- misc --#
    def __str__(self) -> str:
        """
        Prints sample : data_item_name : file_path for all shards.
        """
        return self.input_files.to_string()

    __repr__ = __str__

    def _repr_html_(self):
        return self.input_files.to_html()

    @staticmethod
    def _list_files(indir) -> List[str]:
        """
        Recursively lists all files in a directory.
        """
        files = []
        for dirpath, dirnames, filenames in os.walk(indir):
            for filename in filenames:
                files.append(os.path.join(dirpath, filename))
        return files

    #-- properties --#
    @property
    def samples(self):
        if self._samples is None:
            self._samples = set(self._shards.keys())
        return self._samples
    
    @property
    def num_shards(self):
        return len(self._shards)

    @property
    def shards(self):
        yield from self._shards.values()

    @property
    def input_files(self) -> pd.DataFrame:
        if self._input_files is None:
            self._input_files = pd.concat(
                [shard.input_files for shard in self.shards]
            )
        return self._input_files

    @property
    def input(self) -> pd.DataFrame:
        """
        Returns a DataFrame with all data items and file paths, merged with metadata.
        """
        if self._input is None:
            self._input = pd.merge(
                self.input_files.copy(), 
                self.metadata, on='Sample', how='left'
            )
        return self._input

    @property
    def rise_time_data(self):
        DF = self._get_shard_data('_rise_time_data')
        if DF is None:
            return None
        # explode columns, if they exist
        cols = ['Rise Times', 'Rise Positions']        
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        # return after merging with metadata
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def fwhm_data(self):
        DF = self._get_shard_data('_fwhm_data')
        if DF is None:
            return None
        # explode columns, if they exist
        cols = ['FWHM Backward Positions', 'FWHM Forward Positions', 'FWHM Values', 'Spike Counts']
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def frpm_data(self):
        DF = self._get_shard_data('_frpm_data')
        if DF is None:
            return None
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def mask_metrics_data(self):
        DF = self._get_shard_data('_mask_metrics_data')
        if DF is None:
            return None
        return DF.merge(self.metadata, on='Sample', how='left')
    
    @property
    def silhouette_scores_data(self):
        DF = self._get_shard_data('_silhouette_scores_data')
        if DF is None:
            return None
        return DF.merge(self.metadata, on='Sample', how='left')
