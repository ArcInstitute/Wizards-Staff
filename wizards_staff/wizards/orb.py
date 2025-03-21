# import
## batteries
import os
import sys
import logging
import pickle
from typing import Dict, Any, Generator, Tuple, List, Optional
from dataclasses import dataclass, field
from functools import wraps
## 3rd party
import numpy as np
import pandas as pd
from tifffile import imread
from tqdm.notebook import tqdm
## package
from wizards_staff.logger import init_custom_logger
from wizards_staff.pwc import run_pwc as ws_run_pwc
from wizards_staff.wizards.shard import Shard
from wizards_staff.wizards.cauldron import run_all as ws_run_all

# Functions
def npy_loader(infile, allow_pickle=True):
    return np.load(infile, allow_pickle=allow_pickle)

# Data item mapping (how to load each data item)
DATA_ITEM_MAPPING = {
    'cnm_A': {
        'suffixes': ['_cnm-A.npy'],
        'loader': npy_loader,
        'required': True
    },
    'cnm_C': {
        'suffixes': ['_cnm-C.npy'],
        'loader': npy_loader,
        'required': True
    },
    'cnm_S': {
        'suffixes': ['_cnm-S.npy'],
        'loader': npy_loader,
        'required': True
    },
    'cnm_idx': {
        'suffixes': ['_cnm-idx.npy'],
        'loader': npy_loader,
        'required': True
    },
    'df_f0_graph': {
        'suffixes': ['_df-f0-graph.tif'],
        'loader': imread,
        'required': False
    },
    'dff_dat': { 
        'suffixes': ['_dff-dat.npy'],
        'loader': npy_loader,
        'required': True
    },
    'f_dat': { 
        'suffixes': ['_f-dat.npy'],
        'loader': npy_loader,
        'required': True
    },
    'minprojection': {    # aka: im_min
        'suffixes': ['_minprojection.tif'],
        'loader': imread,
        'required': True
    },
    'mask': {  
        'suffixes': ['_masks.tif'],
        'loader': imread,
        'required': True
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
    allow_missing: bool = False
    quiet: bool = False
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
    _df_mn_pwc: pd.DataFrame = field(default=None, init=False)  # pairwise correlation results
    _df_mn_pwc_intra: pd.DataFrame = field(default=None, init=False)  # pairwise correlation results
    _df_mn_pwc_inter: pd.DataFrame = field(default=None, init=False)  # pairwise correlation results
    _pwc_plots: Dict[str, Any] = field(default_factory=dict, init=False)  # pairwise correlation results
    
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
        mask_suffixes = {}
        # load files 
        for file_path in self._list_files(self.results_folder):
            # skip concatenated/output/ folder
            file_path_parts = file_path.split(os.path.sep)
            try:
                if file_path_parts[-3] == "concatenated" and file_path_parts[-2] == "output":
                    continue
            except IndexError:
                pass
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

            # check for mask suffix consistency 
            # ie make sure we arent pulling both masked and unmasked files
            mask_part = file_suffix.split('_')[0]
            if sample_name in mask_suffixes:
                #print(sample_name, file_path, mask_suffixes[sample_name], mask_part);
                assert mask_suffixes[sample_name] == mask_part, (
                    f"Inconsistent mask suffix for sample {sample_name}: "
                    f"expected '{mask_suffixes[sample_name]}', got '{mask_part}'"
                )
            else:
                mask_suffixes[sample_name] = mask_part
        
            ## categorize file based on suffix
            for item_name, data_info in self._data_mapping.items():
                # file suffix matches data item suffix?
                if any(file_suffix.endswith(suffix) for suffix in data_info['suffixes']):
                    shard = self._shards.setdefault(
                        sample_name,
                        Shard(
                            sample_name,
                            metadata=self.metadata[self.metadata['Sample'] == sample_name],
                            files={},
                            quiet=self.quiet,
                            allow_missing=self.allow_missing
                        )
                    )
                    if item_name in shard.files:
                        existing_file, _ = shard.files[item_name]
                        # Extract the previously recorded suffix
                        existing_suffix = existing_file.split(sample_name, 1)[-1]
                        # Assert that the new suffix is identical to the existing one
                        assert existing_suffix == file_suffix, (
                            f"Inconsistent suffix for sample {sample_name} and data item '{item_name}': "
                            f"'{existing_suffix}' vs '{file_suffix}'"
                        )
                    shard.files[item_name] = (file_path, data_info['loader'])
                    break
        # check for missing files
        for item_name, item_info in self._data_mapping.items():
            if not item_info.get('required', True):  # Skip warning if item is not required
                continue
            missing_samples = []
            for sample in self._samples:
                try:
                    # sample has file?
                    if self._shards[sample].has_file(item_name) is False:
                        missing_samples.append(sample)
                except KeyError:
                    # no sample?
                    missing_samples.append(sample)
            if len(missing_samples) > 0 and not self.quiet:
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
        """
        Dynamically generate a DataFrame for the given attribute from shards.
        """
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

    #-- check input --#
    def has_input(self, data_item_name: str) -> bool:
        """
        Checks if the input data item exists.
        Args:
            data_item_name: Name of the data item.
        Returns:
            True if the input data item exists.
        """
        return any(self.input_files["DataItem"] == data_item_name)

    #-- save data --#
    def save_results(self, outdir: str, result_names: 
                  list=["rise_time_data", "fwhm_data", "frpm_data", 
                        "mask_metrics_data", "silhouette_scores_data",
                        "df_mn_pwc", "df_mn_pwc_intra", "df_mn_pwc_inter"]):
        """
        Saves data items to disk.
        Args:
            outdir: Output directory.
            result_names: List of results to save.
        """
        self._logger.info(f"Saving data items to: {outdir}")
        # output directory
        if outdir != "" and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        # save each data item
        outfiles = []
        for name in result_names:
            # get property
            data = getattr(self, name)
            if data is None:
                self._logger.warning(f"Data item '{name}' not found")
                continue
            # write to disk
            outfile = os.path.join(outdir, name.replace("_", "-") + ".csv")
            data.to_csv(outfile, index=False)
            self._logger.info(f"  '{name}' saved to: {outfile}")
            outfiles.append(outfile)
        # save plots
        for label, plot in self._pwc_plots.items():
            outfile = os.path.join(outdir, f"{label}.png")
            plot.savefig(outfile, bbox_inches='tight')
            self._logger.info(f"  Plot saved to: {outfile}")

    #-- data processing --#
    @wraps(ws_run_all)
    def run_all(self, *args, **kwargs) -> None:
        """
        Runs all data processing steps.
        """
        ws_run_all(self, *args, **kwargs)

    @wraps(ws_run_pwc)
    def run_pwc(self, *args, **kwargs) -> None:
        """
        Runs pairwise correlation analysis on all samples.
        """
        ws_run_pwc(self, *args, **kwargs)

    def save(self, outfile: str) -> None:
        """
        Saves the Orb object to disk via pickle.
        Args:
            outfile: Output file path.
        """
        # output directory
        outdir = os.path.dirname(outfile)
        if outdir != "" and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        # save object
        with open(outfile, 'wb') as f:
            pickle.dump(self, f)
        self._logger.info(f"Orb saved to: {outfile}")

    #-- misc --#
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
            try:
                self._input_files = pd.concat(
                    [shard.input_files for shard in self.shards]
                )
            except ValueError:
                logging.warning("No shards found")
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
    def results(self) -> Dict[str, pd.DataFrame]:
        """
        Return all results
        """
        yield from {
            'rise_time_data': self.rise_time_data,
            'fwhm_data': self.fwhm_data,
            'frpm_data': self.frpm_data,
            'mask_metrics_data': self.mask_metrics_data,
            'silhouette_scores_data': self.silhouette_scores_data,
            'df_mn_pwc': self.df_mn_pwc,
            'df_mn_pwc_intra': self.df_mn_pwc_intra,
            'df_mn_pwc_inter': self.df_mn_pwc_inter
        }.items()

    @property
    def rise_time_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with rise time data.
        """
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
    def fwhm_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with FWHM data.
        """
        DF = self._get_shard_data('_fwhm_data')
        if DF is None:
            return None
        # explode columns, if they exist
        cols = ['FWHM Backward Positions', 'FWHM Forward Positions', 'FWHM Values', 'Spike Counts']
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def frpm_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with FRPM data.
        """
        DF = self._get_shard_data('_frpm_data')
        if DF is None:
            return None
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def mask_metrics_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with mask metrics data.
        """
        DF = self._get_shard_data('_mask_metrics_data')
        if DF is None:
            return None
        return DF.merge(self.metadata, on='Sample', how='left')
    
    @property
    def silhouette_scores_data(self) -> pd.DataFrame:
        """
        Returns a DataFrame with silhouette scores data.
        """
        DF = self._get_shard_data('_silhouette_scores_data')
        if DF is None:
            return None
        return DF.merge(self.metadata, on='Sample', how='left')

    # pairwise correlations
    @property
    def df_mn_pwc(self) -> pd.DataFrame:
        """
        Returns a DataFrame with pairwise correlation results.
        """
        return self._df_mn_pwc

    @property
    def df_mn_pwc_intra(self) -> pd.DataFrame:
        """
        Returns a DataFrame with pairwise correlation results for intra-sample comparisons.
        """
        return self._df_mn_pwc_intra
    
    @property
    def df_mn_pwc_inter(self) -> pd.DataFrame:
        """
        Returns a DataFrame with pairwise correlation results for inter-sample comparisons.
        """
        return self._df_mn_pwc_inter

    @property
    def df_mn_pwc_all(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary of pairwise correlation DataFrames.
        """
        return {
            'all': self.df_mn_pwc,
            'intra': self.df_mn_pwc_intra,
            'inter': self.df_mn_pwc_inter
        }

    #-- dunders --#
    def __str__(self) -> str:
        """
        Returns the input file summary table as a string
        """
        return self.input_files.to_string()

    __repr__ = __str__