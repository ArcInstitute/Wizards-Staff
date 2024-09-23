# import
## batteries
import os
import sys
import logging
from typing import Callable, Dict, Any, Generator, Tuple, List
from dataclasses import dataclass, field
## 3rd party
import numpy as np
import pandas as pd
from tifffile import imread
## package
from wizards_staff.wizards.shard import Shard
from wizards_staff.pwc import run_pwc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Functions
def npy_loader(infile, allow_pickle=True):
    return np.load(infile, allow_pickle=allow_pickle)

# Data item mapping (how to load each data item)
DATA_ITEM_MAPPING = {
    # 'cn_filter': {
    #     'suffixes': ['cn-filter.npy'],
    #     'loader': lambda x: np.load(x, allow_pickle=True)
    # },
    # 'pnr_filter': {
    #     'suffixes': ['pnr-filter.npy'],
    #     'loader': lambda x: np.load(x, allow_pickle=True)
    # },
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
    _rise_time_data: pd.DataFrame = field(default=None, init=False)
    _fwhm_data: pd.DataFrame = field(default=None, init=False)
    _frpm_data: pd.DataFrame = field(default=None, init=False)
    _mask_metrics_data: pd.DataFrame = field(default=None, init=False)
    _silhouette_scores_data: pd.DataFrame = field(default=None, init=False)
    _shards: Dict[str, Shard] = field(default_factory=dict, init=False)   # loaded data
    _data_mapping: Dict[str, Any] = field(default_factory=lambda: DATA_ITEM_MAPPING, init=False)  # data item mapping
    
    def __post_init__(self):
        # load metadata
        self._load_metadata(self.metadata_file_path)
        self._samples = set(self.metadata['Sample'])
        # run categorization upon initialization
        self._categorize_files()   

    def _categorize_files(self):
        """
        Categorizes files into corresponding data items for each sample.
        """
        logger.info("Categorizing files...")
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

    def list_samples(self) -> List[str]:
        """
        Returns a list of all sample names.
        """
        return list(self._shards.keys())

    def list_data_items(self, sample: str) -> List[str]:
        """
        Lists all data items for a given sample.
        """
        shard = self._shards.get(sample)
        if shard:
            return list(shard.get_data_items())
        else:
            logger.warning(f"Sample '{sample}' not found")
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

    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame with all data items and file paths.
        """
        DF = []
        for sample_name, shard in self._shards.items():
            for data_item_name, (file_path, _) in shard.files.items():
                DF.append([sample_name, data_item_name, file_path])
        DF = pd.DataFrame(DF, columns=['Sample', 'DataItem', 'FilePath'])
        # merge with metadata
        return pd.merge(DF, self.metadata, on='Sample', how='left')

    def save_data(self, outdir: str, data_items: 
                  list=["rise_time_data", "fwhm_data", "frpm_data", 
                        "mask_metrics_data", "silhouette_scores_data"]
                 ) -> List[str]:
        """
        Saves data items to disk.
        Args:
            outdir: Output directory.
            data_items: List of data items to save.
        Returns:
            List of saved file paths.
        """
        logging.info(f"Saving data items to: {outdir}")
        # output directory
        if outdir != "" and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        # save each data item
        outfiles = []
        for data_item_name in data_items:
            # get property
            data = getattr(self, data_item_name)
            if data is None:
                logger.warning(f"Data item '{data_item_name}' not found")
                continue
            print(data.shape); 
            # write to disk
            outfile = os.path.join(outdir, data_item_name.replace("_", "-") + ".csv")
            data.to_csv(outfile, index=False)
            logger.info(f"'{data_item_name}' saved to: {outfile}")
            outfiles.append(outfile)
        return outfiles

    def run_pwc(self, **kwargs) -> None:
        """
        Runs pairwise correlation analysis on all samples.
        Args:
            **kwargs: Keyword arguments to pass to `run_pwc
        """
        # run PWC on each sample
        run_pwd(self, **kwargs)

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
            attr = pd.DataFrame(DF)
            # Cache the result
            setattr(self, attr_name, attr)  
        return attr

    def __str__(self) -> str:
        """
        Prints sample : data_item_name : file_path for all shards.
        """
        ret = []
        for sample_name, shard in self._shards.items():
            for data_item_name, (file_path, _) in shard.files.items():
                ret.append(f"{sample_name} : {data_item_name} : {file_path}")
        return '\n'.join(ret)

    @staticmethod
    def _list_files(indir) -> List[str]:
        files = []
        for dirpath, dirnames, filenames in os.walk(indir):
            for filename in filenames:
                files.append(os.path.join(dirpath, filename))
        return files

    # properties
    @property
    def num_shards(self):
        return len(self._shards)

    @property
    def shards(self):
        yield from self._shards.values()

    @property
    def rise_time_data(self):
        DF = self._get_shard_data('_rise_time_data')
        # explode columns, if they exist
        cols = ['Rise Times', 'Rise Positions']        
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        # return after merging with metadata
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def fwhm_data(self):
        DF = self._get_shard_data('_fwhm_data')
        # explode columns, if they exist
        cols = ['FWHM Backward Positions', 'FWHM Forward Positions', 'FWHM Values', 'Spike Counts']
        if all(col in DF.columns for col in cols):
            DF = DF.explode(cols)
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def frpm_data(self):
        DF = self._get_shard_data('_frpm_data')
        return DF.merge(self.metadata, on='Sample', how='left')

    @property
    def mask_metrics_data(self):
        DF = self._get_shard_data('_mask_metrics_data')
        return DF.merge(self.metadata, on='Sample', how='left')
    
    @property
    def silhouette_scores_data(self):
        DF = self._get_shard_data('_silhouette_scores_data')
        return DF.merge(self.metadata, on='Sample', how='left')
