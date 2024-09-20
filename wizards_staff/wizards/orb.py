import os
import sys
import logging
from typing import Callable, Dict, Any, Generator, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from tifffile import imread

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data item mapping (how to load each data item)
DATA_ITEM_MAPPING = {
    'cnm_A': {
        'suffixes': ['cnm-A.npy'],
        'loader': lambda x: np.load(x, allow_pickle=True)
    },
    'cnm_C': {
        'suffixes': ['cnm-C.npy'],
        'loader': lambda x: np.load(x, allow_pickle=True)
    },
    'cnm_S': {
        'suffixes': ['cnm-S.npy'],
        'loader': lambda x: np.load(x, allow_pickle=True)
    },
    'cnm_idx': {
        'suffixes': ['cnm-idx.npy'],
        'loader': lambda x: np.load(x, allow_pickle=True)
    },
    'df_f0_graph': {
        'suffixes': ['df-f0-graph.tif'],
        'loader': lambda x: imread(x)
    },
    'dff_f_mean': {
        'suffixes': ['dff-f-mean.npy'],
        'loader': lambda x: np.load(x, allow_pickle=True)
    },
    'f_mean': {
        'suffixes': ['f-mean.npy'],
        'loader': lambda x: np.load(x, allow_pickle=True)
    },
    'minprojection': {    # aka: im_min
        'suffixes': ['minprojection.tif'],
        'loader': lambda x: imread(x)
    },
    'mask': {  
        'suffixes': ['masks.tif'],
        'loader': lambda x: imread(x)
    }
}

#-- classes --#
@dataclass
class Shard:
    """
    A per-sample component of a Wizard Orb object.
    """
    sample_name: str
    metadata: pd.DataFrame
    files: dict
    _data_items: dict = field(default_factory=dict, init=False) 
    
    def get_data_item(self, item_name: str) -> Any:
        """
        Retrieves the data item for the given name, loading it if not already loaded.
        """
        if item_name not in self._data_items:
            # load file
            file_info = self.files.get(item_name)
            if file_info:
                file_path, loader = file_info
                # check that file exists
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                # load via the loader
                try:
                    self._data_items[item_name] = loader(file_path)
                except Exception as e:
                    logger.error(f"Failed to load data item '{item_name}' for sample '{self.sample_name}': {e}")
                    return None
            else:
                logger.warning(f"File not found for data item '{item_name}' and sample '{self.sample_name}'")
                return None
        return self._data_items[item_name]

    def has_file(self, item_name: str) -> bool:
        """
        Checks if a data item is available for this sample.
        """
        return item_name in self.files

    def get_data_items(self) -> Generator[str, None, None]:
        """
        Yields all data item names available for this sample.
        """
        yield from self.files.keys()

    def items(self) ->  Generator[Tuple[str, Any], None, None]:
        """
        Yields tuples of data item names and their loaded data.
        """
        for key in self._data_items:
            yield key, self._data_items[key]

    def get(self, item_name: str) -> Any:
        """
        Retrieves the data item, loading it if necessary.
        """
        return self.get_data_item(item_name)

    def __str__(self):
        """
        Prints data_item_name : file_path for this shard.
        """
        ret = []
        for data_item_name, (file_path, _) in self.files.items():
            ret.append(f"{data_item_name} : {file_path}")
        return '\n'.join(ret)

@dataclass
class Orb:
    """
    Represents a collection of samples and their associated data shards.
    """
    results_folder: str
    metadata_file_path: str
    metadata: pd.DataFrame = field(init=False)
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
        # load files 
        #self._file_paths = defaultdict(dict)
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
            file_suffix = file_basename[len(sample_name)+1:]
            ## categorize file based on suffix
            for item_name, data_info in self._data_mapping.items():
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

    def list_samples(self) -> list:
        """
        Returns a list of all sample names.
        """
        return list(self._shards.keys())

    def list_data_items(self, sample: str) -> list:
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

    def __str__(self):
        """
        Prints sample : data_item_name : file_path for all shards.
        """
        ret = []
        for sample_name, shard in self._shards.items():
            for data_item_name, (file_path, _) in shard.files.items():
                ret.append(f"{sample_name} : {data_item_name} : {file_path}")
        return '\n'.join(ret)

    @staticmethod
    def _list_files(indir):
        files = []
        for dirpath, dirnames, filenames in os.walk(indir):
            for filename in filenames:
                files.append(os.path.join(dirpath, filename))
        return files

    # def __getattr__(self, data_item):
    #     if data_item in self._data_mapping:
    #         # lazy loading
    #         if data_item not in self._data:
    #             self._data[data_item] = {}
    #             # load data for each sample
    #             for sample,file_path in self.file_paths.get(data_item, {}).items():
    #                 if file_path is not None:
    #                     self._data[data_item][sample] = self._load_file(file_path)
    #                 else:
    #                     self._data[data_item][sample] = None
    #         return self._data[data_item]
    #     else:
    #         raise AttributeError(f"'WizardOrb' object has no attribute '{data_item}'")

