import os
import sys
from typing import Callable
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from tifffile import imread

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
    # 'cn_filter': {
    #     'suffixes': ['cn_filter'],
    #     'extensions': ['.npy', '.tif']
    # },
    # 'corr_pnr_histograms': {
    #     'suffixes': ['corr-pnr-histograms.tif']
    # },
    # 'pnr_filter': {
    #     'suffixes': ['pnr_filter'],
    #     'extensions': ['.npy', '.tif']
    # },
}

@dataclass
class Shard:
    """
    A file component of a Wizard Orb object
    """
    sample_name: str
    metadata: pd.DataFrame
    files: dict
    _data_items: dict = field(default_factory=dict, init=False) 
    
    def get_data_item(self, item_name: str) -> object:
        """
        Get data item for a sample.
        """
        if self._data_items.get(item_name) is None:
            # load the data
            file_path,loader = self._files.get(item_names, (None,None))
            if file_path is not None and loader is not None:
                self._data_items[item_name] = loader(file_path)
            else:
                #raise ValueError(f"File not found for data item '{data_item}' and sample '{sample}'")
                return None
        else:
            data = self._data.get(item_name, {})
            if data is None:
                print(f"File not found for data item '{item_name}' and sample '{self.sample}'", file=sys.stderr)
            return data

    def has_file(self, item_name: str) -> bool:
        """
        Check if a data item is available for a sample.
        """
        return item_name in self.files

    def get_data_items(self):
        for key in self._files.keys():
            yield key

    def items(self):
        for key, value in self._data_items.items():
            yield key, value

    def get(self, item_name):
        return self._data_items.get(key)


@dataclass
class Orb:
    """
    A class to represent a Wizard Orb.
    """
    results_folder: str
    metadata_file_path: str
    metadata: pd.DataFrame = field(init=False)
    _shards: dict = field(default_factory=dict, init=False)   # loaded data
    _data_mapping: dict = field(default_factory=lambda: DATA_ITEM_MAPPING, init=False)  # data item mapping
    
    def __post_init__(self):
        self._shards = dict()
        self._categorize_files()   # run categorization upon initialization

    def _categorize_files(self):
        """
        For all data items, categorize the files into the corresponding data items.
        """
        # Load metadata
        self._load_metadata(self.metadata_file_path)
        self._samples = set(self.metadata['Sample'].tolist())

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
            for data_item, data_info in self._data_mapping.items():
                suffixes = data_info['suffixes']
                if any(file_suffix.endswith(x) for x in suffixes):
                    try:
                        shard = self._shards[sample_name]
                    except KeyError:
                        meta = self.metadata[self.metadata['Sample'] == sample_name]
                        shard = Shard(sample_name, metadata=meta, files={})
                        self._shards[sample_name] = shard
                    # add file to shard
                    shard.files[data_item] = (file_path, data_info['loader'])
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
                msg = f"WARNING: No '{data_item}' files found for samples: {missing_samples}"
                print(msg, file=sys.stderr)
    
    def _load_metadata(self, metadata_file_path):
        # check if metadata file exists
        if not os.path.exists(metadata_file_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file_path}")
        # load metadata
        self.metadata = pd.read_csv(metadata_file_path)
        # check for required columns
        required_columns = ["Sample", "Well", "Frate"]
        for col in required_columns:
            if col not in self.metadata.columns:
                raise ValueError(f"Column '{col}' not found in metadata file: {metadata_file_path}")

    def list_samples(self):
        """
        List all samples
        """
        return list(self._shards.keys())

    def list_data_items(self, sample: str=):
        """
        List all data items for all samples
        """
        try:
            return list(self._shards.get(sample).get_data_items())
        except KeyError:
            print(f"Sample '{sample}' not found", file=sys.stderr)
            return []

    # iter over shards
    def shatter(self):
        for v in self._shards.values():
            yield v

    # iter over samples and shards
    def items(self):
        for sample, shard in self._shards.items():
            yield sample, shard 

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

