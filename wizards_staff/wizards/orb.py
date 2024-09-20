import os
import sys
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

# @dataclass
# class Shard:
#     """
#     A file component of a Wizard Orb object
#     """
#     sample_name: str
#     file_path: str
#     data: object = field(init=False)
    
#     #def __post_init__(self):
#     #    self.data = load_file(self.file_path)


@dataclass
class Orb:
    """
    A class to represent a Wizard Orb.
    """
    results_folder: str
    metadata_file_path: str
    metadata: pd.DataFrame = field(init=False)
    _file_paths: dict = field(init=False)
    _data: defaultdict = field(default_factory=dict, init=False)   # loaded data
    _data_mapping: dict = field(default_factory=lambda: DATA_ITEM_MAPPING, init=False)  # data item mapping
    
    def __post_init__(self):
        self._file_paths = defaultdict(dict)
        self._categorize_files()   # run categorization upon initialization

    def _categorize_files(self):
        """
        For all data items, categorize the files into the corresponding data items.
        """
        # Load metadata
        self._load_metadata(self.metadata_file_path)

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
            if sample_name is None:
                continue
            ## suffix
            file_suffix = file_basename[len(sample_name)+1:]
            ## categorize file based on suffix and extension
            for data_item, info in self._data_mapping.items():
                suffixes = info['suffixes']
                if any(file_suffix.endswith(x) for x in suffixes):
                    self._file_paths[data_item][sample_name] = file_path
                    break

        # check for missing files
        for data_item, sample_info in self._data_mapping.items():
            missing_samples = []
            for sample in self.metadata['Sample'].tolist():
                if sample not in self._file_paths[data_item]:
                    missing_samples.append(sample)
            if len(missing_samples) > 0:
                missing_samples = ', '.join(missing_samples)
                msg = f"No '{data_item}' files found for samples: {missing_samples}"
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

    @staticmethod
    def _list_files(indir):
        files = []
        for dirpath, dirnames, filenames in os.walk(indir):
            for filename in filenames:
                files.append(os.path.join(dirpath, filename))
        return files

    # @staticmethod
    # def _load_file(file_path):
    #     if file_path.endswith('.npy'):
    #         return np.load(file_path, allow_pickle=True)
    #     elif file_path.endswith('.tif'):
    #         return imread(file_path)
    #     else:
    #         raise ValueError(f"Unknown file extension for file: {path}")

    def get_data_item(self, data_item, sample):
        if self._data.get(data_item, {}).get(sample) is None:
            # load the data
            file_path = self._file_paths.get(data_item, {}).get(sample)
            loader = self._data_mapping.get(data_item, {}).get('loader')
            if file_path is not None and loader is not None:
                self._data[data_item][sample] = loader(file_path)
            else:
                #raise ValueError(f"File not found for data item '{data_item}' and sample '{sample}'")
                return None
        else:
            data = self._data.get(data_item, {}).get(sample)
            if data is None:
                print(f"File not found for data item '{data_item}' and sample '{sample}'", file=sys.stderr)
            return data

    def list_data_items(self):
        return list(self._file_paths.keys())

    def list_samples(self):
        samples = dict()
        for data_item,sample_files in self._file_paths.items():
            samples[data_item] = sample_files.keys()
        return samples

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

