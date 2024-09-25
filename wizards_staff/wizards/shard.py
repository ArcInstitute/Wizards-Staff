# import
## batteries
import os
import sys
import logging
from typing import Callable, Dict, Any, Generator, Tuple, List, Optional
from dataclasses import dataclass, field
## 3rd party
import numpy as np
import pandas as pd
## package
from wizards_staff.logger import init_custom_logger
from wizards_staff.wizards.spellbook import convert_f_to_cs as ws_convert_f_to_cs

# classes
@dataclass
class Shard:
    """
    A per-sample component of a Wizard Orb object.
    """
    sample_name: str
    metadata: pd.DataFrame
    files: dict
    _input_items: dict = field(default=None, init=False)
    _input_files: pd.DataFrame = field(default=None, init=False)
    _logger: Optional[logging.Logger] = field(default=None, init=False)
    _rise_time_data: list = field(default_factory=list, init=False)
    _fwhm_data: list = field(default_factory=list, init=False)
    _frpm_data: list = field(default_factory=list, init=False)
    _mask_metrics_data: list = field(default_factory=list, init=False)
    _silhouette_scores_data: list = field(default_factory=list, init=False)
    
    def __post_init__(self):
        self._logger = init_custom_logger(__name__)
        self._input_items = {}

    def get_input(self, item_name: str, req: bool=False) -> Any:
        """
        Retrieves the input item for the given name, loading it if not already loaded.
        """
        if item_name not in self._input_items:
            # load file
            file_info = self.files.get(item_name)
            if file_info:
                # unpack file info
                file_path, loader = file_info
                # check that file exists
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                # load via the loader
                try:
                    # cache the loaded data
                    self._input_items[item_name] = loader(file_path)
                except Exception as e:
                    logger.error(
                        f"Failed to load input item '{item_name}' for sample '{self.sample_name}': {e}"
                    )
                    return None
            else:
                msg = f"Input input'{item_name}' not found for '{self.sample_name}'"
                if req:
                    raise ValueError(msg)
                self._logger.warning(msg)
                return None
        return self._input_items[item_name]

    def has_file(self, item_name: str) -> bool:
        """
        Checks if a data item is available for this sample.
        """
        return item_name in self.files

    #-- data analysis --#
    

    #-- properties --#
    @property
    def input_files(self) -> pd.DataFrame:
        if self._input_files is None:
            # get all input files from all shards
            ret = []
            for data_item_name, (file_path, _) in self.files.items():
                ret.append([self.sample_name, data_item_name, file_path])
            # convert to a DataFrame
            self._input_files = pd.DataFrame(
                ret, columns=['Sample', 'DataItem', 'FilePath']
            )
        return self._input_files

    @property
    def rise_time_data(self):
        return self._rise_time_data
    
    @property
    def fwhm_data(self):
        return self._fwhm_data
    
    @property
    def frpm_data(self):
        return self._frpm_data

    @property
    def mask_metrics_data(self):
        return self._mask_metrics_data
    
    @property
    def silhouette_scores_data(self):  
        return self._silhouette_scores_data

    #-- dunders --#
    def __str__(self) -> str:
        """
        Prints data_item_name : file_path for this shard.
        """
        return self.input_files.to_string()

    __repr__ = __str__


    # def _get_data(self, attr_name: str) -> pd.DataFrame:
    #     """Dynamically generate a DataFrame for the given attribute from shards."""
    #     print('here');
    #     attr = getattr(self, attr_name)
    #     if attr is None or :
    #         # Create DataFrame if it doesn't exist
    #         shard_data = getattr(self, attr_name, None)
    #         if shard_data is not None:
    #             return None
    #         print(shard_data); exit()
    #         attr = pd.DataFrame(shard_data)
    #         print(attr); exit()
    #         # Cache the result
    #         setattr(self, attr_name, attr)  
    #     return attr