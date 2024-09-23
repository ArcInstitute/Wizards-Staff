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

# classes
@dataclass
class Shard:
    """
    A per-sample component of a Wizard Orb object.
    """
    sample_name: str
    metadata: pd.DataFrame
    files: dict
    _rise_time_data: list = field(default_factory=list, init=False)
    _fwhm_data: list = field(default_factory=list, init=False)
    _frpm_data: list = field(default_factory=list, init=False)
    _mask_metrics_data: list = field(default_factory=list, init=False)
    _silhouette_scores_data: list = field(default_factory=list, init=False)
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

    def items(self) -> Generator[Tuple[str, Any], None, None]:
        """
        Yields tuples of data item names and their loaded data.
        """
        for key in self._data_items:
            yield key, self._data_items[key]

    def get(self, item_name: str, req: bool=False, file_name: bool=False) -> Any:
        """
        Retrieves the data item, loading it if necessary.
        """
        # get data item
        if file_name:
            data_item = self.files.get(item_name)
        else:
            data_item = self.get_data_item(item_name)
        # check if required
        if req and data_item is None:
            raise ValueError(f"Data item '{item_name}' not found for sample '{self.sample_name}'")
        return data_item

    def __str__(self) -> str:
        """
        Prints data_item_name : file_path for this shard.
        """
        ret = []
        for data_item_name, (file_path, _) in self.files.items():
            ret.append(f"{data_item_name} : {file_path}")
        return '\n'.join(ret)

    # properties
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