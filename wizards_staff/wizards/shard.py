# import
## batteries
import os
import sys
import logging
from typing import Callable, Dict, Any, Generator, Tuple, List, Optional
from dataclasses import dataclass, field
from functools import wraps
## 3rd party
import numpy as np
import pandas as pd
## package
from wizards_staff.logger import init_custom_logger
from wizards_staff.wizards.familiars import (
    spatial_filtering as ws_spatial_filtering
) 
from wizards_staff.wizards.spellbook import (
    convert_f_to_cs as ws_convert_f_to_cs,
    calc_rise_tm as ws_calc_rise_tm,
    calc_fwhm_spikes as ws_calc_fwhm_spikes,
    calc_frpm as ws_calc_frpm,
    calc_mask_shape_metrics as ws_calc_mask_shape_metrics,
)

# classes
@dataclass
class Shard:
    """
    A per-sample component of a Wizard Orb object.
    """
    sample_name: str
    metadata: pd.DataFrame
    allow_missing: bool = False
    files: Dict[str, Tuple[str, Callable[[str], Any]]] 
    quiet: bool = False
    _input_files: pd.DataFrame = field(default=None, init=False)
    _input_items: dict = field(default_factory=dict)
    _logger: Optional[logging.Logger] = field(default=None, init=False)
    _rise_time_data: list = field(default_factory=list, init=False)
    _fwhm_data: list = field(default_factory=list, init=False)
    _frpm_data: list = field(default_factory=list, init=False)
    _mask_metrics_data: list = field(default_factory=list, init=False)
    _silhouette_scores_data: list = field(default_factory=list, init=False)
    
    def __post_init__(self):
        self._logger = init_custom_logger(__name__)

    def get_input(self, item_name: str, req: bool=False) -> Any:
        """
        Retrieves the input item for the given name, loading it if not already loaded.
        Args:
            item_name: The name of the input item to retrieve.
            req: Whether the input item is required.
        Returns:
            The loaded input item, or None if the input item is not found and allow_missing is True.
        Raises:
            FileNotFoundError: If the input item is not found and allow_missing is False.
        """
        if item_name not in self._input_items:
            # load file
            file_info = self.files.get(item_name)
            if file_info:
                # unpack file info
                file_path, loader = file_info
                # check that file exists
                if not os.path.exists(file_path):
                    if self.allow_missing:
                        self._logger.warning(f"File not found: {file_path}; skipping due to --allow-missing")
                        return None
                    raise FileNotFoundError(f"File not found: {file_path}")
                # load via the loader
                try:
                    # cache the loaded data
                    self._input_items[item_name] = loader(file_path)
                except Exception as e:
                    self._logger.error(
                        f"Failed to load input item '{item_name}' for sample '{self.sample_name}': {e}"
                    )
                    return None
            else:
                msg = f"Input '{item_name}' not found for '{self.sample_name}'"
                if req:
                    raise ValueError(msg)
                self._logger.warning(msg)
                return None
        return self._input_items[item_name]

    def has_file(self, item_name: str) -> bool:
        """
        Checks if a data item is available for this sample.
        Args:
            item_name: The data item name to check.
        Returns:
            True if the data item is available, False otherwise.
        """
        return item_name in self.files

    #-- data analysis --#
    @wraps(ws_convert_f_to_cs)
    def convert_f_to_cs(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return ws_convert_f_to_cs(
            self.get_input('dff_dat', req=True) + 0.0001,
            *args, **kwargs
        )

    @wraps(ws_spatial_filtering)
    def spatial_filtering(self, *args, **kwargs) -> np.ndarray:
        return ws_spatial_filtering(
            *args,
            cnm_A=self.get_input('cnm_A', req=True), 
            cnm_idx=self.get_input('cnm_idx', req=True), 
            im_min=self.get_input('minprojection', req=True), 
            **kwargs
        )

    @wraps(ws_calc_rise_tm)
    def calc_rise_tm(self, calcium_signals, spike_events, *args, **kwargs
                     ) -> Tuple[np.ndarray, np.ndarray]:
        return ws_calc_rise_tm(
            calcium_signals, spike_events, *args, **kwargs
        )

    @wraps(ws_calc_fwhm_spikes)
    def calc_fwhm_spikes(self, calcium_signals, 
                         zscored_spike_events,
                         *args, **kwargs
                         ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
        return ws_calc_fwhm_spikes(
            calcium_signals, zscored_spike_events, *args, **kwargs
        )

    @wraps(ws_calc_frpm)
    def calc_frpm(self, zscored_spike_events, filtered_idx, frate, 
                  *args, **kwargs) -> float:
        return ws_calc_frpm(
            zscored_spike_events, filtered_idx, frate, *args, **kwargs
        )

    @wraps(ws_calc_mask_shape_metrics)
    def calc_mask_shape_metrics(self, *args, **kwargs) -> Dict[str, float]:
        return ws_calc_mask_shape_metrics(self.get_input('mask'), *args, **kwargs)

    #-- properties --#
    @property
    def input_files(self) -> pd.DataFrame:
        """
        Returns a DataFrame of input files for the shard object.
        """
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
        Prints the input files for the shard.
        """
        return self.input_files.to_string()

    __repr__ = __str__

