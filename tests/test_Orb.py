# imports
import os
import pytest
import pandas as pd
import numpy as np
from tifffile import imread
from wizards_staff.wizards.orb import Orb
from wizards_staff.wizards.shard import Shard

# Fixture to initialize the Orb object
@pytest.fixture
def orb(setup_test_data):
    """
    Initialize the Orb object
    """
    metadata_path = os.path.join(setup_test_data, 'metadata.csv')
    orb = Orb(setup_test_data, metadata_path)
    return orb

# Test to ensure the Orb object is initialized correctly
def test_init(orb):
    """
    Assess the initialization of the Orb class
    """
    assert isinstance(orb.samples, set)
    samples = {'10xGCaMP-6wk-Baseline-Stream_Stream_G03_s1_FITC_full', '10xGCaMP-6wk-Baseline-Stream_Stream_F07_s1_FITC_full'}
    assert orb.samples == samples
    # assert attributes
    assert isinstance(orb.input, pd.DataFrame)
    assert isinstance(orb.metadata, pd.DataFrame)

def test_shatter(orb):
    """
    Test the shatter method of the Orb class
    """
    for shard in orb.shatter():
        assert isinstance(shard, Shard)

def test_run_all(orb):
    """
    Test the run_all method of the Orb class
    """
    orb.run_all(group_name="Well", show_plots=False, save_files=False, debug=True)
    # check the output
    result_names = {
        'rise_time_data', 'fwhm_data', 'frpm_data', 'mask_metrics_data',
        'silhouette_scores_data', 'df_mn_pwc', 'df_mn_pwc_intra', 'df_mn_pwc_inter'
    }
    for result_name,result_data in orb.results:
        assert result_name in result_names
        assert isinstance(result_data, pd.DataFrame)
