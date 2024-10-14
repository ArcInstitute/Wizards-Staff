# imports
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
from tifffile import imread
from wizards_staff.wizards.shard import Shard

# Fixtures for the test data
@pytest.fixture
def sample_name():
    return '10xGCaMP-6wk-Baseline-Stream_Stream_F07_s1'

@pytest.fixture
def mock_metadata(sample_name):
    return pd.DataFrame({
        'Sample': [sample_name],
        'Well': ['A1'],
        'Frate': [30]
    })

@pytest.fixture
def mock_files(setup_test_data, sample_name):
    input_dir = setup_test_data
    return {
        'dff_dat': (
            os.path.join(input_dir, 'caiman_calc-dff-f0', f'{sample_name}_FITC_full_dff-dat.npy'), 
            np.load
        ),
        'cnm_A': (
            os.path.join(input_dir, 'caiman', f'{sample_name}_FITC_full_masked_cnm-A.npy'), 
            np.load
        ),
        'cnm_idx': (
            os.path.join(input_dir, 'caiman', f'{sample_name}_FITC_full_masked_cnm-idx.npy'), 
            np.load
        ),
        'minprojection': (
            os.path.join(input_dir, 'mask', f'{sample_name}_FITC_full_minprojection.tif'), 
            imread
        )
    }

@pytest.fixture
def shard(sample_name, mock_metadata, mock_files, setup_test_data):
    return Shard(sample_name, mock_metadata, mock_files)

# Test to ensure the Shard object is initialized correctly
def test_init(shard, sample_name, mock_metadata, mock_files):
    assert shard.sample_name == sample_name
    pd.testing.assert_frame_equal(shard.metadata, mock_metadata)
    assert shard.files == mock_files

# Test the get_input method of the Shard class
def test_get_input(shard, mock_files):
    for file_type in mock_files.keys():
        result = shard.get_input(file_type)
        assert isinstance(result, np.ndarray)


