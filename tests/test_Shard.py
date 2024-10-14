# import
## batteries
import os
import unittest
from unittest.mock import patch, MagicMock, mock_open
## 3rd party
import pandas as pd
import numpy as np
from tifffile import imread
## package
from wizards_staff.wizards.shard import Shard

# testing
class TestShard(unittest.TestCase):
    def setUp(self):
        # Mock sample name and metadata
        self.sample_name = '10xGCaMP-6wk-Baseline-Stream_Stream_F07_s1'
        self.mock_metadata = pd.DataFrame({
            'Sample': [self.sample_name],
            'Well': ['A1'],
            'Frate': [30]
        })
        # Mock files dictionary
        input_dir = "/large_storage/multiomics/projects/lizard_wizard/test_output/Calcium_AAV-GCAMP_6wk_20240416/"
        self.mock_files = {
            'dff_dat': (
                os.path.join(input_dir, 'caiman_calc-dff-f0', f'{self.sample_name}_FITC_full_dff-dat.npy'), 
                np.load
            ),
            'cnm_A': (
                os.path.join(input_dir, 'caiman', f'{self.sample_name}_FITC_full_masked_cnm-A.npy'), 
                np.load
            ),
            'cnm_idx': (
                os.path.join(input_dir, 'caiman', f'{self.sample_name}_FITC_full_masked_cnm-idx.npy'), 
                np.load
            ),
            'minprojection': (
                os.path.join(input_dir, 'mask', f'{self.sample_name}_FITC_full_minprojection.tif'), 
                imread
            )
        }
        # Initialize Shard
        self.shard = Shard(self.sample_name, self.mock_metadata, self.mock_files)
    
    # Ensure that the Shard object is initialized correctly
    def test_init(self):
        self.assertEqual(self.shard.sample_name, self.sample_name)
        pd.testing.assert_frame_equal(self.shard.metadata, self.mock_metadata)
        self.assertEqual(self.shard.files, self.mock_files)
    
    # test self.get_input
    def test_get_input(self):
        for file_type in self.mock_files.keys():
            # Test reading a numpy file
            result = self.shard.get_input(file_type)
            self.assertIsInstance(result, np.ndarray)

if __name__ == '__main__':
    unittest.main()