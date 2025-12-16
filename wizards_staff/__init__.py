# disable multithreading for numpy
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Hide INFO and WARNING messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimization warnings

# import the Orb class
from .wizards.orb import Orb

# import stats module for convenient access
from . import stats

# import drug_response module for baseline-normalized drug response analysis
from . import drug_response
from .drug_response import compare_baseline_dosing

__all__ = ['Orb', 'stats', 'drug_response', 'compare_baseline_dosing']