# disable multithreading for numpy
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# import the Orb class
from .wizards.orb import Orb
__all__ = ['Orb']