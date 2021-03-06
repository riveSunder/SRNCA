import os

import unittest

import numpy as np
import torch
from srnca.nca import NCA

from testing.srnca.test_nca import TestNCA
from testing.srnca.test_utils import TestSeedAll, \
        TestReadImage, \
        TestTensorToImage, \
        TestComputeGrams

if __name__ == "__main__":

    unittest.main(verbosity=2)
