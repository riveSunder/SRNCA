import os

import unittest

import numpy as np
import torch
from srnca.nca import NCA

class TestNCA(unittest.TestCase):

    def setUp(self):
        pass

    def test_count_parameters(self):
        

        for hidden in [2,8,32,64]:
            for channels in [1,3,5,15]:
                for filters in [3,4,5]:

                    expected_value = channels * hidden * filters \
                            +  hidden + hidden*channels

                    nca = NCA(number_channels=channels, \
                            number_filters=filters, \
                            number_hidden=hidden) 

                    number_parameters = nca.count_parameters()

                    self.assertEqual(number_parameters, expected_value)

    def test_save_parameters(self):
        
        nca_0 = NCA()
        nca_1 = NCA()
        
        root_path = os.path.join(os.path.split(\
                os.path.split(\
                os.path.split(os.path.abspath(__file__))[0])[0])[0])
        save_path = os.path.join(root_path, "parameters", "temp_test.pt")

        print("\n", root_path)
        print(save_path)

        nca_0.save_parameters(save_path)

        nca_1.load_parameters(save_path)

        for param_0, param_1 in zip(nca_0.parameters(), nca_1.parameters()):
            
            self.assertTrue(\
                    np.allclose(param_0.detach().numpy(), param_1.detach().numpy()))

if __name__ == "__main__":

    unittest.main(verbosity=2)
