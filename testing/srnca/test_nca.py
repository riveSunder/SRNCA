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
                for filters in [3,4,5, 15]:

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

        nca_0.save_parameters(save_path)

        nca_1.load_parameters(save_path)

        for param_0, param_1 in zip(nca_0.parameters(), nca_1.parameters()):
            
            self.assertTrue(\
                    np.allclose(param_0.detach().numpy(), \
                    param_1.detach().numpy(), atol=1e-7))

        my_command = f"rm {save_path}"

        os.system(my_command)

    def test_to_device(self):

        nca = NCA()
        nca.to_device("cpu")
        nca.to_device("cuda")
        

    def test_fit(self):

        nca = NCA(number_channels=3)

        target = torch.rand(1,3,64,64)

        this_filepath = os.path.realpath(__file__)
        temp_tag = os.path.split(this_filepath)[0]

        temp_tag = os.path.join(temp_tag, "temp")
        nca.fit(target, max_steps=3, lr=1e-3, max_ca_steps=16, batch_size=4, exp_tag=temp_tag)


        find_log = False
        find_params = False
        for elem in os.listdir(os.path.split(this_filepath)[0]):
            if elem.startswith("temp") and elem.endswith(".npy"):
                find_log = True
            if elem.startswith("temp") and elem.endswith(".pt"):
                find_params = True

        self.assertTrue(find_log)
        self.assertTrue(find_params)
        import pdb; pdb.set_trace()
        os.system(f"rm {os.path.split(this_filepath)[0]}/temp*_log_dict.npy")
        os.system(f"rm {os.path.split(this_filepath)[0]}/temp*.pt")



if __name__ == "__main__": #pragma: no cover

    unittest.main(verbosity=2)
