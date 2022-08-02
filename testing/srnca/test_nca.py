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
                    for use_bias in [0,1]:

                    

                        nca = NCA(number_channels=channels, \
                                number_filters=filters, \
                                number_hidden=hidden,\
                                use_bias=use_bias, clamp=use_bias) 

                        expected_value = channels * hidden * filters \
                                + nca.use_bias * channels \
                                +  nca.use_bias * hidden + hidden*channels

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
        os.system(f"rm {os.path.split(this_filepath)[0]}/temp*_log_dict.npy")
        os.system(f"rm {os.path.split(this_filepath)[0]}/temp*.pt")


    def test_command_line(self):

        this_filepath = os.path.realpath(__file__)
        this_dir = os.path.split(this_filepath)[0]

        dir_list_0 = os.listdir(this_dir)
        test_tag = "test_delete"

        exp_tag = os.path.join(this_dir, test_tag)

        my_cmd = f"python -m srnca.nca -t {exp_tag} -a 2"
        os.system(my_cmd)

        dir_list_1 = os.listdir(this_dir)

        self.assertGreater(len(dir_list_1), len(dir_list_0))
        
        check_pt = False
        check_npy = False

        for elem in dir_list_1:
            if test_tag in elem and elem.endswith("npy"):
                check_npy = True
            if test_tag in elem and elem.endswith("pt"):
                check_pt = True

        self.assertTrue(check_npy)
        self.assertTrue(check_pt)

        cleanup_command = f"rm {exp_tag}*"
        os.system(cleanup_command)


        

if __name__ == "__main__": #pragma: no cover

    unittest.main(verbosity=2)
