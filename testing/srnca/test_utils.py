import os

import unittest

import numpy as np
import torch

from srnca.utils import seed_all, \
        read_image, \
        tensor_to_image, \
        image_to_tensor


class TestSeedAll(unittest.TestCase):

    def setUp(self):
        pass
    
    def test_seed_all(self):

        my_seed = 42

        np.random.seed(my_seed)

        my_seeds = np.random.choice(np.arange(0, 1000), size=8, replace=False)

        for my_seed in my_seeds:

            seed_all(my_seed)

            np_norm_sample_a = np.random.randn(31,32,33,34)
            np_uniform_sample_a = np.random.rand(31,32,33,34)
            np_int_sample_a = np.random.randint(1,5, size=(32,33))

            torch_norm_sample_a = torch.randn(31,32,33,34)
            torch_uniform_sample_a = torch.rand(31,32,33,34)
            
            seed_all(my_seed)

            np_norm_sample_b = np.random.randn(31,32,33,34)
            np_uniform_sample_b = np.random.rand(31,32,33,34)
            np_int_sample_b = np.random.randint(1,5, size=(32,33))

            torch_norm_sample_b = torch.randn(31,32,33,34)
            torch_uniform_sample_b = torch.rand(31,32,33,34)

            self.assertTrue(np.alltrue(np_norm_sample_a == np_norm_sample_b))
            self.assertTrue(np.alltrue(np_uniform_sample_a == np_uniform_sample_b))
            self.assertTrue(np.alltrue(np_int_sample_a == np_int_sample_b))

            self.assertTrue(torch.all(torch_norm_sample_a == torch_norm_sample_b))
            self.assertTrue(torch.all(torch_uniform_sample_a == torch_uniform_sample_b))

class TestReadImage(unittest.TestCase):

    def setUp(self):
        pass


    def test_read_image(self):

        # this level of testing is two levels down from root
        file_path = os.path.join("/", *os.path.abspath(__file__).split("/")[:-3])

        image_path = os.path.join(file_path, "data", "images", \
                "jwst_segment_alignment.jpg")

        for max_size in [32, 64, 96]:

            image_from_file = read_image(image_path, max_size=max_size)

            self.assertTrue(np.alltrue(\
                    image_from_file.shape == np.array([max_size,max_size, 3])))


class TestTensorToImage(unittest.TestCase):

    def setUp(self):
        # this folder is two levels down from root
        self.file_path = os.path.join("/", *os.path.abspath(__file__).split("/")[:-3])

        self.image_path = os.path.join(self.file_path, "data", "images", \
                "jwst_segment_alignment.jpg")
        
        max_size = 64

    def test_tensor_image_parsing(self):
        max_size = 64

        image_from_file = read_image(self.image_path, max_size=max_size)

        tensor_from_image = image_to_tensor(image_from_file) 

        self.assertEqual(len(tensor_from_image.shape), 4)

        image_from_tensor_3d = tensor_to_image(tensor_from_image)

        self.assertEqual(len(image_from_tensor_3d.shape), 3)

if __name__ == "__main__":
    
    unittest.main(verbosity=2)
