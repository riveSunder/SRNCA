import os

import unittest

import numpy as np
import torch

from srnca.utils import seed_all, \
        read_image, \
        tensor_to_image, \
        image_to_tensor, \
        compute_grams

class TestSeedAll(unittest.TestCase):

    def setUp(self):
        pass
    
    def test_seed_all(self):

        my_seed = 42

        np.random.seed(my_seed)

        my_seeds = np.random.choice(np.arange(0, 100), size=8, replace=False)

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

        # file resides in folder two levels down from srnca root
        file_path = os.path.join("/", *os.path.abspath(__file__).split("/")[:-3])

        image_paths = [os.path.join(file_path, "data", "images", \
                "jwst_segment_alignment.jpg")]

        image_paths.append(os.path.join(file_path, "data", "images", \
                "frogs.png"))

        for max_size in [32, 64, 96]:

            image_from_file = read_image(image_paths[0], max_size=max_size)

            self.assertTrue(np.alltrue(\
                    image_from_file.shape == np.array([max_size,max_size, 3])))

            image_from_file = read_image(image_paths[1], max_size=max_size)

            self.assertTrue(np.alltrue(\
                    image_from_file.shape == np.array([max_size, max_size, 1])))

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

        # tensor from rgb image
        tensor_from_image = image_to_tensor(image_from_file) 
        self.assertEqual(len(tensor_from_image.shape), 4)

        # tensor from monochrome image
        tensor_from_2d_image = image_to_tensor(image_from_file[:,:,0]) 
        self.assertEqual(len(tensor_from_2d_image.shape), 4)

        # rgb image tensor case
        image_from_tensor_3d = tensor_to_image(tensor_from_image)
        self.assertEqual(len(image_from_tensor_3d.shape), 3)

        # monochrome image tensor case
        image_from_tensor_2d = tensor_to_image(tensor_from_2d_image)
        self.assertEqual(len(image_from_tensor_2d.shape), 2)

        # multichannel tensor case
        image_from_tensor_multi = tensor_to_image(torch.rand(1,32,16,17))
        self.assertEqual(len(image_from_tensor_multi.shape), 3)


class TestComputeGrams(unittest.TestCase):

    def setUp(self):
        pass

    def test_compute_grams(self):

        for number_channels in [1,3,7,16]:

            temp_a = torch.rand(16, number_channels, 32, 33)
            temp_b = torch.rand(16, number_channels, 32, 33)

            grams_a = compute_grams(temp_a)
            grams_aa = compute_grams(temp_a)
            grams_b = compute_grams(temp_b)

            for gram_a, gram_aa in zip(grams_a, grams_aa):
                self.assertTrue(torch.all(gram_aa == gram_a))

            for gram_a, gram_b in zip(grams_a, grams_b):
                self.assertFalse(torch.all(gram_a == gram_b))
            


if __name__ == "__main__": #pragma: no cover
    
    unittest.main(verbosity=2)
