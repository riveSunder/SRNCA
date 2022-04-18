import unittest

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

if __name__ == "__main__":

    unittest.main(verbosity=2)
