import unittest
import numpy as np
from mnist import ConvLayer


class TestConvLayer(unittest.TestCase):
    def test_forward_helper(self):
        size = 10
        conv_size = 5
        input = np.eye(size)
        conv_layer = ConvLayer(conv_size)
        result = conv_layer.forward_helper(input)
        self.assertEqual(result.shape, (5, 5))

    def test_max_pooling(self):
        size = 4
        inp = np.arange(size * size).reshape(size, size)
        conv_layer = ConvLayer(5)
        output = conv_layer.max_pooling(inp)
        print(output)
        self.assertTrue(np.all(np.equal(np.asarray([5, 7, 13, 15]).reshape(2, 2), output)))