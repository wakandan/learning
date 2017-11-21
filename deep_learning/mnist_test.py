import unittest
import numpy as np
from mnist import ConvLayer


class TestConvLayer(unittest.TestCase):
    def test_forward_helper(self):
        size = 10
        conv_size = 5
        inp = np.arange(size * size).reshape(size, size)
        conv_layer = ConvLayer(conv_size)
        result = conv_layer.forward_helper(inp)
        self.assertEqual(result.shape, (6, 6))

    def test_max_pooling(self):
        size = 4
        inp = np.arange(size * size).reshape(size, size)
        conv_layer = ConvLayer(5)
        output = conv_layer.max_pooling(inp)
        self.assertTrue(np.all(np.equal(np.asarray([5, 7, 13, 15]).reshape(2, 2), output)))

    def test_forward(self):
        size = 10
        inp = np.random.normal(size=(size, size))
        conv_layer = ConvLayer(size=3)
        output = conv_layer.forward(inp)
        self.assertEqual(output.shape, (4, 4))
