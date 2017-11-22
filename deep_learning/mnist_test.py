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
        conv_layer = ConvLayer(filter_size=3)
        output = conv_layer.forward(inp)
        self.assertEqual(output.shape, (4, 4))

    def test_backprop(self):
        size = 10
        inp = np.random.normal(size=(size, size))
        conv_layer = ConvLayer(filter_size=3)
        output_size = size - conv_layer.filter_size + 1
        delta_array_size = output_size * output_size
        delta_array = np.random.normal(size=(delta_array_size, 1))
        weight_gradient_result, past_delta_result = conv_layer.backprop(delta_array, inp)
        self.assertEqual(weight_gradient_result.shape, (conv_layer.filter_size, conv_layer.filter_size))
        self.assertEqual(past_delta_result.shape, (size, size))
