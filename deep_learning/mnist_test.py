import unittest

import numpy as np

from mnist import ConvLayer, FullyConnectedLayer
from mnist import ConvoNetwork


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
        conv_layer = ConvLayer()
        output = conv_layer.forward(inp)
        self.assertEqual(output.shape, (4, 4))

    def test_backprop(self):
        size = 6
        inp = np.random.normal(size=(size, size))
        conv_layer = ConvLayer()
        output_size = (size - conv_layer.filter_size + 1) / 2
        delta_array_size = output_size * output_size
        delta_array = np.random.normal(size=(delta_array_size, 1))
        weight_gradient_result, past_delta_result = conv_layer.backprop(delta_array, inp)
        self.assertEqual(weight_gradient_result.shape, (conv_layer.filter_size, conv_layer.filter_size))
        self.assertEqual(past_delta_result.shape, (size, size))

    def test_max_position_array(self):
        size = 10
        inp = np.arange(size * size).reshape(size, size)
        conv_layer = ConvLayer()
        out = conv_layer.get_max_position(inp)
        self.assertEqual(len(out), 10)
        self.assertEqual(len(out[0]), 10)
        self.assertEqual(out[2][2], (3, 3))


class TestFullyConnected(unittest.TestCase):
    def test_init(self):
        layer = FullyConnectedLayer(10, 5)
        self.assertEqual((5, 10), layer.weight_array.shape)
        self.assertEqual((5, 1), layer.bias_array.shape)

    def test_forward(self):
        layer = FullyConnectedLayer(10, 5)
        inp = np.random.normal(size=(10, 1))
        forward = layer.forward(inp)
        print('forward, ', forward)
        self.assertEqual((5, 1), forward.shape)

    def test_backprop(self):
        layer = FullyConnectedLayer(10, 5)
        inp_array = np.random.normal(size=(10, 1))
        delta_array = np.random.normal(size=(5, 1))
        gradient_w_array, gradient_b_array, back_delta_array = layer.backprop(inp_array, delta_array)
        self.assertEqual((10, 1), back_delta_array.shape)
        self.assertEqual((10, 5), gradient_w_array.shape)
        self.assertEqual((5, 1), gradient_b_array.shape)


class TestConvoNetwork(unittest.TestCase):
    def test_forward(self):
        size = 4
        inp = np.random.normal(size=(size, 1))
        network = ConvoNetwork(layers=(
            FullyConnectedLayer(inp_size=size, out_size=6),
            FullyConnectedLayer(inp_size=6, out_size=3)
        ))
        activation = network.forward(inp)
        print(activation)
        self.assertEqual((3, 1), activation.shape)
