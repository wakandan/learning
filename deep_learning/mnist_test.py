import unittest

import numpy as np

from mnist import ConvoLayer, FullyConnectedLayer
from mnist import ConvoNetwork


class TestConvLayer(unittest.TestCase):
    def test_forward_helper(self):
        size = 10
        conv_size = 5
        inp = np.arange(size * size).reshape(size, size)
        conv_layer = ConvoLayer(conv_size)
        result = conv_layer.forward_helper(inp)
        self.assertEqual(result.shape, (6, 6))

    def test_max_pooling(self):
        size = 4
        inp = np.arange(size * size).reshape(size, size)
        conv_layer = ConvoLayer(5)
        output = conv_layer.max_pooling(inp)
        self.assertTrue(np.all(np.equal(np.asarray([5, 7, 13, 15]).reshape(2, 2), output)))

    def test_forward(self):
        size = 10
        inp = np.random.normal(size=(size, size))
        conv_layer = ConvoLayer()
        output = conv_layer.forward(inp)
        self.assertEqual(output.shape, (4, 4))

    def test_backprop(self):
        size = 6
        inp = np.random.normal(size=(size, size))
        conv_layer = ConvoLayer()
        output_size = (size - conv_layer.filter_size + 1) / 2
        delta_array_size = output_size * output_size
        delta_array = np.random.normal(size=(delta_array_size, 1))
        weight_gradient_result, _, past_delta_result = conv_layer.backprop(inp, delta_array)
        self.assertEqual(weight_gradient_result.shape, (conv_layer.filter_size, conv_layer.filter_size))
        self.assertEqual(past_delta_result.shape, (size, size))

    def test_max_position_array(self):
        size = 10
        inp = np.arange(size * size).reshape(size, size)
        conv_layer = ConvoLayer()
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
    def test_forward_1(self):
        size = 4
        inp = np.random.normal(size=(size, 1))
        network = ConvoNetwork(layers=(
            FullyConnectedLayer(inp_size=size, out_size=6),
            FullyConnectedLayer(inp_size=6, out_size=3)
        ))
        activation = network.forward(inp)
        print(activation)
        self.assertEqual((3, 1), activation.shape)

    def test_backprop_1(self):
        inp_size = 4
        out_size = 3
        inp = np.random.normal(size=(inp_size, 1))
        out = np.random.normal(size=(out_size, 1))
        network = ConvoNetwork(layers=(
            FullyConnectedLayer(inp_size=inp_size, out_size=6),
            FullyConnectedLayer(inp_size=6, out_size=out_size)
        ))
        backprop = network.backprop(inp, out)

    def test_forward_2(self):
        size = 10
        inp = np.random.normal(size=(size, size))
        network = ConvoNetwork(layers=(
            ConvoLayer(),
            FullyConnectedLayer(inp_size=16, out_size=3)
        ))
        activation = network.forward(inp)
        print('activation', activation)
        print('activation shape', activation.shape)
        self.assertEqual((3, 1), activation.shape)

    def test_backprop_2(self):
        inp_size = 10
        out_size = 3
        inp_array = np.random.normal(size=(inp_size, inp_size))
        network = ConvoNetwork(layers=(
            ConvoLayer(),
            FullyConnectedLayer(inp_size=16, out_size=out_size)
        ))
        out_array = np.random.normal(size=(out_size, 1))
        network.backprop(inp_array, out_array)

    def test_forward_3(self):
        network = ConvoNetwork(layers=(
            FullyConnectedLayer(inp_size=5, out_size=100),
            ConvoLayer()
        ))
        size = 5
        inp = np.random.normal(size=(size, 1))
        activation = network.forward(inp)
        print('activation', activation)
        print('activation shape', activation.shape)
        self.assertEqual((4, 4), activation.shape)

    def test_backprop_3(self):
        network = ConvoNetwork(layers=(
            FullyConnectedLayer(inp_size=5, out_size=100),
            ConvoLayer()
        ))
        inp_size = 5
        out_size = 4
        inp_array = np.random.normal(size=(inp_size, 1))
        out_array = np.random.normal(size=(out_size, out_size))
        network.backprop(inp_array, out_array)

    def test_forward_4(self):
        network = ConvoNetwork(layers=(
            ConvoLayer(),
            ConvoLayer()
        ))
        inp = np.random.normal(size=(10, 10))
        activation = network.forward(inp)
        print('activation', activation)
        print('activation shape', activation.shape)
        self.assertEqual((1, 1), activation.shape)

    def test_backprop_4(self):
        network = ConvoNetwork(layers=(
            ConvoLayer(),
            ConvoLayer()
        ))
        inp = np.random.normal(size=(14, 14))
        out = np.random.normal(size=(2, 2))
        network.backprop(inp, out)
