import random

import mnist_loader
import numpy as np


class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        """
        Call nan_to_num so that numpy deals correctly with ln number, which is very close to zero
        :param a:
        :param y:
        :return:
        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return a - y


class QuadraticCost:
    @staticmethod
    def fn(a, y):
        # return 0.5 * np.linalg.norm(a - y) ** 2
        return 0.5 * (a - y).dot(a - y)

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)


def sigmoid(z):  # sigmoid function
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):  # sigmoid prime
    return sigmoid(z) * (1.0 - sigmoid(z))


class ConvLayer:
    def __init__(self, filter_size, fn_activation=sigmoid):
        self.filter_size = filter_size
        self.weight_array = np.random.normal(size=(filter_size, filter_size))
        self.bias = np.random.normal()
        self.fn_activation = fn_activation

    def forward_helper(self, inp, stride=1):
        """
        :param inp:
        :return:
        """
        output_size = inp.shape[0] - self.filter_size + 1
        output = np.zeros([output_size, output_size])
        weight_len = self.filter_size * self.filter_size
        weight = self.weight_array.reshape([1, weight_len])
        for i in range(0, output_size, stride):
            for j in range(0, output_size, stride):
                output[i][j] = np.dot(weight,
                                      inp[i:i + self.filter_size][:, j:j + self.filter_size].reshape([weight_len, 1])) + \
                               self.bias
        return self.fn_activation(output)

    def max_pooling(self, inp, pool_size=2):
        output_size = inp.shape[0] / pool_size
        output = np.zeros([output_size, output_size])
        for i in range(output_size):
            for j in range(output_size):
                output[i][j] = np.max(inp[i * pool_size:(i + 1) * pool_size][:, j * pool_size:(j + 1) * pool_size])
        return output

    def forward(self, inp, stride=1, pool_size=2):
        output = self.forward_helper(inp, stride)
        self.input_array_shape = inp.shape
        self.input_array_size = inp.shape[0]
        return self.max_pooling(output, pool_size)

    def backprop(self, delta_array, inp, stride=1):
        """
        delta_array is a (input_array_size - filter_size + 1) x 1. It should be the result of reshaping the max-pooled
        output matrix
        :return:
        delta_array_last: a input_array_size x input_array_size
        """
        # forward step
        output_size = inp.shape[0] - self.filter_size + 1
        input_size = inp.shape[0]
        weighted_input = np.zeros([output_size, output_size])
        weight_len = self.filter_size * self.filter_size
        weight = self.weight_array.reshape([1, weight_len])
        for i in range(0, output_size, stride):
            for j in range(0, output_size, stride):
                weighted_input[i][j] = np.dot(weight,
                                              inp[i:i + self.filter_size][:, j:j + self.filter_size].reshape(
                                                  [weight_len, 1])) + \
                                       self.bias
        activation = self.fn_activation(weighted_input)
        delta_array_size = int(np.sqrt(delta_array.shape[0]))
        delta_array = delta_array.reshape(delta_array_size, delta_array_size)
        weight_gradient_array = np.zeros((self.filter_size, self.filter_size))
        print('input size', inp.shape)
        print('delta array shape', delta_array.shape)
        print('delta array size', delta_array_size)
        for a in range(self.filter_size):
            for b in range(self.filter_size):
                weight_gradient = 0
                for i in range(delta_array_size):
                    for j in range(delta_array_size):
                        # print(a, b, i, j)
                        x = delta_array[i][j] * sigmoid_prime(weighted_input[i][j]) * inp[i + a][j + b]
                        weight_gradient += x
                weight_gradient_array[a][b] = weight_gradient

        back_delta_array = np.zeros((input_size, input_size))
        for i in range(input_size):
            for j in range(input_size):
                for a in range(self.filter_size):
                    for b in range(self.filter_size):
                        if not (0 <= i - a <= input_size - self.filter_size) or not (
                                        0 <= j - b <= input_size - self.filter_size):
                            continue
                        back_delta_array[i][j] += delta_array[i - a][j - b] * sigmoid_prime(inp[i - a][j - b]) * \
                                                  self.weight_array[a][b]
        back_delta_array = back_delta_array / (input_size * input_size)
        print(back_delta_array)
        # getting average of gradient array
        weight_gradient_array = weight_gradient_array / (self.filter_size * self.filter_size)
        print(weight_gradient_array)
        return weight_gradient_array, back_delta_array


class Network:
    def __init__(self, layer_sizes, cost=CrossEntropyCost):
        self.weights = []
        self.biases = []
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.normal(size=(output_size, input_size)) for input_size, output_size in
                        zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.normal(size=(output_size, 1)) for input_size, output_size in
                       zip(layer_sizes[:-1], layer_sizes[1:])]
        self.cost = cost

    def forward(self, x):
        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def backprop(self, x, y):
        a = x
        zs = []
        activations = [x]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta  # important
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # important
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_w, nabla_b)

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Stochastic gradient descent: estimate delta C based on a sample of smaller training size. Though it still going
        through all the examples, the weights & biases are updated faster, leads to faster training
        :param training_data:
        :param epochs:
        :param mini_batch_size:
        :param eta:
        :param test_data: if presents print validation %
        :return:
        """
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        For each mini batch, do full feed forward and then backprop
        :param mini_batch:
        :param eta: learning rate
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            # sum all delta weights and delta biases for each training example
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # update the weights and biases using gradient descent
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]


if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print('loaded data', len(training_data))

    n = len(training_data[0][0])
    network = Network([n, 30, 10])
    network.sgd(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
    print(network.evaluate(test_data))
