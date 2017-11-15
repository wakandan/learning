import random

import mnist_loader
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print('loaded data', len(training_data))


def sigmoid(z):  # sigmoid function
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):  # sigmoid prime
    return sigmoid(z) * (1.0 - sigmoid(z))


class Network:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.normal(size=(output_size, input_size)) for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.normal(size=(output_size, 1)) for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:])]
        # for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            # weights = np.random.normal(size=(output_size, input_size))
            # self.weights.append(weights)
            # bias = np.random.normal(size=(output_size, 1))
            # self.biases.append(bias)

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
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])  # final detal_L
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


n = len(training_data[0][0])
network = Network([n, 30, 10])
network.sgd(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
print(network.evaluate(test_data))
