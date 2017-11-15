import mnist_loader
import numpy as np
from pprint import pprint

#print len(training_data)
def s(z): # sigmoid function
    return 1/(1+np.exp(-z))

def sp(z): # sigmoid prime
    return s(z)*(1-s(z))

class Network:
    def __init__(self, layer_sizes):
        self.layers = []
        self.weights = []
        self.biases = []
        self.layers = layer_sizes
        for row, col in zip(layer_sizes[:-1], layer_sizes[1:]):
            weights = np.random.normal(size=(row, col))
            self.weights.append(weights)
            bias = np.random.normal(size=row)
            self.biases.append(bias)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

n = input_size = training_data[0][0]
network = Network([len(training_data[0][0]),3,1])
epoc = 0
max_epoc = 20
while epoc < max_epoc:
    epoc += 1
    for x, y in training_data:
        for i in network.sizes:
            pass

pprint(network.weights)
pprint(network.biases)
