import numpy as np
import math
import random
from pprint import pprint
#import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
np.seterr(all='raise')



# train a neuron to output 0 when input is 1
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def label(x):
    return x*0.5+10

def generate_values():
    xs = []
    ys = []
    for x in xrange(-50, 51):
        #x = random.uniform(-5, 5) 
        xs.append(x)
        ys.append(label(x))
    return xs, ys

scaler = StandardScaler()

def calc_derivatives(inp):
    """
    input x is an array of (1,)
    """
    derivatives = np.concatenate([inp**3, inp**2, inp, np.ones(inp.size).reshape(-1, 1)], axis=1)
    return derivatives

xs_dat, ys = generate_values()
xs_dat = np.asarray(xs_dat).reshape(-1, 1) #scaler only takes input of matrix
xs = scaler.fit_transform(xs_dat)
xs_powers = calc_derivatives(xs)
ys = np.asarray(ys)
ys = ys.reshape(-1, 1)
weights = np.asarray([2, 3, 10, -10])
max_epoc = 1e3
learning_rate = 1e-3
epoc = 0
while epoc < max_epoc:
    y = xs_powers.dot(weights.reshape(-1, 1))
    error = y - ys
    cost = np.sum(error**2)
    gradients = error.transpose().dot(xs_powers) 
    weights = weights - (gradients * learning_rate)
    for var in 'epoc cost weights'.split():
       print "%s = %3s; " % (var, vars()[var]), 
    print 
    epoc += 1
test_input = np.asarray([-1, 5, 10, 100]).reshape(-1, 1)
expected_output = label(test_input)
scaled_test_output = scaler.transform(test_input)
observed_output = calc_derivatives(scaled_test_output).dot(weights.reshape(-1, 1))
print 'test output = %s, expected = %s, error = %s' % (expected_output, observed_output, sum(observed_output-expected_output))
