import numpy as np
import math
import random
from pprint import pprint
import matplotlib
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

def normalize(xs):
    result = []
    min_value = min(xs)
    max_value = max(xs)
    for x in xs:
        result.append((x-min_value)*1.0/(max_value-min_value))
    return min_value, max_value, result

def denormalize(x, minx, maxx):
    return x * (maxx - minx) + minx

def f(x, weights):
    return np.dot(weights, np.array([x**3, x**2, x, 1]).transpose() )


xs, ys = generate_values()
minx, maxx, xs = normalize(xs)
xs = np.asarray(xs)
weights = np.array([4, 5, 10, -10])
max_epoc = 5*1e1
learning_rate = 1.2*1.e-4
epoc = 0
while epoc < max_epoc:
    gradient = np.zeros(4)
    cost = 0
    a = np.zeros([len(xs), 4])
    for i in range(4):
        a[:, i] = xs**(3-i)
    error = np.dot(a, weights) - ys
    error = sum(error)
    print error
    import sys; sys.exit()
    for x, y in zip(xs, ys):
        error = f(x, weights) - y
        cost += error * error
        derivatives = np.array([x**3, x**2, x, 1])
        gradient += error/100.0 * derivatives
    weights = weights - gradient * learning_rate
    for var in 'epoc cost gradient weights'.split():
       print "%s = %3s; " % (var, vars()[var]), 
    print 
    epoc += 1
x1 = -1
x1p = (x1 - minx)*1.0/(maxx-minx)
y1 = f(x1, weights)
print 'test output = %s, expected = %s, error = %s' % (y1, label(x1), y1-label(x1))
