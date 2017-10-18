import numpy as np
import math
import random
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler



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

def transform(x):
    return x*0.1

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

def f(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

scaler = StandardScaler()
xs_dat, ys = generate_values()
xs_dat = np.asarray(xs_dat)
xs_dat = xs_dat.reshape(-1, 1)
xs = scaler.fit_transform(xs_dat)
#minx, maxx, xs = normalize(xs)
a = 2 
b = 3
c = 10
d= -10
max_epoc = 1e3
learning_rate = 6*1.e-4
epoc = 0
while epoc < max_epoc:
    gradient_a = gradient_b = gradient_c = gradient_d = 0
    cost = 0
    for x, y in zip(xs, ys):
        error = f(x, a, b, c, d) - y
        cost += error * error
        gradient_a += error * x**3
        gradient_b += error * x**2
        gradient_c += error * x
        gradient_d += error 
    a -= gradient_a*learning_rate
    b -= gradient_b*learning_rate
    c -= gradient_c*learning_rate
    d -= gradient_d*learning_rate
    for var in 'epoc cost gradient_a a b c d'.split():
       print "%s = %3s; " % (var, vars()[var]), 
    print 
    epoc += 1
test_input = np.asarray([-1, 5, 10, 100])
test_input = test_input.reshape(-1, 1)
expected_output = label(test_input)
scaled_test_output = scaler.transform(test_input)
observed_output = f(scaled_test_output, a, b, c, d)
print 'test output = %s, expected = %s, error = %s' % (expected_output, observed_output, sum(observed_output-expected_output))
#plt.plot(xs, [label(x) for x in xs_dat], 'r-', [f(x, a, b, c, d) for x in xs])
#plt.show()
