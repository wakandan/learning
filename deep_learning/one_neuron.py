import numpy as np
import matplotlib.pyplot as plt
# train a neuron to output 0 when input is 1
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

w = 2
b = 2
epoc = 0
max_epoc = 100000
x = 1
y = 0
learning_rate = 0.15
outputs = []
while epoc < max_epoc:
    z = w*x + b
    a = sigmoid(z)
    cost = a - y
    error = cost * sigmoid_prime(z)
    gradient_b = error
    gradient_w = error * a
    b = b - gradient_b * learning_rate
    w = w - gradient_w * learning_rate
    output = sigmoid(w*x + b)
    outputs.append(cost)
    print('epoc: %3s, b = %s, w = %s, cost = %s => output = %s' % (epoc, b, w, cost, output))
    epoc += 1
print x*w+b
print len(range(max_epoc))
plt.plot(range(max_epoc), outputs)
plt.show()
