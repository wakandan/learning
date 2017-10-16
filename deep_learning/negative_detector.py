import numpy as np
def generate_data():
    xs = []
    ys = []
    for x in range(-10, 10):
        xs.append(x)
        if x < 0:
            ys.append(1)
        else:
            ys.append(0)
    return xs, ys

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

xs, ys = generate_data()

epoc = 0
MAX_EPOC = 2000
w = 1 
b = 1 
learning_rate = 0.1

while epoc < MAX_EPOC:
    epoc += 1
    gradient_b = gradient_w = 0
    cost = 0
    for x, y in zip(xs, ys):
        z = w*x+b
        a = sigmoid(z)
        cost += (a-y)**2
        gradient_w += 2*(a-y)*sigmoid_prime(z)*x 
        gradient_b += 2*(a-y)*sigmoid_prime(z)
    gradient_w /= len(xs)
    gradient_b /= len(xs)
    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b 
    for var in 'epoc gradient_w gradient_b cost w b'.split():
        print "%s = %s; " % (var, vars()[var]),
    print 

for test in (-20, -40, -1, 1, 30, 0):
    value = sigmoid(test*w+b)
    print 'test %s = %s => %s' % (test, value, value > 0.5)
