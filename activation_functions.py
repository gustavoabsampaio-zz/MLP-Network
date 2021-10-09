import numpy as np

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def relu(x):
    return np.maximum(0, x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def sigmoid_prime(x):
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds

def relu_prime(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
    xs = x - np.max(x)
    num = np.exp(xs)
    den = np.sum(num, axis = 1).reshape(-1, 1)
    return num/den

def softmax_prime(x):
    s = x.reshape(-1, 1)
    J = np.diagflat(s) - np.dot(s, s.T)
    return J