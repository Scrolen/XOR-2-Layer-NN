import numpy as np
# Activation Functions and derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def sigmoid_derivative_from_activation(A):
    return A * (1 - A)

def relu(z):
    return np.maximum(0, z)

def relu_derivative_from_activation(A):
    # The derivative is 1 for any positive value, 0 otherwise.
    return A > 0
