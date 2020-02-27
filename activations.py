import numpy as np

def sigmoid(Z):
    #function calculate sigmoid of Z

    A = 1 / ( 1 + np.exp(-Z) )
    return A, Z


def relu(Z):
    #function calculate relu of Z

    A = np.maximum( 0, Z)
    return A, Z

def relu_derivative(Z):

    gd = np.where(Z<=0, 0, 1)

    return gd

def sigmoid_derivative(Z):

    sigmZ, _ = sigmoid(Z)
    gd = sigmZ * (1 - sigmZ)

    return gd
