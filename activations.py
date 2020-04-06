import numpy as np

def sigmoid(Z):
    #function calculate sigmoid of Z

    A = 1 / ( 1 + np.exp(-Z) )
    return A, Z


def relu(Z):
    #function calculate relu of Z

    A = np.maximum( 0, Z)
    return A, Z


def softmax(Z):
    #function calculate softmax of Z

    t = np.exp(Z)
    A = t / np.sum(t, axis = 0)

    return A, Z


def relu_derivative(Z):
    #function calculate derivative of relu

    dg = np.where(Z<=0, 0, 1)

    return dg

def sigmoid_derivative(Z):
    #function calculate derivative of sigmoid

    sigmZ, _ = sigmoid(Z)
    dg = sigmZ * (1 - sigmZ)

    return dg
