import numpy as np
from activations import sigmoid, relu

def linear_forward(A_prev, W, b):
    # function calculate Z of units in single layer

    Z = np.dot(W, A_prev) + b

    assert(Z.shape == (W.shape[0], A_prev.shape[1]))

    cache = (A_prev, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    # function calculate A  od units in single layer

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)


    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def model_forward(X, parameters, activations):
    # function calculate AL (last A in network)

    caches = []
    A = X
    L = len(parameters) // 2          # number of layers in the neural network


    for l in range(1, L+1):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = activations[l-1])
        caches.append(cache)

    AL = A

    assert(AL.shape == (1,X.shape[1]))

    return AL, caches
