import numpy as np
from activations import sigmoid, relu, softmax

def linear_forward(A_prev, W, b):
    # function calculate Z

    Z = np.dot(W, A_prev) + b

    assert(Z.shape == (W.shape[0], A_prev.shape[1]))

    cache = (A_prev, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    # function calculate A od units in single layer

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)


    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def model_forward(X, parameters, layers):
    # function calculate AL (last A in network)

    caches = []
    A = X
    L = len(parameters) // 2          # # layers


    for l in range(1, L+1):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = layers[l-1].activation)
        caches.append(cache)


    AL = A

    assert(AL.shape == (layers[l-1].units, X.shape[1]))

    return AL, caches
