import numpy as np

def zero_initialize(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


    return parameters

def relu_initialize(layer_dims):             #relu initialization
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
        parameters['b' + str(l)] = np.random.randn(layer_dims[l], 1) * 0.1    #np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


    return parameters


def xavier_initialize(layer_dims):             #xavier initialization
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1. / layer_dims[l-1])
        parameters['b' + str(l)] = np.random.randn(layer_dims[l], 1) * 0.1

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


    return parameters


def initialize(layer_dims, initializer = 'random'):
    parameters = {}

    if initializer == 'random':
        parameters = relu_initialize(layer_dims)
    elif initializer == 'zeros':
        parameters = zero_initialize(layer_dims)
    elif initializer == 'xavier':
        parameters = xavier_initialize(layer_dims)

    return parameters
