import numpy as np

def zero_initialize(layers):
    parameters = {}
    L = len(layers)

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers[l].units, layers[l-1].units))
        parameters['b' + str(l)] = np.zeros((layers[l].units, 1))

        assert(parameters['W' + str(l)].shape == (layers[l].units, layers[l-1].units))
        assert(parameters['b' + str(l)].shape == (layers[l].units, 1))


    return parameters

def relu_initialize(layers):             #relu initialization
    parameters = {}
    L = len(layers)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers[l].units, layers[l-1].units) * np.sqrt(2. / layers[l-1].units)
        parameters['b' + str(l)] = np.random.randn(layers[l].units, 1) * 0.1    #np.zeros((layers[l].units, 1))

        assert(parameters['W' + str(l)].shape == (layers[l].units, layers[l-1].units))
        assert(parameters['b' + str(l)].shape == (layers[l].units, 1))


    return parameters


def xavier_initialize(layers):             #xavier initialization
    parameters = {}
    L = len(layers)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers[l].units, layers[l-1].units) * np.sqrt(1. / layers[l-1].units)
        parameters['b' + str(l)] = np.random.randn(layers[l].units, 1) * 0.1

        assert(parameters['W' + str(l)].shape == (layers[l].units, layers[l-1].units))
        assert(parameters['b' + str(l)].shape == (layers[l].units, 1))


    return parameters


def initialize(layers, initializer = 'random'):
    parameters = {}

    if initializer == 'random':
        parameters = relu_initialize(layers)
    elif initializer == 'zeros':
        parameters = zero_initialize(layers)
    elif initializer == 'xavier':
        parameters = xavier_initialize(layers)

    return parameters
