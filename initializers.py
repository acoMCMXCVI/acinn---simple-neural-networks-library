import numpy as np

def zero_initialize(layers):
    parameters = {}
    L = len(layers)

    for l in range(1, L+1):
        parameters['W' + str(l)] = np.zeros((layers[l-1].units, layers[l-1].input_shape))
        parameters['b' + str(l)] = np.zeros((layers[l-1].units, 1))

        assert(parameters['W' + str(l)].shape == (layers[l-1].units, layers[l-1].input_shape))
        assert(parameters['b' + str(l)].shape == (layers[l-1].units, 1))


    return parameters

def relu_initialize(layers):             #relu initialization
    parameters = {}
    L = len(layers)

    for l in range(1, L+1):
        parameters['W' + str(l)] = np.random.randn(layers[l-1].units, layers[l-1].input_shape) * np.sqrt(2. / layers[l-1].units)
        parameters['b' + str(l)] = np.random.randn(layers[l-1].units, 1) * np.sqrt(2. / layers[l-1].units)    #np.zeros((layers[l].units, 1))

        assert(parameters['W' + str(l)].shape == (layers[l-1].units, layers[l-1].input_shape))
        assert(parameters['b' + str(l)].shape == (layers[l-1].units, 1))


    return parameters


def xavier_initialize(layers):             #xavier initialization
    parameters = {}
    L = len(layers)

    for l in range(1, L+1):
        parameters['W' + str(l)] = np.random.randn(layers[l-1].units, layers[l-1].input_shape) * np.sqrt(1. / layers[l-1].input_shape)
        parameters['b' + str(l)] = np.random.randn(layers[l-1].units, 1) * np.sqrt(1. / layers[l-1].input_shape)

        assert(parameters['W' + str(l)].shape == (layers[l-1].units, layers[l-1].input_shape))
        assert(parameters['b' + str(l)].shape == (layers[l-1].units, 1))


    return parameters


def initialize(layers, initializer):
    parameters = {}

    if initializer == 'random' or initializer == 'relu':
        parameters = relu_initialize(layers)
    elif initializer == 'zeros':
        parameters = zero_initialize(layers)
    elif initializer == 'xavier':
        parameters = xavier_initialize(layers)
    else:
        print('Initializer is not found')

    return parameters
