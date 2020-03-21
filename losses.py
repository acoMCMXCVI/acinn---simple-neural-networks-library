import numpy as np

def binary_crossentropy(AL, Y):

    eps = 1e-15
    ALclip = np.clip(AL, eps, 1 - eps)

    # Compute loss from aL and y.
    loss = -1. *  np.sum ( Y * np.log(ALclip) + (1-Y) * np.log (1-ALclip) )     # We dont use divide wiht m here, becouse minibatchs
    #loss = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    loss = np.squeeze(loss)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(loss.shape == ())

    return loss

def mean_squared_error(AL, Y):

    m = Y.shape[1]

    # Compute loss from aL and y.
    loss = 1. / m * np.sum (np.power((Y-AL), 2))

    loss = np.squeeze(loss)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(loss.shape == ())

    return loss


def model_loss(AL, Y, loss = 'mean_squared_error'):

    if loss == 'mean_squared_error':
        c = mean_squared_error(AL, Y)
    elif loss == 'binary_crossentropy':
        c = binary_crossentropy(AL, Y)

    return c

def regularization_cost(AL, parameters, layers):

    L = len(parameters) // 2          # number of layers in the neural network
    Y.shape[-1]

    L2_regularization_cost = 0

    for l in range(0, L):
        if layers[l].regularization != None:
            # tu treba dodati granjanje na L2 i L1
            L2_regularization_cost += np.sum(np.square(parameters['W' + str(l+1)])*layers[l].regularization.lambd/(2*m)


    return L2_regularization_cost



def binary_crossentropy_derivative(Y, AL):

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    return dAL


def binary_crossentropy_original(AL, Y):

    m = Y.shape[1]

    # Compute loss from aL and y.
    loss = -1. / m * np.sum ( Y * np.log(AL) + (1-Y) * np.log (1-AL) )
    #loss = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    loss = np.squeeze(loss)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(loss.shape == ())

    return loss
