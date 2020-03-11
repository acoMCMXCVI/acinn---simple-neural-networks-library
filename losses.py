import numpy as np

def binary_crossentropy(AL, Y):

    m = Y.shape[1]

    # Compute loss from aL and y.
    loss = -1. / m * np.sum ( Y * np.log(AL) + (1-Y) * np.log (1-AL) )
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


def binary_crossentropy_derivative(Y, AL):

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    return dAL
