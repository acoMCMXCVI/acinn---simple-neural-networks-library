import numpy as np

def binary_crossentropy(AL, Y):

    # clip AL becouse log function
    eps = 1e-15
    ALclip = np.clip(AL, eps, 1 - eps)

    # compute loss from aL and y
    loss = -1. *  np.sum ( Y * np.log(ALclip) + (1-Y) * np.log (1-ALclip) )     # we dont use divide wiht m here, becouse minibatchs
    #loss = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    loss = np.squeeze(loss)      #[[17]] > 17
    assert(loss.shape == ())

    return loss

def categorical_crossentropy(AL, Y):

    # compute loss from aL and y
    loss = -1. *  np.sum ( Y * np.log(AL) )     # we dont use divide wiht m here, becouse minibatchs

    loss = np.squeeze(loss)      #[[17]] > 17
    assert(loss.shape == ())

    return loss

def mean_squared_error(AL, Y):

    m = Y.shape[1]

    # compute loss from aL and y
    loss = 1. / m * np.sum (np.power((Y-AL), 2))

    loss = np.squeeze(loss)      #[[17]] > 17
    assert(loss.shape == ())

    return loss


def model_loss(AL, Y, loss = 'mean_squared_error'):
    #function calculate loss of model

    if loss == 'mean_squared_error':
        c = mean_squared_error(AL, Y)
    elif loss == 'binary_crossentropy':
        c = binary_crossentropy(AL, Y)
    elif loss == 'categorical_crossentropy':
        c = categorical_crossentropy(AL, Y)
    else:
        print('Cost function is not fund')

    return c


def binary_crossentropy_derivative(Y, AL):
    # function caculate dAL with respect of loss function (sigmoid)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    return dAL

def categorical_crossentropy_softmax_derivative(Y, AL):
    # function caculate dZL with respect of loss function (softmax)
    dZ = AL - Y

    return dZ


'''
def regularization_cost(AL, parameters, layers):

    L = len(parameters) // 2          # number of layers in the neural network
    Y.shape[-1]

    L2_regularization_cost = 0

    for l in range(0, L):
        if layers[l].regularization != None:
            # tu treba dodati granjanje na L2 i L1
            L2_regularization_cost += np.sum(np.square(parameters['W' + str(l+1)]))*layers[l].regularization.lambd/(2*m)

    return L2_regularization_cost
'''
