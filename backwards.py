import numpy as np
from activations import sigmoid_derivative, relu_derivative
from losses import binary_crossentropy_derivative, categorical_crossentropy_softmax_derivative

def relu_backward(dA, activation_cache):

    dZ = np.multiply(dA, relu_derivative(activation_cache))

    return dZ

def sigmoid_backward(dA, activation_cache):

    dZ = np.multiply(dA, sigmoid_derivative(activation_cache))

    return dZ


def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]


    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis = 1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)


    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db



def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)


    return dA_prev, dW, db



def model_backward(AL, Y, caches, layers, loss):

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]

    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL


    # Ovo mora odvojeno od for petlje zbog prvog prosledjivanja dA (dAL)
    current_cache = caches[L-1]

    if loss == 'binary_crossentropy':
        dAL = binary_crossentropy_derivative(Y, AL)
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = layers[L-1].activation)

    elif loss == 'mean_squared_error':
        dAL = binary_crossentropy_derivative(Y, AL)
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = layers[L-1].activation)

    elif loss == 'categorical_crossentropy':                                #ovde je malo drugacije resenje backwarda zato sto nije lako proracunati dAL, pa zatim dZL, nego odmah moramo dZL
        linear_cache, activation_cache = current_cache
        dZL = categorical_crossentropy_softmax_derivative(Y, AL)
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZL, linear_cache)


    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = layers[l].activation)

        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp


    return grads
