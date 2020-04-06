import numpy as np

class Optimizer():

    optimizer = None
    learning_rate_init = None
    beta = None                     # beta for momentum
    beta1 = None
    beta2 = None
    decay = None

    t = 1                           # adam counter

    v = {}                          # velocity for momentum, Adam
    s = {}                          # second moment for RMSprop, Adam

    def __init__(self, optimizer = 'SGD', learning_rate = 0.001, beta = 0.9, beta1 = 0.9, beta2 = 0.99, decay = 0):

        self.optimizer = optimizer
        self.learning_rate_init = learning_rate
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay
        self.t = 1


    def optimize(self, parameters, grads, epoch):

        learning_rate = 1 / (1 + self.decay * epoch) * self.learning_rate_init          #standard learning decay ( da li jako remeti performanse? )

        if self.optimizer == 'SGD':

            parameters = stochastic_gradient_descent(parameters, grads, learning_rate)

        elif self.optimizer == 'momentum':

            if not self.v: self.v = initialize_velocity(parameters)
            parameters = momentum(parameters, grads, self.v, learning_rate, self.beta)

        elif self.optimizer == 'RMSprop':

            if not self.s: self.s = initialize_velocity(parameters)
            parameters = rms_prop(parameters, grads, self.s, learning_rate, self.beta)

        elif self.optimizer == 'Adam':

            if not self.s or not self.v:
                self.s = initialize_velocity(parameters)
                self.v = initialize_velocity(parameters)
            parameters = adam(parameters, grads, self.v, self.s, learning_rate, self.beta1, self.beta2, t = self.t, epsilon = 1e-8)

            self.t += 1

        return parameters



def stochastic_gradient_descent(parameters, grads, learning_rate):


    L = len(parameters) // 2 # # layers

    # update rule
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters


def momentum(parameters, grads, v, learning_rate, beta):


    L = len(parameters) // 2 # # layers

    # update rule
    for l in range(L):
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1-beta) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1-beta) * grads["db" + str(l+1)]

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]

    return parameters


def rms_prop(parameters, grads, s, learning_rate, beta, epsilon = 1e-8):

    L = len(parameters) // 2 # # layers

    # update rule
    for l in range(L):
        s["dW" + str(l+1)] = beta * s["dW" + str(l+1)] + (1 - beta) * (grads["dW" + str(l+1)]**2)
        s["db" + str(l+1)] = beta * s["db" + str(l+1)] + (1 - beta) * (grads["db" + str(l+1)]**2)

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)] / ( np.sqrt(s["dW" + str(l+1)]) + epsilon )
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)] / ( np.sqrt(s["db" + str(l+1)]) + epsilon )


    return parameters



def adam(parameters, grads, v, s, learning_rate, beta1, beta2, epsilon = 1e-8, t=1):

    v_corrected = {}
    s_corrected = {}

    L = len(parameters) // 2 # # layers

    # update rule
    for l in range(L):
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]

        # compute bias corrected first moment estimate
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - np.power(beta1, t))

        # moving average of the squared gradients
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * np.power(grads["dW" + str(l+1)], 2)
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * np.power(grads["db" + str(l+1)], 2)

        # compute bias corrected second raw moment estimate
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - np.power(beta2, t))

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / ( np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon )
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / ( np.sqrt(s_corrected["db" + str(l+1)]) + epsilon )


    return parameters

def initialize_velocity(parameters):

    L = len(parameters) // 2 # # layers
    v = {}

    # initialize velocity
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)

    return v
