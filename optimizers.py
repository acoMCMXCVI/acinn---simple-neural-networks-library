

def stochastic_gradient_descent(parameters, grads, learning_rate):


    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters



class Optimizer():

    optimizer = None
    learning_rate = None
    beta = None
    beta1 = None
    beta2 = None

    def __init__(self, optimizer = 'SGD', learning_rate = 0.001, beta = 0.9, beta1 = 0.9, beta2 = 0.99):

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2


    def optimize(self, parameters, grads):

        if self.optimizer == 'SGD':
            parameters = stochastic_gradient_descent(parameters, grads, self.learning_rate)
        elif self.foptimizer == 'momentum':
            parameters = stochastic_gradient_descent(parameters, grads, self.learning_rate)

        return parameters
