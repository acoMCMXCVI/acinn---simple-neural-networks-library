import numpy as np

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



def stochastic_gradient_descent(parameters, grads, learning_rate):


    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters


def make_dev_minibatch_sets(X, Y, mini_batch_size, validation_split):

    m = X.shape[-1]
    mini_batches = []

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))

    # Create list of data indexes for dev and traint sets
    dev_permutation = permutation[ : int(m * validation_split) ]
    train_permutation = permutation[ int(m * validation_split) : ]

    # Create dev set
    shuffled_dev_X = X[:, dev_permutation]
    shuffled_dev_Y = Y[:, dev_permutation].reshape((1,-1))

    dev_set = (shuffled_dev_X, shuffled_dev_Y)


    # Create train set
    shuffled_train_X = X[:, train_permutation]
    shuffled_train_Y = Y[:, train_permutation].reshape((1,-1))

    m_t = shuffled_train_X.shape[-1]

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(m_t / mini_batch_size) # math.floor(m / mini)  number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_train_X[:, k * mini_batch_size : (k + 1) * mini_batch_size ]
        mini_batch_Y = shuffled_train_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size ]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size) Dodajemo mini batch koji moze da bude i manji od pune vrednosti
    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_train_X[:, - ( m_t - mini_batch_size * int(m_t/mini_batch_size)) :]
        mini_batch_Y = shuffled_train_Y[:, - ( m_t - mini_batch_size * int(m_t/mini_batch_size)) :]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches, dev_set
