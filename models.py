import numpy as np
from initializers import initialize
from forwards import model_forward
from backwards import model_backward
from losses import model_loss
from optimizers import Optimizer
from datasplit import make_dev_train_sets, make_m_batches

from checking import gradient_check

class Acinn:

    layers = []                     # layers of NN (Dense, Convolution,...)
    parameters = {}                 # parameters (W,b) of layers
    loss = ()                       # loss that we use
    optimizer = None                # optimizer of model

    # Function for add new layers to network
    def add(self, layer):
        if self.layers:
            layer.input_shape = self.layers[-1].units

        self.layers.append(layer)

        assert(self.layers[0].input_shape != None), 'Input shape is not defined'

    # Function for compile mode, initialize parms, loss, optimizer,...
    def compile(self, initializer = 'random', loss = 'mean_squared_error', optimizer = Optimizer()):

        self.parameters = initialize(self.layers, initializer)
        self.loss = loss
        self.optimizer = optimizer

    # Function for fitting model
    def fit(self, X, Y, batch_size = 32, epochs = 1, validation_split = 0., info=True):
        assert(X.shape[0] == self.layers[0].input_shape), 'Input shape of X is not equale to input shape of model'  # Check for input shape is same with model input

        costs = []

        train_set, dev_set = make_dev_train_sets(X, Y, validation_split)
        (X_train, Y_train) = train_set
        (X_dev, Y_dev) = dev_set

        for i in range(0, epochs):

            epoch_cost_total = 0
            dev_cost = 0

            minibatches = make_m_batches(X_train, Y_train, batch_size)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                AL, cashe = model_forward(minibatch_X, self.parameters, self.layers)

                epoch_cost_total += model_loss(AL, minibatch_Y, self.loss)

                gradients = model_backward(AL, minibatch_Y, cashe, self.layers, self.loss)

                #if i == 0 or i == 10 or i == 25000 or i == 1000 or i == 10000:
                    #gradient_check(self.parameters, gradients, self.layers, minibatch_X, minibatch_Y, self.loss)

                self.parameters = self.optimizer.optimize(self.parameters, gradients, i)

            epoch_cost_avg = epoch_cost_total / X_train.shape[-1]       # Ovde ukupan train loss delimo sa brojem examplova u train setu

            # calculating the loss of dev set
            if validation_split != 0.:
                AL_dev, _ = model_forward(X_dev, self.parameters, self.layers)
                dev_cost = model_loss(AL_dev, Y_dev, self.loss) / X_dev.shape[-1]   # ovde dev loss delimo sa brojem examplova u dev setu


            if info and i % 10 == 0:
                print ("Train cost after iteration %i: %f" %(i, epoch_cost_avg))
                print ("Dev cost after iteration %i: %f" %(i, dev_cost))

            if i % 10 == 0:
                costs.append((epoch_cost_avg, dev_cost))

        return costs



    def predict(self, x):

        AL, cache = model_forward(x, self.parameters, self.layers)
        predictions = AL > 0.5

        return predictions



    def lay(self):
        L = len(self.layers)

        for l in range(0,L):
            print(self.layers[l].input_shape)
            print(self.layers[l].units)
