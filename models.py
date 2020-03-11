import numpy as np
from initializers import initialize
from forwards import model_forward
from backwards import model_backward
from losses import model_loss
from optimizers import Optimizer, make_dev_minibatch_sets
from checking import gradient_check

class Acinn:

    layers = []                     # dimension of NN
    activations = []                # array of activations for layers
    parameters = {}                 # parameters (W,b) of layers
    loss = ()                       # loss that we use
    optimizer = None


    def add(self, layer):
        if self.layers:
            layer.input_shape = self.layers[-1].units

        self.layers.append(layer)

        assert(self.layers[0].input_shape != None), 'Input shape is not defined'


    def compile(self, initializer = 'random', loss = 'mean_squared_error', optimizer = Optimizer()):

        self.parameters = initialize(self.layers, initializer)
        self.loss = loss
        self.optimizer = optimizer


    def fit(self, X, Y, batch_size = 32, epochs = 1, validation_set = 0.2, info=True):
        assert(X.shape[0] == self.layers[0].input_shape), 'Input shape of X is not equale to input shape of model'  #provera da li je X istog shapea kao i input

        costs = []

        for i in range(0, epochs):

            minibatches, dev_set = make_dev_minibatch_sets(X, Y, batch_size, validation_set)
            (dev_X, dev_Y) = dev_set

            epoch_cost_total = 0

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                AL, cashe = model_forward(minibatch_X, self.parameters, self.layers)

                epoch_cost_total += model_loss(AL, minibatch_Y, self.loss)

                gradients = model_backward(AL, minibatch_Y, cashe, self.layers, self.loss)

                #if i == 0 or i == 10 or i == 100 or i == 1000:
                    #gradient_check(self.parameters, gradients, self.layers, minibatch_X, minibatch_Y, self.loss)

                self.parameters = self.optimizer.optimize(self.parameters, gradients)

            epoch_cost_avg = epoch_cost_total / int(X.shape[-1] * (1 - validation_set)) # Ovde ukupan train loss delimo sa brojem examplova u train setu

            # calculating the loss of dev set
            AL_dev, _ = model_forward(dev_X, self.parameters, self.layers)
            dev_cost = model_loss(AL_dev, dev_Y, self.loss) / dev_X.shape[-1]   # ovde dev loss delimo sa brojem examplova u dev setu


            if info and i % 100 == 0:
                print ("Train cost after iteration %i: %f" %(i, epoch_cost_avg))
                print ("Dev cost after iteration %i: %f" %(i, dev_cost))

            if i % 10 == 0:
                costs.append((epoch_cost_avg, dev_cost))

        return costs



    def predict(self, x):

        AL, cache = model_forward(x, self.parameters, self.activations)
        predictions = AL > 0.5

        return predictions



    def lay(self):
        L = len(self.layers)

        for l in range(0,L):
            print(self.layers[l].input_shape)
            print(self.layers[l].units)
