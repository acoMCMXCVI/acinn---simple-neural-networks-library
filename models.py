import numpy as np
from initializers import initialize
from forwards import model_forward
from backwards import model_backward
from losses import model_loss
from optimizers import Optimizer
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


    def fit(self, X, Y, epochs = 1, info=True):
        assert(X.shape[0] == self.layers[0].input_shape), 'Input shape of X is not equale to input shape of model'  #provera da li je X istog shapea kao i input

        costs = []

        for i in range(0, epochs):

            AL, cashe = model_forward(X, self.parameters, self.layers)

            cost = model_loss(AL, Y, self.loss)

            gradients = model_backward(AL, Y, cashe, self.layers, self.loss)

            if i == 0 or i == 10 or i == 100 or i == 1000:
                gradient_check(self.parameters, gradients, self.layers, X, Y, self.loss)

            self.parameters = self.optimizer.optimize(self.parameters, gradients)



            if info and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

            if i % 100 == 0:
                costs.append(cost)

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
