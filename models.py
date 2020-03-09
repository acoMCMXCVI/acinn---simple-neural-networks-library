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

        self.layers.append(layer)

        assert(self.layers[0].input_shape != None), 'Input shape is not defined'



    def compile(self, initializer = 'random', loss = 'mean_squared_error', optimizer = Optimizer() ):

        self.parameters = initialize(self.layers, initializer)
        self.loss = loss
        self.optimizer = optimizer


    def fit(self, X, Y, epochs = 1, info=True):
        if X.shape[0] == self.layers[0].input_shape:                                    #provera da li je X istog shapea kao i input

            costs = []

            for i in range(0, epochs):

                AL, cashe = model_forward(X, self.parameters, self.activations)

                cost = model_loss(AL, Y, self.loss)

                gradients = model_backward(AL, Y, cashe, self.activations, self.loss)

                if i == 0 or i == 10 or i == 100 or i == 1000:
                    gradient_check(self.parameters, gradients, self.activations, X, Y, self.loss)

                self.parameters = self.optimizer.optimize(self.parameters, gradients)



                if info and i % 1000 == 0:
                    print ("Cost after iteration %i: %f" %(i, cost))

                if i % 100 == 0:
                    costs.append(cost)


        else:
            print('Shape of X is not shape of input')


    def predict(self, x):

        AL, cache = model_forward(x, self.parameters, self.activations)
        predictions = AL > 0.5

        return predictions
