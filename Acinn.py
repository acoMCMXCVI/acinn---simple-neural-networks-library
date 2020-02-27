import numpy as np
from initializers import initialize
from forwards import L_model_forward
from losess import compute_loss

class Acinn:

    layer_dims = [0]                # dimension of NN
    activations = []                # array of activations for layers
    parameters = {}                 # parameters (W,b) of layers
    loss = ()                       # loss that we use


    def add(self, num_units = 0, activation = 'relu', input_shape = 0):
        if self.layer_dims[0] is not 0 or input_shape is not 0:                 #provera da li je input shape definisan

            self.layer_dims.append(num_units)
            self.activations.append(activation)

            if input_shape is not 0:
                self.layer_dims[0] = input_shape

        else:
            print('Input shape is not defined')



    def compile(self, initializer = 'random', loss = 'mean_squared_error' ):

        self.parameters = initialize(self.layer_dims, initializer)
        self.loss = loss


    def fit(self, X, Y, epochs = 1):
        if X.shape[0] is self.layer_dims[0]:                                    #provera da li je X istog shapea kao i input

            costs = []

            for i in range(0, epochs):
                AL, cashe = L_model_forward(X, self.parameters, self.activations)
                cost = compute_loss(AL, Y, self.loss)



                print(cost)
                costs.append(cost)


        else:
            print('Shape of X is not shape of input')
