import numpy as np
import pickle
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

    # function for add new layers to network
    def add(self, layer):
        if self.layers:
            layer.input_shape = self.layers[-1].units

        self.layers.append(layer)

        assert(self.layers[0].input_shape != None), 'Input shape is not defined'

    # function for compile mode, initialize parms, loss, optimizer,...
    def compile(self, initializer = 'random', loss = 'mean_squared_error', optimizer = Optimizer()):

        self.parameters = initialize(self.layers, initializer)
        self.loss = loss
        self.optimizer = optimizer

    # function for fitting model
    def fit(self, X, Y, batch_size = 32, epochs = 1, validation_split = 0., info=True):
        assert(X.shape[0] == self.layers[0].input_shape), 'Input shape of X is not equale to input shape of model'  # check for input shape is same with model input

        costs = []
        accuracies = []

        train_set, dev_set = make_dev_train_sets(X, Y, validation_split)
        (X_train, Y_train) = train_set
        (X_dev, Y_dev) = dev_set

        for i in range(0, epochs):

            epoch_cost_total = 0
            val_cost = 0
            val_acc = 0

            minibatches = make_m_batches(X_train, Y_train, batch_size)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                AL, cashe = model_forward(minibatch_X, self.parameters, self.layers)

                epoch_cost_total += model_loss(AL, minibatch_Y, self.loss)

                gradients = model_backward(AL, minibatch_Y, cashe, self.layers, self.loss)

                #if i == 0 or i == 10 or i == 25000 or i == 1000 or i == 10000:
                    #gradient_check(self.parameters, gradients, self.layers, minibatch_X, minibatch_Y, self.loss)

                self.parameters = self.optimizer.optimize(self.parameters, gradients, i)

            epoch_cost_avg = epoch_cost_total / X_train.shape[-1]       # entire train loss divaded with number of examples in train set

            # calculating the loss and acc of dev set
            if validation_split != 0.:
                val_cost, val_acc = self.evaluate(X_dev, Y_dev)


            if info and i % 10 == 0:
                print('Epoch %i / %i \t train loss - %f \t dev loss - %f \t dec acc - %f' %(i, epochs, epoch_cost_avg, val_cost, val_acc))
                #print ("Train cost after iteration %i: %f" %(i, epoch_cost_avg))
                #print ("Dev cost after iteration %i: %f" %(i, val_cost))
                #print ("Dev acc after iteration %i: %f" %(i, val_acc))

            if i % 10 == 0:
                costs.append((epoch_cost_avg, val_cost))
                accuracies.append(val_acc)

        return costs, accuracies



    def evaluate(self, X, Y):
        # function evaluete model calculating cost and acc for X and Y

        AL, cashe = model_forward(X, self.parameters, self.layers)

        evoluation_cost = model_loss(AL, Y, self.loss) / X.shape[-1]

        prediction = self.predict(AL, True)
        evoluation_accuracy = self.accuracy(Y, prediction)

        return evoluation_cost, evoluation_accuracy


    def predict(self, x, in_model = False):
        # function calculate prediction for given X

        if in_model == False:
            AL, _ = model_forward(x, self.parameters, self.layers)
        else:
            AL = x

        if self.layers[-1].activation == 'sigmoid':
            predictions = AL > 0.5
        elif self.layers[-1].activation == 'relu':
            predictions = AL
        elif self.layers[-1].activation == 'softmax':
            predictions = AL == np.max(AL, axis = 0)

        return predictions



    def accuracy(self, Y, predictions):
        # function calculate acc for Y and predicted values from X

        acc = (Y == predictions).all(axis=0)
        acc = float(np.sum(acc) / Y.shape[-1]) * 100

        return acc

    def save_weights(self, path):
        # function save model to disc

        with open(path + '.pkl', 'wb') as f:
            pickle.dump(self.parameters, f, pickle.HIGHEST_PROTOCOL)

    def load_weights(self, path):
        # function load model from disc

        with open(path + '.pkl', 'rb') as f:
            self.parameters = pickle.load(f)
