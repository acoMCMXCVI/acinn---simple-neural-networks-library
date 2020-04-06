import numpy as np
import matplotlib.pyplot as plt
from models import Acinn
from optimizers import Optimizer
from layers import Dense
import h5py
from utilities import load_cat_data, load_2D_dataset, load_hand_softmax, convert_to_one_hot

'''
X = np.load('data\X_train.npy')
Y = np.load('data\y_train.npy')
X = X.T
Y = Y.reshape(1, len(Y))

x = np.load('data\X_test.npy')
y = np.load('data\y_test.npy')
x = x.T
y = y.reshape(1, len(y))



X = np.arange(20).reshape(1,-1)
Y = np.arange(20).reshape(1,-1)
print(X)




train_set, dev_set = make_dev_train_sets(X,Y,0.5)
(train_X, train_Y) = train_set
(dev_X, dev_Y) = dev_set

print(train_X.shape)
print(dev_X.shape)

print(train_X)
print(dev_X)

minibatches = make_m_batches(train_X, train_Y, 0)
print(minibatches)



Y = np.identity(5)
Yhat = np.identity(5)
print(Y.size)

model = Acinn()

print(model.accuracy(Y, Yhat))
'''

lr2 = 0.0001

#hand data set
train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_hand_softmax()
train_y = convert_to_one_hot(train_y_orig, 6)
test_y = convert_to_one_hot(test_y_orig, 6)

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.


model = Acinn()

model.add(Dense(25, 'relu', train_x.shape[0]))
model.add(Dense(12))
model.add(Dense(6, 'softmax'))

model.compile(initializer = 'xavier', loss = 'categorical_crossentropy', optimizer = Optimizer(optimizer = 'Adam', learning_rate=lr2, decay = 0))

model.load_weights('model')

cost_acc_train = model.evaluate(train_x, train_y)
cost_acc_test = model.evaluate(test_x, test_y)
print('model cost and acc is for train:' + str(cost_acc_train))
print('model cost and acc is for test:' + str(cost_acc_test))
