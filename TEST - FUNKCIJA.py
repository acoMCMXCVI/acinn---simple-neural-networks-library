import numpy as np
import matplotlib.pyplot as plt
from models import Acinn
from optimizers import Optimizer, make_dev_train_sets, make_m_batches
from layers import Dense


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

