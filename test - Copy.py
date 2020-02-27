import numpy as np
import matplotlib.pyplot as plt
from acinn import Acinn
from optimizers import Optimizer

X = np.load('data\X_train.npy')
Y = np.load('data\y_train.npy')
X = X.T
Y = Y.reshape(1, len(Y))



model = Acinn()

print(model.layer_dims)

model.add(2, activation = 'relu', input_shape = 2)
model.add(15)
model.add(15)
model.add(7)
model.add(1, activation = 'sigmoid')


model.compile(initializer = 'random', loss = 'binary_crossentropy', optimizer = Optimizer(learning_rate=0.1) )



model.fit(X, Y, 100)

