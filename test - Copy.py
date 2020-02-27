import numpy as np
import matplotlib.pyplot as plt
from Acinn import Acinn
import backwards
import activations

X = np.load('data\X_train.npy')
Y = np.load('data\y_train.npy')
X = X.T
Y = Y.reshape(1, len(Y))


'''
model = Acinn()

print(model.layer_dims)

model.add(2, activation = 'relu', input_shape = 3)
model.add(15)
model.add(1, activation = 'sigmoid')


print(model.layer_dims)
print(model.activations)

model.compile(initializer = 'zeros', loss = 'binary_crossentropy')

print(model.parameters)


model.fit(X, Y, 1)

'''
print(backwards.relu_backward(np.array([[-2,1,0,2]]).T, np.array([[1,3,2,1]]).T))
