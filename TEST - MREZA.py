import numpy as np
import matplotlib.pyplot as plt
from models import Acinn
from optimizers import Optimizer

X = np.load('data\X_train.npy')
Y = np.load('data\y_train.npy')
X = X.T
Y = Y.reshape(1, len(Y))

x = np.load('data\X_test.npy')
y = np.load('data\y_test.npy')
x = x.T
y = y.reshape(1, len(y))



model = Acinn()

model.add(2, activation = 'relu', input_shape = 2)
model.add(2)
model.add(2)
model.add(1, activation = 'sigmoid')

model.compile(initializer = 'random', loss = 'binary_crossentropy', optimizer = Optimizer(learning_rate=0.01) )

model.fit(X, Y, 10000)


'''
predictions = model.predict(x)



print ('Accuracy: %d' % float((np.dot(y,predictions.T) + np.dot(1-y,1-predictions.T))/float(y.size)*100) + '%')
predictions = model.predict(X)

print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

'''

