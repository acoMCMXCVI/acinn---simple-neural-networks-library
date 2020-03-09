import numpy as np
import matplotlib.pyplot as plt
from models import Acinn
from optimizers import Optimizer
from layers import Dense


X = np.load('data\X_train.npy')
Y = np.load('data\y_train.npy')
X = X.T
Y = Y.reshape(1, len(Y))

x = np.load('data\X_test.npy')
y = np.load('data\y_test.npy')
x = x.T
y = y.reshape(1, len(y))



model = Acinn()

model.add(Dense(2, 'relu', 2))
model.add(Dense(1, 'sigmoid'))

#model.lay()

model.compile(initializer = 'relu', loss = 'binary_crossentropy', optimizer = Optimizer(learning_rate=0.01) )

history = model.fit(X, Y, 10000)

plt.plot(np.squeeze(history))
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(0.01))
plt.show()
