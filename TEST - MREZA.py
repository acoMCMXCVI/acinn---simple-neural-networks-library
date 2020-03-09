import numpy as np
import matplotlib.pyplot as plt
from models import Acinn
from optimizers import Optimizer
from layers import Dense


train_x_orig = np.load('data\train_x_orig.npy')
train_y = np.load('data\train_y.npy')


test_x_orig = np.load('data\test_x_orig.npy')
test_y = np.load('data\test_y.npy')




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



'''
predictions = model.predict(x)



print ('Accuracy: %d' % float((np.dot(y,predictions.T) + np.dot(1-y,1-predictions.T))/float(y.size)*100) + '%')
predictions = model.predict(X)

print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

'''
