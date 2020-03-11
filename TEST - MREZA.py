import numpy as np
import matplotlib.pyplot as plt
from models import Acinn
from optimizers import Optimizer
from layers import Dense
import h5py
from utilities import load_data


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
print(train_x_orig.shape)
print(train_y.shape)

plt.imshow(train_x_orig[50])
#plt.show()
print(train_x_orig[50])

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T 
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print(train_x[0])
print(train_y.shape)

model = Acinn()

model.add(Dense(7, 'relu', train_x.shape[0]))
model.add(Dense(1, 'sigmoid'))

#model.lay()

model.compile(initializer = 'he', loss = 'binary_crossentropy', optimizer = Optimizer(learning_rate=0.01) )

history = model.fit(train_x, train_y, 10000)

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
