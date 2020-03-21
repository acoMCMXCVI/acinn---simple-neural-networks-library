import numpy as np
import matplotlib.pyplot as plt
from models import Acinn
from optimizers import Optimizer
from layers import Dense
import h5py
from utilities import load_data, load_2D_dataset


lr = 0.0007


train_x_orig = np.load('data/X_train.npy')
train_y = np.load('data/y_train.npy').reshape(1,-1)
test_x_orig = np.load('data/X_test.npy')
test_y = np.load('data/y_test.npy').reshape(1,-1)


'''
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
print(train_x_orig.shape)
print(train_y.shape)

#plt.imshow(train_x_orig[50])
#plt.show()
'''

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.


'''
train_x, train_y, test_x, test_y = load_2D_dataset()
'''


model = Acinn()

model.add(Dense(20, 'relu', train_x.shape[0]))
model.add(Dense(3))
model.add(Dense(1, 'sigmoid'))

#model.lay()

model.compile(initializer = 'he', loss = 'binary_crossentropy', optimizer = Optimizer(optimizer = 'Adam', learning_rate=lr, decay = 0))

history = model.fit(train_x, train_y, batch_size = 256, epochs = 40000, validation_split = 0.15)

plt.plot(np.squeeze(history))
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(lr))
plt.legend(('train', 'dev'))
plt.show()

predictions = model.predict(train_x)

print ('Accuracy: %d' % float((np.dot(train_y,predictions.T) + np.dot(1-train_y,1-predictions.T))/float(train_y.size)*100) + '%')

predictions = model.predict(test_x)

print ('Accuracy: %d' % float((np.dot(test_y,predictions.T) + np.dot(1-test_y,1-predictions.T))/float(test_y.size)*100) + '%')
