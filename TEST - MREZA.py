import numpy as np
import matplotlib.pyplot as plt
from models import Acinn
from optimizers import Optimizer
from layers import Dense
import h5py
from utilities import load_cat_data, load_2D_dataset, load_hand_softmax, convert_to_one_hot


lr = 0.000098
lr2 = 0.0001

#moj data set
train_x_orig = np.load('data/X_train.npy')
train_y = np.load('data/y_train.npy').reshape(1,-1)
test_x_orig = np.load('data/X_test.npy')
test_y = np.load('data/y_test.npy').reshape(1,-1)


'''
#cat data set
train_x_orig, train_y, test_x_orig, test_y, classes = load_cat_data()
print(train_x_orig.shape)
print(train_y.shape)

#plt.imshow(train_x_orig[50])
#plt.show()
'''

#hand data set
train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_hand_softmax()
print(train_x_orig.shape)
print(train_y.shape)


train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

train_y = convert_to_one_hot(train_y_orig, 6)
test_y = convert_to_one_hot(test_y_orig, 6)

print(train_y.T[125][:])
print(train_x.shape)
print(train_y.shape)


'''
# njihov 2d data set
train_x, train_y, test_x, test_y = load_2D_dataset()


# logistic regresion

model = Acinn()

model.add(Dense(20, 'relu', train_x.shape[0]))
model.add(Dense(3))
model.add(Dense(1, 'sigmoid'))

#model.lay()

model.compile(initializer = 'he', loss = 'binary_crossentropy', optimizer = Optimizer(optimizer = 'Adam', learning_rate=lr, decay = 0))

history = model.fit(train_x, train_y, batch_size = 256, epochs = 40000, validation_split = 0.15)
'''

# softmax regresio

model = Acinn()

model.add(Dense(25, 'relu', train_x.shape[0]))
model.add(Dense(12))    
model.add(Dense(6, 'softmax'))

#model.lay()

model.compile(initializer = 'he', loss = 'categorical_crossentropy', optimizer = Optimizer(optimizer = 'Adam', learning_rate=lr2, decay = 0))

history = model.fit(train_x, train_y, batch_size = 32, epochs = 1500, validation_split = 0)

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

