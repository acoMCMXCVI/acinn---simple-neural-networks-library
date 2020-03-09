import numpy as np
import matplotlib.pyplot as plt
from models import Acinn
from optimizers import Optimizer
from layers import Dense

model = Acinn()

model.add(Dense(15, 'relu', 5))
model.add(Dense(15, 'relu'))

model.compile(initializer = 'relu', loss = 'binary_crossentropy', optimizer = Optimizer(learning_rate=0.01) )
