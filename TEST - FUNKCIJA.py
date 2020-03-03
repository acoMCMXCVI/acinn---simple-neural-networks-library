import numpy as np
import matplotlib.pyplot as plt
from models import Acinn
from optimizers import Optimizer
import checking
import initializers

layer_dims = [2,3,2,1]

parameters = initializers.relu_initialize(layer_dims)


parameters_values, parameters_shapes = checking.dictionary_to_vector(parameters)
parametersN = checking.vector_to_dictionary(parameters_values, parameters_shapes)


print(parametersN)
print(parameters)

