# layers witch we can use in networks arh.

class Dense():

    units = 0
    activation = None
    input_shape = None

    def __init__(self, units, activation = 'relu', input_shape = None):

        self.units = units
        self.activation = activation
        self.input_shape = input_shape
