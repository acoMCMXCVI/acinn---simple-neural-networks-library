class Dense():

    units = 0
    activation = None
    input_shape = None

    def __init__(self, units, activation = 'relu', input_shape = None):

        self.units = 0
        self.activation = activation
        self.input_shape = input_shape
