import numpy as np
from forwards import model_forward
from backwards import model_backward
from losses import model_loss

def gradient_check(parameters, gradients, layers, X, Y, loss, epsilon = 1e-7):

    # Set-up variables
    parameters_values, parameters_shapes = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)

    num_parameters = parameters_values.shape[0]

    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters,))


    # Compute gradapprox
    for i in range(num_parameters):

        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one

        thetaplus = np.copy(parameters_values)                                      # Step 1
        thetaplus[i] = thetaplus[i] + epsilon                                # Step 2
        AL, _ = model_forward(X, vector_to_dictionary(thetaplus, parameters_shapes), layers)
        J_plus[i][0] = model_loss(AL, Y, loss) / X.shape[-1]


        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        thetaminus = np.copy(parameters_values)                                     # Step 1
        thetaminus[i] = thetaminus[i] - epsilon
        AL, _ = model_forward(X, vector_to_dictionary(thetaminus, parameters_shapes), layers)  # Step 2
        J_minus[i][0] = model_loss(AL, Y, loss) / X.shape[-1]

        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i][0] - J_minus[i][0]) / (2*epsilon)

    #print(grad)
    #print(gradients)
    #print(gradapprox)


    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(gradapprox - grad)                                           # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                                        # Step 2'
    difference = numerator / denominator                                          # Step 3'

    if difference > 2e-7:
        print ("Mistake! difference = " + str(difference))
    else:
        print ("Working! difference = " + str(difference))

    return difference


def dictionary_to_vector(parameters):

    L = len(parameters) // 2                            # number of layers in the neural network

    parameters_values = []
    parameters_shapes = []

    for l in range(1, L+1):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]

        parameters_values.append(tuple(W.reshape(-1,)))
        parameters_values.append(tuple(b.reshape(-1,)))

        parameters_shapes.append(W.shape)
        parameters_shapes.append(b.shape)

    parameters_values = np.concatenate((parameters_values), axis=None)

    #print(parameters_shapes)
    #print(parameters_values)

    return parameters_values, parameters_shapes



def vector_to_dictionary(parameters_values, parameters_shapes):

    L = len(parameters_shapes)                            # number of layers in the neural network

    dictionary = {}

    new_values = []

    start = 0
    for shape in parameters_shapes:
        lenght = shape[0]*shape[1]
        new_values.append(parameters_values[start:start+lenght].reshape(shape))
        start += lenght

    #print(len(new_values))
    #print(L)

    for l in range(0, L, 2):
        dictionary['W' + str(int(l/2 + 1))] = new_values[l]
        dictionary['b' + str(int(l/2 + 1))] = new_values[l+1]

    return dictionary


def gradients_to_vector(gradients):

    L = len(gradients) // 3                            # number of layers in the neural network

    gradients_values = []

    for l in range(1, L+1):
        dW = gradients['dW' + str(l)]
        db = gradients['db' + str(l)]

        #print(dW)
        #print(db)

        gradients_values.append(tuple(dW.reshape(-1,)))
        gradients_values.append(tuple(db.reshape(-1,)))

    gradients_values = np.concatenate((gradients_values), axis=None)

    #print(gradients_values)

    return gradients_values
