import numpy as np

def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1

        layer_inout_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_inout_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1

    return params_values

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')

    return activation_func(Z_curr), Z_curr


def full_forward_propagation(X, params_values, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = X

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr

        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        W_curr = params_values["W" + str(layer_idx)]
        # extraction of b for the current layer
        b_curr = params_values["b" + str(layer_idx)]
        # calculation of activation for the current layer
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

        # saving calculated values in the memory
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory

def A(x):
    if ((x[0] <= 1. and x[0]>= 0.5) and (x[1] <= 1. and x[1]>= 0.5)):

        f0 = 2
        f1 = 2
        g0 = 2
        g1 = 2

        #Dirichlet Boundary condition
        #out = (1 - x[0]) * f0 + x[0] * f1 + (1 - x[1]) * (g0 - ((1 - x[0]) * g0 + x[0] * g0)) + x[1] * (g1 - ((1 - x[0]) * g1 + x[0] * g1))

        #example dirichlet boundary condition
        out = (1 - x[0]) * f0 + x[0] * f1 + (1 - x[1]) * (g0 - (2 * (1 - x[0]) + 2 * x[0])) + x[1] * (g1 - (2 * (1 - x[0]) + 2 * x[0]))
    else:
        out = (1 - x[0]) + x[0] + (1 - x[1]) * (1 - ((1 - x[0]) + x[0])) + x[1] * (1 - ((1 - x[0]) + x[0]))

    #out = x[1] * np.sin(np.pi * x[0])
    #out = x[0] * 2 * x[1]
    return out

def f(x):
    out = 0.
    return out

def psy_trial(x, net_out):
    out = A(x) + x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * net_out
    return out


def loss_function(W, x, y):
    loss_sum = 0.

    for xi in x:
        for yi in y:
            input_point = np.array([xi, yi])
            net_out = single_layer_forward_propagation(input_point, W, b, sigmoid)[0]

            psy_t_hessian = jacobian(jacobian(psy_trial))(input_point, net_out)

            gradient_of_trial_d2x = psy_t_hessian[0][0]
            gradient_of_trial_d2y = psy_t_hessian[1][1]

            func = f(input_point)

            err_sqr = ((gradient_of_trial_d2x + gradient_of_trial_d2y) - func)**2
            loss_sum = loss_sum + err_sqr
    return loss_sum