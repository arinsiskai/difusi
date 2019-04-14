
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr
from matplotlib import pyplot, cm
np.random.seed(1)
from mpl_toolkits.mplot3d import Axes3D

#plate size, mm

w = h = 2
#intervals in x-, y-, directions, mm
dx = dy = 0.1
#thermal diffusivity of stell, mm2.s-1
D = 4

nx, ny = 10, 10
print nx
print ny


x_space = np.linspace(0, w, nx)
y_space = np.linspace(0, h, ny)

### Neural Network ###

NN_ARCHITECTURE = [
    {"input_dim" : 20, "output_dim" : 20, "activation": "sigmoid"}
]

def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1

        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['W' + str(layer_idx)] = np.random.rand(layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1

    return params_values


def f(x):
    out = 0.
    return out

def sigmoid(x):
    out = 1. / (1. + np.exp(-x))
    return out

def relu(x):
    return np.maximum(0,x)

def sigmoid_backward(dA, x):
    sig = sigmoid(x)
    return dA * sig * (1 - sig)

def relu_backward(dA, x):
    dZ = np.array(dA, copy = True)
    dZ[x <= 0] = 0;
    return dZ

def neural_network_forward_propagation(W_curr, x_prev, b_curr, activation="relu"):

    Z_curr = np.dot(W_curr, x_prev) + b_curr
    # selection of activation function
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')

    # return of calculated activation A and the intermediate Z matrix
    return activation_func(Z_curr), Z_curr

'''
def neural_network_x(x):
    a1 = sigmoid(np.dot(x, W[0]))
    out = np.dot(a1, W[1])
    return out
'''

def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    x_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        x_prev = x_curr

        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        x_curr, Z_curr = neural_network_forward_propagation(W_curr, x_prev, b_curr, activ_function_curr)

        memory["A" + str(idx)] = x_prev
        memory["Z" + str(layer_idx)] = Z_curr

    return x_curr, memory

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

def psy_trial(x, net_out):
    out = A(x) + x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * net_out
    return out

def loss_function(params_values, x, y, nn_architecture):
    loss_sum = 0.

    for xi in x:
        for yi in y:
            input_point = np.array([xi, yi])
            net_out = full_forward_propagation(input_point, params_values, nn_architecture)

            psy_t_hessian = jacobian(jacobian(psy_trial))(input_point, net_out)

            gradient_of_trial_d2x = psy_t_hessian[0][0]
            gradient_of_trial_d2y = psy_t_hessian[1][1]

            func = f(input_point)

            err_sqr = ((gradient_of_trial_d2x + gradient_of_trial_d2y) - func)**2
            loss_sum = loss_sum + err_sqr
    return loss_sum


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    # number of examples
    m = A_prev.shape[1]

    # selection of activation function
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    # calculation of the activation function derivative
    dZ_curr = backward_activation_func(dA_curr, Z_curr)

    # derivative of the matrix W
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    # derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    # derivative of the matrix A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

'''
params_values = [npr.rand(2, 10), npr.randn(10, 1)]
nn_architecture = "relu"
learning_rate = 0.001

print full_forward_propagation(X ,params_values, nn_architecture)

print("init weight...")
for i in range(100):
    print('%d' % i)
    print loss_function(W, x_space, y_space)
    #exit(0)
    loss_grad = grad(loss_function)(W, x_space, y_space)
    W[0] = W[0] - learning_rate * loss_grad[0]
    W[1] = W[1] - learning_rate * loss_grad[1]

print loss_function(W, x_space, y_space)
surface2 = np.zeros((ny, nx))
'''


print("neural net solution...")
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        net_outt = neural_network(W, [x, y])[0]
        surface2[i][j] = psy_trial([x, y], net_outt)

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface2, rstride=1, cstride=1, cmap=cm.viridis,
                       linewidth=0, antialiased=False)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()