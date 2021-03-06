#Code for 2D difusion, so we need x and y axis for the model

import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

nx = 10
ny = 10
nt = 10
nu = 0.5
sigma = 0.25
dx = 1. / nx
dy = 1. / ny
dt = sigma * dx * dy / nu

x_space = np.linspace(0, 1, nx)
y_space = np.linspace(0, 1, ny)
t_space = np.linspace(0, 1, nt)

u = np.zeros((ny, nx))
un = np.zeros((ny, nx))


def f(x):
    return 0.

def sigmoid(x):
    out = 1. / (1. + np.exp(-x))
    return out

def neural_network(W, x):
    a1 = sigmoid(np.dot(x, W[0]))
    out = np.dot(a1, W[1])
    return out

def neural_network_x(x):
    a1 = sigmoid(np.dot(x, W[0]))
    out = np.dot(a1, W[1])
    return out

def A(x):
    out = x[1] * np.sin(np.pi * x[0])
    return out

def psy_trial(x, net_out):
    out = A(x) + x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * net_out
    return out

def loss_function(W, x, y, t):
    loss_sum = 0.

    for xi in x:
        for yi in y:
            input_point = np.array([xi, yi])
            t_input = np.array(t)
            net_out = neural_network(W, input_point)[0]

            net_out_jacobian = jacobian(neural_network_x)(input_point)
            net_out_hessian = jacobian(jacobian(neural_network_x))(input_point)

            psy_t = psy_trial(input_point, net_out)
            psy_t_jacobian = jacobian(psy_trial)(input_point, net_out)
            psy_t_hessian = jacobian(jacobian(psy_trial))(input_point, net_out)
            psy_time_jacobian = jacobian(psy_trial)(t_input, net_out)

            gradient_of_trial_d2x = psy_t_hessian[0][0]
            gradient_of_trial_d2y = psy_t_hessian[1][1]
            gradient_of_trial_dt = psy_time_jacobian[0]

            func = f(input_point)

            err_sqr = ((gradient_of_trial_d2x + gradient_of_trial_d2y - gradient_of_trial_dt) - func)**2
            loss_sum = loss_sum + err_sqr
    return loss_sum


W = [npr.rand(2, 10), npr.randn(10, 1)]
learning_rate = 0.01

print neural_network(W, np.array([1, 1]))

print("init weight...")
for i in range(50):
    print('%d' % i)
    loss_grad = grad(loss_function)(W, x_space, y_space, t_space)
    W[0] = W[0] - learning_rate * loss_grad[0]
    W[1] = W[1] - learning_rate * loss_grad[1]

print loss_function(W, x_space, y_space, t_space)

surface2 = np.zeros((ny, nx))
surface = np.zeros((ny, nx))


print("neural net solution...")
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        net_outt = neural_network(W, [x, y])[0]
        surface2[i][j] = psy_trial([x, y], net_outt)

print surface2[2]

print("plotting...")
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface2, rstride=1, cstride=1, cmap=cm.viridis,
                       linewidth=0, antialiased=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 3)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()
