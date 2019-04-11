
'''Solving Laplace Equation with PDE'''

import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr
from matplotlib import pyplot, cm
np.random.seed(1)
from mpl_toolkits.mplot3d import Axes3D


def plot2D(x, y, p):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis,
            linewidth=0, antialiased=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.view_init(30, 225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.show()

def laplace2d(p, y, dx, dy, l1norm_target):
    l1norm = 1
    pn = np.empty_like(p)

    while l1norm > l1norm_target:
        pn = p.copy()
        p[1:-1, 1:-1] = ((dy ** 2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2]) + dx ** 2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1])) / (2. * (dx ** 2 + dy ** 2)))

        p[:, 0] = 0  # p = 0 @ x = 0
        p[:, -1] = y  # p = y @ x = 2
        p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
        p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1
        l1norm = (np.sum(np.abs(p[:]) - np.abs(pn[:])) /
                  np.sum(np.abs(pn[:])))
    return p


#variabel declaration
x = 1
y = 1

nx, ny = 31, 31
c = 1
dx = 2. / (nx - 1)
dy = 2. / (ny - 1)

x_space = np.linspace(0, x, nx)
y_space = np.linspace(0, y, ny)

##initial conditions
p = np.zeros((ny, nx))  # create a XxY vector of 0's

##boundary conditions
p[:, 0] = 0  # p = 0 @ x = 0
p[:, -1] = y_space  # p = y @ x = 2
p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1

p = laplace2d(p, y_space, dx, dy, 1e-4)
plot2D(x_space, y_space, p)


### Neural Network ###

def f(x):
    out = 0
    return out

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
    out = x[0] * x[1]
    return out

def psy_trial(x, net_out):
    out = A(x) + x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * net_out
    return out

def loss_function(W, x, y):
    loss_sum = 0.

    for xi in x:
        for yi in y:
            input_point = np.array([xi, yi])
            net_out = neural_network(W, input_point)[0]

            net_out_jacobian = jacobian(neural_network_x)(input_point)
            net_out_hessian = jacobian(jacobian(neural_network_x))(input_point)

            psy_t = psy_trial(input_point, net_out)
            psy_t_jacobian = jacobian(psy_trial)(input_point, net_out)
            psy_t_hessian = jacobian(jacobian(psy_trial))(input_point, net_out)

            gradient_of_trial_d2x = psy_t_hessian[0][0]
            gradient_of_trial_d2y = psy_t_hessian[1][1]

            func = f(input_point)

            err_sqr = ((gradient_of_trial_d2x + gradient_of_trial_d2y) - func)**2
            loss_sum = loss_sum + err_sqr
    return loss_sum


W = [npr.rand(2, 10), npr.randn(10, 1)]
learning_rate = 0.001

print neural_network(W, np.array([1, 1]))

print("init weight...")
for i in range(200):
    print('%d' % i)
    print loss_function(W, x_space, y_space)
    #exit(0)
    loss_grad = grad(loss_function)(W, x_space, y_space)
    W[0] = W[0] - learning_rate * loss_grad[0]
    W[1] = W[1] - learning_rate * loss_grad[1]

print loss_function(W, x_space, y_space)
surface2 = np.zeros((ny, nx))

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