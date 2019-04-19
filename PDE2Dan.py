
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr
from matplotlib import pyplot, cm
np.random.seed(1)
from mpl_toolkits.mplot3d import Axes3D

### Neural Network ###

def f(x):
    out = 0.
    return out


def relu(X):
    out = np.maximum(0, X)
    return out


def sigmoid(x):
    out = 1. / (1. + np.exp(-x))
    return out

def neural_network(W, x, b):
    a1 = sigmoid(np.dot(x, W[0]) + b[0])
    out = np.dot(a1, W[1]) + b[1]
    return out

def neural_network_x(W, x, b):
    a1 = sigmoid(np.dot(x, W[0]) + b[0])
    out = np.dot(a1, W[1]) + b[1]
    return out

def A(x):
    '''if ((x[0] <= 1. and x[0]>= 0.5) and (x[1] <= 1. and x[1]>= 0.5)):

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
    '''

    #out = x[1] * np.sin(np.pi * x[0])
    #out = x[0] * 2 * x[1]
    #out = x[0] * x[1]
    out = x[1] * np.sin(np.pi * x[0])
    return out

def psy_trial(x, net_out):
    out = A(x) + x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * net_out
    return out

def loss_function(W, x, y, b):
    loss_sum = 0.

    for xi in x:
        for yi in y:
            input_point = np.array([xi, yi])
            net_out = neural_network(W, input_point, b)[0]

            psy_t_hessian = jacobian(jacobian(psy_trial))(input_point, net_out)

            gradient_of_trial_d2x = psy_t_hessian[0][0]
            gradient_of_trial_d2y = psy_t_hessian[1][1]

            func = f(input_point)

            err_sqr = ((gradient_of_trial_d2x + gradient_of_trial_d2y) - func)**2
            loss_sum = loss_sum + err_sqr
    return loss_sum


def analytic_solution(x):
    return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * \
    		np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1]))


def main(a):
    #plate size, mm

    w = h = 1
    #intervals in x-, y-, directions, mm
    dx = dy = 0.1
    #thermal diffusivity of stell, mm2.s-1
    D = 4

    nx, ny = 10, 10
    print nx
    print ny


    x_space = np.linspace(0, w, nx)
    y_space = np.linspace(0, h, ny)
    W = [npr.rand(2, 10), npr.rand(10, 1)]
    #b = [npr.rand(np.size(W[0][0])), npr.rand(np.size(W[1]))]
    b = [np.zeros(np.size(W[0][0])), np.zeros(np.size(W[1]))]


    learning_rate = 0.01

    #print neural_network(W, np.array([1, 1]), b([1, 1]))

    print("init weight...")

    loss = []
    epoch = []
    iter = 0
    for i in range(a):
        print('%d' % i)
        L = loss_function(W, x_space, y_space, b)
        loss.append(L)

        iter = iter + 1
        epoch.append(iter)

        loss_grad = grad(loss_function)(W, x_space, y_space, b)
        W[0] = W[0] - learning_rate * loss_grad[0]
        W[1] = W[1] - learning_rate * loss_grad[1]
        #b[0] = b[0] - learning_rate * loss_grad[0]
        #b[1] = b[1] - learning_rate * loss_grad[0]

        print loss[i]

    surface2 = np.zeros((ny, nx))

    print("neural net solution...")
    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            net_outt = neural_network(W, [x, y], b)[0]
            surface2[i][j] = psy_trial([x, y], net_outt)

    fig1 = plt.figure()
    ax = fig1.gca(projection='3d')
    X, Y = np.meshgrid(x_space, y_space)
    surf = ax.plot_surface(X, Y, surface2, rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 2)


    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.savefig('gambar-dengan-bias-nol-iterasi-{}.png'.format(a))
    plt.close()

    surface = np.zeros((ny, nx))

    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            surface[i][j] = analytic_solution([x, y])

    fig2 = plt.figure()
    ax = fig2.gca(projection='3d')
    X, Y = np.meshgrid(x_space, y_space)
    surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 2)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')


    fig3 = plt.figure()
    plt.plot(epoch[10:], loss[10:])

    plt.show()



x = [100]
for a in x:
    main(a)


