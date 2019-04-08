
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

nx, ny = 20, 20
print nx
print ny


x_space = np.linspace(0, w, nx)
y_space = np.linspace(0, h, ny)

### Neural Network ###

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
    if ((x[0] <= 1 and x[0]>= 0.5) and (x[1] <= 1 and x[1]>= 0.5)):

        f0 = 2
        f1 = 2
        g0 = 2
        g1 = 2
        out = (1 - x[0]) * f0 + x[0] * f1 + (1 - x[1]) * (g0 - ((1 - x[0]) * g0 + x[0] * g0)) + x[1] * (g1 - ((1 - x[0]) * g1 + x[0] * g1))
        #out = (1 - x[0]) * f0 + x[0] * f1 + (1 - x[1]) * (g0 - (2 * (1 - x[0]) + 2 * x[0])) + x[1] * (g1 - (2 * (1 - x[0]) + 2 * x[0]))
    else:
        out = 0

    #out = x[0]
    #out = x[1] * np.sin(np.pi * x[0])
    #out = x[0] * 2 * x[1]
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