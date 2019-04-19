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
    out = (1 - x[0]) + (1 - x[1]) * 50 * x[0] * (1 - x[0]) + x[1] * (50 * x[0] * (1 - x[0]))
    return out

def psy_trial(x, net_out):
    out = A(x) + x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * net_out
    return out


nx = 10
nt = 10
w = 1
t = 10
x_space = np.linspace(0, w, nx)
t_space = np.linspace(0, t, nt)
W = [npr.rand(2, 10), npr.rand(10, 1)]
b = [np.zeros(np.size(W[0][0])), np.zeros(np.size(W[1]))]


def loss_function(W, x, y, b):
    loss_sum = 0.
    D = 0.003

    for xi in x:
        for yi in y:
            input_point = np.array([xi, yi])

            net_out = neural_network(W, input_point, b)[0]
            psy_t_hessian = jacobian(jacobian(psy_trial))(input, net_out)
            gradient_of_trial_d2x = psy_t_hessian[0][0]
            gradient_of_trial_dt = jacobian(psy_trial)(input, net_out)[0]
            
            func = f(input_point)

            err_sqr = ((D * gradient_of_trial_d2x - gradient_of_trial_dt) - func)**2
            loss_sum = loss_sum + err_sqr

    return loss_sum


loss = loss_function(W, x_space, t_space, b)


loss = []
epoch = []
loss = []
epoch = []
iter = 0
learning_rate = 0.01
for i in range(10):
    print('%d' % i)
    L = loss_function(W, x_space, t_space, b)
    loss.append(L)

    iter = iter + 1
    epoch.append(iter)

    loss_grad = grad(loss_function)(W, x_space, t_space, b)
    W[0] = W[0] - learning_rate * loss_grad[0]
    W[1] = W[1] - learning_rate * loss_grad[1]
    #b[0] = b[0] - learning_rate * loss_grad[0]
    #b[1] = b[1] - learning_rate * loss_grad[0]

    print loss[i]


surface = np.zeros((nx, nt))
for i, t in enumerate(t_space):
    for j, x in enumerate(x_space):
        net_out_result = neural_network(W, [x, t], b)[0]
        surface[i][j] = psy_trial([x, t], net_out_result) * 0.003

print surface


X, Y = np.meshgrid(x_space, t_space)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, surface, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
plt.show()


exit(0)

'''
def main(a):
    #plate size, mm

    w = 1
    t = 10
    #intervals in x-, y-, directions, mm
    dx = dy = 0.1
    #thermal diffusivity of stell, mm2.s-1
    D = 0.003

    nx = 10
    nt = 10
    print nx
    print nt


    x_space = np.linspace(0, w, nx)
    t_space = np.linspace(0, t, nt)
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
        L = loss_function(W, x_space, t_space, b)
        loss.append(L)

        iter = iter + 1
        epoch.append(iter)

        loss_grad = grad(loss_function)(W, x_space, t_space, b)
        W[0] = W[0] - learning_rate * loss_grad[0]
        W[1] = W[1] - learning_rate * loss_grad[1]
        #b[0] = b[0] - learning_rate * loss_grad[0]
        #b[1] = b[1] - learning_rate * loss_grad[0]

        print loss[i]

    surface2 = np.zeros((nt, nx))



    X, Y = np.meshgrid(x_space, t_space)
    net_outt = neural_network(W, X, b)[0]
    R = psy_trial(X, net_outt)

    print (R)


x = [10]
for a in x:
    main(a)
'''





