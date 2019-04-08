import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from numpy import pi
from mpl_toolkits.mplot3d import Axes3D

nx = 20

dx = 1. / nx
C = 1
nt = 20
dt = 1 / nt
k = 1

x_space = np.linspace(0, 1, nx)
t_space = np.linspace(0, 5, nt)

def analytic_solution(x, t):
    out = np.exp(-t) * 1 / (2 * np.sqrt(pi + 3)) * np.exp(-1/(4 * t)) + np.exp(-16 * t) * np.sin(4*x)
    return out

u = np.zeros((nx, nt))

for i, x in enumerate(x_space):
    for j, t in enumerate(t_space):
        u[i][j] = analytic_solution(x, t)

print (np.shape(u))
print (u[:, 10])

plt.plot(x_space, u[:, 1], label='0 analitik')
plt.legend()


def f(x):
    return 0.

def sigmoid(x):
    out = 1. / (1. + np.exp(-x))
    return out

def neural_network(W, x):
    a1 = sigmoid(np.dot(x, W[0]))
    out = np.dot(a1, W[1])
    return out

def A(x):
    out = np.sin(4 * x)
    #out = 5 * x
    return out

def psy_trial(x, net_out):
    out = A(x) + x * (1 - x) * net_out
    #out = A(x) + x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * net_out (for 2D)
    return out

def loss_function(W, x):
    loss_sum = 0.

    for xi in x:
        input_point = np.array(xi)
        net_out = neural_network(W, input_point)[0]

        psy_t_jacobian = jacobian(psy_trial)(input_point, net_out)
        psy_t_hessian = jacobian(jacobian(psy_trial))(input_point, net_out)

        gradient_of_trial_d2x = psy_t_hessian[0]
        gradient_of_trial_dt = psy_t_jacobian[0]

        func = f(input_point)

        err_sqr = ((gradient_of_trial_d2x - gradient_of_trial_dt) - func)**2
        loss_sum = loss_sum + err_sqr
    return loss_sum



W = [npr.rand(2, 10), npr.randn(10, 1)]
learning_rate = 0.01
print loss_function(W, x_space)

print neural_network(W, np.array([1, 1]))

print("init weight...")
for i in range(200):
    print('%d' % i)
    print loss_function(W, x_space)

    loss_grad = grad(loss_function)(W, x_space)
    W[0] = W[0] - learning_rate * loss_grad[0]
    W[1] = W[1] - learning_rate * loss_grad[1]


surface2 = np.zeros((nx, nt))


print("neural net solution...")
for i, x in enumerate(x_space):
    for j, t in enumerate(t_space):
        net_outt = neural_network(W, t)[0]
        surface2[i] = psy_trial(x, net_outt)

print surface2[:, 2]
plt.plot(x_space, surface2[:, 2], 'r--', label="prediksi")
plt.title("Per bandingan hasil Analitik dan prediksi dari NN")
plt.legend()
plt.show()
