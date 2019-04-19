import autograd.numpy as np
from autograd import grad, jacobian
import matplotlib.pyplot as plt

nData = 100     # input size
H1 = 100     # Hidden layer size
H2 = 50
Out = 1      # Output size

x_space = np.linspace(0, 1, nData)
y_space = np.linspace(0, 1, nData)

inputN = np.array((x_space, y_space))

N = np.reshape(inputN, np.size(inputN))

W1 = np.random.randn(np.size(N), H1)

b1 = np.random.randn(H1)

W2 = np.random.randn(H1, H2)
b2 = np.random.randn(H2)

W3 = np.random.randn(H2, Out)
b3 = np.random.randn(Out)


def sigmoid(X):
    out = 1 / (1 + np.exp(-X))
    return out


def relu(X):
    out = np.maximum(0, X)
    return out


def sigmoid_backward(dA, X):
    sig = sigmoid(X)
    out = dA * sig * (1 - sig)
    return out


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def ffpass_np(x):
    '''
    a1 = np.dot(x, W1) + b1     # affine
    r = np.maximum(0, a1)       # ReLU
    a2 = np.dot(r, W2) + b2     # affine
    '''
    a1 = np.dot(x, W1) + b1
    r = relu(a1)
    a2 = np.dot(r, W2) + b2
    s = sigmoid(a2)
    out = np.dot(s, W3) + b3
    return out


out_ffpassx = ffpass_np(x_space)
out_ffpassy = ffpass_np(y_space)


def A(x):
    out = x[1] * np.sin(np.pi * x[0])
    return out

def psy_trial(x, net_out):
    out = A(x) + x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * net_out
    return out


out_psy_trial = psy_trial(N, ffpass_np(N))


def f(x):
    return 0.


def loss_function(x, y):
    loss_sum = 0.

    for xi in x:
        for yi in y:
            input_point = np.array([xi, yi])
            net_out = ffpass_np(input_point)

            psy_t_hessian = jacobian(jacobian(psy_trial))(input_point, net_out)

            gradient_of_trial_d2x = psy_t_hessian
            gradient_of_trial_d2y = psy_t_hessian

            func = f(input_point)

            err_sqr = ((gradient_of_trial_d2x + gradient_of_trial_d2y) - func)**2
            loss_sum = loss_sum + err_sqr
    return loss_sum


learning_rate = 0.01

print loss_function(x_space, y_space)



print("init weight...")
for i in range(500):
    print('%d' % i)
    print loss_function(x_space, y_space)
    loss_grad = grad(loss_function)(x_space, y_space)

    W1 = W1 - learning_rate * loss_grad[0]
    W2 = W2 - learning_rate * loss_grad[1]
    W3 = W3 - learning_rate * loss_grad[0]
print loss_function(x_space, y_space)

