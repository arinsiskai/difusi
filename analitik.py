import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

# t = 0
D = 0.003
n = 1000
def analytic_solution(x, t):
    analitik = 0
    for i in range(1, n, 2):
        #sum = 400. / ((np.pi**3) * (2.0 * n - 1)) * np.sin((2. * n - 1) * np.pi * x)
        #ganjil = (2 * n) - 1
        a = 400.0 / ((np.pi ** 3) * (i ** 3))
        b = np.sin((i * np.pi * x))
        c = np.exp(-D * (i ** 2) * (np.pi ** 2) * t)
        sum = a * b * c
        analitik += sum

    return analitik


nx = 100
nt = 100
x_space = np.linspace(0, 1, nx)
t_space = np.linspace(0, 100, nt)

X, Y = np.meshgrid(x_space, t_space)
R = analytic_solution(X, Y)


fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, t_space)
Z = analytic_solution(X, Y)
surf = ax.plot_surface(X, Y, R, cmap=cm.coolwarm, linewidth=0, antialiased=False)

'''
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 2)

'''

ax.set_xlabel('$x$')
ax.set_ylabel('$t$')

plt.show()







'''
D = 0.003
n = 1000
analitik = 0
for i in range(n):
    ganjil = (2. * n) - 1.
    a = 400.0 / ((np.pi ** 3) * ganjil ** 3)
    b = np.sin(np.deg2rad(ganjil * np.pi * x))
    c = np.exp(-(ganjil ** 2 * np.pi ** 2 * D * t))
    sum = a * b * c
    analitik += sum
    
print analitik
#plt.plot(t_space, surface)
#plt.show()

ganjil = 1
x = 1
b = np.sin(np.deg2rad(ganjil * np.pi * x))


print b
'''




