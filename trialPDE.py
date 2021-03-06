import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

#plate size, mm

w = h = 5
#intervals in x-, y-, directions, mm
dx = dy = 0.1

x_space = np.linspace(0, w, w/dx)
y_space = np.linspace(0, h, h/dy)

print np.size(x_space)
print np.size(y_space)

#thermal diffusivity of stell, mm2.s-1
D = 4

Tcool, Thot = 300, 700

nx, ny = int(w/dx), int(h/dy)
print nx
print ny

dx2, dy2 = dx*dx, dy*dy
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

u0 = Tcool * np.ones((nx, ny))
u = np.empty((nx, ny))

#initial condition ring of inner radius r, width dr centred at (cx, cy)
r = 2
cx = 5
cy = 5
r2 = r**2
for i in range(nx):
    for j in range(ny):
        p2 = (i * dx - cx)**2 + (j * dy - cy)**2
        if p2 < r2:
            u0[i, j] = Thot

def do_timestep(u0, u):
    # Propagate with forward-difference in time, central-difference in space
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * ((u0[2:, 1:-1] - 2 * u0[1:-1, 1:-1] + \
                                                u0[:-2, 1:-1]) / dx2 + (u0[1:-1, 2:] - 2 * u0[1:-1, 1:-1] + u0[1:-1, :-2]) / dy2)
    u0 = u.copy()
    return u0, u

# Number of timesteps
nsteps = 101
# Output 4 figures at these timesteps
mfig = [0, 10, 50, 100]
fignum = 0
fig = plt.figure()
for m in range(nsteps):
    u0, u = do_timestep(u0, u)
    if m in mfig:
        fignum += 1
        print(m, fignum)
        ax = fig.add_subplot(220 + fignum)
        im = ax.imshow(u.copy(), cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
        ax.set_axis_off()
        ax.set_title('{:.1f} ms'.format(m*dt*1000))
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel('$T$ / K', labelpad=20)
fig.colorbar(im, cax=cbar_ax)
plt.show()
