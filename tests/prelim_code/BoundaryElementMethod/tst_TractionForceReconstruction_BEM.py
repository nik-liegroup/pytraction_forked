import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.signal import convolve
from scipy.fft import *
from tests.prelim_code.tst_utilis import *
from tests.prelim_code.tst_example_fields import *

# Define parameters
elastic_modulus = 1000
s = 0.48
pix_per_mu = 1
beta = 0.1
lambda_2 = 0.0001

sigma = 5
x0, y0 = 5, 5
width_x, width_y = 4, 4

point_dens = 50
x_min, y_min = -10, -10
x_max, y_max = 10, 10

# Create grid of points in spatial coordinates
x_val, y_val = np.linspace(x_min, x_max, point_dens), np.linspace(y_min, y_max, point_dens)
xx, yy = np.meshgrid(x_val, y_val, indexing='xy')

# Calculate BEM matrix
gamma_glob = traction_bem(xx, yy, 'fft', point_dens, s, elastic_modulus)

# Define forward Fredholm term f(x, y) under integral
forward_fx, forward_fy, forward_glob_norm = tri_pole(xx, yy, x0, y0, sigma)
forward_glob_f = np.concatenate([forward_fx.reshape(point_dens ** 2, 1),
                                 forward_fy.reshape(point_dens ** 2, 1)])

# Define inverse Fredholm term u(x,y) on left side of integral equation
inverse_ux, inverse_uy, inverse_norm = tri_pole(xx, yy, x0, y0, sigma)
inverse_glob_u = np.concatenate([inverse_ux.reshape(point_dens ** 2, 1),
                                 inverse_uy.reshape(point_dens ** 2, 1)])

# Calculate forward solution
bem_forward_u = (gamma_glob @ forward_glob_f)
bem_forward_ux = bem_forward_u[:point_dens ** 2].reshape(point_dens, point_dens).T
bem_forward_uy = bem_forward_u[point_dens ** 2:].reshape(point_dens, point_dens).T

# Calculate inverse solution
bem_inverse_f = tikhonov(gamma_glob, inverse_glob_u, lambda_2)
bem_inverse_fx = bem_inverse_f[:point_dens ** 2].reshape(point_dens, point_dens).T
bem_inverse_fy = bem_inverse_f[point_dens ** 2:].reshape(point_dens, point_dens).T

# Plots
# Create subplot for forward solution
fig_forward, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle('Boundary element method: Forward solution')

# Quiver plot for the first vector field
axs[0].quiver(xx, yy, forward_fx, forward_fy, color='blue')
axs[0].set_title('Input: 2D traction field f(x,y)')

# Quiver plot for the second vector field
axs[1].quiver(xx, yy, bem_forward_ux, bem_forward_uy, color='red')
axs[1].set_title('BEM forward solution u(x,y) to Fredholm integral')

# Create subplot for inverse solution
fig_inverse, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle('Boundary element method: Inverse solution')

# Quiver plot for the third vector field
axs[0].quiver(xx, yy, inverse_ux, inverse_uy, color='blue')
axs[0].set_title('Input: 2D displacement field u(x,y)')

# Quiver plot for the fourth vector field
axs[1].quiver(xx, yy, bem_inverse_fx, bem_inverse_fy, color='red')
axs[1].set_title('BEM inverse solution f(x,y) to Fredholm integral')

plt.suptitle('Boundary element method')
plt.tight_layout()
plt.show()
