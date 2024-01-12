import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.signal import convolve
from scipy.fft import *
from tests.prelim_code.tst_utilis import *

# Define parameters
elastic_modulus = 1000
s = 0.48
pix_per_mu = 1
beta = 0.1

sigma = 5
x0, y0 = 5, 5
width_x, width_y = 4, 4

point_dens = 50
x_min, y_min = -10, -10
x_max, y_max = 10, 10

# Create grid of points in spatial coordinates
x_val, y_val = np.linspace(x_min, x_max, point_dens), np.linspace(y_min, y_max, point_dens)
xx, yy = np.meshgrid(x_val, y_val)

# Calculate BEM matrix
gamma_glob = traction_force_bem(xx, yy, point_dens, s, elastic_modulus)

# Invert BEM matrix
gamma_glob_inv = scipy.linalg.inv(gamma_glob)

# Define forward Fredholm term f(x, y) under integral
forward_glob_fx, forward_glob_fy, forward_glob_norm = tri_pole(xx, yy, x0, y0, sigma)
forward_glob_fx, forward_glob_fy = (forward_glob_fx.reshape(point_dens ** 2, 1),
                                    forward_glob_fy.reshape(point_dens ** 2, 1))
forward_glob_f = np.concatenate([forward_glob_fx, forward_glob_fy])

# Define inverse Fredholm term u(x,y) on left side of integral equation
inverse_glob_ux, inverse_glob_uy, inverse_glob_norm = tri_pole(xx, yy, x0, y0, sigma)
inverse_glob_ux, inverse_glob_uy = (inverse_glob_ux.reshape(point_dens ** 2, 1),
                                    inverse_glob_uy.reshape(point_dens ** 2, 1))
inverse_glob_u = np.concatenate([inverse_glob_ux, inverse_glob_ux])

# Calculate finite element solution
bem_forward = (gamma_glob @ forward_glob_f)
bem_forward_ux = bem_forward[0:point_dens ** 2].reshape(point_dens, point_dens)
bem_forward_uy = bem_forward[point_dens ** 2:2 * point_dens ** 2].reshape(point_dens, point_dens)

bem_inverse = (gamma_glob_inv @ inverse_glob_u)
bem_inverse_fx = bem_inverse[0:point_dens ** 2].reshape(point_dens, point_dens)
bem_inverse_fy = bem_inverse[point_dens ** 2:2 * point_dens ** 2].reshape(point_dens, point_dens)

# Plots
# Create subplots
fig_quiver, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.subplots_adjust(hspace=0.27)

# Quiver plot for the first vector field
axs[0].quiver(xx, yy, forward_glob_fx, forward_glob_fy, color='blue')
axs[0].set_title('BEM forward solution to Fredberg integral')

# Quiver plot for the second vector field
axs[1].quiver(yy, xx, bem_forward_ux, bem_forward_uy, color='red')
axs[1].set_title('BEM inverse solution to Fredberg integral')

plt.suptitle('Boundary element method')
plt.tight_layout()

# Create a 3D plot
fig_surface = plt.figure(figsize=(15, 6))

# First row
ax1 = fig_surface.add_subplot(131, projection='3d')
ax1.plot_surface(xx, yy, boxcar2dim(xx, yy, width_x, width_y), cmap='viridis', alpha=0.8)
ax1.set_title('Input function to Fredberg integral')

ax2 = fig_surface.add_subplot(132, projection='3d')
ax2.plot_surface(xx, yy, bem_forward_ux, cmap='viridis', alpha=0.8)
ax2.set_title('BEM forward solution to Fredberg integral')

ax3 = fig_surface.add_subplot(133, projection='3d')
ax3.plot_surface(xx, yy, bem_inverse_fx, cmap='viridis', alpha=0.8)
ax3.set_title('BEM inverse solution to Fredberg integral')

# Set labels
for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

plt.suptitle('Boundary element method')
plt.tight_layout()
plt.show()
