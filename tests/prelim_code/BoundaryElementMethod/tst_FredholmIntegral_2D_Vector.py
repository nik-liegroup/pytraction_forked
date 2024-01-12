import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.signal import convolve
from scipy.fft import *
from tests.prelim_code.tst_utilis import *
from tests.prelim_code.tst_example_fields import *

# Create grid of points in spatial coordinates
point_dens = 30
x_val, y_val = np.linspace(-10, 10, point_dens), np.linspace(-10, 10, point_dens)
xx, yy = np.meshgrid(x_val, y_val)
meshsize_x, meshsize_y = x_val[1] - x_val[0], y_val[1] - y_val[0]

# Calculate point density in integration interval
int_den_x = len(x_val) / (x_val[-1] - x_val[0])
int_den_y = len(y_val) / (y_val[-1] - y_val[0])

# Set box width
width_x = 4
width_y = 4

# Define empty array to store convolution results
gamma_glob_xx, gamma_glob_xy, gamma_glob_yy = (np.zeros((point_dens ** 2, point_dens ** 2)),
                                               np.zeros((point_dens ** 2, point_dens ** 2)),
                                               np.zeros((point_dens ** 2, point_dens ** 2)))

# Loop over grid in spatial domain and calculate BEM matrix components
for i, x_k in enumerate(x_val):
    for j, y_k in enumerate(y_val):
        box_kernel = boxcar2dim(xx, yy, width_x, width_y)
        pyr_shape = pyramid2dim(xx - x_k, yy - y_k, meshsize_x, meshsize_y)
        gamma_xx = convolve(box_kernel,
                            pyr_shape,
                            mode='same')
        gamma_xy = convolve(box_kernel,
                            pyr_shape,
                            mode='same')
        gamma_yy = convolve(box_kernel,
                            pyr_shape,
                            mode='same')

        # Sum up BEM matrices
        block_i = point_dens * i
        block_j = point_dens * j

        # Swapping rows and columns in matrix equals multiplication by permutation matrix (Involutory matrix)
        # This justifies to separate vector components as the matrices inverse is invariant under this operation
        gamma_glob_xx[block_i:block_i + point_dens:, block_j:block_j + point_dens] = gamma_xx
        gamma_glob_xy[block_i:block_i + point_dens:, block_j:block_j + point_dens] = gamma_xy
        gamma_glob_yy[block_i:block_i + point_dens:, block_j:block_j + point_dens] = gamma_yy
        print(f'Index {i, j}')

# Concatenate along the columns (axis=1)
gamma_r1 = np.concatenate((gamma_glob_xx, gamma_glob_xy), axis=1)
gamma_r2 = np.concatenate((gamma_glob_xy, gamma_glob_yy), axis=1)

# Concatenate along the rows (axis=0)
gamma_glob = np.concatenate((gamma_r1, gamma_r2), axis=0)

# Invert BEM matrix
gamma_glob_inv = scipy.linalg.pinv(gamma_glob)

# Define forward Fredholm term f(x, y) under integral
forward_glob_fx = boxcar2dim(xx, yy, width_x, width_y).reshape(point_dens ** 2, 1)
forward_glob_fy = boxcar2dim(xx, yy, width_x, width_y).reshape(point_dens ** 2, 1)
forward_glob_f = np.concatenate([forward_glob_fx, forward_glob_fy])

# Define inverse Fredholm term u(x,y) on left side of integral equation
inverse_glob_ux = pyramid2dim(xx, yy, width_x, width_y).reshape(point_dens ** 2, 1)
inverse_glob_uy = pyramid2dim(xx, yy, width_x, width_y).reshape(point_dens ** 2, 1)
inverse_glob_u = np.concatenate([inverse_glob_ux, inverse_glob_uy])

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
axs[0].quiver(xx, yy, bem_forward_ux, bem_forward_uy, color='blue')
axs[0].set_title('BEM forward solution to Fredberg integral')

# Quiver plot for the second vector field
axs[1].quiver(xx, yy, bem_inverse_fx, bem_inverse_fy, color='red')
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
