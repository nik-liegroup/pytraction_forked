import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.signal import convolve
from scipy.fft import *
from tests.prelim_code.tst_utilis import *


# Create grid of points in spatial coordinates
point_dens = 50
x_val, y_val = np.linspace(-10, 10, point_dens), np.linspace(-10, 10, point_dens)
xx, yy = np.meshgrid(x_val, y_val)
meshsize_x, meshsize_y = x_val[1] - x_val[0], y_val[1] - y_val[0]

# Calculate point density in integration interval
int_den_x = len(x_val) / (x_val[-1] - x_val[0])
int_den_y = len(y_val) / (y_val[-1] - y_val[0])

# Set box width
width_x = 2
width_y = 2

# Define empty array to store convolution results
gamma_glob = np.zeros((point_dens ** 2, point_dens ** 2))

# Loop over grid in spatial domain and calculate BEM matrix components
for i, x_k in enumerate(x_val):
    for j, y_k in enumerate(y_val):
        gamma_loc = convolve(boxcar2dim(xx, yy, width_x, width_y),
                             pyramid2dim(xx - x_k, yy - y_k, meshsize_x, meshsize_y),
                             mode='same')

        # Sum up BEM matrices
        block_i = point_dens * i
        block_j = point_dens * j
        gamma_glob[block_i:block_i + point_dens, block_j:block_j + point_dens] = gamma_loc
        print(f'Index {i, j}')

# Invert BEM matrix
gamma_glob_inv = scipy.linalg.inv(gamma_glob)

# Define forward Fredholm term u(x, y) on left side of integral equation
forward_glob = boxcar2dim(xx, yy, width_x, width_y)
forward_glob = forward_glob.reshape(point_dens ** 2, 1)

# Define inverse Fredholm term f(x, y) under integral
inverse_glob = pyramid2dim(xx, yy, width_x, width_y)
inverse_glob = inverse_glob.reshape(point_dens ** 2, 1)

# Calculate finite element solution
bem_forward = (gamma_glob @ forward_glob).reshape(point_dens, point_dens)
bem_inverse = (gamma_glob_inv @ inverse_glob).reshape(point_dens, point_dens)

# Create a plot
fig = plt.figure(figsize=(15, 12))

# First row
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(xx, yy, boxcar2dim(xx, yy, width_x, width_y), cmap='viridis', alpha=0.8)
ax1.set_title('Input function to Fredberg integral')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(xx, yy, bem_forward, cmap='viridis', alpha=0.8)
ax2.set_title('BEM forward solution to Fredberg integral')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(xx, yy, bem_inverse, cmap='viridis', alpha=0.8)
ax3.set_title('BEM inverse solution to Fredberg integral')

# Set labels
for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')

plt.suptitle('Boundary element method')
plt.tight_layout()
plt.show()
