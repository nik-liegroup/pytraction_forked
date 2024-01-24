import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.fft import *
from tests.prelim_code.prelim_utilis import *
from tests.prelim_code.tst_example_fields import *

# Create grid of points in spatial coordinates
point_dens = 30
x_val, y_val = np.linspace(-10, 10, point_dens), np.linspace(-10, 10, point_dens)
xx, yy = np.meshgrid(x_val, y_val)
meshsize_x, meshsize_y = x_val[1] - x_val[0], y_val[1] - y_val[0]

k_x, k_y = (np.fft.fftfreq(x_val.shape[0], d=meshsize_x),
            np.fft.fftfreq(y_val.shape[0], d=meshsize_y))
kxx, kyy = np.meshgrid(k_x, k_y)

# Set box width
width_x = 4
width_y = 4

# Define kernel
kernel = boxcar2dim_ft(kxx, kyy, width_x, width_y)
kernel_flat = kernel.reshape(point_dens ** 2, 1)
gamma_glob = np.diagflat(kernel_flat)

# Invert Fourier matrix
gamma_glob_inv = scipy.linalg.inv(gamma_glob)

# Define forward Fredholm term f(x, y) under integral
forward_glob = fft2(boxcar2dim(xx, yy, width_x, width_y))
forward_glob = forward_glob.reshape(point_dens ** 2, 1)

# Define inverse Fredholm term u(x,y) on left side of integral equation
inverse_glob = fft2(pyramid2dim(xx, yy, width_x, width_y))
inverse_glob = inverse_glob.reshape(point_dens ** 2, 1)

# Calculate solution using convolution theorem
ft_forward = ifft2((gamma_glob @ forward_glob).reshape(point_dens, point_dens)).real
ft_inverse = ifft2((gamma_glob_inv @ inverse_glob).reshape(point_dens, point_dens)).real

# Create a plot
fig = plt.figure(figsize=(15, 12))

# First row
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(xx, yy, boxcar2dim(xx, yy, width_x, width_y), cmap='viridis', alpha=0.8)
ax1.set_title('Input function to Fredberg integral')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(xx, yy, ft_forward, cmap='viridis', alpha=0.8)
ax2.set_title('Fourier forward solution to Fredberg integral')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(xx, yy, ft_inverse, cmap='viridis', alpha=0.8)
ax3.set_title('Fourier inverse solution to Fredberg integral')

# Set labels
for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')

plt.suptitle('Fourier method')
plt.tight_layout()
plt.show()
