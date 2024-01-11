import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.signal import convolve
from scipy.fft import *
from tests.prelim_code.tst_utilis import *


# Define Fredholm integral function
def boxcar1dim(x, width_x):
    return np.where((np.abs(x) <= width_x / 2), 1, 0)


# Define Fredholm kernel
def kernel(x, width_x):
    return np.where((np.abs(x) <= width_x / 2), 1, 0)


# Define shape function
def pyramid1dim(x, width_x):
    return np.maximum(0, width_x - np.abs(x))


# Create grid of points in spatial coordinates
point_dens = 1000
x_val = np.linspace(-10, 10, point_dens)
meshsize_x = x_val[1] - x_val[0]

# Calculate point density in integration interval
int_den_x = len(x_val) / (x_val[-1] - x_val[0])

# Set gaussian width
width_x = 2

# Define empty array to store convolution results
gamma_glob = np.zeros((point_dens, point_dens))

# Loop over grid in spatial domain and calculate BEM matrix components
for k, x_k in enumerate(x_val):
    gamma_loc = convolve(kernel(x_val, width_x),
                         pyramid1dim(x_val - x_k, meshsize_x),
                         mode='same')

    # Sum up BEM matrices
    block = k
    gamma_glob[block, :] = gamma_loc
    print(f'Index {k}')

# Invert BEM matrix
gamma_glob_inv = scipy.linalg.inv(gamma_glob)

# Define forward Fredholm term u(x) on left side of integral equation
forward_glob = boxcar1dim(x_val, width_x).flatten()

# Define inverse Fredholm term f(x) under integral
inverse_glob = pyramid1dim(x_val, width_x).flatten()

# Calculate finite element solution
bem_forward = (gamma_glob @ forward_glob).flatten()
bem_inverse = (gamma_glob_inv @ inverse_glob).flatten()

# Create a plot
fig = plt.figure(figsize=(15, 12))

# First row
ax1 = fig.add_subplot(131)
ax1.plot(x_val, boxcar1dim(x_val, width_x), alpha=0.8)
ax1.set_title('Input function to Fredberg integral')

ax2 = fig.add_subplot(132)
ax2.plot(x_val, bem_forward, alpha=0.8)
ax2.set_title('BEM forward solution to Fredberg integral')

ax3 = fig.add_subplot(133)
ax3.plot(x_val, bem_inverse, alpha=0.8)
ax3.set_title('BEM inverse solution to Fredberg integral')

# Set labels
for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')

plt.suptitle('Boundary element method')
plt.tight_layout()
plt.show()
