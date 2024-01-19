import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.signal import convolve
from scipy.sparse import spdiags
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

# Create grid of points in Fourier coordinates
meshsize_x, meshsize_y = x_val[1] - x_val[0], y_val[1] - y_val[0]
k_x, k_y = fftfreq(x_val.shape[0], d=meshsize_x), fftfreq(y_val.shape[0], d=meshsize_y)
kxx, kyy = np.meshgrid(k_x, k_y)

# Define forward Fredholm term f(x, y) under integral
forward_fx, forward_fy, forward_glob_norm = tri_pole(xx, yy, x0, y0, sigma)

# Define inverse Fredholm term u(x,y) on left side of integral equation
inverse_ux, inverse_uy, inverse_norm = vortex(xx, yy, x0, y0)

# Calculate fourier transform of 2D field components
forward_fft_fx = fft2(forward_fx)
forward_fft_fy = fft2(forward_fy)
forward_fft_f = np.concatenate([forward_fft_fx.reshape(point_dens ** 2, 1),
                                forward_fft_fy.reshape(point_dens ** 2, 1)])

inverse_fft_ux = fft2(inverse_ux)
inverse_fft_uy = fft2(inverse_uy)
inverse_fft_u = np.concatenate([inverse_fft_ux.reshape(point_dens ** 2, 1),
                                inverse_fft_uy.reshape(point_dens ** 2, 1)])

# Calculate Green's matrix in Fourier space
g_xx, g_xy, g_yy = kernel_ft(kxx, kyy, s, elastic_modulus)

# Set all zero frequency components in greens function to zero
g_xx[0, 0] = 0
g_xy[0, 0] = 0
g_yy[0, 0] = 0

i_max = len(kxx[0, :])
j_max = len(kyy[:, 0])

g_xy[int(i_max // 2), :] = 0  # Set values in middle row to zero
g_xy[:, int(j_max // 2)] = 0  # Set values in middle column to zero

# Reshape matrices from dim(i_max, j_max) to dim(1, i_max * j_max) by concatenating rows
g1 = g_xx.reshape(1, i_max * j_max)
g2 = g_yy.reshape(1, i_max * j_max)
g3 = g_xy.reshape(1, i_max * j_max)

# Create zero filled matrix in the shape of g3
g4 = np.zeros(g3.shape)

# Concatenate and transposing g1 & g2 along first axis resulting in array of dim(i_max * j_max, 1) and flatten by
# concatenating rows
x1 = np.array([g1, g2]).T.flatten()
x2 = np.array([g3, g4]).T.flatten()

# Transpose and add dummy dimension to get array with dim(i_max * j_max * 2, 1)
x1 = np.expand_dims(x1, axis=1)
x2 = np.expand_dims(x2, axis=1)

# Eliminate the padding of zeros in x3 that was added during the construction of g4
x3 = x2[1:]

# Create a column vector (pad) containing a single element, which is 0
pad = np.expand_dims(np.array([0]), axis=1)

# Concatenate three arrays along the first axis
data = np.array([np.concatenate([x3, pad]).T, x1.T, np.concatenate([pad, x3]).T])
data = np.squeeze(data, axis=1)  # Removes the unnecessary singleton dimension introduced by np.expand_dims

# Create 2D sparse matrix representing the differential operator acting on Fourier-transformed displacement fields
X = spdiags(data, (-1, 0, 1), len(x1), len(x1))

# ********** MAGIC *************
inverse_glob_u = np.concatenate([inverse_ux.reshape(point_dens ** 2, 1),
                                 inverse_uy.reshape(point_dens ** 2, 1)])

fourier_ux = ifft2(forward_fft_fx * g_xx + forward_fft_fy * g_xy).real
fourier_uy = ifft2(forward_fft_fx * g_xy + forward_fft_fy * g_yy).real

fourier_u = X.T @ forward_fft_f

fourier_fx = ifft2(fourier_u[point_dens ** 2:].reshape(point_dens, point_dens)).real
fourier_fy = ifft2(fourier_u[:point_dens ** 2].reshape(point_dens, point_dens)).real


# Plots
# Create subplot for forward solution
fig_forward, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle('Boundary element method: Forward solution')

# Quiver plot for the first vector field
im = axs[0].imshow(np.rot90(np.sqrt(forward_fx ** 2 + forward_fy ** 2), 3),
                   extent=[np.min(xx), np.max(xx), np.min(yy), np.max(yy)],
                   interpolation="bicubic",
                   cmap="jet")
axs[0].quiver(xx, yy, forward_fx, forward_fy, color='black')
cbar = fig_forward.colorbar(im, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Traction stress [Pa]", rotation=270, labelpad=20, size=14)
cbar.ax.tick_params(labelsize=14)

# Quiver plot for the second vector field
im = axs[1].imshow(np.rot90(np.sqrt(fourier_ux ** 2 + fourier_uy ** 2), 3),
                   extent=[np.min(xx), np.max(xx), np.min(yy), np.max(yy)],
                   interpolation="bicubic",
                   cmap="jet")
axs[1].quiver(xx, yy, fourier_ux, fourier_uy, color='black')
axs[0].set_axis_off()
axs[1].set_axis_off()
cbar = fig_forward.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Displacement field [\u03bcm]", rotation=270, labelpad=20, size=14)
cbar.ax.tick_params(labelsize=14)

fig_forward.savefig('bem_forward.png', dpi=300, bbox_inches="tight")
axs[0].set_title('Input: 2D traction field f(x,y)')
axs[1].set_title('BEM forward solution u(x,y) to Fredholm integral')

# Create subplot for inverse solution
fig_inverse, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle('Boundary element method: Inverse solution')

# Quiver plot for the third vector field
im = axs[0].imshow(np.rot90(np.sqrt(inverse_ux ** 2 + inverse_uy ** 2), 3),
                   extent=[np.min(xx), np.max(xx), np.min(yy), np.max(yy)],
                   interpolation="bicubic",
                   cmap="jet")
axs[0].quiver(xx, yy, inverse_ux, inverse_uy, color='black')
cbar = fig_inverse.colorbar(im, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Displacement field [\u03bcm]", rotation=270, labelpad=20, size=14)
cbar.ax.tick_params(labelsize=14)

# Quiver plot for the fourth vector field
im = axs[1].imshow(np.rot90(np.sqrt(fourier_fx ** 2 + fourier_fy ** 2), 3),
                   extent=[np.min(xx), np.max(xx), np.min(yy), np.max(yy)],
                   interpolation="bicubic",
                   cmap="jet")
axs[1].quiver(xx, yy, fourier_fx, fourier_fy, color='black')
axs[0].set_axis_off()
axs[1].set_axis_off()
cbar = fig_inverse.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Traction stress [Pa]", rotation=270, labelpad=20, size=14)
cbar.ax.tick_params(labelsize=14)

fig_inverse.savefig('bem_inverse.png', dpi=300, bbox_inches="tight")
axs[0].set_title('Input: 2D displacement field u(x,y)')
axs[1].set_title('BEM inverse solution f(x,y) to Fredholm integral')

plt.suptitle('Boundary element method')
plt.tight_layout()
plt.show()
