import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.linalg import pinv, inv
from scipy.fft import *
from tests.prelim_code.tst_example_fields import *
from tests.prelim_code.prelim_inversion import *
from tests.prelim_code.prelim_regularization import *

# Define parameters
elastic_modulus = 1000
s = 0.48
pix_per_mu = 9.68
beta = 0.1
lambda_2 = 0.001

sigma = 5
x0, y0 = 5, 5
width_x, width_y = 4, 4

point_dens = 50
x_min, y_min = -10, -10
x_max, y_max = 10, 10

# Create grid of points in spatial coordinates
x_val, y_val = np.linspace(x_min, x_max, point_dens), np.linspace(y_min, y_max, point_dens)
xx, yy = np.meshgrid(x_val, y_val)

# Define forward Fredholm term f(x, y) under integral
forward_fx, forward_fy, forward_glob_norm = tri_pole(xx, yy, x0, y0, sigma)

# Define inverse Fredholm term u(x,y) on left side of integral equation
inverse_ux, inverse_uy, inverse_norm = vortex(xx, yy, x0, y0)
inverse_u = np.concatenate([inverse_ux.reshape(point_dens ** 2, 1),
                            inverse_uy.reshape(point_dens ** 2, 1)])

# Calculate fourier transform of 2D field components
forward_fft_fx = fft2(forward_fx)
forward_fft_fy = fft2(forward_fy)
forward_fft_f = np.concatenate([forward_fft_fx.reshape(point_dens ** 2, 1),
                                forward_fft_fy.reshape(point_dens ** 2, 1)])

inverse_fft_ux = fft2(inverse_ux)
inverse_fft_uy = fft2(inverse_uy)
inverse_fft_u = np.concatenate([inverse_fft_ux.reshape(point_dens ** 2, 1),
                                inverse_fft_uy.reshape(point_dens ** 2, 1)])

fx, fy, ft_fx, ft_fy, ft_ux, ft_uy, kxx, kyy, gamma_glob = (
    traction_fourier(xx, yy, inverse_ux, inverse_uy, s, elastic_modulus, None, pix_per_mu))

# Calculate forward solution
fourier_u = gamma_glob @ forward_fft_f
fourier_ux = ifft2(fourier_u[:point_dens ** 2].reshape(point_dens, point_dens)).real
fourier_uy = ifft2(fourier_u[point_dens ** 2:].reshape(point_dens, point_dens)).real

# Calculate inverse solution
fourier_f = tikhonov_simple(gamma_glob, inverse_fft_u, 0.0001)
fourier_fx = ifft2(fourier_f[:point_dens ** 2].reshape(point_dens, point_dens)).real
fourier_fy = ifft2(fourier_f[point_dens ** 2:].reshape(point_dens, point_dens)).real

# Plots
# Create subplot for forward solution
fig_forward, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle('Fourier method: Forward solution')

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

fig_forward.savefig('fourier_forward.png', dpi=300, bbox_inches="tight")
axs[0].set_title('Input: 2D traction field f(x,y)')
axs[1].set_title('Fourier forward solution u(x,y) to Fredholm integral')

# Create subplot for inverse solution
fig_inverse, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle('Fourier method: Inverse solution')

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

fig_inverse.savefig('fourier_inverse.png', dpi=300, bbox_inches="tight")
axs[0].set_title('Input: 2D displacement field u(x,y)')
axs[1].set_title('Fourier inverse solution f(x,y) to Fredholm integral')

plt.suptitle('Fourier method')
plt.tight_layout()
plt.show()
