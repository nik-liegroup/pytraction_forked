import numpy as np
import matplotlib.pyplot as plt
from tests.prelim_code.prelim_utilis import *
from tests.prelim_code.tst_example_fields import *

# Define parameters
elastic_modulus = 1000
s = 0.48
pix_per_mu = 1
beta = 1 / 1e-6
lambda_2 = 0.0001

sigma = 5
x0, y0 = 5, 5

point_dens = 30
x_min, y_min = -10, -10
x_max, y_max = 10, 10

# Create grid of points in spatial coordinates
x_val, y_val = np.linspace(x_min, x_max, point_dens), np.linspace(y_min, y_max, point_dens)
xx, yy = np.meshgrid(x_val, y_val, indexing='xy')

# Calculate BEM matrix
gamma_glob = traction_bem(xx, yy, 'conv', point_dens, s, elastic_modulus)

# Define forward Fredholm term f(x, y) under integral
forward_fx, forward_fy, forward_glob_norm = tri_pole(xx, yy, x0, y0, sigma)
forward_glob_f = np.concatenate([forward_fx.reshape(point_dens ** 2, 1),
                                 forward_fy.reshape(point_dens ** 2, 1)])

# Define inverse Fredholm term u(x,y) on left side of integral equation
inverse_ux, inverse_uy, inverse_norm = vortex(xx, yy, x0, y0)
inverse_glob_u = np.concatenate([inverse_ux.reshape(point_dens ** 2, 1),
                                 inverse_uy.reshape(point_dens ** 2, 1)])

# Calculate forward solution
bem_forward_u = (gamma_glob @ forward_glob_f)
bem_forward_ux = bem_forward_u[:point_dens ** 2].reshape(point_dens, point_dens).T
bem_forward_uy = bem_forward_u[point_dens ** 2:].reshape(point_dens, point_dens).T

# Calculate optimal lambda for inverse solution
lambd = bayesian_regularization(gamma_glob, inverse_glob_u, beta)
bem_inverse_f = tikhonov_simple(gamma_glob, inverse_glob_u, lambd)


bem_inverse_fx = bem_inverse_f[:point_dens ** 2].reshape(point_dens, point_dens).T
bem_inverse_fy = bem_inverse_f[point_dens ** 2:].reshape(point_dens, point_dens).T

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
im = axs[1].imshow(np.rot90(np.sqrt(bem_forward_ux ** 2 + bem_forward_uy ** 2), 3),
                   extent=[np.min(xx), np.max(xx), np.min(yy), np.max(yy)],
                   interpolation="bicubic",
                   cmap="jet")
axs[1].quiver(xx, yy, bem_forward_ux, bem_forward_uy, color='black')
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
im = axs[1].imshow(np.rot90(np.sqrt(bem_inverse_fx ** 2 + bem_inverse_fy ** 2), 3),
                   extent=[np.min(xx), np.max(xx), np.min(yy), np.max(yy)],
                   interpolation="bicubic",
                   cmap="jet")
axs[1].quiver(xx, yy, bem_inverse_fx, bem_inverse_fy, color='black')
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
