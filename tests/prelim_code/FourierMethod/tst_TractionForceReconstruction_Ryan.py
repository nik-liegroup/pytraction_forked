import numpy as np
import matplotlib.pyplot as plt
from tests.prelim_code.tst_example_fields import *
from tests.prelim_code.prelim_regularization import *
from tests.prelim_code.prelim_inversion import *

# Define parameters
elastic_modulus = 1000
s = 0.48
pix_per_mu = 9.68
beta = 0.1

sigma = 5
x0, y0 = 5, 5
width_x, width_y = 4, 4

point_dens = 50
x_min, y_min = -20, -20
x_max, y_max = 20, 20

# Create grid of points in spatial coordinates
x_val, y_val = (np.linspace(x_min, x_max, point_dens),
                np.linspace(y_min, y_max, point_dens))
xx, yy = np.meshgrid(x_val, y_val)

i_max = (np.shape(xx)[0])
j_max = (np.shape(yy)[1])
meshsize = x_val[1] - x_val[0]

# Define inverse Fredholm term u(x,y) on left side of integral equation
inverse_ux, inverse_uy, inverse_norm = vortex(xx, yy, x0, y0)
inverse_u = np.array([inverse_ux.flatten(), inverse_uy.flatten()])

# Calculate inverse solution
_, _, _, _, ft_ux, ft_uy, kxx, kyy, X = traction_fourier(
    xx, yy, inverse_ux, inverse_uy, s, elastic_modulus, lambd=None, scaling_factor=pix_per_mu, zdepth=0
)

# Calculate lambda from bayesian model
L, evidencep, evidence_one = bayesian_regularization_ft(
    beta, ft_ux, ft_uy, elastic_modulus, s, i_max, j_max, X, xx, yy, inverse_ux, inverse_uy
)

# Calculate traction field in fourier space and transform back to spatial domain
fx, fy, ft_fx, ft_fy, ft_ux, ft_uy, kxx, kyy, X = traction_fourier(
    xx, yy, inverse_ux, inverse_uy, s, elastic_modulus, lambd=L, scaling_factor=pix_per_mu, zdepth=0
)

# Define forward Fredholm term f(x, y) under integral
forward_fx, forward_fy, forward_glob_norm = tri_pole(xx, yy, x0, y0, sigma)
forward_f = np.array([forward_fx.flatten(), forward_fy.flatten()])
fourier_u = X @ forward_f.flatten()
fourier_ux = fourier_u[:i_max * j_max].reshape(i_max, j_max)
fourier_uy = fourier_u[i_max * j_max:].reshape(i_max, j_max)

# Plots
# Create subplot for forward solution
fig_forward, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle('Ryans fourier method: Forward solution')

# Quiver plot for the first vector field
im = axs[0].imshow(np.rot90(np.sqrt(forward_fx ** 2 + forward_fy ** 2), 3),
                   extent=[np.min(xx), np.max(xx),
                           np.min(yy), np.max(yy)],
                   interpolation="bicubic",
                   cmap="jet")
axs[0].quiver(xx, yy, forward_fx, forward_fy, color='black')
cbar = fig_forward.colorbar(im, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Traction stress [Pa]", rotation=270, labelpad=20, size=14)
cbar.ax.tick_params(labelsize=14)

# Quiver plot for the second vector field
im = axs[1].imshow(np.rot90(np.sqrt(fourier_ux ** 2 + fourier_uy ** 2), 3),
                   extent=[np.min(xx), np.max(xx),
                           np.min(yy), np.max(yy)],
                   interpolation="bicubic",
                   cmap="jet")
axs[1].quiver(xx, yy, fourier_ux, fourier_uy, color='black')
axs[0].set_axis_off()
axs[1].set_axis_off()
cbar = fig_forward.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Displacement field [\u03bcm]", rotation=270, labelpad=20, size=14)
cbar.ax.tick_params(labelsize=14)

fig_forward.savefig('ryan_fourier_forward.png', dpi=300, bbox_inches="tight")
axs[0].set_title('Input: 2D traction field f(x,y)')
axs[1].set_title('Fourier forward solution u(x,y) to Fredholm integral')

# Create subplot for inverse solution
fig_inverse, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle('Fourier method: Inverse solution')

# Quiver plot for the third vector field
im = axs[0].imshow(np.rot90(np.sqrt(inverse_ux ** 2 + inverse_uy ** 2), 3),
                   extent=[np.min(xx), np.max(xx),
                           np.min(yy), np.max(yy)],
                   interpolation="bicubic",
                   cmap="jet")
axs[0].quiver(xx, yy, inverse_ux, inverse_uy, color='black')
cbar = fig_inverse.colorbar(im, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Displacement field [\u03bcm]", rotation=270, labelpad=20, size=14)
cbar.ax.tick_params(labelsize=14)

# Quiver plot for the fourth vector field
im = axs[1].imshow(np.rot90(np.sqrt(fx ** 2 + fy ** 2), 3),
                   extent=[np.min(xx), np.max(xx),
                           np.min(yy), np.max(yy)],
                   interpolation="bicubic",
                   cmap="jet")
axs[1].quiver(xx, yy, fx, fy, color='black')
axs[0].set_axis_off()
axs[1].set_axis_off()
cbar = fig_inverse.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Traction stress [Pa]", rotation=270, labelpad=20, size=14)
cbar.ax.tick_params(labelsize=14)

fig_inverse.savefig('ryan_fourier_inverse.png', dpi=300, bbox_inches="tight")
axs[0].set_title('Input: 2D displacement field u(x,y)')
axs[1].set_title('Fourier inverse solution f(x,y) to Fredholm integral')

plt.suptitle('Ryans fourier element method')
plt.tight_layout()
plt.show()
