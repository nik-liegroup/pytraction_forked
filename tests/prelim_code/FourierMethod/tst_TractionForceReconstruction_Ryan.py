import numpy as np
import matplotlib.pyplot as plt
from tests.prelim_code.tst_example_fields import *
from pytraction.process import *
from tests.prelim_code.prelim_utilis import *

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
pos = np.array([xx.flatten(), yy.flatten()])

# Define inverse Fredholm term u(x,y) on left side of integral equation
inverse_ux, inverse_uy, inverse_norm = vortex(xx, yy, x0, y0)
inverse_u = np.array([inverse_ux.flatten(), inverse_uy.flatten()])

# Calculate inverse solution
meshsize = xx[0, 1] - xx[0, 0]

grid_mat, u, i_max, j_max = interp_vec2grid(pos, inverse_u, meshsize, [])

ftux, ftuy, kxx, kyy, i_max, j_max, X = fourier_xu(u,
                                                   i_max,
                                                   j_max,
                                                   elastic_modulus,
                                                   s,
                                                   meshsize)
point_dens = int(np.shape(u[:, :, 0])[0])
gamma_glob = traction_fourier(u[:, :, 0], u[:, :, 1], point_dens, s, elastic_modulus)

# Calculate lambda from bayesian model
L, evidencep, evidence_one = optimal_lambda(
    beta, ftux, ftuy, kxx, kyy, elastic_modulus, s, meshsize, i_max, j_max, X, 1
)

# Calculate traction field in fourier space and transform back to spatial domain
f_pos, f_nm_2, fourier_f, f_n_m, ftfx, ftfy = reg_fourier_tfm(
    ftux, ftuy, kxx, kyy, L, elastic_modulus, s, meshsize, i_max, j_max, pix_per_mu, 0, grid_mat
)

fourier_fx = f_n_m[:, :, 0]
fourier_fy = f_n_m[:, :, 1]

# Flip shapes back into position
fourier_f = fourier_f.reshape(i_max, j_max).T
fourier_f = np.flip(fourier_f, axis=0)

# Define forward Fredholm term f(x, y) under integral
forward_fx, forward_fy, forward_glob_norm = tri_pole(grid_mat[:, :, 0], grid_mat[:, :, 1], x0, y0, sigma)
forward_f = np.array([forward_fx.flatten(), forward_fy.flatten()])
fourier_u = X @ forward_f.flatten()

fourier_ux = fourier_u[:len(grid_mat[:, :, 0]) ** 2].reshape(len(grid_mat[:, :, 0]), len(grid_mat[:, :, 0]))
fourier_uy = fourier_u[len(grid_mat[:, :, 1]) ** 2:].reshape(len(grid_mat[:, :, 1]), len(grid_mat[:, :, 1]))

# Plots
# Create subplot for forward solution
fig_forward, axs = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle('Ryans fourier method: Forward solution')

# Quiver plot for the first vector field
im = axs[0].imshow(np.rot90(np.sqrt(forward_fx ** 2 + forward_fy ** 2), 3),
                   extent=[np.min(grid_mat[:, :, 0]), np.max(grid_mat[:, :, 0]),
                           np.min(grid_mat[:, :, 1]), np.max(grid_mat[:, :, 1])],
                   interpolation="bicubic",
                   cmap="jet")
axs[0].quiver(grid_mat[:, :, 0], grid_mat[:, :, 1], forward_fx, forward_fy, color='black')
cbar = fig_forward.colorbar(im, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Traction stress [Pa]", rotation=270, labelpad=20, size=14)
cbar.ax.tick_params(labelsize=14)

# Quiver plot for the second vector field
im = axs[1].imshow(np.rot90(np.sqrt(fourier_ux ** 2 + fourier_uy ** 2), 3),
                   extent=[np.min(grid_mat[:, :, 0]), np.max(grid_mat[:, :, 0]),
                           np.min(grid_mat[:, :, 1]), np.max(grid_mat[:, :, 1])],
                   interpolation="bicubic",
                   cmap="jet")
axs[1].quiver(grid_mat[:, :, 0], grid_mat[:, :, 1], fourier_ux, fourier_uy, color='black')
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
im = axs[0].imshow(np.rot90(np.sqrt(u[:, :, 0] ** 2 + u[:, :, 1] ** 2), 3),
                   extent=[np.min(grid_mat[:, :, 0]), np.max(grid_mat[:, :, 0]),
                           np.min(grid_mat[:, :, 1]), np.max(grid_mat[:, :, 1])],
                   interpolation="bicubic",
                   cmap="jet")
axs[0].quiver(grid_mat[:, :, 0], grid_mat[:, :, 1], u[:, :, 0], u[:, :, 1], color='black')
cbar = fig_inverse.colorbar(im, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label("Displacement field [\u03bcm]", rotation=270, labelpad=20, size=14)
cbar.ax.tick_params(labelsize=14)

# Quiver plot for the fourth vector field
im = axs[1].imshow(np.rot90(np.sqrt(fourier_fx ** 2 + fourier_fy ** 2), 3),
                   extent=[np.min(grid_mat[:, :, 0]), np.max(grid_mat[:, :, 0]),
                           np.min(grid_mat[:, :, 1]), np.max(grid_mat[:, :, 1])],
                   interpolation="bicubic",
                   cmap="jet")
axs[1].quiver(grid_mat[:, :, 0], grid_mat[:, :, 1], fourier_fx, fourier_fy, color='black')
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
