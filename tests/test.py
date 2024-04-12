import matplotlib.pyplot as plt
from pytraction.optimal_lambda import optimal_lambda
from pytraction.optimal_lambda2 import optimal_lambda2
from pytraction.inversion import traction_fourier
from pytraction.fourier import *

# Save example field #
# displ_field = log[0]["deformation_interpolated"][0]
# pos_field = log[0]["position_interpolated"][0]
# np.save('example_displacement.npy', displ_field)
# np.save('example_position.npy', pos_field)

# Read real displacement field
pos = np.load('example_position.npy')
vec = np.load('example_displacement.npy')
xx = pos[:, :, 0]
x_val = xx[0, :]
meshsize = x_val[1] - x_val[0]

# Plot displacement field
plt.quiver(pos[:, :, 0], pos[:, :, 1], vec[:, :, 0], vec[:, :, 1])
plt.show()

# Parameters
beta = 868
E = 300
s = 0.5

# OLD CODE
ft_ux_old, ft_uy_old, kxx_old, kyy_old, gamma_glob_old = fourier_xu(vec, E, s, meshsize)
lamd_old, evidence_old, evidence_one_old = optimal_lambda(
    beta, ft_ux_old, ft_uy_old, kxx_old, kyy_old, E, s, meshsize, gamma_glob_old
)

# NEW CODE
fx, fy, ft_fx, ft_fy, ft_ux, ft_uy, kxx, kyy, gamma_glob = traction_fourier(
    pos=pos, vec=vec, s=s, elastic_modulus=E, lambd=None, scaling_z=1, zdepth=0)
lamd, evidence, evidence_one = optimal_lambda2(
    beta, ft_ux, ft_uy, kxx, kyy, E, s, meshsize, gamma_glob, vec=vec, pos=pos
)

print(f"Lambda old: {lamd_old}, Evidence_old: {evidence_one_old}")
print(f"Lambda new: {lamd}, Evidence_new: {evidence_one}")
