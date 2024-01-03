import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv
from pytraction.fourier import fourier_xu, reg_fourier_tfm
from pytraction.optimal_lambda import optimal_lambda
from pytraction.utils import interp_vec2grid
from pytraction.postprocess import *
from typing import Callable


# Define tri-pole vector field component functions
def L_x(x_p, y_p):
    lx = (np.exp(-((x_p + x0) ** 2 + (y_p + y0) ** 2) / (sigma ** 2)) -
          np.exp(-((x_p - x0) ** 2 + (y_p - y0) ** 2) / (sigma ** 2)) +
          np.exp(-((x_p - x0) ** 2 + (y_p + y0) ** 2) / (sigma ** 2)))
    return lx


def L_y(x_p, y_p):
    ly = (np.exp(-((x_p + x0) ** 2 + (y_p + y0) ** 2) / (sigma ** 2)) -
          np.exp(-((x_p - x0) ** 2 + (y_p - y0) ** 2) / (sigma ** 2)) -
          np.exp(-((x_p - x0) ** 2 + (y_p + y0) ** 2) / (sigma ** 2)))
    return ly


# Define tri-pole vector field
def tripole(x_p, y_p):
    # Vector field component functions
    lx = L_x(x_p, y_p)
    ly = L_y(x_p, y_p)

    # Remove small values to avoid division by 0
    lx_norm = np.where((lx < 0.01) & (lx > - 0.01), np.nan, lx)
    ly_norm = np.where((ly < 0.01) & (ly > - 0.01), np.nan, ly)

    # Calculate normalization coefficients
    l_norm = np.sqrt(lx_norm ** 2 + ly_norm ** 2)
    return lx, ly, l_norm


# Define forward kernel component functions in fourier space to calculate displacement field
def G_ft(k_xx, k_yy):
    k = np.sqrt(k_xx ** 2 + k_yy ** 2)
    gxx_ft = coeff_ft * ((1 - s) / k + s * k_yy ** 2 / k ** 3)
    gyy_ft = coeff_ft * ((1 - s) / k + s * k_xx ** 2 / k ** 3)
    gxy_ft = coeff_ft * (s * k_xx * k_yy / k ** 3)
    return np.nan_to_num(gxx_ft), np.nan_to_num(gxy_ft), np.nan_to_num(gyy_ft)


def pyramid_shape(k_xx, k_yy, w):
    return (w ** 2 / 4) * (np.sinc(k_xx * w / 4) ** 2) * (np.sinc(k_yy * w / 4) ** 2)


def matrix_map(matrix, vec):
    # Get dimension of NxM matrix
    n = matrix.shape[0]
    m = matrix.shape[1]
    # Create output vector of length n
    result = np.empty((n, 1))
    # Check if input vector has correct dimension
    if vec.shape[0] != m:
        raise ValueError(f'Vector should be of dimension {m} but has dimension {vec.shape[0]}')
    else:
        for i in range(n):
            for j in range(m):
                result[i] += matrix[i][j] * vec[j]

    return result


# Define parameters
point_dens = 50
E = 1000
s = 0.48
sigma = 15
x0, y0 = 30, 30
coeff_ft = 2 * (1 + s) / E

# Vector field dimensions
x_min, x_max = -100, 100
y_min, y_max = -100, 100
meshsize = int((x_max - x_min)/point_dens)

# Create positional coordinates
x_val = np.linspace(x_min, x_max, point_dens)
y_val = np.linspace(y_min, y_max, point_dens)
x_k_val = x_val
y_k_val = y_val

# Calculate even number of mesh intervals
i_max = point_dens - np.mod(point_dens, 2)
j_max = point_dens - np.mod(point_dens, 2)

k_x = (
        2 * np.pi / (i_max * meshsize)
        * np.concatenate([np.arange(0, (i_max - 1) / 2, 1), -np.arange(i_max / 2, 0, -1)])
    )

k_y = (
        2 * np.pi / (j_max * meshsize)
        * np.concatenate([np.arange(0, (j_max - 1) / 2, 1), -np.arange(j_max / 2, 0, -1)])
    )

kxx_val, kyy_val = np.meshgrid(k_x, k_y)

gamma_11_ft, gamma_12_ft, gamma_22_ft = G_ft(kxx_val, kyy_val) * pyramid_shape_ft(kxx_val, kyy_val, meshsize)

gamma_glob = np.zeros((2 * point_dens, 2 * point_dens))
for i, x_k_i in enumerate(x_k_val):
    for j, y_k_j in enumerate(y_k_val):
        shift_theo = kxx_val * x_k_i + kyy_val * y_k_j
        gamma_11_shift = gamma_11_ft * np.exp(-1j * shift_theo)
        gamma_12_shift = gamma_12_ft * np.exp(-1j * shift_theo)
        gamma_22_shift = gamma_22_ft * np.exp(-1j * shift_theo)

        gamma_11 = np.real(np.fft.ifft2(gamma_11_shift))
        gamma_12 = np.real(np.fft.ifft2(gamma_12_shift))
        gamma_21 = gamma_12
        gamma_22 = np.real(np.fft.ifft2(gamma_22_shift))

        # Concatenate along the columns (axis=1)
        gamma_r1 = np.concatenate((gamma_11, gamma_12), axis=1)
        gamma_r2 = np.concatenate((gamma_21, gamma_22), axis=1)

        # Concatenate along the rows (axis=0)
        gamma_loc = np.concatenate((gamma_r1, gamma_r2), axis=0)
        gamma_glob += np.nan_to_num(gamma_loc)

# Analytic definition of displacement field
xx, yy = np.meshgrid(x_val, y_val)
ux, uy, u_norm = tripole(x_val, y_val)

u_glob = np.array([val for pair in zip(ux, uy) for val in pair])
u_glob = u_glob.reshape(2 * point_dens, 1)

# Invert global BEM matrix
gamma_inv = np.linalg.pinv(gamma_glob)

# Calculate approximate traction field
t_glob = matrix_map(gamma_inv, u_glob)
t_glob = t_glob.reshape(point_dens, 2)
tx = t_glob[:, 0]
ty = t_glob[:, 1]

# Create plot of vector fields
fig, axs = plt.subplots(2, 2)

# Quiver plot for the first vector field
axs[0, 0].quiver(y_val, x_val, ux, uy, color='blue')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('y')
axs[0, 0].set_title('Analytic displacement field')

# Quiver plot for the second vector field
axs[0, 1].quiver(y_val, x_val, tx, ty, color='blue')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('y')
axs[0, 1].set_title('Approximated traction field (Matrix inversion)')

# Quiver plot for the third vector field
axs[1, 0].quiver(y_val, x_val, ux, uy, color='green')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('y')
axs[1, 0].set_title('Recovered displacement field (Forward problem)')

# Scalar plot for the fourth field
txx, tyy = np.meshgrid(tx, ty)
traction_magnitude = np.sqrt(txx ** 2 + tyy ** 2)
traction_magnitude = np.flip(traction_magnitude, axis=0)

c_x = traction_magnitude.shape[0] // 2
c_y = traction_magnitude.shape[1] // 2
axs[1, 1].imshow(traction_magnitude, cmap='viridis')
axs[1, 1].set_title('Traction scalar field')

# Show the plot
plt.show()
