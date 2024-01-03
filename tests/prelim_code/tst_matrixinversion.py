import matplotlib.pyplot as plt
import numpy as np
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


# Define forward kernel component functions to calculate displacement field
def Gxx(x, y, x_p, y_p):
    r = np.sqrt((x - x_p) ** 2 + (y - y_p) ** 2)
    gxx = coeff * ((1 - s) / r + s * x ** 2 / r ** 3)
    return gxx


def Gyy(x, y, x_p, y_p):
    r = np.sqrt((x - x_p) ** 2 + (y - y_p) ** 2)
    gyy = coeff * ((1 - s) / r + s * y ** 2 / r ** 3)
    return gyy


def Gxy(x, y, x_p, y_p):
    r = np.sqrt((x - x_p) ** 2 + (y - y_p) ** 2)
    gxy = coeff * (s * x * y / r ** 3)
    return gxy


def pyramid_shape(x_p, y_p, x_k, y_k, w):
    d_x = x_p - x_k
    d_y = y_p - y_k

    pyramid1 = (np.heaviside(d_x, 1) * np.heaviside(d_y, 1) * np.heaviside(w - d_x, 1)
                * np.heaviside(w - d_y, 1) * (1 - d_x / w) * (1 - d_y / w))
    pyramid2 = (np.heaviside(-d_x, 1) * np.heaviside(d_y, 1) * np.heaviside(w + d_x, 1)
                * np.heaviside(w - d_y, 1) * (1 + d_x / w) * (1 - d_y / w))
    pyramid3 = (np.heaviside(d_x, 1) * np.heaviside(-d_y, 1) * np.heaviside(w - d_x, 1)
                * np.heaviside(w + d_y, 1) * (1 - d_x / w) * (1 + d_y / w))
    pyramid4 = (np.heaviside(-d_x, 1) * np.heaviside(-d_y, 1) * np.heaviside(w + d_x, 1)
                * np.heaviside(w + d_y, 1) * (1 + d_x / w) * (1 + d_y / w))

    return pyramid1 + pyramid2 + pyramid3 + pyramid4


def convolution(integr: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int], np.ndarray],
                x,
                y,
                x_p,
                y_p,
                x_k: int,
                y_k: int):
    conv = np.zeros((len(x), len(y)))

    global conv_counter
    conv_counter += 1
    print("Current count:", conv_counter)

    for i, x_i in enumerate(x):
        for j, y_j in enumerate(y):
            for k in range(len(x_p) - 1):
                for l in range(len(y_p) - 1):
                    xp1, xp2 = x_p[k], x_p[k + 1]
                    yp1, yp2 = y_p[l], y_p[l + 1]

                    dxp = xp2 - xp1
                    dyp = yp2 - yp1

                    integrand_values = [
                        integr(x_i, y_j, x_p, y_p, x_k, y_k)
                        for x_p in (xp1, xp2)
                        for y_p in (yp1, yp2)
                    ]

                    conv[i, j] += 0.25 * sum(integrand_values) * dxp * dyp

            print(f'Convolution at {x_i}, {y_j} completed')
    return conv


def integrandKxx(x, y, x_p, y_p, x_k, y_k):
    intrg = Gxx(x, y, x_p, y_p) * pyramid_shape(x_p, y_p, x_k, y_k, w)
    return intrg


def integrandKxy(x, y, x_p, y_p, x_k, y_k):
    intrg = Gxy(x, y, x_p, y_p) * pyramid_shape(x_p, y_p, x_k, y_k, w)
    return intrg


def integrandKyy(x, y, x_p, y_p, x_k, y_k):
    intrg = Gyy(x, y, x_p, y_p) * pyramid_shape(x_p, y_p, x_k, y_k, w)
    return intrg


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
point_dens = 10
w = 20
E = 1000
s = 0.48
pix_per_mu = 1
beta = 0.1
sigma = 15
coeff = (1 + s) / (np.pi * E)
x0, y0 = 30, 30
conv_counter = 0
conv_param = 1

# Vector field dimensions
x_min, x_max = -100, 100
y_min, y_max = -100, 100

# Create positional coordinates
x_val = np.linspace(x_min, x_max, point_dens)
y_val = np.linspace(y_min, y_max, point_dens)
x_k_val = np.linspace(x_min, x_max, point_dens)
y_k_val = np.linspace(y_min, y_max, point_dens)

gamma_glob = np.zeros((2 * point_dens, 2 * point_dens))
for i, x_k_i in enumerate(x_k_val):
    x_p_floor = i + len(x_k_val) - conv_param - 1
    x_p_ceil = i + len(x_k_val) + conv_param

    x_p_val = np.concatenate((np.flip(x_k_val), x_k_val[1:]))[x_p_floor:x_p_ceil]

    for j, y_k_j in enumerate(y_k_val):
        y_p_floor = i + len(y_k_val) - conv_param - 1
        y_p_ceil = i + len(y_k_val) + conv_param

        y_p_val = np.concatenate((np.flip(y_k_val), y_k_val[1:]))[y_p_floor:y_p_ceil]

        gamma_11 = convolution(integrandKxx, x_val, y_val, x_p_val, y_p_val, x_k_i, y_k_j)
        gamma_12 = convolution(integrandKxy, x_val, y_val, x_p_val, y_p_val, x_k_i, y_k_j)
        gamma_21 = gamma_12
        gamma_22 = convolution(integrandKyy, x_val, y_val, x_p_val, y_p_val, x_k_i, y_k_j)

        # Concatenate along the columns (axis=1)
        gamma_r1 = np.concatenate((gamma_11, gamma_12), axis=1)
        gamma_r2 = np.concatenate((gamma_21, gamma_22), axis=1)

        # Concatenate along the rows (axis=0)
        gamma_loc = np.concatenate((gamma_r1, gamma_r2), axis=0)
        gamma_loc = np.nan_to_num(gamma_loc)
        gamma_glob += gamma_loc

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
