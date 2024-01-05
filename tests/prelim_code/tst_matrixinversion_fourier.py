import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv
from itertools import product
from tests.prelim_code.tst_utilis import *
import cvxpy


# Define forward kernel component functions in fourier space to calculate displacement field
def G_ft(k_xx, k_yy):
    k = np.sqrt(k_xx ** 2 + k_yy ** 2)
    gxx_ft = coeff_ft * ((1 - s) / k + s * k_yy ** 2 / k ** 3)
    gyy_ft = coeff_ft * ((1 - s) / k + s * k_xx ** 2 / k ** 3)
    gxy_ft = coeff_ft * (s * k_xx * k_yy / k ** 3)
    return np.nan_to_num(gxx_ft), np.nan_to_num(gyy_ft), np.nan_to_num(gxy_ft)


# Define material parameters
E = 1000
s = 0.48
sigma = 15
x0, y0 = 30, 30
coeff_ft = 2 * (1 + s) / E

# Vector field dimensions
x_min, x_max = -100, 100
y_min, y_max = -100, 100

# Create positional coordinates for displacement field
point_dens = 20
x_val = np.linspace(x_min, x_max, point_dens)
y_val = np.linspace(y_min, y_max, point_dens)
xx, yy = np.meshgrid(x_val, y_val)

# Calculate meshsize and pyramid width
meshsize_x = x_val[1] - x_val[0]
meshsize_y = x_val[1] - x_val[0]
width_x = 2 * meshsize_x
width_y = 2 * meshsize_y

# Create positional coordinates fo nodes
x_k_val = x_val
y_k_val = y_val

# Get wave vector components
k_x = np.fft.fftfreq(x_val.shape[0], d=meshsize_x)
k_y = np.fft.fftfreq(y_val.shape[0], d=meshsize_y)

# Create wave vector grid
kxx, kyy = np.meshgrid(k_x, k_y)

# Multiplication in fourier space equals convolution in spatial domain
gamma_11_ft, gamma_22_ft, gamma_12_ft = G_ft(kxx, kyy) * pyramid2dim_ft(kxx, kyy, width_x, width_y)

# Define empty array to store convolution results
gamma_glob = np.zeros((2 * point_dens ** 2, 2 * point_dens ** 2))

# Loop over grid in spatial domain and calculate BEM matrix components
for i, x_k_i in enumerate(x_k_val):
    for j, y_k_j in enumerate(y_k_val):
        # Use shift theorem for each node to center pyramid shape above
        shift = kxx * x_k_i + kyy * y_k_j
        gamma_11_shift = gamma_11_ft * np.exp(-1j * shift)
        gamma_12_shift = gamma_12_ft * np.exp(-1j * shift)
        gamma_22_shift = gamma_22_ft * np.exp(-1j * shift)

        # Transform BEM matrix components back to spatial domain
        gamma_11 = np.fft.ifft2(gamma_11_shift).real
        gamma_12 = np.fft.ifft2(gamma_12_shift).real
        gamma_21 = gamma_12
        gamma_22 = np.fft.ifft2(gamma_22_shift).real

        # Concatenate along the columns (axis=1)
        gamma_r1 = np.concatenate((gamma_11, gamma_12), axis=1)
        gamma_r2 = np.concatenate((gamma_21, gamma_22), axis=1)

        # Concatenate along the rows (axis=0)
        gamma_loc = np.concatenate((gamma_r1, gamma_r2), axis=0)

        # Sum up BEM matrices
        block_i = 2*point_dens*i
        block_j = 2*point_dens*j
        gamma_glob[block_i:block_i+2*point_dens, block_j:block_j+2*point_dens] = gamma_loc
        print(f'Index {i}, {j}')

# Analytic definition of displacement field
uxx, uyy, u_norm = tri_pole(xx, yy, x0, y0, sigma)
uxx = uxx.ravel()
uyy = uyy.ravel()

# Concatenate displacement vector pairs (ux, uy) into global displacement vector
u_glob = np.array([val for pair in zip(uxx, uyy) for val in pair])
u_glob = u_glob.reshape(2 * point_dens ** 2, 1)

# Pseudo-invert global BEM matrix
gamma_inv = np.linalg.inv(gamma_glob)

# Calculate approximate traction field
t_glob = matrix_map(gamma_inv, u_glob)

# Reshape global traction vector to extract components
t_glob = t_glob.reshape(point_dens ** 2, 2)
txx = t_glob[:, 0]
tyy = t_glob[:, 1]


def plot_greens_functions():
    # Create a 3D plot
    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(131)
    ax1.quiver(xx, yy, uxx, uyy, cmap='viridis', alpha=0.8)
    ax1.set_title('Displacement field')

    ax2 = fig.add_subplot(132)
    ax2.quiver(xx, yy, txx, tyy, cmap='viridis', alpha=0.8)
    ax2.set_title('Traction field')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(xx, yy, gamma_22_ft.imag, cmap='viridis', alpha=0.8)
    ax3.set_title('Gamma yy')

    # Set labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

    plt.tight_layout()
    plt.show()

plot_greens_functions()