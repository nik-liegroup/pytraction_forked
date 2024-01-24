import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd, norm, cholesky
from typing import Callable
from tests.prelim_code.prelim_utilis import *


def bayesian_regularization(X, u_glob, ux, uy):
    # Concatenate displacement vector components in noise window
    noise_window = 10
    ux, uy = ux[noise_window:noise_window], uy[noise_window:noise_window]
    noise_vec = np.array([ux.flatten(), uy.flatten()])

    # Calculate inverse noise variance
    beta = 1 / np.var(noise_vec)

    # Standardize the input data along columns to their spread
    sd_X = np.std(X, axis=0)
    X = (X - np.mean(X, axis=0)) / sd_X
    u_glob = u_glob - np.mean(u_glob)

    # Singular Value Decomposition of X where U and V are orthogonal matrices and S is diagonal with singular values
    U, S, V = svd(X, full_matrices=False, compute_uv=True)

    # Matrix multiplication X.T @ X = V @ (S.T @ S) @ V.T
    XX = X.T @ X

    # Prepare identity matrix
    aa = X.shape[1]
    c = np.ones(aa)
    C = np.diag(c)

    # Parameters for Golden Section Search
    alpha1 = 200  # initial left alpha
    alpha2 = 40000  # initial right alpha
    step_size = 30
    max_n = 200

    # Golden Section search to find maximum of log evidence
    n = 0
    nn = []
    lambdas = []

    while True:
        # Find middle value of alpha1 and alpha2
        middle = 0.5 * (alpha1 + alpha2)

        # Calculate evidence for alpha value closer to alpha1 and for closer to alpha2
        middle_up = middle + 0.5 * step_size
        evidence_up = log_evidence(middle_up, beta, U, S, V, u_glob, C, X, aa, XX)
        middle_down = middle - 0.5 * step_size
        evidence_down = log_evidence(middle_down, beta, U, S, V, u_glob, C, X, aa, XX)

        # Choose value yielding smaller evidence to reduce the possible interval for alpha
        if evidence_up >= evidence_down:
            alpha1 = middle_down
        else:
            alpha2 = middle_up

        n += 1
        nn.append(n)
        lambdas.append(alpha1 / beta)

        if norm(alpha1 - alpha2) / norm(alpha1) < 1e-5 or n == max_n:
            break

    # Calculation of final solution with optimal parameters
    f_glob = tikhonov(X, u_glob, lambdas[-1])

    # Undo standardization
    f_glob /= sd_X[:, np.newaxis]

    return f_glob


def log_evidence(alpha, beta, U, S, V, u_glob, C, X, aa, XX):
    lambd = alpha / beta
    f_glob = tikhonov(X, u_glob, lambd)

    # Calculate log(det(A))
    A = alpha * C + beta * XX
    L = cholesky(A)
    log_det_A = 2 * np.sum(np.log(np.diag(L)))

    # Formula for log evidence
    evidence_value = -0.5 * alpha * f_glob.T @ f_glob - 0.5 * beta * (X @ f_glob - u_glob).T @ (X @ f_glob - u_glob) \
                     - 0.5 * log_det_A + 0.5 * aa * np.log(beta) + 0.5 * aa * np.log(alpha) \
                     - 0.5 * aa * np.log(2 * np.pi)

    return evidence_value


# Tikhonov regularization
def tikhonov(X, u_glob, lambda_2):
    aa = X.shape[1]
    c = np.ones(aa)
    C = np.diag(c)

    f_glob = np.linalg.inv(X.T @ X + lambda_2 * C) @ (X.T @ u_glob)
    return f_glob


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
uxx, uyy, u_norm = vortex(xx, yy, x0, y0)
uxx = uxx.ravel()
uyy = uyy.ravel()

# Concatenate displacement vector pairs (ux, uy) into global displacement vector
u_glob = np.array([val for pair in zip(uxx, uyy) for val in pair])
u_glob = u_glob.reshape(2 * point_dens ** 2, 1)

# Calculate approximate traction field
ux, uy, u_norm = vortex(x_val, y_val, x0, y0)
# t_glob = bayesian_regularization(gamma_glob, u_glob, ux, uy)
t_glob = tikhonov(gamma_glob, u_glob, 100000000)

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