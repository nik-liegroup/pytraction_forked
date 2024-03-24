import numpy as np
from numpy.linalg import cholesky
from scipy.signal import convolve
from scipy.fft import fftfreq, ifft2
import scipy.optimize as optimize
from functools import partial


# Shape function for boundary element method
def pyramid2dim(xx, yy, width_x, width_y):
    return (np.maximum(0, width_x - np.abs(xx)) *
            np.maximum(0, width_y - np.abs(yy)))


def pyramid2dim_ft(kxx, kyy, width_x, width_y):
    return (width_x * np.sinc(kxx * width_x) *
            width_y * np.sinc(kyy * width_y)) ** 2


# Define forward kernel component functions
def kernel_real(x, y, s, elastic_modul):
    coeff = (1 + s) / (np.pi * elastic_modul)
    r = np.sqrt(x ** 2 + y ** 2)
    gxx = coeff * ((1 - s) / r + s * x ** 2 / r ** 3)
    gxy = coeff * (s * x * y / r ** 3)
    gyy = coeff * ((1 - s) / r + s * y ** 2 / r ** 3)
    return np.nan_to_num(gxx), np.nan_to_num(gxy), np.nan_to_num(gyy)


# Define forward kernel component functions in fourier space
def kernel_ft(k_x, k_y, s, elastic_modul):
    coeff = 2 * (1 + s) / elastic_modul
    k = np.sqrt(k_x ** 2 + k_y ** 2)
    k[0, 0] = 1

    gxx = coeff * ((1 - s) / k + s * k_y ** 2 / k ** 3)
    gyy = coeff * ((1 - s) / k + s * k_x ** 2 / k ** 3)
    gxy = coeff * (s * k_x * k_y / k ** 3)

    # Set all zero frequency components in greens function to zero
    gxx[0, 0] = 0
    gxy[0, 0] = 0
    gyy[0, 0] = 0

    #  Set values in middle row and column to zero
    i_max = len(k_x[0, :])
    j_max = len(k_y[:, 0])

    gxy[int(i_max // 2), :] = 0
    gxy[:, int(j_max // 2)] = 0

    return np.nan_to_num(gxx), np.nan_to_num(gxy), np.nan_to_num(gyy)


def traction_bem(xx, yy, method, point_dens, s, elastic_modulus):
    x_val, y_val = xx[0, :], yy[:, 0]
    meshsize_x, meshsize_y = x_val[1] - x_val[0], y_val[1] - y_val[0]

    k_x, k_y = fftfreq(x_val.shape[0], d=meshsize_x), fftfreq(y_val.shape[0], d=meshsize_y)
    kxx, kyy = np.meshgrid(k_x, k_y)

    # Multiplication in fourier space equals convolution in spatial domain
    gamma_kxx, gamma_kxy, gamma_kyy = (kernel_ft(kxx, kyy, s, elastic_modulus) *
                                       pyramid2dim_ft(kxx, kyy, 2 * meshsize_x, 2 * meshsize_y))

    gamma_glob_xx, gamma_glob_xy, gamma_glob_yy = (np.zeros((point_dens ** 2, point_dens ** 2)),
                                                   np.zeros((point_dens ** 2, point_dens ** 2)),
                                                   np.zeros((point_dens ** 2, point_dens ** 2)))

    # Loop over grid in spatial domain and calculate BEM matrix components
    for i, x_k in enumerate(x_val):
        for j, y_k in enumerate(y_val):
            if method == 'conv':
                kernel_xx, kernel_xy, kernel_yy = kernel_real(xx, yy, s, elastic_modulus)
                pyr_shape = pyramid2dim(xx - x_k, yy - y_k, 2 * meshsize_x, 2 * meshsize_y)
                gamma_xx = convolve(kernel_xx,
                                    pyr_shape,
                                    mode='same')
                gamma_xy = convolve(kernel_xy,
                                    pyr_shape,
                                    mode='same')
                gamma_yy = convolve(kernel_yy,
                                    pyr_shape,
                                    mode='same')
            elif method == 'fft':
                # Use shift theorem for each node to center pyramid shape above
                shift = kxx * x_k + kyy * y_k
                gamma_xx = ifft2(gamma_kxx * np.exp(-1j * shift))
                gamma_xy = ifft2(gamma_kxy * np.exp(-1j * shift))
                gamma_yy = ifft2(gamma_kyy * np.exp(-1j * shift))

            else:
                raise ValueError(f"{method} method is not available.")

            # Define BEM matrix block sizes
            block_i = point_dens * i
            block_j = point_dens * j

            # Swapping rows and columns in matrix equals multiplication by permutation matrix (Involutory matrix)
            # This justifies to separate vector components as the matrices inverse is invariant under this operation
            gamma_glob_xx[block_i:block_i + point_dens:, block_j:block_j + point_dens] = gamma_xx
            gamma_glob_xy[block_i:block_i + point_dens:, block_j:block_j + point_dens] = gamma_xy
            gamma_glob_yy[block_i:block_i + point_dens:, block_j:block_j + point_dens] = gamma_yy
            print(f'Index {i, j}')

    # Concatenate along the columns (axis=1)
    gamma_r1 = np.concatenate((gamma_glob_xx, gamma_glob_xy), axis=1)
    gamma_r2 = np.concatenate((gamma_glob_xy, gamma_glob_yy), axis=1)

    # Concatenate along the rows (axis=0)
    gamma_glob = np.concatenate((gamma_r1, gamma_r2), axis=0)

    return gamma_glob


def traction_fourier(xx, yy, point_dens, s, elastic_modulus):
    x_val, y_val = xx[0, :], yy[:, 0]
    meshsize_x, meshsize_y = x_val[1] - x_val[0], y_val[1] - y_val[0]

    k_x, k_y = fftfreq(x_val.shape[0], d=meshsize_x), fftfreq(y_val.shape[0], d=meshsize_y)
    kxx, kyy = np.meshgrid(k_x, k_y)

    # Calculate Green's matrix in Fourier space
    g_xx, g_xy, g_yy = kernel_ft(kxx, kyy, s, elastic_modulus)

    # Crate differential operator matrix
    gamma_1 = np.diagflat(g_xx.reshape(1, point_dens ** 2))
    gamma_2 = np.diagflat(g_xy.reshape(1, point_dens ** 2))

    gamma_3 = np.diagflat(g_xy.reshape(1, point_dens ** 2))
    gamma_4 = np.diagflat(g_yy.reshape(1, point_dens ** 2))

    gamma_glob = np.block([[gamma_1, gamma_2],
                           [gamma_3, gamma_4]])

    return gamma_glob


def log_evidence(alpha, beta, X, u_glob):
    # Calculate traction field given current lambda
    lambd = alpha / beta
    f_glob = tikhonov_simple(X, u_glob, lambd)

    # Create identity matrix from X
    m, n = X.shape
    Id = np.eye(m, n)

    # Calculate cholesky decomposition for hessian matrix A
    A = alpha * Id + beta * X.T @ X
    L = cholesky(A)
    log_detA = 2 * np.sum(np.log(np.diag(L)))

    # Formula for log evidence
    evidence_value = - 0.5 * alpha * f_glob.T @ f_glob \
                     - 0.5 * beta * (X @ f_glob - u_glob).T @ (X @ f_glob - u_glob) \
                     - 0.5 * log_detA \
                     - m * np.log(2 * np.pi) \
                     + m * np.log(beta) \
                     + n * np.log(alpha)

    minus_evidence = - evidence_value.item()

    return minus_evidence


def bayesian_regularization(X, u_glob, beta):
    # Standardize the differential operator X along its columns to their spread
    X_sd = np.std(X, axis=0)
    X_mean = np.mean(X, axis=0)
    X = (X - X_mean) / X_sd

    # Standardize the deformation field u to its mean
    u_glob = (u_glob - np.mean(u_glob))

    # Parameters for Golden Section Search
    alpha_left = 1e-6  # Initial left alpha
    alpha_right = 1e6  # Initial right alpha

    # Set target function for optimization
    target_func = partial(
        log_evidence,
        beta=beta,
        X=X,
        u_glob=u_glob
    )
    alpha_opt, fval, ierr, numfunc = optimize.fminbound(target_func, alpha_left, alpha_right, disp=3, full_output=True)

    lambd = alpha_opt / beta
    print(f"Final regularization lambda: {lambd}")
    return lambd


# Tikhonov regularization
def tikhonov_simple(X, u_glob, lambda_2):
    aa = X.shape[1]
    c = np.ones(aa)
    C = np.diag(c)

    f_glob = np.linalg.inv(X.T @ X + lambda_2 * C) @ (X.T @ u_glob)
    return f_glob


# Center 2D array to remove padding
def center_padding(array2dim, x, y):
    # Determine the start and end indices to get the central part
    center_i = (np.shape(array2dim)[0]) // 2
    center_j = (np.shape(array2dim)[1]) // 2

    result = array2dim[center_i - len(x) // 2: center_i + len(x) // 2,
                       center_j - len(y) // 2: center_j + len(y) // 2]
    return result