import numpy as np
from scipy.signal import convolve


# Example fields
# Define tri-pole vector field component functions
def tri_pole_x(x_p, y_p, x0, y0, sigma):
    lx = (np.exp(-((x_p + x0) ** 2 + (y_p + y0) ** 2) / (sigma ** 2)) -
          np.exp(-((x_p - x0) ** 2 + (y_p - y0) ** 2) / (sigma ** 2)) +
          np.exp(-((x_p - x0) ** 2 + (y_p + y0) ** 2) / (sigma ** 2)))
    return lx


def tri_pole_y(x_p, y_p, x0, y0, sigma):
    ly = (np.exp(-((x_p + x0) ** 2 + (y_p + y0) ** 2) / (sigma ** 2)) -
          np.exp(-((x_p - x0) ** 2 + (y_p - y0) ** 2) / (sigma ** 2)) -
          np.exp(-((x_p - x0) ** 2 + (y_p + y0) ** 2) / (sigma ** 2)))
    return ly


# Define tri-pole vector field
def tri_pole(x_p, y_p, x0, y0, sigma):
    # Vector field component functions
    lx = tri_pole_x(x_p, y_p, x0, y0, sigma)
    ly = tri_pole_y(x_p, y_p, x0, y0, sigma)

    # Remove small values to avoid division by 0
    lx_norm = np.where((lx < 0.01) & (lx > - 0.01), np.nan, lx)
    ly_norm = np.where((ly < 0.01) & (ly > - 0.01), np.nan, ly)

    # Calculate normalization coefficients
    l_norm = np.sqrt(lx_norm ** 2 + ly_norm ** 2)
    return lx, ly, l_norm


# Define vortex vector field component functions
def vortex_x(x_p, y_p, x0, y0):
    r1 = np.sqrt((x_p - x0) ** 2 + (y_p - y0) ** 2)
    r2 = np.sqrt((x_p + x0) ** 2 + (y_p + y0) ** 2)
    vx = - (y_p - y0) / np.exp(0.3 * r1) + (y_p + y0) / np.exp(0.3 * r2)
    return vx


def vortex_y(x_p, y_p, x0, y0):
    r1 = np.sqrt((x_p - x0) ** 2 + (y_p - y0) ** 2)
    r2 = np.sqrt((x_p + x0) ** 2 + (y_p + y0) ** 2)
    vy = (x_p - x0) / np.exp(0.3 * r1) - (x_p + x0) / np.exp(0.3 * r2)
    return vy


# Define vortex vector field
def vortex(x_p, y_p, x0, y0):
    # Vector field component functions
    vx = vortex_x(x_p, y_p, x0, y0)
    vy = vortex_y(x_p, y_p, x0, y0)

    # Calculate normalization coefficients
    v_norm = np.sqrt(vx ** 2 + vy ** 2)
    return vx, vy, v_norm


# 2D box function with width_x and width_y centered around (0,0)
def boxcar2dim(xx, yy, width_x, width_y):
    return np.where((np.abs(xx) <= width_x / 2) &
                    (np.abs(yy) <= width_y / 2), 1, 0)


def boxcar2dim_ft(kxx, kyy, width_x, width_y):
    return (width_x * np.sinc(kxx * width_x) *
            width_y * np.sinc(kyy * width_y))


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
    gxx = coeff * ((1 - s) / k + s * k_y ** 2 / k ** 3)
    gyy = coeff * ((1 - s) / k + s * k_x ** 2 / k ** 3)
    gxy = coeff * (s * k_x * k_y / k ** 3)
    return np.nan_to_num(gxx), np.nan_to_num(gxy), np.nan_to_num(gyy)


# Center 2D array to remove padding
def center_padding(array2dim, x, y):
    # Determine the start and end indices to get the central part
    center_i = (np.shape(array2dim)[0]) // 2
    center_j = (np.shape(array2dim)[1]) // 2

    result = array2dim[center_i - len(x) // 2: center_i + len(x) // 2,
                       center_j - len(y) // 2: center_j + len(y) // 2]
    return result


def traction_force_bem(xx, yy, point_dens, s, elastic_modulus):
    x_val, y_val = xx[0, :], yy[:, 0]
    meshsize_x, meshsize_y = x_val[1] - x_val[0], y_val[1] - y_val[0]

    gamma_glob_xx, gamma_glob_xy, gamma_glob_yy = (np.zeros((point_dens ** 2, point_dens ** 2)),
                                                   np.zeros((point_dens ** 2, point_dens ** 2)),
                                                   np.zeros((point_dens ** 2, point_dens ** 2)))

    # Loop over grid in spatial domain and calculate BEM matrix components
    for i, x_k in enumerate(x_val):
        for j, y_k in enumerate(y_val):
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


# Tikhonov regularization
def tikhonov(X, u_glob, lambda_2):
    aa = X.shape[1]
    c = np.ones(aa)
    C = np.diag(c)

    f_glob = np.linalg.inv(X.T @ X + lambda_2 * C) @ (X.T @ u_glob)
    return f_glob
