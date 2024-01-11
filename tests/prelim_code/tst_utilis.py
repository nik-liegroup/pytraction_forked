import numpy as np


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
