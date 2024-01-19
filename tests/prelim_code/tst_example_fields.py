import numpy as np


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
def vortex_x(x_p, y_p, x0, y0, alpha):
    r1 = np.sqrt((x_p - x0) ** 2 + (y_p - y0) ** 2)
    r2 = np.sqrt((x_p + x0) ** 2 + (y_p + y0) ** 2)
    vx = - (y_p - y0) / np.exp(alpha * r1) + (y_p + y0) / np.exp(alpha * r2)
    return vx


def vortex_y(x_p, y_p, x0, y0, alpha):
    r1 = np.sqrt((x_p - x0) ** 2 + (y_p - y0) ** 2)
    r2 = np.sqrt((x_p + x0) ** 2 + (y_p + y0) ** 2)
    vy = (x_p - x0) / np.exp(alpha * r1) - (x_p + x0) / np.exp(alpha * r2)
    return vy


# Define vortex vector field
def vortex(x_p, y_p, x0, y0):
    alpha = 0.45
    # Vector field component functions
    vx = vortex_x(x_p, y_p, x0, y0, alpha)
    vy = vortex_y(x_p, y_p, x0, y0, alpha)

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
