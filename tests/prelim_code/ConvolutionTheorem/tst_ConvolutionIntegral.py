import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from typing import Callable
from tests.prelim_code.prelim_utilis import *
from tests.prelim_code.tst_example_fields import *


# Define kernel function
def boxcar2dim(x, y, x_p, y_p):
    return np.where((np.abs(x - x_p) <= width_x / 2) &
                    (np.abs(y - y_p) <= width_y / 2), 1, 0)


def pyramid_shape(x_p, y_p):
    return (np.maximum(0, width_x - np.abs(x_p)) *
            np.maximum(0, width_y - np.abs(y_p)))


def integrand_quad(x, y, x_p, y_p):
    return boxcar2dim(x, y, x_p, y_p) * x_p ** 2 * y_p ** 2


def convolution(integr, x, y, x_p, y_p):
    conv = np.zeros((len(x), len(y)))

    for i, x_i in enumerate(x):
        for j, y_j in enumerate(y):
            for k in range(len(x_p) - 1):
                for m in range(len(y_p) - 1):
                    xp1, xp2 = x_p[k], x_p[k + 1]
                    yp1, yp2 = y_p[m], y_p[m + 1]

                    dxp = xp2 - xp1
                    dyp = yp2 - yp1

                    integrand_values = [
                        integr(x_i, y_j, x_p_value, y_p_value)
                        for x_p_value in (xp1, xp2)
                        for y_p_value in (yp1, yp2)
                    ]

                    conv[i, j] += sum(integrand_values) * dxp * dyp

            print(f'Convolution at {x_i}, {y_j} completed')

    return conv


# Create grid of points with spatial coordinates
x_val, y_val = np.linspace(-10, 10, 20), np.linspace(-10, 10, 20)
xx, yy = np.meshgrid(x_val, y_val)

xp_val, yp_val = np.linspace(-10, 10, 20), np.linspace(-10, 10, 20)
xxp, yyp = np.meshgrid(xp_val, yp_val)

# Calculate point density in integration interval
int_den_x = len(x_val) / (x_val[-1] - x_val[0])
int_den_y = len(y_val) / (y_val[-1] - y_val[0])

# Set box width
width_x = 2
width_y = 2

# Calculate numerical solution
conv_result_numeric = convolution(integrand_quad, x_val, y_val, xp_val, yp_val)

# Calculate using inbuilt convolution function
conv_result_conv2d = convolve2d(boxcar2dim(xx, yy, 0, 0),
                                xx ** 2 * yy ** 2,
                                mode='same',
                                boundary='wrap') / (int_den_x * int_den_y)

# Create a 3D plot
fig = plt.figure(figsize=(15, 12))

# First row
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(xx, yy, conv_result_numeric, cmap='viridis', alpha=0.8)
ax1.set_title('Convolution (Numeric)')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(xx, yy, conv_result_conv2d, cmap='viridis', alpha=0.8)
ax2.set_title('Convolution (Inbuilt)')

# Set labels
for ax in [ax1, ax2]:
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

plt.suptitle('Convolution theorem')
plt.tight_layout()
plt.show()
