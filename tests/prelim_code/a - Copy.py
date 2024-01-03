import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from mpl_toolkits.mplot3d import Axes3D

def boxcar2D(x, y, w):
    return np.where((np.abs(x) <= w / 2) & (np.abs(y) <= w / 2), 1, 0)

# Create a grid of points in 2D space
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)

# Set the width of the 2D boxcar functions
width_w = 2

# Generate the 2D boxcar functions
boxcar1 = boxcar2D(x, y, width_w)
boxcar2 = boxcar2D(x, y, width_w)

# Convolve the 2D boxcar functions using convolution2d
convolution_result_conv = convolve2d(boxcar1, boxcar2, mode='same') / np.sum(boxcar1)

# Perform Fourier transform on the boxcar functions
fft_boxcar1 = fft2(boxcar1)
fft_boxcar2 = fft2(boxcar2)

# Apply fftshift to bring zero frequency components to the center
fft_boxcar1 = fftshift(fft_boxcar1)
fft_boxcar2 = fftshift(fft_boxcar2)

# Multiply the Fourier transforms
fft_result = fft_boxcar1 * fft_boxcar2

# Inverse Fourier transform to obtain the convolution result
convolution_result_fft = np.real(ifft2(ifftshift(fft_result))) / np.sum(boxcar1)

# Create a 3D plot
fig = plt.figure(figsize=(15, 6))

# Plot the convolution result obtained using convolution2d
z = np.sinc(x*np.pi/width_w) ** 2 * np.sinc(y*np.pi/width_w) ** 2
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
ax1.set_title('Convolution Result (Convolution2D)')

# Plot the Fourier space multiplication result
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x, y, convolution_result_fft, cmap='viridis', alpha=0.8)
ax2.set_title('Convolution Result (Fourier Space Multiplication)')

# Set labels
for ax in [ax1, ax2]:
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Convolution Result')

plt.suptitle('Convolution2D vs. Fourier Space Multiplication')
plt.tight_layout()
plt.show()
