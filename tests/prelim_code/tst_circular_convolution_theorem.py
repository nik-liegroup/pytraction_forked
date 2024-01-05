import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.fft import *
from tests.prelim_code.tst_utilis import *


# Create grid of points in spatial coordinates for convolution
x_val = np.linspace(-5, 5, 50)
y_val = np.linspace(-5, 5, 50)
xx, yy = np.meshgrid(x_val, y_val)

# Calculate meshsize and pyramid width
meshsize_x = x_val[1] - x_val[0]
meshsize_y = x_val[1] - x_val[0]
width_x = 2 * meshsize_x
width_y = 2 * meshsize_y

# Generate 2D boxcar functions
boxcar1_conv = boxcar2dim(xx, yy, width_x, width_y)
boxcar2_conv = boxcar2dim(xx, yy, width_x, width_y)

# Generate 2D pyramid function
pyramid_conv = pyramid2dim(xx, yy, width_x, width_y)

# Analytic Fourier transform of 2D pyramid function
k_x = np.fft.fftfreq(xx.shape[0], d=meshsize_x)
k_y = np.fft.fftfreq(yy.shape[1], d=meshsize_y)
kxx, kyy = np.meshgrid(k_x, k_y)
pyramid_conv_fft = pyramid2dim_ft(kxx, kyy, width_x, width_y)

# Convolve the 2D boxcar functions where the result has size 2*len(x_val)*len(y_val) - 1)
convolution_result_conv = convolve2d(boxcar1_conv, boxcar2_conv, mode='full', boundary='fill', fillvalue=0)

# Trim convolution result to same size as input functions
convolution_result_conv = center_padding(convolution_result_conv, x_val, y_val)

# Pad the 2D boxcar functions with zeros to same length as output of convolve2d
# Reason:  Product of two vector DFTs is DFT of their circular convolution, not their linear convolution
# Both convolution types are equal if vectors for circular conv. are extended to length of linear convolution output
boxcar1_ft = np.pad(boxcar1_conv, pad_width=(0, len(x_val) - 1), mode='constant', constant_values=(0, 0))
boxcar2_ft = np.pad(boxcar2_conv, pad_width=(0, len(x_val) - 1), mode='constant', constant_values=(0, 0))

# Perform Fourier transform on the boxcar functions
fft_boxcar1 = fft2(boxcar1_ft)
fft_boxcar2 = fft2(boxcar2_ft)

# Multiply the Fourier transforms
fft_result = fft_boxcar1 * fft_boxcar2

# Bring zero frequency components to the center
fft_result = fftshift(fft_result)

# Inverse Fourier transform to obtain the convolution result
convolution_result_fft = np.real(ifft2(ifftshift(fft_result)))

# Trim convolution result to same size as input functions
convolution_result_fft = center_padding(convolution_result_fft, x_val, y_val)

# Create a 3D plot
fig = plt.figure(figsize=(15, 12))

# First row
ax1 = fig.add_subplot(241, projection='3d')
ax1.plot_surface(xx, yy, boxcar1_conv, cmap='plasma', alpha=0.8)
ax1.set_title('2D box shape function b(x,y)')

ax2 = fig.add_subplot(242, projection='3d')
ax2.plot_surface(xx, yy, convolution_result_conv, cmap='viridis', alpha=0.8)
ax2.set_title('Auto-convolution (SUM) of b(x,y)')

ax3 = fig.add_subplot(243, projection='3d')
ax3.plot_surface(xx, yy, convolution_result_fft, cmap='viridis', alpha=0.8)
ax3.set_title('Auto-convolution (FFT) of b(x,y)')

ax4 = fig.add_subplot(244, projection='3d')
ax4.plot_surface(xx, yy, pyramid_conv, cmap='plasma', alpha=0.8)
ax4.set_title('Auto-convolution (Analytic) of b(x,y)')

# Second row - Fourier transforms
ax5 = fig.add_subplot(245, projection='3d')
ax5.plot_surface(xx, yy, np.abs(fftshift(fft2(boxcar1_conv, norm='forward'))), cmap='viridis', alpha=0.8)
# ax5.set_title('Fourier Transform of b(x,y)')

ax6 = fig.add_subplot(246, projection='3d')
ax6.plot_surface(xx, yy, np.abs(fftshift(fft2(convolution_result_conv, norm='forward'))), cmap='viridis', alpha=0.8)
# ax6.set_title('Fourier Transform of b(x,y)')

ax7 = fig.add_subplot(247, projection='3d')
ax7.plot_surface(xx, yy, np.abs(fftshift(fft2(convolution_result_fft, norm='forward'))), cmap='viridis', alpha=0.8)
# ax7.set_title('Product of Fourier Transforms')

ax8 = fig.add_subplot(248, projection='3d')
ax8.plot_surface(kxx, kyy, pyramid_conv_fft, cmap='plasma', alpha=0.8)
# ax8.set_title('Auto-convolution (Analytic) of b(x,y)')

# Set labels
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Result')

plt.suptitle('Convolution2D vs. Fourier Space Multiplication')
plt.tight_layout()
plt.show()
