import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.fft import *
from tests.prelim_code.prelim_utilis import *
from tests.prelim_code.tst_example_fields import *

# Create grid of points in spatial and fourier coordinates
x_val, y_val = np.linspace(-10, 10, 100), np.linspace(-10, 10, 100)
xx, yy = np.meshgrid(x_val, y_val)

meshsize_x, meshsize_y = x_val[1] - x_val[0], y_val[1] - y_val[0]

k_x, k_y = np.fft.fftfreq(x_val.shape[0], d=meshsize_x), np.fft.fftfreq(y_val.shape[0], d=meshsize_y)
kxx, kyy = np.meshgrid(k_x, k_y)

# Calculate point density in integration interval
int_den_x = len(x_val) / (x_val[-1] - x_val[0])
int_den_y = len(y_val) / (y_val[-1] - y_val[0])

# Set box width
width_x = 2
width_y = 2

# Generate 2D box function
boxcar2D = boxcar2dim(xx, yy, width_x, width_y)

# Analytic Fourier transform of 2D box function
boxcar2D_fft = boxcar2dim_ft(kxx, kyy, width_x, width_y)

# Generate pyramid function
pyramid2D = pyramid2dim(xx, yy, width_x, width_y)

# Analytic Fourier transform of pyramid function
pyramid2D_fft = pyramid2dim_ft(kxx, kyy, width_x, width_y)

# Convolve the boxcar functions where the result has size 2*len(x_val) - 1
# Scaled by point density in integration interval
convolution_result = convolve2d(boxcar2D, boxcar2D, mode='full', boundary='fill', fillvalue=0) / (int_den_x * int_den_y)

# Trim convolution result to same size as input functions
convolution_result = center_padding(convolution_result, xx, yy)

# Pad the boxcar functions with zeros to same length as output of convolve
# Reason:  Product of two vector DFTs is DFT of their circular convolution, not their linear convolution
# Both convolution types are equal if vectors for circular conv. are extended to length of linear convolution output
boxcar_sci_fft = np.pad(boxcar2D,
                        pad_width=((0, len(x_val) - 1), (0, len(y_val) - 1)),
                        mode='constant',
                        constant_values=(0, 0))

# Perform Fourier transform on the boxcar functions
boxcar_sci_fft = fft2(boxcar_sci_fft)

# Multiply the Fourier transforms
fft_sci_result = boxcar_sci_fft * boxcar_sci_fft

# Bring zero frequency components to the center
fft_sci_result = fftshift(fft_sci_result)

# Inverse Fourier transform to obtain the convolution result
# Scaled by point density in integration interval
sci_result = np.real(ifft2(ifftshift(fft_sci_result))) / (int_den_x * int_den_y)

# Trim convolution result to same size as input functions
center_i = (np.shape(sci_result)[0]) // 2
center_j = (np.shape(sci_result)[1]) // 2
sci_result = center_padding(sci_result, xx, yy)

# Create a 3D plot
fig = plt.figure(figsize=(15, 12))

# First row
ax1 = fig.add_subplot(241, projection='3d')
ax1.plot_surface(xx, yy, boxcar2D, cmap='viridis', alpha=0.8)
ax1.set_title('2D box shape function b(x,y)')

ax2 = fig.add_subplot(242, projection='3d')
ax2.plot_surface(xx, yy, convolution_result, cmap='viridis', alpha=0.8)
ax2.set_title('Auto-convolution (SUM) of b(x,y)')

ax3 = fig.add_subplot(243, projection='3d')
ax3.plot_surface(xx, yy, sci_result, cmap='viridis', alpha=0.8)
ax3.set_title('Auto-convolution (FFT) of b(x,y)')

ax4 = fig.add_subplot(244, projection='3d')
ax4.plot_surface(xx, yy, pyramid2D, cmap='viridis', alpha=0.8)
ax4.set_title('Auto-convolution (Analytic) of b(x,y)')

# Second row - Fourier transforms
ax5 = fig.add_subplot(245, projection='3d')
ax5.plot_surface(kxx, kyy, boxcar2D_fft, cmap='viridis', alpha=0.8)
# ax5.set_title('Fourier Transform of b(x,y)')

ax6 = fig.add_subplot(246, projection='3d')
ax6.plot_surface(kxx, kyy, np.abs(fft2(convolution_result)) / (int_den_x * int_den_y), cmap='viridis', alpha=0.8)
# ax6.set_title('Fourier Transform of b(x,y)')

ax7 = fig.add_subplot(247, projection='3d')
ax7.plot_surface(kxx, kyy, np.abs(fft2(sci_result)) / (int_den_x * int_den_y), cmap='viridis', alpha=0.8)
# ax7.set_title('Product of Fourier Transforms')

ax8 = fig.add_subplot(248, projection='3d')
ax8.plot_surface(kxx, kyy, np.abs(pyramid2D_fft), cmap='viridis', alpha=0.8)
# ax8.set_title('Auto-convolution (Analytic) of b(x,y)')

# Set labels
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

for ax in [ax5, ax6, ax7, ax8]:
    ax.set_xlabel('k_x-axis')
    ax.set_ylabel('k_y-axis')

plt.suptitle('Convolution theorem')
plt.tight_layout()
plt.show()
