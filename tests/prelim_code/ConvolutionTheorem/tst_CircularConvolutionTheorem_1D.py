import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.fft import *
from tests.prelim_code.prelim_utilis import *
from tests.prelim_code.tst_example_fields import *


# Create grid of points in spatial and fourier coordinates
x_val = np.linspace(-10, 10, 1000)
meshsize_x = x_val[1] - x_val[0]
k_x = np.fft.fftfreq(x_val.shape[0], d=meshsize_x)

# Calculate point density in integration interval
int_den_x = len(x_val) / (x_val[-1] - x_val[0])

# Set box width
width_x = 2

# Generate box function
boxcar = np.where((np.abs(x_val) <= width_x / 2), 1, 0)

# Analytic Fourier transform of box function (Scaled by point density in integration interval)
boxcar_fft = width_x * np.sinc(k_x * width_x)

# Generate pyramid function
pyramid = np.maximum(0, width_x - np.abs(x_val))

# Analytic Fourier transform of pyramid function
pyramid_fft = (width_x * np.sinc(k_x * width_x)) ** 2

# Convolve the boxcar functions where the result has size 2*len(x_val) - 1
convolution_result = convolve(boxcar, boxcar, mode='full', method='direct') / int_den_x

# Trim convolution result to same size as input functions
center_i = (np.shape(convolution_result)[0]) // 2
convolution_result = convolution_result[center_i - len(x_val) // 2: center_i + len(x_val) // 2]

# Pad the boxcar functions with zeros to same length as output of convolve
# Reason:  Product of two vector DFTs is DFT of their circular convolution, not their linear convolution
# Both convolution types are equal if vectors for circular conv. are extended to length of linear convolution output
boxcar_sci_fft = np.pad(boxcar, pad_width=(0, len(x_val) - 1), mode='constant', constant_values=0)

# Perform Fourier transform on the boxcar functions
boxcar_sci_fft = fft(boxcar_sci_fft)

# Multiply the Fourier transforms
fft_sci_result = boxcar_sci_fft * boxcar_sci_fft

# Bring zero frequency components to the center
fft_sci_result = fftshift(fft_sci_result)

# Inverse Fourier transform to obtain the convolution result
sci_result = np.real(ifft(ifftshift(fft_sci_result))) / int_den_x

# Trim convolution result to same size as input functions
center_i = (np.shape(sci_result)[0]) // 2
sci_result = sci_result[center_i - len(x_val) // 2: center_i + len(x_val) // 2]

# Create a plot
fig = plt.figure(figsize=(15, 12))

# First row
ax1 = fig.add_subplot(241)
ax1.plot(x_val, boxcar, alpha=0.8)
ax1.set_title('1D box shape function b(x)')

ax2 = fig.add_subplot(242)
ax2.plot(x_val, convolution_result, alpha=0.8)
ax2.set_title('Auto-convolution (SUM) of b(x)')

ax3 = fig.add_subplot(243)
ax3.plot(x_val, sci_result, alpha=0.8)
ax3.set_title('Auto-convolution (FFT) of b(x)')

ax4 = fig.add_subplot(244)
ax4.plot(x_val, pyramid, alpha=0.8)
ax4.set_title('Auto-convolution (Analytic) of b(x)')

# Second row - Fourier transforms
ax5 = fig.add_subplot(245)
ax5.plot(x_val, fftshift(boxcar_fft), alpha=0.8)
# ax5.set_title('Fourier Transform of b(x,y)')

ax6 = fig.add_subplot(246)
ax6.plot(x_val, np.abs(fftshift(fft(convolution_result))) / int_den_x, alpha=0.8)
# ax6.set_title('Fourier Transform of b(x,y)')

ax7 = fig.add_subplot(247)
ax7.plot(x_val, np.abs(fftshift(fft(sci_result))) / int_den_x, alpha=0.8)
# ax7.set_title('Product of Fourier Transforms')

ax8 = fig.add_subplot(248)
ax8.plot(k_x, pyramid_fft, alpha=0.8)
# ax8.set_title('Auto-convolution (Analytic) of b(x,y)')

# Set labels
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

for ax in [ax5, ax6, ax7, ax8]:
    ax.set_xlabel('k_x-axis')
    ax.set_ylabel('k_y-axis')

plt.suptitle('Convolution vs. Fourier Space Multiplication')
plt.tight_layout()
plt.show()
