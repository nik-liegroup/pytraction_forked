import matplotlib.pyplot as plt
import numpy as np

alpha = np.linspace(0.1, 10, 1000)
logevidence = np.sin(alpha)

grad_logevidence = np.gradient(logevidence, alpha)
angle_logevidence = -np.rad2deg(np.arctan(grad_logevidence))

angle_diff = np.abs(angle_logevidence - 45)
idx = angle_diff.argmin()
alpha_val = logevidence[idx]

#plt.plot(xpoints, ypoints, 'r')
plt.plot(alpha, 10*logevidence, 'r')
plt.plot(alpha, angle_logevidence, 'g')
plt.axvline(x=alpha_val, color='b')
plt.show()
