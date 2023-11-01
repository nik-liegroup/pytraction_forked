import numpy as np
import matplotlib.pyplot as plt
import os

from pytraction import (TractionForceConfig, process_stack, plot, Dataset)

pix_per_mu = 9.64  # The number of pixels per micron
E = 1000  # Young's modulus in Pa
config_path = os.path.join('.', 'config', 'config.yaml')

img_path = '1kPa_PAA_Gel_1_Before_Tryp_Pos9.tif'
ref_path = '1kPa_PAA_Gel_1_After_Tryp_Pos9.tif'
roi_path  = '1kPa_PAA_Gel_1_Pos9.roi'
traction_config = TractionForceConfig(E=E, scaling_factor=pix_per_mu, min_window_size=64, config=config_path)
img, ref, roi = traction_config.load_data(img_path, ref_path, roi_path)

print(f'The expected shape of the image is {img.shape}')
print(f'The expected shape of the reference is {ref.shape}')

fig, ax = plt.subplots(2,2)
ax[0,0].set_title('img frame 0 channel 0 (beads)')
ax[0,0].imshow(img[0,0,:,:], cmap='gray')
ax[0,0].set_axis_off()

ax[0,1].set_title('img frame 0 channel 1 (brightfield)')
ax[0,1].imshow(img[0,1,:,:], cmap='gray')
ax[0,1].set_axis_off()

ax[1,0].set_title('ref channel 0 (beads)')
ax[1,0].imshow(ref[0,:,:], cmap='gray')
ax[1,0].set_axis_off()

ax[1,1].set_title('ref channel 1 (cell)')
ax[1,1].imshow(ref[1,:,:], cmap='gray')
ax[1,1].set_axis_off()

plt.tight_layout()
#plt.show()

log = process_stack(img[:1, :, :, :], ref, traction_config, roi=roi, crop=False)
plot(log, frame=0, mask=True)
plt.show()

log.save('1kPa_PAA_Gel_Tryp_Before_Position_6.h5')

dataset_loaded = Dataset('1kPa_PAA_Gel_Tryp_Before_Position_6.h5')
print(dataset_loaded)